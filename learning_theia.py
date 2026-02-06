#!/usr/bin/env python3

import os
import sys
import argparse
import zipfile
import pickle
import json
from pathlib import Path

def parse_args_early():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids (e.g. 0,1,2); default empty = use all available")
    args, _ = parser.parse_known_args()
    return args.gpus


gpus = parse_args_early()
if gpus and gpus.strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import random
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from sklearn.utils import class_weight
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from gnn import GCN

STEP_LLM_DIR = os.path.dirname(script_dir)
BIGDATA_DIR = os.path.join(STEP_LLM_DIR, "BIGDATA")
DEFAULT_ARTIFACTS_DIR = os.path.join(BIGDATA_DIR, "FLASH_artifacts")

DATASET = "THEIA"
MODEL_36_EPOCHS = 50
MODEL_36_HYPERPARAMETERS = {
    "units": 32,
    "num_layers": 4,
    "dropout": 0.5,
    "num_neighbors": "-1,-1",
    "learning_rate": 0.01,
    "batch_size": 50000,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Train baseline model.")
    parser.add_argument("--autoprov", action="store_true", help="Train AutoProv model")
    parser.add_argument("--artifacts_dir", type=str, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--embedding", type=str, default="word2vec", choices=["word2vec", "mpnet", "minilm", "roberta", "distilbert", "fasttext"])
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids; default empty = use all available GPUs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for baseline (default 20 so epoch 19 checkpoint exists for eval; ignored for autoprov)")
    return parser.parse_args()


def load_pickle_maybe_zipped(base_dir, stem):
    base = Path(base_dir)
    pkl_path = base / f"{stem}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    zip_path = base / f"{stem}.pkl.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m and not m.endswith("/")]
            members = sorted(members, key=lambda s: (s.count("/"), len(s)))
            if not members:
                raise FileNotFoundError(f"No member in {zip_path}")
            return pickle.loads(zf.read(members[0]))
    raise FileNotFoundError(f"Missing {pkl_path} or {zip_path}")


def zip_artifact(file_path):
    path = Path(file_path)
    if not path.is_file():
        return
    zip_path = path.with_suffix(path.suffix + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(path, arcname=path.name)
    path.unlink()


def zip_all_in_dir(dir_path, extensions=(".pth", ".json")):
    d = Path(dir_path)
    if not d.is_dir():
        return
    for f in d.iterdir():
        if f.is_file() and f.suffix in extensions and not f.name.endswith(".zip"):
            zip_artifact(f)


def run_baseline_training(args, base_dir, model_dir, labels, edges, nodes, device, data_device):
    dataset = DATASET
    labels = np.array(labels) if isinstance(labels, list) else labels
    edges = np.array(edges) if isinstance(edges, list) else edges
    nodes = np.array(nodes) if isinstance(nodes, list) else nodes
    input_dim = nodes.shape[1] if len(nodes) > 0 else 30
    unique_classes = np.unique(labels)
    num_classes = int(unique_classes.max() + 1) if labels.size > 0 else 2

    num_neighbors = [int(x) for x in "-1,-1".split(",")]
    batch_size = 25000
    units, num_layers, dropout = 32, 2, 0.5

    model = GCN(input_dim, num_classes, hidden_units=units, num_layers=num_layers, dropout_rate=dropout).to(device)
    if torch.cuda.device_count() > 1 and args.data_gpu is None:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    l = np.array(labels)
    u = np.unique(l)
    if len(u) < num_classes:
        cw = np.ones(num_classes)
        for c in u:
            cw[c] = len(l) / (len(u) * np.sum(l == c))
    else:
        cw = class_weight.compute_class_weight(class_weight="balanced", classes=u, y=l)
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)
    criterion = CrossEntropyLoss(weight=class_weights, reduction="mean")

    graph = Data(
        x=torch.tensor(nodes, dtype=torch.float).to(data_device),
        y=torch.tensor(labels, dtype=torch.long).to(data_device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(data_device),
    )
    graph.n_id = torch.arange(graph.num_nodes).to(data_device)
    mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool).to(data_device)

    epochs = args.epochs
    best_loss = float("inf")
    patience = 10
    patience_counter = 0
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        loader = NeighborLoader(graph, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=mask)
        total_loss = 0.0
        for subg in tqdm(loader, desc=f"Train epoch {epoch}"):
            model.train()
            optimizer.zero_grad()
            if data_device != device:
                subg = subg.to(device)
            out = model(subg.x, subg.edge_index)
            loss = criterion(out, subg.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * subg.batch_size
        del loader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        loader = NeighborLoader(graph, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=mask)
        for subg in tqdm(loader, desc=f"Validate epoch {epoch}"):
            model.eval()
            with torch.no_grad():
                if data_device != device:
                    subg = subg.to(device)
                out = model(subg.x, subg.edge_index)
                sorted_vals, indices = out.sort(dim=1, descending=True)
                conf = (sorted_vals[:, 0] - sorted_vals[:, 1]) / sorted_vals[:, 0]
                conf = (conf - conf.min()) / (conf.max() - conf.min())
                pred = indices[:, 0]
                cond = (pred == subg.y) | (conf >= 0.53)
                if data_device != device:
                    mask[subg.n_id.to(data_device)[cond.to(data_device)]] = False
                else:
                    mask[subg.n_id[cond]] = False
        del loader

        current_loss = total_loss / (graph.num_nodes if graph.num_nodes > 0 else 1)
        remaining = mask.sum().item()

        torch.save(model.state_dict(), os.path.join(model_dir, f"word2vec_gnn_{dataset}_{epoch}_E3.pth"))
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, f"word2vec_gnn_{dataset}_best_E3.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience and remaining == 0:
                break

    zip_all_in_dir(model_dir, (".pth",))


def run_autoprov_training(args, base_dir, model_dir, labels, edges, nodes, device, data_device):
    dataset = DATASET
    h = MODEL_36_HYPERPARAMETERS
    labels = np.array(labels) if isinstance(labels, list) else labels
    edges = np.array(edges) if isinstance(edges, list) else edges
    nodes = np.array(nodes) if isinstance(nodes, list) else nodes
    input_dim = nodes.shape[1] if len(nodes) > 0 else 30
    unique_classes = np.unique(labels)
    num_classes = int(unique_classes.max() + 1) if labels.size > 0 else 2

    num_neighbors = [int(x) for x in h["num_neighbors"].split(",")]
    batch_size = h["batch_size"]
    units = h["units"]
    num_layers = h["num_layers"]
    dropout = h["dropout"]
    lr = h["learning_rate"]

    model = GCN(input_dim, num_classes, hidden_units=units, num_layers=num_layers, dropout_rate=dropout).to(device)
    if torch.cuda.device_count() > 1 and args.data_gpu is None:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    l = np.array(labels)
    u = np.unique(l)
    if len(u) < num_classes:
        cw = np.ones(num_classes)
        for c in u:
            cw[c] = len(l) / (len(u) * np.sum(l == c))
    else:
        cw = class_weight.compute_class_weight(class_weight="balanced", classes=u, y=l)
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)
    criterion = CrossEntropyLoss(weight=class_weights, reduction="mean")

    graph = Data(
        x=torch.tensor(nodes, dtype=torch.float).to(data_device),
        y=torch.tensor(labels, dtype=torch.long).to(data_device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(data_device),
    )
    graph.n_id = torch.arange(graph.num_nodes).to(data_device)
    mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool).to(data_device)

    epochs = MODEL_36_EPOCHS
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "hyperparameters.json"), "w") as f:
        json.dump(h, f, indent=2)

    for epoch in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        loader = NeighborLoader(graph, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=mask)
        total_loss = 0.0
        for subg in tqdm(loader, desc=f"Model 36 epoch {epoch}"):
            model.train()
            optimizer.zero_grad()
            if data_device != device:
                subg = subg.to(device)
            out = model(subg.x, subg.edge_index)
            loss = criterion(out, subg.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * subg.batch_size
        del loader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        loader = NeighborLoader(graph, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=mask)
        for subg in loader:
            model.eval()
            with torch.no_grad():
                if data_device != device:
                    subg = subg.to(device)
                out = model(subg.x, subg.edge_index)
                sorted_vals, indices = out.sort(dim=1, descending=True)
                conf = (sorted_vals[:, 0] - sorted_vals[:, 1]) / sorted_vals[:, 0]
                conf = (conf - conf.min()) / (conf.max() - conf.min())
                pred = indices[:, 0]
                cond = (pred == subg.y) | (conf >= 0.53)
                if data_device != device:
                    mask[subg.n_id.to(data_device)[cond.to(data_device)]] = False
                else:
                    mask[subg.n_id[cond]] = False
        del loader

        current_loss = total_loss / (graph.num_nodes if graph.num_nodes > 0 else 1)
        remaining = mask.sum().item()
        torch.save(model.state_dict(), os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    zip_all_in_dir(model_dir, (".pth", ".json"))


def main():
    args = parse_args()
    if args.baseline and args.autoprov:
        print("Error: use exactly one of --baseline or --autoprov")
        sys.exit(1)
    if not args.baseline and not args.autoprov:
        print("Error: must specify --baseline or --autoprov")
        sys.exit(1)

    dataset = DATASET
    artifacts_dir = os.path.abspath(args.artifacts_dir)
    embedding = args.embedding

    if args.baseline:
        suffix = ""
        atr_dir = "train_attr"
    else:
        suffix = "_rulellm_llmlabel"
        atr_dir = "train_attr_rulellm_llmlabel"

    base_dir = f"{artifacts_dir}/{dataset}_graphs{suffix}/{embedding}/{atr_dir}"
    if not os.path.isdir(base_dir):
        print(f"Error: base_dir not found: {base_dir}")
        sys.exit(1)

    labels = load_pickle_maybe_zipped(base_dir, "labels")
    edges = load_pickle_maybe_zipped(base_dir, "edges")
    nodes = load_pickle_maybe_zipped(base_dir, "nodes")

    nodes_arr = np.array(nodes) if isinstance(nodes, list) else nodes
    labels_arr = np.array(labels) if isinstance(labels, list) else labels
    input_dim = int(nodes_arr.shape[1]) if nodes_arr.size > 0 else 30
    num_classes = int(np.unique(labels_arr).max() + 1) if labels_arr.size > 0 else 2

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)
    data_device = device
    if args.gpus and args.gpus.strip():
        visible_gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
        if visible_gpus and args.data_gpu is not None and torch.cuda.is_available():
            num_visible = len(visible_gpus)
            if args.data_gpu in visible_gpus:
                data_idx = visible_gpus.index(args.data_gpu)
                model_idx = 0 if data_idx != 0 else (1 if num_visible > 1 else 0)
            else:
                data_idx = model_idx = 0
            data_device = torch.device(f"cuda:{data_idx}")
            device = torch.device(f"cuda:{model_idx}")
    if args.baseline:
        model_dir = os.path.join(base_dir, "model")
        run_baseline_training(args, base_dir, model_dir, labels, edges, nodes, device, data_device)
    else:
        model_dir = os.path.join(base_dir, "hypersearch_models", "model_36")
        run_autoprov_training(args, base_dir, model_dir, labels, edges, nodes, device, data_device)


if __name__ == "__main__":
    main()
