#!/usr/bin/env python3

import os
import sys
import json
import io
import gc
import argparse
import warnings
import zipfile
import pickle
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm

from gnn import GCN, load_from_zip

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        device = torch.device("cuda:0")
        use_data_parallel = True
    else:
        device = torch.device("cuda:0")
        use_data_parallel = False
else:
    device = torch.device("cpu")
    use_data_parallel = False

DATASET = "THEIA"
BASELINE_EPOCH = 19
AUTOPROV_MODEL_ID = 36
AUTOPROV_EPOCH = 50

DATASET_DATES = {
    "theia": {
        "train_start_date": "2018-04-03",
        "train_end_date": "2018-04-05",
        "test_start_date": "2018-04-09",
        "test_end_date": "2018-04-12",
    },
}


def get_dataset_dates(dataset: str, train_start_date: str = None, train_end_date: str = None,
                      test_start_date: str = None, test_end_date: str = None):
    dataset_lower = dataset.lower()
    defaults = DATASET_DATES.get(dataset_lower, {})
    return {
        "train_start_date": train_start_date or defaults.get("train_start_date"),
        "train_end_date": train_end_date or defaults.get("train_end_date"),
        "test_start_date": test_start_date or defaults.get("test_start_date"),
        "test_end_date": test_end_date or defaults.get("test_end_date"),
    }


def get_attack_scenarios(dataset):
    step_llm_dir = os.path.dirname(script_dir)
    pids_gt_bigdata = os.path.join(step_llm_dir, "BIGDATA", "PIDS_GT")
    pids_gt_dir = pids_gt_bigdata if os.path.isdir(pids_gt_bigdata) else os.path.join(step_llm_dir, "PIDS_GT")
    attacks = {
        "THEIA": [
            ("Firefox_Backdoor_Drakon", os.path.join(pids_gt_dir, "THEIA", "node_Firefox_Backdoor_Drakon_In_Memory.csv"), "2018-04-10"),
            ("Browser_Extension_Drakon", os.path.join(pids_gt_dir, "THEIA", "node_Browser_Extension_Drakon_Dropper.csv"), "2018-04-12")
        ],
    }
    return attacks.get(dataset.upper(), [])


def load_malicious_ids_from_csv(csv_path: str) -> set:
    malicious_ids = set()
    with open(csv_path, "r") as f:
        for line in f:
            entry = line.strip()
            if entry:
                malicious_ids.add(entry.split(",")[0])
    return malicious_ids


def find_best_f1_metrics(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    y_pred = (y_scores >= best_threshold).astype(int)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "threshold": best_threshold,
    }


def parse_num_neighbors(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def clean_state_dict(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def infer_gcn_architecture(state_dict: dict, num_classes: int) -> Tuple[int, int]:
    for suffix in ("lin_rel.weight", "lin_l.weight"):
        weight_keys = [
            key
            for key in state_dict.keys()
            if key.startswith("convs.") and key.endswith(suffix)
        ]
        if weight_keys:
            break
    if not weight_keys:
        raise ValueError("Unable to infer architecture from state_dict")
    weight_keys.sort()
    first_weight = state_dict[weight_keys[0]]
    hidden_units = first_weight.shape[0]
    final_weight = state_dict[weight_keys[-1]]
    if final_weight.shape[0] != num_classes:
        raise ValueError(
            f"Final layer output mismatch: expected {num_classes}, got {final_weight.shape[0]}"
        )
    num_layers = len(weight_keys)
    return hidden_units, num_layers


def compute_adp(
    all_confidences_by_uuid: Dict[str, float],
    uuid_attack_map: Dict[str, set],
    total_attacks: int,
) -> float:
    if not all_confidences_by_uuid:
        return 0.0
    uuids = list(all_confidences_by_uuid.keys())
    scores = np.array([all_confidences_by_uuid[uuid] for uuid in uuids], dtype=np.float64)
    num_nodes = scores.size
    if num_nodes == 0:
        return 0.0
    index_to_attacks: Dict[int, set] = {}
    attack_names: set = set()
    for idx, uuid in enumerate(uuids):
        attacks = uuid_attack_map.get(uuid)
        if attacks:
            index_to_attacks[idx] = set(attacks)
            attack_names.update(attacks)
    if not attack_names:
        return 0.0
    total_attacks = len(attack_names)
    y_global = np.zeros(num_nodes, dtype=np.int8)
    for idx in index_to_attacks.keys():
        y_global[idx] = 1
    sorted_idx = np.argsort(-scores)
    tp = 0.0
    fp = 0.0
    seen_attacks: set = set()
    precisions: List[float] = [0.0]
    detections: List[float] = [0.0]
    for rank, original_idx in enumerate(sorted_idx):
        if y_global[original_idx] == 1:
            tp += 1.0
        else:
            fp += 1.0
        if original_idx in index_to_attacks:
            seen_attacks.update(index_to_attacks[original_idx])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        detection_fraction = len(seen_attacks) / total_attacks
        precisions.append(precision)
        detections.append(detection_fraction)
    precisions_arr = np.array(precisions, dtype=np.float64)
    detections_arr = np.array(detections, dtype=np.float64)
    precision_to_detection: Dict[float, float] = {}
    for precision, detection in zip(precisions_arr, detections_arr):
        key = round(float(precision), 6)
        if key in precision_to_detection:
            precision_to_detection[key] = max(precision_to_detection[key], detection)
        else:
            precision_to_detection[key] = detection
    unique_precisions = np.array(sorted(precision_to_detection.keys()), dtype=np.float64)
    unique_detections = np.array(
        [precision_to_detection[key] for key in unique_precisions], dtype=np.float64
    )
    if unique_precisions.size == 0:
        return 0.0
    if unique_precisions[0] > 0.0:
        unique_precisions = np.insert(unique_precisions, 0, 0.0)
        unique_detections = np.insert(unique_detections, 0, 0.0)
    if unique_precisions[-1] < 1.0:
        unique_precisions = np.append(unique_precisions, 1.0)
        unique_detections = np.append(unique_detections, unique_detections[-1])
    adp_val = np.trapz(unique_detections, unique_precisions)
    return float(max(0.0, min(1.0, adp_val)))


def load_training_artifacts(base_dir: Path) -> Tuple[int, int]:
    labels = load_pickle_maybe_zipped(base_dir, "labels")
    nodes = load_pickle_maybe_zipped(base_dir, "nodes")
    labels = np.array(labels) if isinstance(labels, list) else labels
    nodes = np.array(nodes) if isinstance(nodes, list) else nodes
    if nodes.size == 0:
        raise ValueError(f"No training nodes found in {base_dir}")
    input_dim = nodes.shape[1]
    num_classes = int(labels.max() + 1) if labels.size > 0 else 2
    return input_dim, num_classes


def load_test_windows(
    args,
    dataset_upper: str,
    suffix: str,
    data_device: torch.device,
    dataset_lower: str,
) -> Tuple[Dict[str, Data], Dict[str, dict], Path]:
    test_graphs_by_window: Dict[str, Data] = {}
    test_data_by_window: Dict[str, dict] = {}
    base_test_dir = (
        Path(args.artifacts_dir)
        / f"{dataset_upper}_graphs{suffix}"
        / args.embedding
    )
    current_date = datetime.strptime(args.test_start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.test_end_date, "%Y-%m-%d")
    hourly_windows = []
    temp_date = current_date
    while temp_date <= end_date:
        for hour in range(24):
            hourly_windows.append(f"{temp_date.strftime('%Y-%m-%d')}_{hour:02d}")
        temp_date += timedelta(days=1)
    for window_key in tqdm(hourly_windows, desc="Loading test windows", leave=False):
        window_dir = base_test_dir / f"test_attr_{window_key}{suffix}"
        if not window_dir.exists():
            continue
        try:
            window_labels = load_pickle_maybe_zipped(window_dir, "labels")
            window_edges = load_pickle_maybe_zipped(window_dir, "edges")
            window_nodes = load_pickle_maybe_zipped(window_dir, "nodes")
            window_mapp = load_pickle_maybe_zipped(window_dir, "mapp")
        except FileNotFoundError:
            continue
        window_labels = np.array(window_labels) if isinstance(window_labels, list) else window_labels
        window_edges = np.array(window_edges) if isinstance(window_edges, list) else window_edges
        window_nodes = np.array(window_nodes) if isinstance(window_nodes, list) else window_nodes
        if window_nodes.size == 0:
            continue
        graph = Data(
            x=torch.tensor(window_nodes, dtype=torch.float).to(data_device),
            y=torch.tensor(window_labels, dtype=torch.long).to(data_device),
            edge_index=torch.tensor(window_edges, dtype=torch.long).to(data_device),
        )
        graph.n_id = torch.arange(graph.num_nodes, device=data_device)
        test_graphs_by_window[window_key] = graph
        test_data_by_window[window_key] = {"mapp": window_mapp}
    results_dir = Path(f"./results{suffix}")
    results_dir.mkdir(exist_ok=True)
    dataset_dir = results_dir / dataset_lower
    dataset_dir.mkdir(exist_ok=True)
    csv_path = dataset_dir / "hyperparameter_search_results.csv"
    return test_graphs_by_window, test_data_by_window, csv_path


def _read_zip_single_file(zip_path: Path) -> Tuple[str, bytes]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m and not m.endswith("/")]
        if not members:
            raise FileNotFoundError(f"No files found inside zip: {zip_path}")
        members_sorted = sorted(members, key=lambda s: (s.count("/"), len(s)))
        member = members_sorted[0]
        return member, zf.read(member)


def _resolve_plain_or_zip(path: Path) -> Path:
    if path.exists():
        return path
    zip_path = Path(str(path) + ".zip")
    if zip_path.exists():
        return zip_path
    raise FileNotFoundError(f"Missing file (and .zip): {path}")


def load_pickle_maybe_zipped(base_dir: Path, stem: str):
    pkl_path = base_dir / f"{stem}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    zip_path = base_dir / f"{stem}.pkl.zip"
    if zip_path.exists():
        _, data = _read_zip_single_file(zip_path)
        return pickle.loads(data)
    alt_zip = base_dir / f"{stem}.zip"
    if alt_zip.exists():
        member, data = _read_zip_single_file(alt_zip)
        if not member.endswith(".pkl"):
            raise ValueError(f"Expected a .pkl inside {alt_zip}, found {member}")
        return pickle.loads(data)
    raise FileNotFoundError(f"Missing artifact: {pkl_path} or {zip_path}")


def load_json_maybe_zipped(path: Path) -> dict:
    resolved = _resolve_plain_or_zip(path)
    if resolved.suffix != ".zip":
        with open(resolved, "r") as f:
            return json.load(f)
    member, data = _read_zip_single_file(resolved)
    if not member.endswith(".json"):
        raise ValueError(f"Expected a .json inside {resolved}, found {member}")
    return json.loads(data.decode("utf-8"))


def torch_load_maybe_zipped(path: Path, *, map_location, weights_only: bool = True):
    resolved = _resolve_plain_or_zip(path)
    if resolved.suffix != ".zip":
        return torch.load(resolved, map_location=map_location, weights_only=weights_only)
    
    gpu_available = False
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gpu_available = True
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            gpu_available = False
        else:
            raise
    gc.collect()
    
    member, data = _read_zip_single_file(resolved)
    bio = io.BytesIO(data)
    result = torch.load(bio, map_location="cpu", weights_only=weights_only)
    
    target_device = str(map_location) if isinstance(map_location, torch.device) else map_location
    if target_device != "cpu" and target_device is not None and gpu_available:
        if isinstance(result, dict):
            moved_result = {}
            try:
                for idx, (k, v) in enumerate(result.items()):
                    if isinstance(v, torch.Tensor):
                        moved_result[k] = v.to(map_location, non_blocking=False)
                        del v
                        if idx % 5 == 0:
                            try:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                            except RuntimeError:
                                pass
                        gc.collect()
                    else:
                        moved_result[k] = v
                result = moved_result
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except RuntimeError:
                    pass
                gc.collect()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
                    gc.collect()
                else:
                    raise
        elif isinstance(result, torch.Tensor):
            try:
                result = result.to(map_location, non_blocking=False)
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    pass
                else:
                    raise
        elif hasattr(result, "to"):
            try:
                result = result.to(map_location)
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    pass
                else:
                    raise
    elif target_device != "cpu" and target_device is not None and not gpu_available:
        pass
    
    return result


def run_inference_full(
    model: torch.nn.Module,
    test_graphs_by_window: Dict[str, Data],
    test_data_by_window: Dict[str, dict],
    attack_scenarios: List[Tuple[str, str, str]],
    device: torch.device,
    data_device: torch.device,
    num_neighbors_list: List[int],
    batch_size: int,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    all_confidences_by_uuid: Dict[str, float] = {}
    uuid_to_windows: Dict[str, set] = defaultdict(set)
    uuid_attack_map: Dict[str, set] = defaultdict(set)
    for window_key in tqdm(
        sorted(test_graphs_by_window.keys()),
        desc="Windows",
        leave=False,
    ):
        test_graph = test_graphs_by_window[window_key]
        test_data = test_data_by_window[window_key]
        test_mapp = test_data["mapp"]
        with torch.no_grad():
            loader = NeighborLoader(
                test_graph,
                num_neighbors=num_neighbors_list,
                batch_size=batch_size,
            )
            test_confidences = torch.zeros(test_graph.num_nodes, device=data_device)
            for subg in loader:
                if data_device != device:
                    subg = subg.to(device)
                out = model(subg.x, subg.edge_index)
                sorted_vals, indices = out.sort(dim=1, descending=True)
                denom = (sorted_vals[:, 0] - sorted_vals[:, 1])
                conf = denom / (sorted_vals[:, 0] + 1e-12)
                conf_max = conf.max()
                conf_min = conf.min()
                if (conf_max - conf_min).abs() > 1e-12:
                    conf = (conf - conf_min) / (conf_max - conf_min)
                else:
                    conf = torch.zeros_like(conf)
                if data_device != device:
                    test_confidences[subg.n_id.to(data_device)] = conf.to(data_device)
                else:
                    test_confidences[subg.n_id] = conf
            del loader
        window_confidences = test_confidences.cpu().numpy()
        for idx, uuid in enumerate(test_mapp):
            uuid_to_windows[uuid].add(window_key)
            if uuid not in all_confidences_by_uuid or window_confidences[idx] > all_confidences_by_uuid[uuid]:
                all_confidences_by_uuid[uuid] = window_confidences[idx]
    results: Dict[str, Dict[str, float]] = {}
    total_attacks = len(attack_scenarios)
    for attack_name, csv_path_attack, attack_date in attack_scenarios:
        GT_mal = load_malicious_ids_from_csv(csv_path_attack)
        GT_mal.intersection_update(all_confidences_by_uuid.keys())
        y_true: List[int] = []
        y_scores: List[float] = []
        for uuid in sorted(all_confidences_by_uuid.keys()):
            uuid_windows = uuid_to_windows.get(uuid, set())
            is_on_attack_date = any(window.startswith(attack_date) for window in uuid_windows)
            if uuid in GT_mal and is_on_attack_date:
                y_true.append(1)
                uuid_attack_map[uuid].add(attack_name)
            else:
                y_true.append(0)
            y_scores.append(all_confidences_by_uuid[uuid])
        if not y_scores:
            continue
        y_true_arr = np.array(y_true)
        y_scores_arr = np.array(y_scores)
        if y_true_arr.sum() > 0:
            try:
                fpr_arr, tpr_arr, _ = roc_curve(y_true_arr, y_scores_arr)
                auc_roc = float(auc(fpr_arr, tpr_arr))
                auc_pr = float(average_precision_score(y_true_arr, y_scores_arr))
            except Exception:
                auc_roc = 0.0
                auc_pr = 0.0
            f1_metrics = find_best_f1_metrics(y_true_arr, y_scores_arr)
        else:
            auc_roc = 0.0
            auc_pr = 0.0
            f1_metrics = {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "TP": 0,
                "FP": 0,
                "TN": int((y_true_arr == 0).sum()),
                "FN": 0,
            }
        results[attack_name] = {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "f1": f1_metrics["f1"],
            "precision": f1_metrics["precision"],
            "recall": f1_metrics["recall"],
            "TP": f1_metrics["TP"],
            "FP": f1_metrics["FP"],
            "TN": f1_metrics["TN"],
            "FN": f1_metrics["FN"],
        }
    adp_value = compute_adp(all_confidences_by_uuid, uuid_attack_map, total_attacks)
    return results, adp_value


def _make_args(artifacts_dir, embedding, test_start_date, test_end_date, atr_dir="train_attr", suffix="", batch_size=5000, num_neighbors="-1,-1"):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpus_str = ",".join(str(i) for i in range(num_gpus))
    else:
        gpus_str = ""
    
    return SimpleNamespace(
        artifacts_dir=artifacts_dir,
        embedding=embedding,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        atr_dir=atr_dir,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        device="cuda" if torch.cuda.is_available() else "cpu",
        gpus=gpus_str,
        data_gpu=None,
    )


def evaluate_baseline(artifacts_dir, embedding="word2vec"):
    suffix = ""
    dataset_upper = DATASET
    dataset_lower = DATASET.lower()
    dates = get_dataset_dates(dataset_lower)
    test_start = dates["test_start_date"]
    test_end = dates["test_end_date"]

    base_dir = Path(artifacts_dir) / f"{dataset_upper}_graphs{suffix}" / embedding / f"train_attr{suffix}"
    model_dir = base_dir / f"model{suffix}"
    checkpoint_path = model_dir / f"word2vec_gnn_{DATASET}_{BASELINE_EPOCH}_E3.pth"

    if not base_dir.exists():
        return pd.DataFrame()
    try:
        _ = _resolve_plain_or_zip(checkpoint_path)
    except FileNotFoundError:
        return pd.DataFrame()

    input_dim, num_classes = load_training_artifacts(base_dir)
    num_classes = 6

    state_dict = torch_load_maybe_zipped(checkpoint_path, map_location=device, weights_only=True)
    state_dict = clean_state_dict(state_dict)
    hidden_units, num_layers = infer_gcn_architecture(state_dict, num_classes)

    model = GCN(
        input_dim,
        num_classes,
        hidden_units=hidden_units,
        num_layers=num_layers,
        dropout_rate=0.5,
    ).to(device)
    model.load_state_dict(state_dict)
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.eval()

    args = _make_args(artifacts_dir, embedding, test_start, test_end, "train_attr", suffix)
    data_device = device
    test_graphs_by_window, test_data_by_window, _ = load_test_windows(
        args, dataset_upper, suffix, data_device, dataset_lower
    )
    if not test_graphs_by_window:
        return pd.DataFrame()

    attack_scenarios = get_attack_scenarios(DATASET)
    if not attack_scenarios:
        return pd.DataFrame()

    num_neighbors_list = parse_num_neighbors(args.num_neighbors)
    if not num_neighbors_list:
        num_neighbors_list = [-1, -1]

    results_map, adp_value = run_inference_full(
        model,
        test_graphs_by_window,
        test_data_by_window,
        attack_scenarios,
        device,
        data_device,
        num_neighbors_list,
        args.batch_size,
    )

    rows = []
    for attack_name, metrics in results_map.items():
        rows.append({
            "attack_name": attack_name,
            "AUC_ROC": metrics["auc_roc"],
            "AUC_PR": metrics["auc_pr"],
            "ADP": adp_value,
        })
    return pd.DataFrame(rows)


def evaluate_autoprov(artifacts_dir, embedding="word2vec"):
    suffix = "_rulellm_llmlabel"
    dataset_upper = DATASET
    dataset_lower = DATASET.lower()
    dates = get_dataset_dates(dataset_lower)
    test_start = dates["test_start_date"]
    test_end = dates["test_end_date"]

    base_dir = Path(artifacts_dir) / f"{dataset_upper}_graphs{suffix}" / embedding / f"train_attr{suffix}"
    model_dir = base_dir / "hypersearch_models" / f"model_{AUTOPROV_MODEL_ID}"
    checkpoint_path = model_dir / f"model_epoch_{AUTOPROV_EPOCH - 1}.pth"
    hyperparams_path = model_dir / "hyperparameters.json"

    if not base_dir.exists():
        return pd.DataFrame()
    try:
        _ = _resolve_plain_or_zip(checkpoint_path)
    except FileNotFoundError:
        return pd.DataFrame()
    try:
        _ = _resolve_plain_or_zip(hyperparams_path)
    except FileNotFoundError:
        return pd.DataFrame()

    hparams = load_json_maybe_zipped(hyperparams_path)

    input_dim, num_classes = load_training_artifacts(base_dir)
    label_encoder_path = base_dir / "label_encoder.pkl"
    try:
        _ = _resolve_plain_or_zip(label_encoder_path)
        label_encoder = load_pickle_maybe_zipped(base_dir, "label_encoder")
        num_classes = len(label_encoder.classes_)
    except FileNotFoundError:
        pass

    units = int(hparams.get("units", 32))
    num_layers = int(hparams.get("num_layers", 2))
    dropout = float(hparams.get("dropout", 0.5))
    batch_size = int(hparams.get("batch_size", 5000))
    num_neighbors_str = hparams.get("num_neighbors", "-1,-1")
    num_neighbors_list = parse_num_neighbors(num_neighbors_str)
    if not num_neighbors_list:
        num_neighbors_list = [-1, -1]

    state_dict = torch_load_maybe_zipped(checkpoint_path, map_location=device, weights_only=True)
    state_dict = clean_state_dict(state_dict)
    model = GCN(
        input_dim,
        num_classes,
        hidden_units=units,
        num_layers=num_layers,
        dropout_rate=dropout,
    ).to(device)
    model.load_state_dict(state_dict)
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.eval()

    args = _make_args(artifacts_dir, embedding, test_start, test_end, "train_attr", suffix, batch_size=batch_size, num_neighbors=num_neighbors_str)
    data_device = device
    test_graphs_by_window, test_data_by_window, _ = load_test_windows(
        args, dataset_upper, suffix, data_device, dataset_lower
    )
    if not test_graphs_by_window:
        return pd.DataFrame()

    attack_scenarios = get_attack_scenarios(DATASET)
    if not attack_scenarios:
        return pd.DataFrame()

    results_map, adp_value = run_inference_full(
        model,
        test_graphs_by_window,
        test_data_by_window,
        attack_scenarios,
        device,
        data_device,
        num_neighbors_list,
        batch_size,
    )

    rows = []
    for attack_name, metrics in results_map.items():
        rows.append({
            "attack_name": attack_name,
            "AUC_ROC": metrics["auc_roc"],
            "AUC_PR": metrics["auc_pr"],
            "ADP": adp_value,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate THEIA baseline (epoch 19) and AutoProv (model 36, epoch 50)")
    _default_artifacts = os.path.join(os.path.dirname(script_dir), "BIGDATA", "FLASH_artifacts")
    parser.add_argument("--artifacts_dir", type=str, default=_default_artifacts, help="FLASH artifacts root (default: AutoProv/BIGDATA/FLASH_artifacts)")
    parser.add_argument("--embedding", type=str, default="word2vec", help="Embedding type used for training")
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    embedding = args.embedding

    results_baseline = evaluate_baseline(artifacts_dir, embedding)
    results_autoprov = evaluate_autoprov(artifacts_dir, embedding)

    if not results_baseline.empty:
        display_cols = ["attack_name", "AUC_ROC", "AUC_PR", "ADP"]
        baseline_display = results_baseline[display_cols].copy()
        baseline_display["AUC_ROC"] = baseline_display["AUC_ROC"].round(3)
        baseline_display["AUC_PR"] = baseline_display["AUC_PR"].round(3)
        baseline_display["ADP"] = baseline_display["ADP"].round(3)
        print(baseline_display.to_string(index=False))

    if not results_autoprov.empty:
        display_cols = ["attack_name", "AUC_ROC", "AUC_PR", "ADP"]
        autoprov_display = results_autoprov[display_cols].copy()
        autoprov_display["AUC_ROC"] = autoprov_display["AUC_ROC"].round(3)
        autoprov_display["AUC_PR"] = autoprov_display["AUC_PR"].round(3)
        autoprov_display["ADP"] = autoprov_display["ADP"].round(3)
        print(autoprov_display.to_string(index=False))


if __name__ == "__main__":
    main()
