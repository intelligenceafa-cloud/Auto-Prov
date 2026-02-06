import os
import sys
import argparse
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
try:
    from pygod.detector import CoLA as PyGODCoLA, GAE as PyGODGAE
    PYGOD_AVAILABLE = True
except ImportError:
    PYGOD_AVAILABLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OCRGCNBase(nn.Module):
    def __init__(self, in_dim, hid_dim, num_relations, num_layers=3, dropout=0.0):
        super(OCRGCNBase, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_dim, hid_dim, num_relations))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hid_dim, hid_dim, num_relations))
        self.convs.append(RGCNConv(hid_dim, hid_dim, num_relations))
        self.c = None
    
    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def init_center_c(self, data, edge_index, edge_type, eps=0.1):
        self.eval()
        with torch.no_grad():
            z = self.forward(data, edge_index, edge_type)
            c = z.mean(dim=0)
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c >= 0)] = eps
        self.c = c
        return c

class OCRGCN:
    def __init__(self, in_dim, hid_dim=32, num_relations=10, num_layers=3, dropout=0.0,
                 lr=0.005, epoch=100, beta=0.5, contamination=0.001, warmup=10, eps=0.1, device='cuda'):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epoch = epoch
        self.beta = beta
        self.contamination = contamination
        self.warmup = warmup
        self.eps = eps
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.config = {
            'hid_dim': hid_dim, 'num_layers': num_layers, 'dropout': dropout,
            'lr': lr, 'epoch': epoch, 'beta': beta, 'contamination': contamination,
            'warmup': warmup, 'eps': eps
        }
        self.model = OCRGCNBase(in_dim, hid_dim, num_relations, num_layers, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.radius = None
    
    def fit(self, data, edge_index, edge_type, target_mask=None, val_data=None):
        data = data.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        if target_mask is not None:
            target_mask = target_mask.to(self.device)
        self.model.init_center_c(data, edge_index, edge_type, eps=self.eps)
        c = self.model.c
        pbar = tqdm(range(1, self.epoch + 1), desc="Training OCRGCN", leave=False)
        for ep in pbar:
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model(data, edge_index, edge_type)
            dist = torch.sum((z - c) ** 2, dim=1)
            if target_mask is not None:
                dist = dist[target_mask]
            if ep <= self.warmup:
                loss = torch.mean(dist)
            else:
                if self.radius is None:
                    with torch.no_grad():
                        sorted_dist, _ = torch.sort(dist)
                        quantile_idx = int((1 - self.contamination) * len(sorted_dist))
                        self.radius = sorted_dist[quantile_idx]
                scores = dist - self.radius ** 2
                loss_dist = self.radius ** 2 + (1 / self.contamination) * torch.mean(F.relu(scores))
                radius_loss = self.beta * torch.abs(self.radius)
                loss = loss_dist + radius_loss
            loss.backward()
            self.optimizer.step()
            phase = "warmup" if ep <= self.warmup else "training"
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'phase': phase})
        pbar.close()
    
    def predict(self, data, edge_index, edge_type, target_mask=None):
        self.model.eval()
        data = data.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        if target_mask is not None:
            target_mask = target_mask.to(self.device)
        with torch.no_grad():
            z = self.model(data, edge_index, edge_type)
            c = self.model.c
            dist = torch.sum((z - c) ** 2, dim=1)
            scores = dist.cpu().numpy()
        return scores
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'center': self.model.c,
            'radius': self.radius,
            'model_name': 'ocrgcn',
            'config': self.config,
            'hyperparameters': {
                'in_dim': self.in_dim, 'hid_dim': self.hid_dim, 'num_relations': self.num_relations,
                'num_layers': self.num_layers, 'dropout': self.dropout, 'contamination': self.contamination
            }
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.c = checkpoint['center']
        self.radius = checkpoint['radius']
        if 'config' in checkpoint:
            self.config = checkpoint['config']

class CoLAWrapper:
    def __init__(self, in_dim, hid_dim=64, num_layers=2, dropout=0.0, lr=0.005, epoch=100, contamination=0.001, device='cuda'):
        if not PYGOD_AVAILABLE:
            raise ImportError("PyGOD is required for CoLA wrapper")
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.config = {'in_dim': in_dim, 'hid_dim': hid_dim, 'num_layers': num_layers, 'dropout': dropout, 'lr': lr, 'epoch': epoch, 'contamination': contamination}
        self.contamination = contamination
        self.model = PyGODCoLA(hid_dim=hid_dim, num_layers=num_layers, dropout=dropout, lr=lr, epoch=epoch, contamination=contamination)
        self.model.device = self.device
    
    def _build_data(self, x, edge_index):
        return Data(x=x.to(self.device), edge_index=edge_index.to(self.device))
    
    def fit(self, data, edge_index, edge_type, target_mask=None, val_data=None):
        graph = self._build_data(data, edge_index)
        self.model.fit(graph)
    
    def predict(self, data, edge_index, edge_type, target_mask=None):
        graph = self._build_data(data, edge_index)
        scores = self.model.decision_function(graph)
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        return scores
    
    def save_model(self, path):
        torch.save({'config': self.config, 'model_name': 'cola', 'model_obj': self.model}, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint.get('config', self.config)
        self.contamination = self.config.get('contamination', self.contamination)
        self.model = checkpoint.get('model_obj', self.model)
        self.model.device = self.device

class GAEWrapper:
    def __init__(self, in_dim, hid_dim=64, num_layers=2, dropout=0.0, lr=0.005, epoch=100, contamination=0.001, device='cuda'):
        if not PYGOD_AVAILABLE:
            raise ImportError("PyGOD is required for GAE wrapper")
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.config = {'in_dim': in_dim, 'hid_dim': hid_dim, 'num_layers': num_layers, 'dropout': dropout, 'lr': lr, 'epoch': epoch, 'contamination': contamination}
        self.contamination = contamination
        self.model = PyGODGAE(hid_dim=hid_dim, num_layers=num_layers, lr=lr, epoch=epoch, contamination=contamination)
        self.model.device = self.device
    
    def _build_data(self, x, edge_index):
        return Data(x=x.to(self.device), edge_index=edge_index.to(self.device))
    
    def fit(self, data, edge_index, edge_type, target_mask=None, val_data=None):
        graph = self._build_data(data, edge_index)
        self.model.fit(graph)
    
    def predict(self, data, edge_index, edge_type, target_mask=None):
        graph = self._build_data(data, edge_index)
        scores = self.model.decision_function(graph)
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        return scores
    
    def save_model(self, path):
        torch.save({'config': self.config, 'model_name': 'gae', 'model_obj': self.model}, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint.get('config', self.config)
        self.contamination = self.config.get('contamination', self.contamination)
        self.model = checkpoint.get('model_obj', self.model)
        if hasattr(self.model, 'device'):
            self.model.device = self.device

def build_detector(model_name, config, in_dim, num_relations, device):
    model_name = model_name.lower()
    if model_name == 'ocrgcn':
        return OCRGCN(in_dim=in_dim, hid_dim=config['hid_dim'], num_relations=num_relations,
                     num_layers=config['num_layers'], dropout=config['dropout'], lr=config['lr'],
                     epoch=config['epoch'], beta=config.get('beta', 0.5), contamination=config['contamination'],
                     warmup=config.get('warmup', 2), eps=config.get('eps', 0.1), device=device)
    if model_name == 'cola':
        return CoLAWrapper(in_dim=in_dim, hid_dim=config['hid_dim'], num_layers=config['num_layers'],
                          dropout=config['dropout'], lr=config['lr'], epoch=config['epoch'],
                          contamination=config['contamination'], device=device)
    if model_name == 'gae':
        return GAEWrapper(in_dim=in_dim, hid_dim=config['hid_dim'], num_layers=config['num_layers'],
                         dropout=config['dropout'], lr=config['lr'], epoch=config['epoch'],
                         contamination=config['contamination'], device=device)
    raise ValueError(f"Unsupported model: {model_name}")

def create_base_config(args):
    is_original_mode = not (hasattr(args, 'rulellm') and (args.rulellm or (hasattr(args, 'llmlabel') and args.llmlabel) or (hasattr(args, 'llmfunc') and args.llmfunc)))
    warmup = args.warmup if hasattr(args, 'warmup') and args.warmup is not None else (2 if is_original_mode else 10)
    return {
        'hid_dim': args.hid_dim, 'num_layers': args.num_layers, 'dropout': args.dropout,
        'lr': args.lr, 'epoch': args.epoch, 'beta': args.beta, 'contamination': args.contamination,
        'warmup': warmup, 'eps': args.eps
    }

def get_model_file_name(model_name, suffix):
    base = 'ocrgcn' if model_name.lower() == 'ocrgcn' else model_name.lower()
    return f'{base}_model{suffix}.pth'

def get_mapping_file_name(model_name, suffix):
    if model_name.lower() == 'ocrgcn':
        return f'mappings{suffix}.pkl'
    return f'mappings{suffix}_{model_name.lower()}.pkl'

def detect_per_vtype_files(base_dir):
    vtypes_list_file = os.path.join(base_dir, 'vtypes_list.pkl')
    if os.path.exists(vtypes_list_file):
        with open(vtypes_list_file, 'rb') as f:
            vtype_info = pickle.load(f)
        return vtype_info['vtypes'], vtype_info['vtype_counts']
    if os.path.exists(os.path.join(base_dir, 'train_data.pkl')):
        return None, None
    print(f"⚠️ No data files found in {base_dir}")
    return None, None

def load_vtype_data(base_dir, vtype):
    vtype_safe = vtype.replace('/', '_').replace('+', '_')
    train_file = os.path.join(base_dir, f'train_data_{vtype_safe}.pkl')
    features_file = os.path.join(base_dir, f'features_train_{vtype_safe}.pkl')
    edge_types_file = os.path.join(base_dir, 'edge_types.pkl')
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(features_file, 'rb') as f:
        features_train = pickle.load(f)
    with open(edge_types_file, 'rb') as f:
        edge_types = pickle.load(f)
    return train_data, features_train, edge_types

def split_data_by_vtype(graph_data, features_df, edge_types, target_vtype):
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    target_vtype_nodes = {nid: ninfo for nid, ninfo in nodes.items() if ninfo.get('type') == target_vtype}
    target_node_ids = set(target_vtype_nodes.keys())
    vtype_graph_data = {
        'nodes': nodes, 'edges': edges, 'graph': None,
        'target_vtype': target_vtype, 'target_node_ids': target_node_ids,
    }
    return vtype_graph_data, features_df, len(target_vtype_nodes)

def prepare_pyg_data(graph_data, features_df, edge_types):
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    target_node_ids = graph_data.get('target_node_ids', None)
    node_ids = sorted(list(nodes.keys()))
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    edge_type_mapping = {etype: idx for idx, etype in enumerate(sorted(edge_types))}
    features_df = features_df.set_index('node_uuid')
    features_df = features_df.reindex(node_ids, fill_value=0)
    x = torch.FloatTensor(features_df.values)
    edge_list = []
    edge_type_list = []
    for src, dst, etype, timestamp in tqdm(edges, desc="Processing edges", leave=False):
        if src in node_id_to_idx and dst in node_id_to_idx:
            edge_list.append([node_id_to_idx[src], node_id_to_idx[dst]])
            edge_type_list.append(edge_type_mapping.get(etype, 0))
    if len(edge_list) > 0:
        edge_index = torch.LongTensor(edge_list).t()
        edge_type_tensor = torch.LongTensor(edge_type_list)
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index_undirected = torch.cat([edge_index, reverse_edge_index], dim=1)
        edge_type_undirected = torch.cat([edge_type_tensor, edge_type_tensor], dim=0)
    else:
        edge_index_undirected = torch.LongTensor(2, 0)
        edge_type_undirected = torch.LongTensor([])
    data = Data(x=x, edge_index=edge_index_undirected, edge_type=edge_type_undirected)
    if target_node_ids is not None:
        target_mask = torch.zeros(len(node_ids), dtype=torch.bool)
        for i, nid in enumerate(node_ids):
            if nid in target_node_ids:
                target_mask[i] = True
        data.target_mask = target_mask
    return data, edge_type_mapping, node_id_to_idx

def get_unique_vtypes(graph_data):
    nodes = graph_data['nodes']
    vtype_counts = {}
    for node_info in nodes.values():
        vtype = node_info.get('type', 'unknown')
        vtype_counts[vtype] = vtype_counts.get(vtype, 0) + 1
    vtypes = sorted(vtype_counts.keys(), key=lambda x: vtype_counts[x], reverse=True)
    return vtypes, vtype_counts

def train_model_simple(model_name, config, data, edge_index, edge_type, epochs, model_dir, device, num_relations, target_mask=None):
    model_label = os.path.basename(model_dir)
    model_name = model_name.lower()
    in_dim = data.shape[1]
    if model_name == 'ocrgcn':
        detector = build_detector(model_name, config, in_dim, num_relations, device)
        pbar = tqdm(range(1, epochs + 1), desc=f"Training {model_label}")
        data_device = data.to(device)
        edge_index_device = edge_index.to(device)
        edge_type_device = edge_type.to(device)
        target_mask_device = target_mask.to(device) if target_mask is not None else None
        detector.model.init_center_c(data_device, edge_index_device, edge_type_device, eps=detector.eps)
        c = detector.model.c
        for ep in pbar:
            detector.model.train()
            detector.optimizer.zero_grad()
            z = detector.model(data_device, edge_index_device, edge_type_device)
            dist = torch.sum((z - c) ** 2, dim=1)
            if target_mask_device is not None:
                dist = dist[target_mask_device]
            if ep <= detector.warmup:
                loss = torch.mean(dist)
            else:
                if detector.radius is None:
                    with torch.no_grad():
                        sorted_dist, _ = torch.sort(dist)
                        quantile_idx = int((1 - detector.contamination) * len(sorted_dist))
                        detector.radius = sorted_dist[quantile_idx]
                scores = dist - detector.radius ** 2
                loss_dist = detector.radius ** 2 + (1 / detector.contamination) * torch.mean(F.relu(scores))
                radius_loss = detector.beta * torch.abs(detector.radius)
                loss = loss_dist + radius_loss
            loss.backward()
            detector.optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{ep}.pth')
            detector.save_model(checkpoint_path)
        pbar.close()
        return detector
    progress_desc = f"Training {model_label} ({model_name})"
    pbar = tqdm(range(1, epochs + 1), desc=progress_desc)
    last_detector = None
    for ep in pbar:
        config_epoch = config.copy()
        config_epoch['epoch'] = ep
        detector = build_detector(model_name, config_epoch, in_dim, num_relations, device)
        detector.fit(data, edge_index, edge_type, target_mask=target_mask)
        checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{ep}.pth')
        detector.save_model(checkpoint_path)
        last_detector = detector
        torch.cuda.empty_cache()
    pbar.close()
    return last_detector

def train_model_8(base_dir, hypersearch_root, train_data, features_train, edge_types, 
                  vtypes_from_files=None, vtype_counts_from_files=None, 
                  embedding='mpnet', device='cuda', max_epochs=50):
    model_id = 8
    config = {
        'model_id': 8,
        'name': 'with_dropout',
        'hid_dim': 32,
        'num_layers': 3,
        'dropout': 0.1,
        'lr': 0.005,
        'beta': 0.5,
        'contamination': 0.001,
        'warmup': 10,
        'eps': 0.1,
        'epoch': max_epochs
    }
    
    if vtypes_from_files is not None:
        vtypes, vtype_counts = vtypes_from_files, vtype_counts_from_files
        use_per_vtype_files = True
    else:
        vtypes, vtype_counts = get_unique_vtypes(train_data)
        use_per_vtype_files = False
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL 8 (with_dropout) - Epochs: {max_epochs}")
    print(f"{'='*80}")
    print(f"Node Types: {len(vtypes)} ({', '.join([f'{v}: {vtype_counts[v]:,}' for v in vtypes[:5]])}...)")
    print(f"{'='*80}\n")
    
    os.makedirs(hypersearch_root, exist_ok=True)
    
    for vtype in vtypes:
        if vtype is not None:
            model_name = f"model_{model_id}_{vtype}"
            model_dir = os.path.join(hypersearch_root, model_name)
        else:
            model_name = f"model_{model_id}"
            model_dir = os.path.join(hypersearch_root, model_name)
        
        print(f"\n{'='*60}")
        print(f"Model {model_id} ({config['name']}) - VType: {vtype}")
        print(f"{'='*60}")
        
        if use_per_vtype_files:
            print(f"  Loading per-vtype data...", end=' ', flush=True)
            vtype_train_data, vtype_features_train, _ = load_vtype_data(base_dir, vtype)
            print(f"✓ {len(vtype_train_data['nodes'])} nodes, {len(vtype_train_data['edges'])} edges")
        else:
            vtype_train_data, vtype_features_train, num_nodes = split_data_by_vtype(
                train_data, features_train, edge_types, vtype
            )
            print(f"  Filtered to {num_nodes:,} nodes of type '{vtype}'")
        
        os.makedirs(model_dir, exist_ok=True)
        
        config_with_vtype = config.copy()
        config_with_vtype['vtype'] = vtype
        config_with_vtype['model'] = 'ocrgcn'
        hyperparams_path = os.path.join(model_dir, 'hyperparams.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(config_with_vtype, f, indent=2)
        
        data_train_vtype, edge_type_mapping_vtype, node_id_to_idx_vtype = prepare_pyg_data(
            vtype_train_data, vtype_features_train, edge_types
        )
        
        if data_train_vtype.num_nodes == 0:
            print(f"  Skipping - no nodes of type '{vtype}'")
            continue
        
        print(f"  PyG graph: {data_train_vtype.num_nodes:,} nodes, {data_train_vtype.num_edges:,} edges")
        
        train_model_simple(
            'ocrgcn',
            dict(config),
            data_train_vtype.x,
            data_train_vtype.edge_index,
            data_train_vtype.edge_type,
            max_epochs,
            model_dir,
            device,
            len(edge_type_mapping_vtype),
            target_mask=getattr(data_train_vtype, 'target_mask', None)
        )
        
        del data_train_vtype
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"Model 8 training complete!")
    print(f"{'='*80}\n")

def parse_args():
    parser = argparse.ArgumentParser(description='OCR_APT Graph Learning for THEIA')
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--baseline', action='store_true',
                           help='Train baseline model (single model, no hyperparameter search)')
    mode_group.add_argument('--autoprov', action='store_true',
                           help='Train model 8 only (autoprov mode, similar to --rulellm --llmlabel)')
    
    parser.add_argument('--embedding', type=str, default='mpnet',
                       choices=['mpnet', 'minilm', 'roberta', 'distilbert'],
                       help='Embedding model to use for autoprov mode (default: mpnet)')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    parser.add_argument('--gpus', type=str, default='',
                       help='Comma-separated GPU ids (e.g. 0,1,2); default empty = use all available')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.gpus and args.gpus.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    artifacts_root = os.path.join(autoprov_dir, 'BIGDATA', 'OCR_APT_artifacts')
    
    dataset = 'theia'
    
    if args.baseline:
        base_dir = os.path.join(artifacts_root, dataset)
        model_root = os.path.join(base_dir, 'ocrgcn')
        output_dir = os.path.join(model_root, 'original')
        
        print(f"\n{'='*80}")
        print(f"OCR_APT Graph Learning - THEIA (BASELINE)")
        print(f"Mode: Baseline (Regex mode, OCR-APT features)")
        print(f"Device: {args.device}")
        print(f"{'='*80}\n")
        
        vtypes_from_files, vtype_counts_from_files = detect_per_vtype_files(base_dir)
        
        if vtypes_from_files is not None:
            use_per_vtype_files = True
            print(f"✓ Detected per-vtype data files ({len(vtypes_from_files)} vtypes)")
            print(f"  {', '.join([f'{v}: {vtype_counts_from_files[v]:,}' for v in vtypes_from_files[:5]])}...")
            
            with open(os.path.join(base_dir, 'edge_types.pkl'), 'rb') as f:
                edge_types = pickle.load(f)
            
            train_data = None
            features_train = None
            vtypes = vtypes_from_files
            vtype_counts = vtype_counts_from_files
        else:
            use_per_vtype_files = False
            print("Loading training data...", end=' ', flush=True)
            
            train_data_zip_part1 = os.path.join(base_dir, 'train_data_part1.pkl.zip')
            train_data_zip_part2 = os.path.join(base_dir, 'train_data_part2.pkl.zip')
            train_data_path = os.path.join(base_dir, 'train_data.pkl')
            
            features_train_zip_part1 = os.path.join(base_dir, 'features_train_part1.pkl.zip')
            features_train_zip_part2 = os.path.join(base_dir, 'features_train_part2.pkl.zip')
            features_train_path = os.path.join(base_dir, 'features_train.pkl')
            
            import zipfile
            import pandas as pd
            
            if os.path.exists(train_data_zip_part1) and os.path.exists(train_data_zip_part2):
                with zipfile.ZipFile(train_data_zip_part1, 'r') as zf:
                    with zf.open('train_data.pkl', 'r') as f:
                        train_data_part1 = pickle.load(f)
                with zipfile.ZipFile(train_data_zip_part2, 'r') as zf:
                    with zf.open('train_data.pkl', 'r') as f:
                        train_data_part2 = pickle.load(f)
                train_data = {
                    'nodes': {},
                    'edges': [],
                    'graph': None
                }
                train_data['nodes'].update(train_data_part1['nodes'])
                train_data['nodes'].update(train_data_part2['nodes'])
                train_data['edges'].extend(train_data_part1['edges'])
                train_data['edges'].extend(train_data_part2['edges'])
            elif os.path.exists(train_data_path):
                with open(train_data_path, 'rb') as f:
                    train_data = pickle.load(f)
            else:
                raise FileNotFoundError(f"Training data not found in {base_dir}")
            
            if os.path.exists(features_train_zip_part1) and os.path.exists(features_train_zip_part2):
                with zipfile.ZipFile(features_train_zip_part1, 'r') as zf:
                    with zf.open('features_train.pkl', 'r') as f:
                        features_train_part1 = pickle.load(f)
                with zipfile.ZipFile(features_train_zip_part2, 'r') as zf:
                    with zf.open('features_train.pkl', 'r') as f:
                        features_train_part2 = pickle.load(f)
                features_train = pd.concat([features_train_part1, features_train_part2], ignore_index=True)
            elif os.path.exists(features_train_path):
                with open(features_train_path, 'rb') as f:
                    features_train = pickle.load(f)
            else:
                raise FileNotFoundError(f"Features train not found in {base_dir}")
            
            with open(os.path.join(base_dir, 'edge_types.pkl'), 'rb') as f:
                edge_types = pickle.load(f)
            
            print(f"✓ {len(train_data['nodes'])} nodes, {len(train_data['edges'])} edges, {len(edge_types)} edge_types")
            
            vtypes, vtype_counts = get_unique_vtypes(train_data)
        
        os.makedirs(output_dir, exist_ok=True)
        
        max_epochs = args.epochs if args.epochs is not None else 100
        
        print(f"\n{'='*80}")
        print(f"TRAINING SEPARATE MODELS PER VTYPE")
        print(f"{'='*80}")
        print(f"Node Types: {len(vtypes)} ({', '.join([f'{v}: {vtype_counts[v]:,}' for v in vtypes])})")
        print(f"Total Models: {len(vtypes)}")
        print(f"Epochs: {max_epochs}")
        print(f"{'='*80}\n")
        
        for vtype in vtypes:
            print(f"\n{'='*60}")
            print(f"Training model for VType: {vtype}")
            print(f"{'='*60}")
            
            if use_per_vtype_files:
                print(f"  Loading per-vtype data...", end=' ', flush=True)
                vtype_train_data, vtype_features_train, _ = load_vtype_data(base_dir, vtype)
                print(f"✓ {len(vtype_train_data['nodes'])} nodes, {len(vtype_train_data['edges'])} edges")
            else:
                vtype_train_data, vtype_features_train, num_target_nodes = split_data_by_vtype(
                    train_data, features_train, edge_types, vtype
                )
                print(f"  Target nodes: {num_target_nodes:,} of type '{vtype}' (full graph preserved for message passing)")
            
            model_suffix = f"_{vtype}"
            
            print("  Preparing PyG graph...", end=' ', flush=True)
            data, edge_type_mapping, node_id_to_idx = prepare_pyg_data(
                vtype_train_data, vtype_features_train, edge_types
            )
            
            if data.num_nodes == 0:
                print(f"  ⚠ Skipping - no nodes")
                continue
            
            if data.num_edges == 0:
                print(f"  ⚠ Skipping - no edges (vtype has only isolated nodes)")
                continue
            
            print(f"✓ {data.num_nodes:,} nodes, {data.num_edges:,} edges, {len(edge_type_mapping)} relations")
            
            class Args:
                def __init__(self):
                    self.rulellm = False
                    self.llmlabel = False
                    self.llmfunc = False
                    self.hid_dim = 32
                    self.num_layers = 3
                    self.dropout = 0.0
                    self.lr = 0.005
                    self.epoch = max_epochs
                    self.beta = 0.5
                    self.contamination = 0.001
                    self.warmup = 2
                    self.eps = 0.1
            
            args_obj = Args()
            detector_config = create_base_config(args_obj)
            model_config = detector_config.copy()
            
            print(f"  Initializing OCRGCN...", end=' ', flush=True)
            model = build_detector(
                'ocrgcn',
                detector_config,
                data.x.shape[1],
                len(edge_type_mapping),
                args.device
            )
            print(f"✓ (dim: {data.x.shape[1]}→{detector_config['hid_dim']}, layers: {detector_config['num_layers']})")
            
            target_mask = data.target_mask if hasattr(data, 'target_mask') else None
            model.fit(data.x, data.edge_index, data.edge_type, target_mask=target_mask)
            
            print(f"  Saving model...", end=' ', flush=True)
            model_path = os.path.join(output_dir, get_model_file_name('ocrgcn', model_suffix))
            model.save_model(model_path)
            
            mappings = {
                'edge_type_mapping': edge_type_mapping,
                'node_id_to_idx': node_id_to_idx,
                'vtype': vtype,
                'model': 'ocrgcn',
                'model_config': model_config
            }
            mappings_path = os.path.join(output_dir, get_mapping_file_name('ocrgcn', model_suffix))
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings, f)
            print(f"✓ Saved to {output_dir}/\n")
            
            del model, data
            torch.cuda.empty_cache()
        
        print(f"{'='*80}")
        print(f"Baseline training completed successfully!")
        print(f"{'='*80}\n")
    
    elif args.autoprov:
        embedding = args.embedding.lower()
        base_dir = os.path.join(artifacts_root, f'{dataset}_rulellm_llmlabel_{embedding}')
        model_root = os.path.join(base_dir, 'ocrgcn')
        hypersearch_root = os.path.join(model_root, 'hypersearch_models')
        
        print(f"\n{'='*80}")
        print(f"OCR_APT Graph Learning - THEIA (AUTOPROV)")
        print(f"Mode: RuleLLM + LLM Type Embeddings ({embedding})")
        print(f"Model: 8 (with_dropout) | Epochs: 50")
        print(f"Device: {args.device}")
        print(f"{'='*80}\n")
        
        vtypes_from_files, vtype_counts_from_files = detect_per_vtype_files(base_dir)
        
        if vtypes_from_files is not None:
            use_per_vtype_files = True
            print(f"✓ Detected per-vtype data files ({len(vtypes_from_files)} vtypes)")
            print(f"  {', '.join([f'{v}: {vtype_counts_from_files[v]:,}' for v in vtypes_from_files[:5]])}...")
            
            with open(os.path.join(base_dir, 'edge_types.pkl'), 'rb') as f:
                edge_types = pickle.load(f)
            
            train_data = None
            features_train = None
        else:
            use_per_vtype_files = False
            print("Loading training data...", end=' ', flush=True)
            
            train_data_zip_part1 = os.path.join(base_dir, 'train_data_part1.pkl.zip')
            train_data_zip_part2 = os.path.join(base_dir, 'train_data_part2.pkl.zip')
            train_data_path = os.path.join(base_dir, 'train_data.pkl')
            
            features_train_zip_part1 = os.path.join(base_dir, 'features_train_part1.pkl.zip')
            features_train_zip_part2 = os.path.join(base_dir, 'features_train_part2.pkl.zip')
            features_train_path = os.path.join(base_dir, 'features_train.pkl')
            
            import zipfile
            import pandas as pd
            
            if os.path.exists(train_data_zip_part1) and os.path.exists(train_data_zip_part2):
                with zipfile.ZipFile(train_data_zip_part1, 'r') as zf:
                    with zf.open('train_data.pkl', 'r') as f:
                        train_data_part1 = pickle.load(f)
                with zipfile.ZipFile(train_data_zip_part2, 'r') as zf:
                    with zf.open('train_data.pkl', 'r') as f:
                        train_data_part2 = pickle.load(f)
                train_data = {
                    'nodes': {},
                    'edges': [],
                    'graph': None
                }
                train_data['nodes'].update(train_data_part1['nodes'])
                train_data['nodes'].update(train_data_part2['nodes'])
                train_data['edges'].extend(train_data_part1['edges'])
                train_data['edges'].extend(train_data_part2['edges'])
            elif os.path.exists(train_data_path):
                with open(train_data_path, 'rb') as f:
                    train_data = pickle.load(f)
            else:
                raise FileNotFoundError(f"Training data not found in {base_dir}")
            
            if os.path.exists(features_train_zip_part1) and os.path.exists(features_train_zip_part2):
                with zipfile.ZipFile(features_train_zip_part1, 'r') as zf:
                    with zf.open('features_train.pkl', 'r') as f:
                        features_train_part1 = pickle.load(f)
                with zipfile.ZipFile(features_train_zip_part2, 'r') as zf:
                    with zf.open('features_train.pkl', 'r') as f:
                        features_train_part2 = pickle.load(f)
                features_train = pd.concat([features_train_part1, features_train_part2], ignore_index=True)
            elif os.path.exists(features_train_path):
                with open(features_train_path, 'rb') as f:
                    features_train = pickle.load(f)
            else:
                raise FileNotFoundError(f"Features train not found in {base_dir}")
            
            with open(os.path.join(base_dir, 'edge_types.pkl'), 'rb') as f:
                edge_types = pickle.load(f)
            
            print(f"✓ {len(train_data['nodes'])} nodes, {len(train_data['edges'])} edges, {len(edge_types)} edge_types")
        
        max_epochs = args.epochs if args.epochs is not None else 50
        
        train_model_8(
            base_dir, hypersearch_root,
            train_data, features_train, edge_types,
            vtypes_from_files, vtype_counts_from_files,
            embedding=embedding,
            device=args.device,
            max_epochs=max_epochs
        )
        
        print(f"{'='*80}")
        print(f"AutoProv training completed successfully!")
        print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

