import os
import sys
import warnings
from contextlib import redirect_stdout
from io import StringIO
import zipfile

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_score, recall_score, f1_score,
    confusion_matrix
)
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

def compute_attack_detection_precision(scores, attack_labels):
    if scores is None or len(scores) == 0 or not attack_labels:
        return None
    attack_indices = {}
    index_to_attacks = {}
    num_nodes = len(scores)
    y_global = np.zeros(num_nodes, dtype=np.int8)
    for attack_name, labels in attack_labels.items():
        if labels is None or len(labels) != num_nodes:
            return None
        positive_idx = np.flatnonzero(labels > 0)
        if positive_idx.size == 0:
            continue
        attack_indices[attack_name] = positive_idx
        y_global[positive_idx] = 1
        for idx in positive_idx:
            index_to_attacks.setdefault(idx, set()).add(attack_name)
    if not attack_indices:
        return 0.0
    total_attacks = len(attack_indices)
    sorted_idx = np.argsort(-scores)
    sorted_labels = y_global[sorted_idx]
    tp = 0.0
    fp = 0.0
    seen_attacks = set()
    precisions = [0.0]
    detections = [0.0]
    for rank, idx in enumerate(sorted_idx):
        if sorted_labels[rank] == 1:
            tp += 1.0
        else:
            fp += 1.0
        if idx in index_to_attacks:
            seen_attacks.update(index_to_attacks[idx])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        detection_fraction = len(seen_attacks) / total_attacks
        precisions.append(precision)
        detections.append(detection_fraction)
    precisions = np.array(precisions, dtype=np.float64)
    detections = np.array(detections, dtype=np.float64)
    precision_to_detection = {}
    for precision, detection in zip(precisions, detections):
        key = round(float(precision), 6)
        if key in precision_to_detection:
            precision_to_detection[key] = max(precision_to_detection[key], detection)
        else:
            precision_to_detection[key] = detection
    unique_precisions = np.array(sorted(precision_to_detection.keys()), dtype=np.float64)
    unique_detections = np.array([precision_to_detection[key] for key in unique_precisions], dtype=np.float64)
    if unique_precisions[0] > 0.0:
        unique_precisions = np.insert(unique_precisions, 0, 0.0)
        unique_detections = np.insert(unique_detections, 0, 0.0)
    if unique_precisions[-1] < 1.0:
        unique_precisions = np.append(unique_precisions, 1.0)
        unique_detections = np.append(unique_detections, unique_detections[-1])
    adp = np.trapz(unique_detections, unique_precisions)
    return float(max(0.0, min(1.0, adp)))

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

def get_attack_scenarios(dataset, pids_gt_dir):
    attacks = {
        "THEIA": [
            ("Firefox_Backdoor_Drakon", os.path.join(pids_gt_dir, "THEIA", "node_Firefox_Backdoor_Drakon_In_Memory.csv"), "2018-04-10"),
            ("Browser_Extension_Drakon", os.path.join(pids_gt_dir, "THEIA", "node_Browser_Extension_Drakon_Dropper.csv"), "2018-04-12")
        ],
        "FIVEDIRECTIONS": [
            ("excel_0409", os.path.join(pids_gt_dir, "FIVEDIRECTIONS", "node_fivedirections_e3_excel_0409.csv"), "2018-04-09"),
            ("firefox_0411", os.path.join(pids_gt_dir, "FIVEDIRECTIONS", "node_fivedirections_e3_firefox_0411.csv"), "2018-04-11"),
            ("browser_0412", os.path.join(pids_gt_dir, "FIVEDIRECTIONS", "node_fivedirections_e3_browser_0412.csv"), "2018-04-12")
        ]
    }
    return attacks.get(dataset.upper(), [])

def load_malicious_ids_from_csv(csv_path):
    malicious_ids = set()
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                uuid = line.split(',')[0]
                malicious_ids.add(uuid)
    return malicious_ids

def build_node_to_dates_mapping(test_data):
    node_to_dates = defaultdict(set)
    for src_uuid, dst_uuid, etype, timestamp in test_data['edges']:
        if isinstance(timestamp, (int, float)):
            if timestamp > 1e15:
                timestamp_sec = timestamp / 1e9
            elif timestamp > 1e12:
                timestamp_sec = timestamp / 1e6
            elif timestamp > 1e9:
                timestamp_sec = timestamp / 1e3
            else:
                timestamp_sec = timestamp
            if timestamp_sec < 0 or timestamp_sec > 4e9:
                continue
            dt = datetime.fromtimestamp(timestamp_sec)
        elif isinstance(timestamp, str):
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        else:
            dt = timestamp
        date_str = dt.strftime("%Y-%m-%d")
        node_to_dates[src_uuid].add(date_str)
        node_to_dates[dst_uuid].add(date_str)
    return node_to_dates

def prepare_test_data(graph_data, features_df, edge_types, edge_type_mapping, node_id_to_idx, malicious_nodes):
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    target_node_ids = graph_data.get('target_node_ids', None)
    node_ids = sorted(list(nodes.keys()))
    test_node_id_to_idx = {}
    for idx, nid in enumerate(node_ids):
        test_node_id_to_idx[nid] = idx
    features_df = features_df.set_index('node_uuid')
    features_df = features_df.reindex(node_ids, fill_value=0)
    x = torch.FloatTensor(features_df.values)
    edge_list = []
    edge_type_list = []
    for src, dst, etype, timestamp in edges:
        if src in test_node_id_to_idx and dst in test_node_id_to_idx:
            edge_list.append([test_node_id_to_idx[src], test_node_id_to_idx[dst]])
            edge_type_idx = edge_type_mapping.get(etype, 0)
            edge_type_list.append(edge_type_idx)
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
    labels = np.zeros(len(node_ids), dtype=int)
    for idx, nid in enumerate(node_ids):
        if nid in malicious_nodes:
            labels[idx] = 1
    return data, labels, node_ids

def compute_metrics(labels, scores, contamination=0.01):
    metrics = {}
    if len(np.unique(labels)) > 1:
        metrics['auc_roc'] = roc_auc_score(labels, scores)
        metrics['auc_pr'] = average_precision_score(labels, scores)
    else:
        metrics['auc_roc'] = 0.0
        metrics['auc_pr'] = 0.0
    threshold = np.percentile(scores, (1 - contamination) * 100)
    pred_labels = (scores > threshold).astype(int)
    metrics['precision'] = precision_score(labels, pred_labels, zero_division=0)
    metrics['recall'] = recall_score(labels, pred_labels, zero_division=0)
    metrics['f1'] = f1_score(labels, pred_labels, zero_division=0)
    cm = confusion_matrix(labels, pred_labels)
    if cm.shape == (1, 1):
        if labels[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    metrics['true_positive'] = int(tp)
    metrics['false_positive'] = int(fp)
    metrics['true_negative'] = int(tn)
    metrics['false_negative'] = int(fn)
    metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return metrics

def compute_metrics_hybrid(labels, scores, node_ids, test_data, contamination=0.01):
    metrics_auc = {}
    if len(np.unique(labels)) > 1:
        metrics_auc['auc_roc'] = roc_auc_score(labels, scores)
        metrics_auc['auc_pr'] = average_precision_score(labels, scores)
    else:
        metrics_auc['auc_roc'] = 0.0
        metrics_auc['auc_pr'] = 0.0
    all_tp, all_tn, all_fp, all_fn = 0, 0, 0, 0
    vtype_to_indices = defaultdict(list)
    for idx, nid in enumerate(node_ids):
        if nid in test_data['nodes']:
            vtype = test_data['nodes'][nid].get('type', 'unknown')
            vtype_to_indices[vtype].append(idx)
    for vtype, indices in vtype_to_indices.items():
        vtype_labels = labels[indices]
        vtype_scores = scores[indices]
        threshold = np.percentile(vtype_scores, (1 - contamination) * 100)
        pred_labels = (vtype_scores > threshold).astype(int)
        cm = confusion_matrix(vtype_labels, pred_labels)
        if cm.shape == (1, 1):
            if vtype_labels[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn, fp, fn, tp = cm.ravel()
        all_tp += tp
        all_tn += tn
        all_fp += fp
        all_fn += fn
    metrics_cm = {}
    metrics_cm['precision'] = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    metrics_cm['recall'] = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    metrics_cm['f1'] = 2 * (metrics_cm['precision'] * metrics_cm['recall']) / (metrics_cm['precision'] + metrics_cm['recall']) if (metrics_cm['precision'] + metrics_cm['recall']) > 0 else 0.0
    metrics_cm['tpr'] = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    metrics_cm['fpr'] = all_fp / (all_fp + all_tn) if (all_fp + all_tn) > 0 else 0.0
    metrics_cm['true_positive'] = int(all_tp)
    metrics_cm['false_positive'] = int(all_fp)
    metrics_cm['true_negative'] = int(all_tn)
    metrics_cm['false_negative'] = int(all_fn)
    metrics = {
        'auc_roc': metrics_auc['auc_roc'], 'auc_pr': metrics_auc['auc_pr'],
        'f1': metrics_cm['f1'], 'precision': metrics_cm['precision'], 'recall': metrics_cm['recall'],
        'tpr': metrics_cm['tpr'], 'fpr': metrics_cm['fpr'],
        'true_positive': metrics_cm['true_positive'], 'false_positive': metrics_cm['false_positive'],
        'true_negative': metrics_cm['true_negative'], 'false_negative': metrics_cm['false_negative']
    }
    return metrics

def load_vtype_test_data(base_dir, vtype):
    vtype_safe = vtype.replace('/', '_').replace('+', '_')
    test_file = os.path.join(base_dir, f'test_data_{vtype_safe}.pkl')
    features_file = os.path.join(base_dir, f'features_test_{vtype_safe}.pkl')
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    with open(features_file, 'rb') as f:
        features_test = pickle.load(f)
    return test_data, features_test

def detect_vtype_models(model_dir, model_name, legacy_dir=None):
    model_name = model_name.lower()
    prefix = 'ocrgcn' if model_name == 'ocrgcn' else model_name
    search_dirs = [model_dir]
    if legacy_dir is not None and legacy_dir not in search_dirs:
        search_dirs.append(legacy_dir)
    for root in search_dirs:
        if not os.path.isdir(root):
            continue
        global_model_file = os.path.join(root, f'{prefix}_model.pth')
        if os.path.exists(global_model_file):
            return [None]
        model_files = glob.glob(os.path.join(root, f'{prefix}_model_*.pth'))
        vtypes = []
        for model_file in model_files:
            basename = os.path.basename(model_file)
            vtype = basename.replace(f'{prefix}_model_', '').replace('.pth', '')
            vtypes.append(vtype)
        if vtypes:
            return sorted(vtypes)
    print(f"⚠️ No models found in {model_dir}")
    return []

def run_inference_per_vtype(base_dir, edge_types, model_dir, legacy_dir, vtypes, args, model_name, use_per_vtype_files=False):
    all_scores = {}
    for vtype in vtypes:
        print(f"\n--- Running inference for VType: {vtype} ---")
        if use_per_vtype_files:
            print(f"  Loading per-vtype test data...", end=' ', flush=True)
            vtype_test_data, vtype_features_test = load_vtype_test_data(base_dir, vtype)
            print(f"✓ {len(vtype_test_data['nodes'])} nodes, {len(vtype_test_data['edges'])} edges")
        else:
            with open(os.path.join(base_dir, 'test_data.pkl'), 'rb') as f:
                test_data = pickle.load(f)
            with open(os.path.join(base_dir, 'features_test.pkl'), 'rb') as f:
                features_test = pickle.load(f)
            vtype_test_data, vtype_features_test, num_target_nodes = split_data_by_vtype(test_data, features_test, edge_types, vtype)
            print(f"  Target nodes: {num_target_nodes:,} of type '{vtype}'")
        model_path = os.path.join(model_dir, get_model_file_name(model_name, f'_{vtype}'))
        if not os.path.exists(model_path):
            legacy_model_path = os.path.join(legacy_dir, f'ocrgcn_model_{vtype}.pth') if legacy_dir else None
            if legacy_model_path and os.path.exists(legacy_model_path):
                model_path = legacy_model_path
        if not os.path.exists(model_path):
            print(f"  ⚠ Model not found: {model_path}")
            continue
        mappings_path = os.path.join(model_dir, get_mapping_file_name(model_name, f'_{vtype}'))
        if not os.path.exists(mappings_path):
            legacy_mappings_path = os.path.join(legacy_dir, f'mappings_{vtype}.pkl') if legacy_dir else None
            if legacy_mappings_path and os.path.exists(legacy_mappings_path):
                mappings_path = legacy_mappings_path
        if not os.path.exists(mappings_path):
            print(f"  ⚠ Mapping file not found: {mappings_path}")
            continue
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        edge_type_mapping = mappings['edge_type_mapping']
        node_id_to_idx = mappings['node_id_to_idx']
        model_config = mappings.get('model_config', create_base_config(args))
        trained_model_name = mappings.get('model', model_name)
        data, _, node_ids_vtype = prepare_test_data(vtype_test_data, vtype_features_test, edge_types, edge_type_mapping, node_id_to_idx, set())
        print(f"  PyG graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
        model = build_detector(trained_model_name, model_config, data.x.shape[1], len(edge_type_mapping), args.device)
        model.load_model(model_path)
        target_mask = data.target_mask if hasattr(data, 'target_mask') else None
        scores_vtype = model.predict(data.x, data.edge_index, data.edge_type, target_mask=target_mask)
        for idx, node_id in enumerate(node_ids_vtype):
            all_scores[node_id] = scores_vtype[idx]
        print(f"  ✓ Inference complete, scores: [{scores_vtype.min():.4f}, {scores_vtype.max():.4f}]")
        del model, data
        torch.cuda.empty_cache()
    all_node_ids = sorted(all_scores.keys())
    scores = np.array([all_scores[nid] for nid in all_node_ids])
    return scores, all_node_ids

DATASET_DATES = {
    "THEIA": {
        "train_start_date": "2018-04-03",
        "train_end_date": "2018-04-05",
        "test_start_date": "2018-04-09",
        "test_end_date": "2018-04-12"
    }
}

def load_from_zip_or_pkl(zip_path_part1, zip_path_part2, pkl_path, internal_name):
    if os.path.exists(zip_path_part1) and os.path.exists(zip_path_part2):
        data_part1 = None
        data_part2 = None
        
        with zipfile.ZipFile(zip_path_part1, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                data_part1 = pickle.load(f)
        
        with zipfile.ZipFile(zip_path_part2, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                data_part2 = pickle.load(f)
        
        if isinstance(data_part1, dict) and 'nodes' in data_part1:
            combined_data = {
                'nodes': {},
                'edges': [],
                'graph': None
            }
            combined_data['nodes'].update(data_part1['nodes'])
            combined_data['nodes'].update(data_part2['nodes'])
            combined_data['edges'].extend(data_part1['edges'])
            combined_data['edges'].extend(data_part2['edges'])
            return combined_data
        elif isinstance(data_part1, pd.DataFrame):
            return pd.concat([data_part1, data_part2], ignore_index=True)
        else:
            return data_part1, data_part2
    elif os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Neither zip files ({zip_path_part1}, {zip_path_part2}) nor pkl file ({pkl_path}) found")

def load_vtype_from_zip_or_pkl(base_dir, vtype_safe, file_prefix):
    zip_path_part1 = os.path.join(base_dir, f'{file_prefix}_{vtype_safe}_part1.pkl.zip')
    zip_path_part2 = os.path.join(base_dir, f'{file_prefix}_{vtype_safe}_part2.pkl.zip')
    pkl_path = os.path.join(base_dir, f'{file_prefix}_{vtype_safe}.pkl')
    internal_name = f'{file_prefix}_{vtype_safe}.pkl'
    
    return load_from_zip_or_pkl(zip_path_part1, zip_path_part2, pkl_path, internal_name)

def detect_per_vtype_test_files(base_dir):
    vtypes_list_file = os.path.join(base_dir, 'vtypes_list.pkl')
    
    if os.path.exists(vtypes_list_file):
        with open(vtypes_list_file, 'rb') as f:
            vtype_info = pickle.load(f)
        return vtype_info['vtypes'], vtype_info['vtype_counts']
    
    test_data_pkl = os.path.join(base_dir, 'test_data.pkl')
    test_data_zip_part1 = os.path.join(base_dir, 'test_data_part1.pkl.zip')
    test_data_zip_part2 = os.path.join(base_dir, 'test_data_part2.pkl.zip')
    
    if os.path.exists(test_data_pkl) or (os.path.exists(test_data_zip_part1) and os.path.exists(test_data_zip_part2)):
        return None, None
    
    print(f"No data files found in {base_dir}")
    return None, None

def load_vtype_test_data(base_dir, vtype):
    vtype_safe = vtype.replace('/', '_').replace('+', '_')
    
    test_file_zip_part1 = os.path.join(base_dir, f'test_data_{vtype_safe}_part1.pkl.zip')
    test_file_zip_part2 = os.path.join(base_dir, f'test_data_{vtype_safe}_part2.pkl.zip')
    test_file = os.path.join(base_dir, f'test_data_{vtype_safe}.pkl')
    
    features_file_zip_part1 = os.path.join(base_dir, f'features_test_{vtype_safe}_part1.pkl.zip')
    features_file_zip_part2 = os.path.join(base_dir, f'features_test_{vtype_safe}_part2.pkl.zip')
    features_file = os.path.join(base_dir, f'features_test_{vtype_safe}.pkl')
    
    if os.path.exists(test_file_zip_part1) and os.path.exists(test_file_zip_part2):
        internal_name = f'test_data_{vtype_safe}.pkl'
        test_data_part1 = None
        test_data_part2 = None
        
        with zipfile.ZipFile(test_file_zip_part1, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                test_data_part1 = pickle.load(f)
        
        with zipfile.ZipFile(test_file_zip_part2, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                test_data_part2 = pickle.load(f)
        
        test_data = {
            'nodes': {},
            'edges': [],
            'graph': None
        }
        test_data['nodes'].update(test_data_part1['nodes'])
        test_data['nodes'].update(test_data_part2['nodes'])
        test_data['edges'].extend(test_data_part1['edges'])
        test_data['edges'].extend(test_data_part2['edges'])
    elif os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
    else:
        raise FileNotFoundError(f"Test data file not found for vtype {vtype}")
    
    if os.path.exists(features_file_zip_part1) and os.path.exists(features_file_zip_part2):
        internal_name = f'features_test_{vtype_safe}.pkl'
        features_part1 = None
        features_part2 = None
        
        with zipfile.ZipFile(features_file_zip_part1, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                features_part1 = pickle.load(f)
        
        with zipfile.ZipFile(features_file_zip_part2, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                features_part2 = pickle.load(f)
        
        features_test = pd.concat([features_part1, features_part2], ignore_index=True)
    elif os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            features_test = pickle.load(f)
    else:
        raise FileNotFoundError(f"Features file not found for vtype {vtype}")
    
    return test_data, features_test

def evaluate_ocrgcn_model(base_dir, model_path, model_config, dataset_dates, dataset, embedding=None, use_per_vtype_files=False, is_hypersearch=False):
    try:
        if not os.path.exists(base_dir):
            return pd.DataFrame()
        
        import traceback
        
        vtypes_from_files, vtype_counts_from_files = detect_per_vtype_test_files(base_dir)
        
        if vtypes_from_files is not None:
            use_per_vtype_files = True
            
            with open(os.path.join(base_dir, 'edge_types.pkl'), 'rb') as f:
                edge_types = pickle.load(f)
            
            test_data = {'nodes': {}, 'edges': []}
            all_test_features = []
            
            for vtype in vtypes_from_files:
                vtype_safe = vtype.replace('/', '_').replace('+', '_')
                vtype_test_data = load_vtype_from_zip_or_pkl(base_dir, vtype_safe, 'test_data')
                vtype_features_test = load_vtype_from_zip_or_pkl(base_dir, vtype_safe, 'features_test')
                test_data['nodes'].update(vtype_test_data['nodes'])
                test_data['edges'].extend(vtype_test_data['edges'])
                all_test_features.append(vtype_features_test)
            
            features_test = pd.concat(all_test_features, ignore_index=True)
        else:
            use_per_vtype_files = False
            
            test_data_zip_part1 = os.path.join(base_dir, 'test_data_part1.pkl.zip')
            test_data_zip_part2 = os.path.join(base_dir, 'test_data_part2.pkl.zip')
            test_data_path = os.path.join(base_dir, 'test_data.pkl')
            
            features_test_zip_part1 = os.path.join(base_dir, 'features_test_part1.pkl.zip')
            features_test_zip_part2 = os.path.join(base_dir, 'features_test_part2.pkl.zip')
            features_test_path = os.path.join(base_dir, 'features_test.pkl')
            
            edge_types_path = os.path.join(base_dir, 'edge_types.pkl')
            
            if not os.path.exists(edge_types_path):
                return pd.DataFrame()
            
            if not ((os.path.exists(test_data_zip_part1) and os.path.exists(test_data_zip_part2)) or os.path.exists(test_data_path)):
                return pd.DataFrame()
            
            if not ((os.path.exists(features_test_zip_part1) and os.path.exists(features_test_zip_part2)) or os.path.exists(features_test_path)):
                return pd.DataFrame()
            
            test_data = load_from_zip_or_pkl(test_data_zip_part1, test_data_zip_part2, test_data_path, 'test_data.pkl')
            features_test = load_from_zip_or_pkl(features_test_zip_part1, features_test_zip_part2, features_test_path, 'features_test.pkl')
            
            with open(edge_types_path, 'rb') as f:
                edge_types = pickle.load(f)
        
        if is_hypersearch:
            model_root = os.path.join(base_dir, 'ocrgcn')
            hypersearch_root = os.path.join(model_root, 'hypersearch_models')
            
            model_8_dirs = glob.glob(os.path.join(hypersearch_root, 'model_8*'))
            has_per_vtype_models = any('_' in os.path.basename(d) for d in model_8_dirs)
            
            if has_per_vtype_models:
                vtypes_for_models = []
                for d in model_8_dirs:
                    basename = os.path.basename(d)
                    if basename.startswith('model_8_'):
                        vtype = basename.replace('model_8_', '')
                        checkpoint_path = os.path.join(d, 'checkpoint_epoch_50.pth')
                        if os.path.exists(checkpoint_path):
                            vtypes_for_models.append(vtype)
                
                if not vtypes_for_models:
                    return pd.DataFrame()
                
                checkpoint = None
                model_config_loaded = model_config.copy()
            else:
                if not os.path.exists(model_path):
                    return pd.DataFrame()
                vtypes_for_models = [None]
                checkpoint = torch.load(model_path, map_location=device)
                if 'config' in checkpoint:
                    model_config_loaded = checkpoint['config'].copy()
                else:
                    model_config_loaded = model_config.copy()
            
            edge_type_mapping = {etype: idx for idx, etype in enumerate(sorted(edge_types))}
            node_id_to_idx = {}
            trained_model_name = 'ocrgcn'
        else:
            has_per_vtype_models = False
            vtypes_for_models = None
            
            model_root = os.path.join(base_dir, 'ocrgcn')
            original_model_dir = os.path.join(model_root, 'original')
            vtypes = detect_vtype_models(original_model_dir, 'ocrgcn', legacy_dir=base_dir)
            
            if not vtypes:
                return pd.DataFrame()
            
            trained_model_name = 'ocrgcn'
            model_config_loaded = model_config.copy()
            
            if vtypes[0] is None:
                mapping_path = os.path.join(original_model_dir, get_mapping_file_name('ocrgcn', ''))
                if not os.path.exists(mapping_path):
                    legacy_mapping_path = os.path.join(base_dir, 'mappings.pkl')
                    if os.path.exists(legacy_mapping_path):
                        mapping_path = legacy_mapping_path
                if not os.path.exists(mapping_path):
                    return pd.DataFrame()
                
                with open(mapping_path, 'rb') as f:
                    mappings = pickle.load(f)
                
                edge_type_mapping = mappings['edge_type_mapping']
                node_id_to_idx = mappings['node_id_to_idx']
                trained_model_name = mappings.get('model', 'ocrgcn')
                model_config_loaded = mappings.get('model_config', model_config_loaded)
            else:
                edge_type_mapping = {etype: idx for idx, etype in enumerate(sorted(edge_types))}
                node_id_to_idx = {}
        
        hash_to_uuid_path = os.path.join(base_dir, 'hash_to_uuid.json')
        if os.path.exists(hash_to_uuid_path):
            with open(hash_to_uuid_path, 'r') as f:
                hash_to_uuid = json.load(f)
            uuid_to_hash = {v: k for k, v in hash_to_uuid.items()}
        else:
            uuid_to_hash = None
        
        node_to_dates = build_node_to_dates_mapping(test_data)
        
        if is_hypersearch and has_per_vtype_models and vtypes_for_models and vtypes_for_models[0] is not None:
            all_scores = {}
            all_node_ids_list = []
            
            for vtype in vtypes_for_models:
                vtype_safe = vtype.replace('/', '_').replace('+', '_')
                vtype_test_data = load_vtype_from_zip_or_pkl(base_dir, vtype_safe, 'test_data')
                vtype_features_test = load_vtype_from_zip_or_pkl(base_dir, vtype_safe, 'features_test')
                
                vtype_model_path = os.path.join(hypersearch_root, f'model_8_{vtype}', 'checkpoint_epoch_50.pth')
                if not os.path.exists(vtype_model_path):
                    continue
                
                data_vtype, _, node_ids_vtype = prepare_pyg_data(
                    vtype_test_data, vtype_features_test, edge_types
                )
                
                vtype_checkpoint = torch.load(vtype_model_path, map_location=device)
                if 'config' in vtype_checkpoint:
                    vtype_config = vtype_checkpoint['config'].copy()
                else:
                    vtype_config = model_config_loaded.copy()
                
                model = build_detector(
                    trained_model_name,
                    vtype_config,
                    data_vtype.x.shape[1],
                    len(edge_type_mapping),
                    device
                )
                model.load_model(vtype_model_path)
                
                target_mask = data_vtype.target_mask if hasattr(data_vtype, 'target_mask') else None
                scores_vtype = model.predict(data_vtype.x, data_vtype.edge_index, data_vtype.edge_type, target_mask=target_mask)
                
                for idx, node_id in enumerate(node_ids_vtype):
                    all_scores[node_id] = scores_vtype[idx]
                    all_node_ids_list.append(node_id)
                
                del model, data_vtype
                torch.cuda.empty_cache()
            
            node_ids = sorted(set(all_node_ids_list))
            scores = np.array([all_scores.get(nid, 0.0) for nid in node_ids])
        elif is_hypersearch:
            all_malicious = set()
            data, _, node_ids = prepare_test_data(
                test_data, features_test, edge_types,
                edge_type_mapping, node_id_to_idx, all_malicious
            )
            
            model = build_detector(
                trained_model_name,
                model_config_loaded,
                data.x.shape[1],
                len(edge_type_mapping),
                device
            )
            model.load_model(model_path)
            
            target_mask = data.target_mask if hasattr(data, 'target_mask') else None
            scores = model.predict(data.x, data.edge_index, data.edge_type, target_mask=target_mask)
        else:
            model_root = os.path.join(base_dir, 'ocrgcn')
            original_model_dir = os.path.join(model_root, 'original')
            vtypes = detect_vtype_models(original_model_dir, 'ocrgcn', legacy_dir=base_dir)
            
            if vtypes[0] is not None:
                class Args:
                    def __init__(self):
                        self.model = trained_model_name
                        self.device = device
                        self.contamination = model_config_loaded.get('contamination', 0.001)
                        self.rulellm = False
                        self.llmlabel = False
                        self.llmfunc = False
                        self.hid_dim = model_config_loaded.get('hid_dim', 32)
                        self.num_layers = model_config_loaded.get('num_layers', 3)
                        self.dropout = model_config_loaded.get('dropout', 0.0)
                        self.lr = model_config_loaded.get('lr', 0.005)
                        self.epoch = model_config_loaded.get('epoch', 100)
                        self.beta = model_config_loaded.get('beta', 0.5)
                        self.warmup = model_config_loaded.get('warmup', 2)
                        self.eps = model_config_loaded.get('eps', 0.1)
                
                with redirect_stdout(StringIO()):
                    scores, node_ids = run_inference_per_vtype(
                        base_dir, edge_types, original_model_dir, base_dir, vtypes, 
                        Args(), trained_model_name, use_per_vtype_files
                    )
                
                if test_data is None or 'nodes' not in test_data or not test_data['nodes']:
                    test_data = {'nodes': {}, 'edges': []}
                    if use_per_vtype_files:
                        for vtype_load in vtypes:
                            if vtype_load is not None:
                                vtype_safe = vtype_load.replace('/', '_').replace('+', '_')
                                vtype_test_data = load_vtype_from_zip_or_pkl(base_dir, vtype_safe, 'test_data')
                            else:
                                vtype_test_data = load_from_zip_or_pkl(
                                    os.path.join(base_dir, 'test_data_part1.pkl.zip'),
                                    os.path.join(base_dir, 'test_data_part2.pkl.zip'),
                                    os.path.join(base_dir, 'test_data.pkl'),
                                    'test_data.pkl'
                                )
                            test_data['nodes'].update(vtype_test_data['nodes'])
                            test_data['edges'].extend(vtype_test_data['edges'])
            else:
                all_malicious = set()
                data, _, node_ids = prepare_test_data(
                    test_data, features_test, edge_types,
                    edge_type_mapping, node_id_to_idx, all_malicious
                )
                
                model = build_detector(
                    trained_model_name,
                    model_config_loaded,
                    data.x.shape[1],
                    len(edge_type_mapping),
                    device
                )
                
                if not os.path.exists(model_path):
                    legacy_model_path = os.path.join(base_dir, 'ocrgcn_model.pth')
                    if os.path.exists(legacy_model_path):
                        model_path = legacy_model_path
                
                if not os.path.exists(model_path):
                    return pd.DataFrame()
                
                model.load_model(model_path)
                
                target_mask = data.target_mask if hasattr(data, 'target_mask') else None
                scores = model.predict(data.x, data.edge_index, data.edge_type, target_mask=target_mask)
        
        attack_scenarios = get_attack_scenarios(dataset, '../PIDS_GT')
        if not attack_scenarios:
            return pd.DataFrame()
        
        node_id_to_test_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        results_list = []
        
        for attack_name, csv_path, attack_test_date in attack_scenarios:
            if not os.path.exists(csv_path):
                continue
            
            GT_mal_uuids = load_malicious_ids_from_csv(csv_path)
            
            if uuid_to_hash:
                GT_mal_hashes = set()
                for uuid in GT_mal_uuids:
                    if uuid in uuid_to_hash:
                        GT_mal_hashes.add(uuid_to_hash[uuid])
                GT_mal_in_test = GT_mal_hashes.intersection(set(node_ids))
            else:
                GT_mal_in_test = GT_mal_uuids.intersection(set(node_ids))
            
            labels = np.zeros(len(node_ids), dtype=int)
            malicious_count = 0
            
            for node_uuid in GT_mal_in_test:
                if node_uuid in node_to_dates and attack_test_date in node_to_dates[node_uuid]:
                    test_idx = node_id_to_test_idx[node_uuid]
                    labels[test_idx] = 1
                    malicious_count += 1
            
            if malicious_count == 0:
                continue
            
            if is_hypersearch and has_per_vtype_models and vtypes_for_models and vtypes_for_models[0] is not None:
                metrics = compute_metrics_hybrid(labels, scores, node_ids, test_data, contamination=model_config_loaded.get('contamination', 0.001))
            else:
                contamination_value = model_config_loaded.get('contamination', 0.001)
                metrics = compute_metrics(labels, scores, contamination=contamination_value)
            
            adp_value = compute_attack_detection_precision(scores, {attack_name: labels})
            metrics['adp'] = adp_value if adp_value is not None else 0.0
            
            results_list.append({
                'attack_name': attack_name,
                'test_date': attack_test_date,
                'AUC_ROC': float(metrics['auc_roc']),
                'AUC_PR': float(metrics['auc_pr']),
                'ADP': float(metrics['adp']),
                'malicious_nodes': int(malicious_count)
            })
        
        return pd.DataFrame(results_list)
    except Exception as e:
        import traceback
        print(f"Error in evaluate_ocrgcn_model for {base_dir}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def main():
    dataset = "THEIA"
    dataset_dates = DATASET_DATES[dataset]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    input_dir = os.path.join(autoprov_dir, 'BIGDATA', 'OCR_APT_artifacts')
    
    try:
        base_dir_baseline = os.path.join(input_dir, dataset.lower())
        model_path_baseline = os.path.join(base_dir_baseline, 'ocrgcn', 'original', 'ocrgcn_model.pth')
        
        baseline_config = {
            'hid_dim': 32,
            'num_layers': 3,
            'dropout': 0.0,
            'lr': 0.005,
            'epoch': 100,
            'beta': 0.5,
            'contamination': 0.001,
            'warmup': 2,
            'eps': 0.1
        }
        
        if not os.path.exists(base_dir_baseline):
            print(f"Warning: Baseline artifacts directory not found: {base_dir_baseline}")
            print("Skipping baseline evaluation.")
            results_baseline = pd.DataFrame()
        else:
            results_baseline = evaluate_ocrgcn_model(
                base_dir_baseline, model_path_baseline, baseline_config,
                dataset_dates, dataset, embedding=None, use_per_vtype_files=False
            )
        
        embedding = 'mpnet'
        base_dir_model8 = os.path.join(input_dir, f'{dataset.lower()}_rulellm_llmlabel_{embedding}')
        model_dir_model8 = os.path.join(base_dir_model8, 'ocrgcn', 'hypersearch_models', 'model_8')
        model_path_model8 = os.path.join(model_dir_model8, 'checkpoint_epoch_50.pth')
        
        model8_config = {
            'hid_dim': 32,
            'num_layers': 3,
            'dropout': 0.1,
            'lr': 0.005,
            'epoch': 50,
            'beta': 0.5,
            'contamination': 0.001,
            'warmup': 10,
            'eps': 0.1
        }
        
        if not os.path.exists(base_dir_model8):
            print(f"Warning: AutoProv artifacts directory not found: {base_dir_model8}")
            print("Skipping AutoProv evaluation.")
            results_model8 = pd.DataFrame()
        else:
            results_model8 = evaluate_ocrgcn_model(
                base_dir_model8, model_path_model8, model8_config,
                dataset_dates, dataset, embedding=embedding, use_per_vtype_files=True, is_hypersearch=True
            )
        
        if not results_baseline.empty:
            display_cols = ['attack_name', 'AUC_ROC', 'AUC_PR', 'ADP']
            baseline_display = results_baseline[display_cols].copy()
            baseline_display['AUC_ROC'] = baseline_display['AUC_ROC'].round(3)
            baseline_display['AUC_PR'] = baseline_display['AUC_PR'].round(3)
            baseline_display['ADP'] = baseline_display['ADP'].round(3)
            print("\nBaseline Model Results:")
            print(baseline_display.to_string(index=False))
        else:
            print("\nNo baseline results to display.")
        
        if not results_model8.empty:
            display_cols = ['attack_name', 'AUC_ROC', 'AUC_PR', 'ADP']
            model8_display = results_model8[display_cols].copy()
            model8_display['AUC_ROC'] = model8_display['AUC_ROC'].round(3)
            model8_display['AUC_PR'] = model8_display['AUC_PR'].round(3)
            model8_display['ADP'] = model8_display['ADP'].round(3)
            print("\n\nAuto-Prov Results:")
            print(model8_display.to_string(index=False))
        else:
            print("\nNo AutoProv results to display.")
    
    except Exception as e:
        import traceback
        print(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

