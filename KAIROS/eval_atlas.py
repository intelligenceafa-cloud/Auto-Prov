#!/usr/bin/env python3

import argparse
import os
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import LastNeighborLoader
from tqdm import tqdm
import pickle
import numpy as np
import glob
import json
import pandas as pd
from collections import defaultdict
import zipfile
import tempfile
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ATTACK_TYPE_MAPPING = {
    'S1': 'Strategic web compromise',
    'S2': 'Malvertising dominate',
    'S3': 'Spam campaign',
    'S4': 'Pony campaign'
}

from sklearn.metrics import roc_auc_score, average_precision_score

def load_from_zip_or_file(file_path):
    if os.path.exists(file_path + '.zip'):
        with zipfile.ZipFile(file_path + '.zip', 'r') as zip_ref:
            file_name = os.path.basename(file_path)
            with zip_ref.open(file_name) as f:
                return f.read()
    elif os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return f.read()
    else:
        return None

def load_pickle_from_zip_or_file(file_path):
    data = load_from_zip_or_file(file_path)
    if data is None:
        return None
    return pickle.loads(data)

def load_json_from_zip_or_file(file_path):
    data = load_from_zip_or_file(file_path)
    if data is None:
        return None
    return json.loads(data.decode('utf-8'))

def load_torch_from_zip_or_file(file_path, map_location='cpu'):
    if file_path.endswith('.zip'):
        zip_path = file_path
        file_name = os.path.basename(file_path).replace('.zip', '')
    elif os.path.exists(file_path + '.zip'):
        zip_path = file_path + '.zip'
        file_name = os.path.basename(file_path)
    else:
        if os.path.exists(file_path):
            return torch.load(file_path, map_location=map_location, weights_only=False)
        else:
            return None
    
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmp_dir:
                extracted_path = os.path.join(tmp_dir, file_name)
                zip_ref.extract(file_name, tmp_dir)
                data = torch.load(extracted_path, map_location=map_location, weights_only=False)
                return data
    else:
        return None

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=8,
                                    dropout=0.0, edge_dim=edge_dim)
        self.conv2 = TransformerConv(in_channels*8, out_channels, heads=1, concat=False,
                             dropout=0.0, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        last_update = last_update.to(device)
        x = x.to(device)
        t = t.to(device)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, num_edge_types=4):
        super(LinkPredictor, self).__init__()
        self.lin_src = nn.Linear(in_channels, in_channels*2)
        self.lin_dst = nn.Linear(in_channels, in_channels*2)
        
        self.lin_seq = nn.Sequential(
            nn.Linear(in_channels*4, in_channels*8),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(in_channels*8, in_channels*2),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(in_channels*2, in_channels//2),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(in_channels//2, num_edge_types)
        )
        
    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)      
        h = self.lin_seq(h)
        return h

def tensor_find(t, x):
    t_np = t.cpu().numpy()
    idx = np.argwhere(t_np == x)
    return idx[0][0] + 1 if len(idx) > 0 else 0

def mean(data):
    return np.mean(np.array(data))

def std(data):
    return np.std(np.array(data))

def compute_attack_detection_precision(scores, attack_label_dict):
    if not attack_label_dict:
        return None
    
    num_edges = len(scores)
    if num_edges == 0:
        return None
    
    attack_indices = {}
    index_to_attacks = {}
    y_global = np.zeros(num_edges, dtype=np.int8)
    
    for attack_name, labels in attack_label_dict.items():
        if labels is None or len(labels) != num_edges:
            return None
        positive_idx = np.flatnonzero(labels > 0)
        if positive_idx.size > 0:
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
    unique_detections = np.array(
        [precision_to_detection[key] for key in unique_precisions], dtype=np.float64
    )
    
    if unique_precisions[0] > 0.0:
        unique_precisions = np.insert(unique_precisions, 0, 0.0)
        unique_detections = np.insert(unique_detections, 0, 0.0)
    if unique_precisions[-1] < 1.0:
        unique_precisions = np.append(unique_precisions, 1.0)
        unique_detections = np.append(unique_detections, unique_detections[-1])
    
    adp = np.trapz(unique_detections, unique_precisions)
    adp = float(max(0.0, min(1.0, adp)))
    return adp

@torch.no_grad()
def test_window(inference_data, memory, gnn, link_pred, neighbor_loader, nodeid2msg, 
                criterion, batch_size=512, num_edge_types=4):
    if hasattr(memory, 'msg_s_store'):
        memory.msg_s_store = {}
    if hasattr(memory, 'msg_d_store'):
        memory.msg_d_store = {}
    
    memory.reset_state()
    neighbor_loader.reset_state()
    
    max_node_num = memory.memory.size(0)
    neighbor_loader._assoc = torch.empty(max_node_num, dtype=torch.long, device='cpu')
    
    if hasattr(neighbor_loader, 'device'):
        neighbor_loader.device = torch.device('cpu')
    
    memory.training = False
    gnn.eval()
    link_pred.eval()
    
    edge_list = []
    
    edge_type_map = {
        1: 'resolve', 2: 'web_request', 3: 'read', 4: 'connect'
    }
    
    num_events = len(inference_data.src)
    
    for batch_idx in tqdm(range(0, num_events, batch_size), desc="Inference", leave=False):
        end_idx = min(batch_idx + batch_size, num_events)
        
        src = inference_data.src[batch_idx:end_idx]
        pos_dst = inference_data.dst[batch_idx:end_idx]
        t = inference_data.t[batch_idx:end_idx]
        msg = inference_data.msg[batch_idx:end_idx]
        
        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        
        src = src.to(device)
        pos_dst = pos_dst.to(device)
        t = t.to(device)
        msg = msg.to(device)
        n_id = n_id.to(device)
        edge_index = edge_index.to(device)
        
        assoc = torch.empty(inference_data.num_nodes, dtype=torch.long, device=device)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index,
                inference_data.t[e_id.cpu()].to(device),
                inference_data.msg[e_id.cpu()].to(device))
        
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        
        y_true = []
        for m in msg:
            edge_type_vec = m[16:16+num_edge_types]
            l = tensor_find(edge_type_vec, 1) - 1
            y_true.append(l)
        y_true = torch.tensor(y_true).reshape(-1).to(torch.long).to(device)
        
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src.cpu(), pos_dst.cpu())
        
        for i in range(len(pos_out)):
            edge_loss = criterion(pos_out[i].reshape(1, -1), y_true[i].reshape(-1))
            
            srcnode = int(src[i])
            dstnode = int(pos_dst[i])
            srcmsg = str(nodeid2msg.get(srcnode, 'unknown'))
            dstmsg = str(nodeid2msg.get(dstnode, 'unknown'))
            
            edge_list.append({
                'loss': float(edge_loss),
                'srcnode': srcnode,
                'dstnode': dstnode,
                'srcmsg': srcmsg,
                'dstmsg': dstmsg,
                'edge_type': edge_type_map.get(tensor_find(msg[i][16:16+num_edge_types], 1), 'UNKNOWN'),
                'time': int(t[i])
            })
        
        del src, pos_dst, t, msg, n_id, edge_index, e_id, assoc, z, last_update, pos_out, y_true
    
    torch.cuda.empty_cache()
    
    return edge_list

def calculate_metrics(y_true, y_scores, attack_label_dict=None):
    auc_roc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.0
    auc_pr = average_precision_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.0
    
    if attack_label_dict is not None and len(attack_label_dict) > 0:
        adp = compute_attack_detection_precision(y_scores, attack_label_dict)
        adp = float(adp) if adp is not None else 0.0
    else:
        adp = None
    
    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'adp': adp
    }

def load_model(model_path, device):
    model_data = load_torch_from_zip_or_file(model_path, map_location='cpu')
    if model_data is None:
        return None
    
    if 'memory' in model_data:
        memory = model_data['memory'].to(device)
        gnn = model_data['gnn'].to(device)
        link_pred = model_data['link_pred'].to(device)
        neighbor_loader = model_data['neighbor_loader']
        num_edge_types = model_data.get('config', {}).get('num_edge_types', 4)
    else:
        return load_model_for_eval_infer(model_path, device)
    
    if hasattr(neighbor_loader, 'device'):
        neighbor_loader.device = torch.device('cpu')
    
    if hasattr(memory, 'msg_s_store'):
        memory.msg_s_store = {}
    if hasattr(memory, 'msg_d_store'):
        memory.msg_d_store = {}
    
    memory.training = False
    memory.reset_state()
    neighbor_loader.reset_state()
    
    max_node_num = memory.memory.size(0)
    neighbor_loader._assoc = torch.empty(max_node_num, dtype=torch.long, device='cpu')
    
    del model_data
    torch.cuda.empty_cache()
    
    return memory, gnn, link_pred, neighbor_loader, num_edge_types

@torch.no_grad()
def test_window_for_eval(inference_data, memory, gnn, link_pred, neighbor_loader, nodeid2msg, 
                         criterion, batch_size=512, num_edge_types=4):
    if hasattr(memory, 'msg_s_store') and len(memory.msg_s_store) > 0:
        memory.msg_s_store.clear()
    if hasattr(memory, 'msg_d_store') and len(memory.msg_d_store) > 0:
        memory.msg_d_store.clear()
    
    memory.reset_state()
    neighbor_loader.reset_state()
    
    max_node_num = memory.memory.size(0)
    neighbor_loader._assoc = torch.empty(max_node_num, dtype=torch.long, device='cpu')
    
    if hasattr(neighbor_loader, 'device'):
        neighbor_loader.device = torch.device('cpu')
    
    memory.training = False
    gnn.eval()
    link_pred.eval()
    
    edge_list = []
    
    num_events = len(inference_data.src)
    
    for batch_idx in range(0, num_events, batch_size):
        end_idx = min(batch_idx + batch_size, num_events)
        
        src = inference_data.src[batch_idx:end_idx]
        pos_dst = inference_data.dst[batch_idx:end_idx]
        t = inference_data.t[batch_idx:end_idx]
        msg = inference_data.msg[batch_idx:end_idx]
        
        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        
        src = src.to(device, non_blocking=True)
        pos_dst = pos_dst.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        msg = msg.to(device, non_blocking=True)
        n_id = n_id.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        
        assoc = torch.empty(inference_data.num_nodes, dtype=torch.long, device=device)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index,
                inference_data.t[e_id.cpu()].to(device),
                inference_data.msg[e_id.cpu()].to(device))
        
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        
        node_feat_dim = 16
        edge_type_vecs = msg[:, node_feat_dim:node_feat_dim+num_edge_types]
        y_true = torch.argmax(edge_type_vecs, dim=1).to(torch.long)
        
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src.cpu(), pos_dst.cpu())
        
        criterion_none = CrossEntropyLoss(reduction='none')
        batch_losses = criterion_none(pos_out, y_true)
        
        batch_losses_list = batch_losses.cpu().tolist()
        src_list = src.cpu().tolist()
        dst_list = pos_dst.cpu().tolist()
        
        if isinstance(batch_losses_list, (int, float)):
            batch_losses_list = [batch_losses_list]
        
        for i in range(len(batch_losses_list)):
            edge_list.append({
                'loss': float(batch_losses_list[i]),
                'srcnode': int(src_list[i]),
                'dstnode': int(dst_list[i]),
            })
        
        del src, pos_dst, t, msg, n_id, edge_index, e_id, assoc, z, last_update, pos_out, y_true, batch_losses
    
    torch.cuda.empty_cache()
    
    return edge_list

def load_model_for_eval_infer(model_path, device):
    model_data = load_torch_from_zip_or_file(model_path, map_location='cpu')
    if model_data is None:
        return None
    
    if 'memory' in model_data:
        memory = model_data['memory'].to(device)
        gnn = model_data['gnn'].to(device)
        link_pred = model_data['link_pred'].to(device)
        neighbor_loader = model_data['neighbor_loader']
        num_edge_types = model_data.get('config', {}).get('num_edge_types', 4)
    else:
        config = model_data.get('config', {})
        max_node_num = config.get('max_node_num')
        msg_dim = config.get('msg_dim')
        memory_dim = config.get('memory_dim', 100)
        time_dim = config.get('time_dim', 100)
        embedding_dim = config.get('embedding_dim', 100)
        num_edge_types = config.get('num_edge_types', 4)
        
        if max_node_num is None:
            if 'memory_state_dict' in model_data:
                memory_state = model_data['memory_state_dict']
                if 'memory' in memory_state:
                    max_node_num = memory_state['memory'].shape[0]
                else:
                    for key, value in memory_state.items():
                        if 'memory' in key.lower() and isinstance(value, torch.Tensor):
                            if len(value.shape) >= 1:
                                max_node_num = value.shape[0]
                                break
            
            if max_node_num is None:
                model_dir = os.path.dirname(model_path)
                base_dir = os.path.dirname(model_dir)
                nodeid2msg_path = f'{base_dir}/processed_data/node_mappings/nodeid2msg.pkl'
                try:
                    nodeid2msg = load_pickle_from_zip_or_file(nodeid2msg_path)
                    if nodeid2msg is not None:
                        max_node_num = len([k for k in nodeid2msg.keys() if isinstance(k, int)])
                except Exception as e:
                    pass
        
        if max_node_num is None or max_node_num <= 0:
            raise ValueError(f"Cannot reconstruct model: max_node_num is {max_node_num} (must be > 0)")
        
        if msg_dim is None:
            if 'memory_state_dict' in model_data:
                memory_state = model_data['memory_state_dict']
                if 'message_module.lin_src.weight' in memory_state:
                    lin_src_input_dim = memory_state['message_module.lin_src.weight'].shape[1]
                    inferred_raw_msg_dim = lin_src_input_dim - memory_dim - time_dim
                    if inferred_raw_msg_dim > 0:
                        msg_dim = inferred_raw_msg_dim
                elif 'message_module.lin.weight' in memory_state:
                    lin_input_dim = memory_state['message_module.lin.weight'].shape[1]
                    inferred_raw_msg_dim = lin_input_dim - memory_dim - time_dim
                    if inferred_raw_msg_dim > 0:
                        msg_dim = inferred_raw_msg_dim
            
            if msg_dim is None:
                model_dir = os.path.dirname(model_path)
                base_dir = os.path.dirname(model_dir)
                nodeid2msg_path = f'{base_dir}/processed_data/node_mappings/nodeid2msg.pkl'
                try:
                    nodeid2msg = load_pickle_from_zip_or_file(nodeid2msg_path)
                    if nodeid2msg:
                        sample_msg = next(iter(nodeid2msg.values()))
                        if isinstance(sample_msg, torch.Tensor):
                            msg_dim = sample_msg.shape[0]
                        elif isinstance(sample_msg, np.ndarray):
                            msg_dim = sample_msg.shape[0]
                except Exception as e:
                    pass
            
            if msg_dim is None:
                msg_dim = 36
        
        if msg_dim is None or msg_dim <= 0:
            raise ValueError(f"Cannot reconstruct model: msg_dim is {msg_dim} (must be > 0)")
        
        from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator, LastNeighborLoader
        
        memory = TGNMemory(
            max_node_num, msg_dim, memory_dim, time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)
        gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=msg_dim,
            time_enc=memory.time_enc,
        ).to(device)
        link_pred = LinkPredictor(in_channels=embedding_dim, num_edge_types=num_edge_types).to(device)
        neighbor_loader = LastNeighborLoader(max_node_num, size=20, device='cpu')
        
        memory.load_state_dict(model_data['memory_state_dict'])
        gnn.load_state_dict(model_data['gnn_state_dict'])
        link_pred.load_state_dict(model_data['link_pred_state_dict'])
    
    if hasattr(neighbor_loader, 'device'):
        neighbor_loader.device = torch.device('cpu')
    
    if hasattr(memory, 'msg_s_store'):
        memory.msg_s_store = {}
    if hasattr(memory, 'msg_d_store'):
        memory.msg_d_store = {}
    
    memory.training = False
    memory.reset_state()
    neighbor_loader.reset_state()
    
    max_node_num = memory.memory.size(0)
    neighbor_loader._assoc = torch.empty(max_node_num, dtype=torch.long, device='cpu')
    
    del model_data
    torch.cuda.empty_cache()
    
    return memory, gnn, link_pred, neighbor_loader, num_edge_types

def load_log_level_ground_truth(dataset, timestamp, ground_truth_dir=None):
    if ground_truth_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        ground_truth_dir = os.path.join(project_root, 'STEP-LLM', 'rule_generator', 'ATLAS', 'ablation', 'log_level_ground_truth')
    
    timestamp_dir = os.path.join(ground_truth_dir, dataset, timestamp)
    
    log_level_labels = {}
    total_logs = {}
    malicious_log_indices = {}
    
    labels_path = os.path.join(timestamp_dir, 'log_level_labels.pkl')
    if os.path.exists(labels_path):
        try:
            with open(labels_path, 'rb') as f:
                log_level_labels = pickle.load(f)
        except Exception as e:
            pass
    
    total_logs_path = os.path.join(timestamp_dir, 'total_logs.json')
    if os.path.exists(total_logs_path):
        try:
            with open(total_logs_path, 'r') as f:
                total_logs = json.load(f)
        except Exception as e:
            pass
    
    malicious_indices_path = os.path.join(timestamp_dir, 'malicious_log_indices.pkl')
    if os.path.exists(malicious_indices_path):
        try:
            with open(malicious_indices_path, 'rb') as f:
                malicious_log_indices = pickle.load(f)
        except Exception as e:
            pass
    
    return log_level_labels, total_logs, malicious_log_indices

def load_edge_to_log_mapping(artifacts_dir, window_name, split='test'):
    mapping_path = os.path.join(artifacts_dir, split, 'edge_to_log_mapping', f'mapping_{window_name}.pkl')
    
    edge_to_log_mapping = {}
    
    try:
        data = load_pickle_from_zip_or_file(mapping_path)
        if data is not None:
            edge_to_log_mapping = data
    except Exception as e:
        pass
    
    return edge_to_log_mapping

def aggregate_edge_losses_to_logs(edge_list, edge_to_log_mapping):
    log_losses = defaultdict(list)
    
    for edge_idx, edge in enumerate(edge_list):
        loss = edge['loss']
        
        if edge_idx in edge_to_log_mapping:
            log_key = edge_to_log_mapping[edge_idx]
            log_losses[log_key].append(loss)
    
    log_scores = {}
    log_edge_counts = {}
    extracted_logs = set()
    
    for log_key, losses in log_losses.items():
        log_edge_counts[log_key] = len(losses)
        log_scores[log_key] = max(losses)
        extracted_logs.add(log_key)
    
    return log_scores, log_edge_counts, extracted_logs

def evaluate_log_level(log_scores, log_level_labels, total_logs, extracted_logs, 
                       all_attack_types, miss_penalty=0.0):
    all_log_keys_set = set(log_level_labels.keys())
    
    if len(all_log_keys_set) == 0:
        return {
            'auc_roc': 0.0,
            'auc_pr': 0.0,
            'adp': 0.0,
            'per_attack_metrics': {},
            'num_logs': 0,
            'num_extracted': 0,
            'num_missed': 0,
            'extraction_rate': 0.0
        }
    
    all_log_keys = sorted(all_log_keys_set, key=lambda x: str(x))
    
    num_total = len(all_log_keys)
    num_extracted = len(extracted_logs & all_log_keys_set)
    num_missed = num_total - num_extracted
    extraction_rate = num_extracted / num_total if num_total > 0 else 0.0
    
    y_true = []
    y_scores = []
    y_true_attack_types = []
    
    for log_key in all_log_keys:
        label = log_level_labels.get(log_key, False)
        
        if label == False:
            y_true.append(0)
            y_true_attack_types.append(None)
        else:
            y_true.append(1)
            y_true_attack_types.append(label)
        
        if log_key in log_scores:
            y_scores.append(log_scores[log_key])
        else:
            y_scores.append(miss_penalty)
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    if len(np.unique(y_true)) < 2:
        return {
            'auc_roc': 0.0,
            'auc_pr': 0.0,
            'adp': 0.0,
            'per_attack_metrics': {},
            'num_logs': num_total,
            'num_extracted': num_extracted,
            'num_missed': num_missed,
            'extraction_rate': extraction_rate,
            'num_malicious': int(sum(y_true))
        }
    
    try:
        auc_roc = roc_auc_score(y_true, y_scores)
    except:
        auc_roc = 0.0
    
    try:
        auc_pr = average_precision_score(y_true, y_scores)
    except:
        auc_pr = 0.0
    
    attack_label_dict = {}
    for attack_type in all_attack_types:
        if attack_type is None or attack_type == 'Overall':
            continue
        attack_labels = np.zeros(len(y_true_attack_types), dtype=np.int8)
        for i, attack_info in enumerate(y_true_attack_types):
            if isinstance(attack_info, str) and attack_info == attack_type:
                attack_labels[i] = 1
            elif isinstance(attack_info, list) and attack_type in attack_info:
                attack_labels[i] = 1
        if np.sum(attack_labels) > 0:
            attack_label_dict[attack_type] = attack_labels
    
    adp = 0.0
    
    if attack_label_dict:
        try:
            adp_result = compute_attack_detection_precision(y_scores, attack_label_dict)
            adp = float(adp_result) if adp_result is not None else 0.0
        except Exception as e:
            print(f"    Warning: Error computing ADP: {e}")
            import traceback
            traceback.print_exc()
            adp = 0.0
    else:
        print(f"    Debug ADP: attack_label_dict is empty! Checking why...")
        for attack_type in all_attack_types:
            if attack_type is None or attack_type == 'Overall':
                print(f"      Skipping {attack_type}")
                continue
            count = sum(1 for i, at in enumerate(y_true_attack_types) 
                       if (isinstance(at, str) and at == attack_type) or 
                          (isinstance(at, list) and attack_type in at))
            print(f"      {attack_type}: matched {count} logs")
    
    per_attack_metrics = {}
    for attack_type in all_attack_types:
        if attack_type is None or attack_type == 'Overall':
            continue
        
        attack_mask = np.array([
            y_true_attack_types[i] == attack_type or 
            (isinstance(y_true_attack_types[i], list) and attack_type in y_true_attack_types[i])
            for i in range(len(y_true_attack_types))
        ])
        
        if not np.any(attack_mask):
            continue
        
        y_binary = np.where(attack_mask, 1, 0)
        
        try:
            if len(np.unique(y_binary)) >= 2:
                attack_auc_roc = roc_auc_score(y_binary, y_scores)
                attack_auc_pr = average_precision_score(y_binary, y_scores)
            else:
                attack_auc_roc = 0.0
                attack_auc_pr = 0.0
        except:
            attack_auc_roc = 0.0
            attack_auc_pr = 0.0
        
        per_attack_metrics[attack_type] = {
            'auc_roc': attack_auc_roc,
            'auc_pr': attack_auc_pr,
            'count': int(np.sum(attack_mask))
        }
    
    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'adp': adp,
        'per_attack_metrics': per_attack_metrics,
        'num_logs': num_total,
        'num_extracted': num_extracted,
        'num_missed': num_missed,
        'extraction_rate': extraction_rate,
        'num_malicious': int(sum(y_true))
    }

def display_results_table(results_list, overall_adp):
    if not results_list:
        return
    
    df = pd.DataFrame(results_list)
    
    if 'ADP' not in df.columns:
        df['ADP'] = overall_adp
    else:
        df['ADP'] = overall_adp
    
    display_cols = ['attack_type', 'AUC_ROC', 'AUC_PR', 'ADP']
    if 'attack_type' not in df.columns:
        display_cols = ['dataset', 'AUC_ROC', 'AUC_PR', 'ADP']
    
    display_df = df[display_cols].copy()
    display_df['AUC_ROC'] = display_df['AUC_ROC'].round(3)
    display_df['AUC_PR'] = display_df['AUC_PR'].round(3)
    display_df['ADP'] = display_df['ADP'].round(3)
    
    print(display_df.to_string(index=False))

def evaluate_baseline_mode(base_dir):
    model_path = f'{base_dir}/models/tgn_model_epoch_50.pt'
    if not os.path.exists(model_path) and not os.path.exists(model_path + '.zip'):
        model_path = f'{base_dir}/models/tgn_model_final.pt'
        if not os.path.exists(model_path) and not os.path.exists(model_path + '.zip'):
            raise FileNotFoundError(f"Model not found: {base_dir}/models/tgn_model_epoch_50.pt or tgn_model_final.pt (or their .zip versions)")
    
    model_components = load_model(model_path, device)
    if model_components is None:
        raise FileNotFoundError(f"Model not found: {model_path} (or {model_path}.zip)")
    
    memory, gnn, link_pred, neighbor_loader, num_edge_types = model_components
    
    nodeid2msg_path = f'{base_dir}/processed_data/node_mappings/nodeid2msg.pkl'
    
    nodeid2msg = load_pickle_from_zip_or_file(nodeid2msg_path)
    if nodeid2msg is None:
        raise FileNotFoundError(f"nodeid2msg.pkl not found at {nodeid2msg_path} (or {nodeid2msg_path}.zip)")
    
    test_graphs_dir = f'{base_dir}/test/temporal_graphs'
    test_metadata_dir = f'{base_dir}/test/window_metadata'
    
    if not os.path.exists(test_graphs_dir):
        raise FileNotFoundError(f"Test graphs directory not found: {test_graphs_dir}")
    
    all_test_graphs = sorted(glob.glob(os.path.join(test_graphs_dir, '*.pt'))) + sorted(glob.glob(os.path.join(test_graphs_dir, '*.pt.zip')))
    
    test_windows_to_process = []
    for graph_path in all_test_graphs:
        graph_name = os.path.basename(graph_path).replace('.pt.zip', '').replace('.pt', '')
        if graph_name.startswith('graph_'):
            window_name_underscore = graph_name.replace('graph_', '')
            window_name = window_name_underscore.replace('_', ' ').replace('-', ':')
            test_windows_to_process.append((window_name, graph_path))
    
    criterion = CrossEntropyLoss()
    
    all_edge_list = []
    test_graphs_data = []
    test_metadata_dict = {}
    
    pbar = tqdm(test_windows_to_process, desc="Processing windows", leave=False)
    for window_name, graph_path in pbar:
        window_name_underscore = window_name.replace(' ', '_').replace(':', '-')
        metadata_path = f'{base_dir}/test/window_metadata/metadata_{window_name_underscore}.json'
        
        graph_data = load_torch_from_zip_or_file(graph_path, map_location='cpu')
        if graph_data is None:
            continue
        
        test_graphs_data.append((window_name, graph_data))
        
        window_metadata = {}
        metadata_data = load_json_from_zip_or_file(metadata_path)
        if metadata_data is not None:
            window_metadata = metadata_data
        test_metadata_dict[window_name] = window_metadata
        
        edge_list = test_window_for_eval(
            graph_data, memory, gnn, link_pred, neighbor_loader,
            nodeid2msg, criterion, batch_size=512, num_edge_types=num_edge_types
        )
        
        all_edge_list.extend(edge_list)
        del graph_data, edge_list
        torch.cuda.empty_cache()
    pbar.close()
    
    if len(all_edge_list) == 0:
        raise RuntimeError("No edges processed")
    
    all_log_scores = {}
    all_extracted_logs = set()
    all_log_level_labels = {}
    all_total_logs = {}
    
    edge_offset = 0
    all_attack_types = set()
    
    for window_idx, (window_name, graph_data) in enumerate(test_graphs_data, 1):
        window_metadata = test_metadata_dict.get(window_name, {})
        dataset = window_metadata.get('dataset', None)
        
        window_name_underscore = window_name.replace(' ', '_').replace(':', '-')
        name_without_prefix = window_name_underscore.replace('graph_', '') if window_name_underscore.startswith('graph_') else window_name_underscore
        parts = name_without_prefix.split('_')
        if len(parts) == 4:
            date1 = parts[0]
            time1 = parts[1].replace('-', ':')
            date2 = parts[2]
            time2 = parts[3].replace('-', ':')
            timestamp = f'{date1} {time1}_{date2} {time2}'
        else:
            timestamp = name_without_prefix.replace('_', ' ')
        
        if dataset is None:
            if '2018-11-' in timestamp:
                dataset = 'S1'
            elif '2018-08-' in timestamp or '2018-09-' in timestamp:
                dataset = 'S2'
            elif '2018-12-01' in timestamp:
                dataset = 'S3'
            elif '2018-12-04' in timestamp or '2018-12-05' in timestamp:
                dataset = 'S4'
            else:
                dataset = 'S1'
        
        formatted_window_name = name_without_prefix
        edge_to_log_mapping = load_edge_to_log_mapping(base_dir, formatted_window_name, 'test')
        
        log_level_labels, total_logs, malicious_log_indices = load_log_level_ground_truth(dataset, timestamp)
        
        window_prefix = f"{dataset}_{timestamp}"
        
        for (log_idx, log_type), label in log_level_labels.items():
            unique_key = (window_prefix, log_idx, log_type)
            all_log_level_labels[unique_key] = label
            if label and label != False:
                if isinstance(label, str):
                    all_attack_types.add(label)
                elif isinstance(label, list):
                    all_attack_types.update(label)
        
        for log_type, count in total_logs.items():
            if log_type not in all_total_logs:
                all_total_logs[log_type] = 0
            all_total_logs[log_type] += count
        
        num_edges_in_window = len(graph_data.src) if hasattr(graph_data, 'src') else 0
        window_edges = all_edge_list[edge_offset:edge_offset + num_edges_in_window]
        edge_offset += num_edges_in_window
        
        log_scores, log_edge_counts, extracted_logs = aggregate_edge_losses_to_logs(
            window_edges, edge_to_log_mapping
        )
        
        for (log_idx, log_type), score in log_scores.items():
            unique_key = (window_prefix, log_idx, log_type)
            if unique_key not in all_log_scores or score > all_log_scores[unique_key]:
                all_log_scores[unique_key] = score
        
        for (log_idx, log_type) in extracted_logs:
            unique_key = (window_prefix, log_idx, log_type)
            all_extracted_logs.add(unique_key)
    
    log_level_attack_types = set()
    for log_key, label in all_log_level_labels.items():
        if label and label != False:
            if isinstance(label, str):
                log_level_attack_types.add(label)
            elif isinstance(label, list):
                log_level_attack_types.update(label)
    
    combined_attack_types = sorted(list(all_attack_types | log_level_attack_types))
    
    log_metrics = evaluate_log_level(
        all_log_scores, all_log_level_labels, all_total_logs,
        all_extracted_logs, combined_attack_types
    )
    
    overall_adp = float(log_metrics.get('adp', 0.0) or 0.0)
    
    results_list = []
    for attack_type in combined_attack_types:
        if attack_type in log_metrics['per_attack_metrics']:
            attack_metrics = log_metrics['per_attack_metrics'][attack_type]
            auc_roc = attack_metrics['auc_roc']
            auc_pr = attack_metrics['auc_pr']
        else:
            auc_roc = 0.0
            auc_pr = 0.0
        
        result_entry = {
            'attack_type': attack_type,
            'AUC_ROC': float(auc_roc),
            'AUC_PR': float(auc_pr),
            'ADP': overall_adp
        }
        results_list.append(result_entry)
    
    del memory, gnn, link_pred, neighbor_loader
    torch.cuda.empty_cache()
    
    if len(results_list) > 0:
        return results_list
    else:
        return None

def evaluate_autoprov_mode(base_dir):
    model_dir = f'{base_dir}/models/model_6/'
    model_path = f'{model_dir}tgn_model_epoch_40.pt'
    
    model_components = load_model_for_eval_infer(model_path, device)
    if model_components is None:
        raise FileNotFoundError(f"Model not found: {model_path} (or {model_path}.zip)")
    
    memory, gnn, link_pred, neighbor_loader, num_edge_types = model_components
    
    nodeid2msg_path = f'{base_dir}/processed_data/node_mappings/nodeid2msg.pkl'
    
    nodeid2msg = load_pickle_from_zip_or_file(nodeid2msg_path)
    if nodeid2msg is None:
        raise FileNotFoundError(f"nodeid2msg.pkl not found at {nodeid2msg_path} (or {nodeid2msg_path}.zip)")
    
    test_graphs_dir = f'{base_dir}/test/temporal_graphs'
    test_metadata_dir = f'{base_dir}/test/window_metadata'
    
    if not os.path.exists(test_graphs_dir):
        raise FileNotFoundError(f"Test graphs directory not found: {test_graphs_dir}")
    
    all_test_graphs = sorted(glob.glob(os.path.join(test_graphs_dir, '*.pt'))) + sorted(glob.glob(os.path.join(test_graphs_dir, '*.pt.zip')))
    
    test_windows_to_process = []
    for graph_path in all_test_graphs:
        graph_name = os.path.basename(graph_path).replace('.pt.zip', '').replace('.pt', '')
        if graph_name.startswith('graph_'):
            window_name_underscore = graph_name.replace('graph_', '')
            window_name = window_name_underscore.replace('_', ' ').replace('-', ':')
            test_windows_to_process.append((window_name, graph_path))
    
    criterion = CrossEntropyLoss()
    
    all_edge_list = []
    test_graphs_data = []
    test_metadata_dict = {}
    
    pbar = tqdm(test_windows_to_process, desc="Processing windows", leave=False)
    for window_name, graph_path in pbar:
        window_name_underscore = window_name.replace(' ', '_').replace(':', '-')
        metadata_path = f'{base_dir}/test/window_metadata/metadata_{window_name_underscore}.json'
        
        graph_data = load_torch_from_zip_or_file(graph_path, map_location='cpu')
        if graph_data is None:
            continue
        
        test_graphs_data.append((window_name, graph_data))
        
        window_metadata = {}
        metadata_data = load_json_from_zip_or_file(metadata_path)
        if metadata_data is not None:
            window_metadata = metadata_data
        test_metadata_dict[window_name] = window_metadata
        
        edge_list = test_window_for_eval(
            graph_data, memory, gnn, link_pred, neighbor_loader,
            nodeid2msg, criterion, batch_size=512, num_edge_types=num_edge_types
        )
        
        all_edge_list.extend(edge_list)
        del graph_data, edge_list
        torch.cuda.empty_cache()
    pbar.close()
    
    if len(all_edge_list) == 0:
        raise RuntimeError("No edges processed")
    
    all_log_scores = {}
    all_extracted_logs = set()
    all_log_level_labels = {}
    all_total_logs = {}
    
    edge_offset = 0
    all_attack_types = set()
    
    for window_idx, (window_name, graph_data) in enumerate(test_graphs_data, 1):
        window_metadata = test_metadata_dict.get(window_name, {})
        dataset = window_metadata.get('dataset', None)
        
        window_name_underscore = window_name.replace(' ', '_').replace(':', '-')
        name_without_prefix = window_name_underscore.replace('graph_', '') if window_name_underscore.startswith('graph_') else window_name_underscore
        parts = name_without_prefix.split('_')
        if len(parts) == 4:
            date1 = parts[0]
            time1 = parts[1].replace('-', ':')
            date2 = parts[2]
            time2 = parts[3].replace('-', ':')
            timestamp = f'{date1} {time1}_{date2} {time2}'
        else:
            timestamp = name_without_prefix.replace('_', ' ')
        
        if dataset is None:
            if '2018-11-' in timestamp:
                dataset = 'S1'
            elif '2018-08-' in timestamp or '2018-09-' in timestamp:
                dataset = 'S2'
            elif '2018-12-01' in timestamp:
                dataset = 'S3'
            elif '2018-12-04' in timestamp or '2018-12-05' in timestamp:
                dataset = 'S4'
            else:
                dataset = 'S1'
        
        formatted_window_name = name_without_prefix
        edge_to_log_mapping = load_edge_to_log_mapping(base_dir, formatted_window_name, 'test')
        
        log_level_labels, total_logs, malicious_log_indices = load_log_level_ground_truth(dataset, timestamp)
        
        window_prefix = f"{dataset}_{timestamp}"
        
        for (log_idx, log_type), label in log_level_labels.items():
            unique_key = (window_prefix, log_idx, log_type)
            all_log_level_labels[unique_key] = label
            if label and label != False:
                if isinstance(label, str):
                    all_attack_types.add(label)
                elif isinstance(label, list):
                    all_attack_types.update(label)
        
        for log_type, count in total_logs.items():
            if log_type not in all_total_logs:
                all_total_logs[log_type] = 0
            all_total_logs[log_type] += count
        
        num_edges_in_window = len(graph_data.src) if hasattr(graph_data, 'src') else 0
        window_edges = all_edge_list[edge_offset:edge_offset + num_edges_in_window]
        edge_offset += num_edges_in_window
        
        log_scores, log_edge_counts, extracted_logs = aggregate_edge_losses_to_logs(
            window_edges, edge_to_log_mapping
        )
        
        for (log_idx, log_type), score in log_scores.items():
            unique_key = (window_prefix, log_idx, log_type)
            if unique_key not in all_log_scores or score > all_log_scores[unique_key]:
                all_log_scores[unique_key] = score
        
        for (log_idx, log_type) in extracted_logs:
            unique_key = (window_prefix, log_idx, log_type)
            all_extracted_logs.add(unique_key)
    
    log_level_attack_types = set()
    for log_key, label in all_log_level_labels.items():
        if label and label != False:
            if isinstance(label, str):
                log_level_attack_types.add(label)
            elif isinstance(label, list):
                log_level_attack_types.update(label)
    
    combined_attack_types = sorted(list(all_attack_types | log_level_attack_types))
    
    log_metrics = evaluate_log_level(
        all_log_scores, all_log_level_labels, all_total_logs,
        all_extracted_logs, combined_attack_types
    )
    
    overall_adp = float(log_metrics.get('adp', 0.0) or 0.0)
    
    results_list = []
    for attack_type in combined_attack_types:
        if attack_type in log_metrics['per_attack_metrics']:
            attack_metrics = log_metrics['per_attack_metrics'][attack_type]
            auc_roc = attack_metrics['auc_roc']
            auc_pr = attack_metrics['auc_pr']
        else:
            auc_roc = 0.0
            auc_pr = 0.0
        
        result_entry = {
            'attack_type': attack_type,
            'AUC_ROC': float(auc_roc),
            'AUC_PR': float(auc_pr),
            'ADP': overall_adp
        }
        results_list.append(result_entry)
    
    del memory, gnn, link_pred, neighbor_loader
    torch.cuda.empty_cache()
    
    if len(results_list) > 0:
        return results_list
    else:
        return None

def parse_args():
    p = argparse.ArgumentParser(description="KAIROS ATLAS evaluation (baseline + autoprov)")

    p.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES to set before importing evaluation code (e.g., '0' or '0,1').",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    default_artifacts_root = os.path.join(autoprov_dir, 'BIGDATA', 'KAIROS_artifacts', 'ATLAS_artifacts')
    
    p.add_argument(
        "--artifacts_root",
        type=str,
        default=default_artifacts_root,
        help="Root directory containing ATLAS artifacts (baseline + rulellm_llmlabel_*).",
    )
    p.add_argument(
        "--baseline_dir",
        type=str,
        default=None,
        help="Path to baseline artifacts (default: {artifacts_root}/original_atlas_graph).",
    )

    p.add_argument(
        "--autoprov_dir",
        type=str,
        default=None,
        help="Path to autoprov artifacts (directory that contains processed_data/ and models/).",
    )
    p.add_argument(
        "--embedding",
        type=str,
        default="mpnet",
        choices=["roberta", "mpnet", "minilm", "distilbert"],
        help="Embedding name for autoprov artifact path construction (when --autoprov_dir is not provided).",
    )
    p.add_argument(
        "--cee",
        type=str,
        default="gpt-4o",
        help="Candidate Edge Extractor name for autoprov artifact path construction.",
    )
    p.add_argument(
        "--rule_generator",
        type=str,
        default="llama3_70b",
        help="Rule Generator name for autoprov artifact path construction.",
    )

    p.add_argument("--skip_baseline", action="store_true", help="Skip baseline evaluation.")
    p.add_argument("--skip_autoprov", action="store_true", help="Skip autoprov evaluation.")

    return p.parse_args()

def main():
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    artifacts_root = args.artifacts_root

    baseline_dir = args.baseline_dir or os.path.join(artifacts_root, "original_atlas_graph")

    if args.autoprov_dir:
        autoprov_dir = args.autoprov_dir
        folder_name = os.path.basename(autoprov_dir)
    else:
        folder_name = f"{args.cee.lower()}_{args.rule_generator.lower()}"
        autoprov_dir = os.path.join(
            artifacts_root,
            f"rulellm_llmlabel_{args.embedding.lower()}",
            folder_name,
        )

    if not args.skip_baseline:
        if not os.path.exists(baseline_dir):
            raise FileNotFoundError(f"Baseline artifacts directory not found: {baseline_dir}")
        results_baseline = evaluate_baseline_mode(baseline_dir)
        if results_baseline:
            overall_adp = results_baseline[0].get('ADP', 0.0) if results_baseline else 0.0
            print("\nBaseline Model Results:")
            display_results_table(results_baseline, overall_adp)

    if not args.skip_autoprov:
        if not os.path.exists(autoprov_dir):
            pass
        else:
            try:
                results_autoprov = evaluate_autoprov_mode(autoprov_dir)
                if results_autoprov:
                    overall_adp = results_autoprov[0].get('ADP', 0.0) if results_autoprov else 0.0
                    print("\n\nAuto-Prov Results:")
                    display_results_table(results_autoprov, overall_adp)
            except Exception as e:
                pass

if __name__ == "__main__":
    main()

