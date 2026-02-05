#!/usr/bin/env python3

import argparse
import os
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import pickle
import json
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import dgl
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from collections import defaultdict

try:
    import faiss
    FAISS_AVAILABLE = True
    FAISS_GPU = faiss.get_num_gpus() > 0
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ATTACK_TYPE_MAPPING = {
    'S1': 'Strategic web compromise',
    'S2': 'Malvertising dominate',
    'S3': 'Spam campaign',
    'S4': 'Pony campaign'
}

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm" or name == "BatchNorm":
        return nn.BatchNorm1d
    else:
        return None

class GATConv(nn.Module):
    def __init__(self, in_dim, e_dim, out_dim, n_heads, feat_drop=0.0, attn_drop=0.0,
                 negative_slope=0.2, residual=False, activation=None, norm=None, concat_out=True):
        super(GATConv, self).__init__()
        self.n_heads = n_heads
        self.out_feat = out_dim
        self.concat_out = concat_out
        
        self.fc = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.edge_fc = nn.Linear(e_dim, out_dim * n_heads, bias=False)
        self.attn_h = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self.bias = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        
        if residual:
            if in_dim != n_heads * out_dim:
                self.res_fc = nn.Linear(in_dim, n_heads * out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        
        self.reset_parameters()
        self.activation = activation
        self.norm = norm
        if norm is not None:
            self.norm = norm(n_heads * out_dim)
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
    
    def forward(self, graph, feat, get_attention=False):
        edge_feature = graph.edata['attr']
        h = self.feat_drop(feat)
        feat_transformed = self.fc(h).view(-1, self.n_heads, self.out_feat)
        
        eh = (feat_transformed * self.attn_h).sum(-1).unsqueeze(-1)
        et = (feat_transformed * self.attn_t).sum(-1).unsqueeze(-1)
        
        graph.ndata.update({'hs': feat_transformed, 'eh': eh, 'et': et})
        
        feat_edge = self.edge_fc(edge_feature).view(-1, self.n_heads, self.out_feat)
        ee = (feat_edge * self.attn_e).sum(-1).unsqueeze(-1)
        
        graph.edata.update({'ee': ee})
        graph.apply_edges(dgl.function.u_add_e('eh', 'ee', 'ee'))
        graph.apply_edges(dgl.function.e_add_v('ee', 'et', 'e'))
        
        e = self.leaky_relu(graph.edata.pop('e'))
        graph.edata['a'] = self.attn_drop(dgl.ops.edge_softmax(graph, e))
        
        graph.update_all(dgl.function.u_mul_e('hs', 'a', 'm'),
                        dgl.function.sum('m', 'hs'))
        
        rst = graph.ndata['hs'].view(-1, self.n_heads, self.out_feat)
        rst = rst + self.bias.view(1, self.n_heads, self.out_feat)
        
        if self.res_fc is not None:
            resval = self.res_fc(h).view(-1, self.n_heads, self.out_feat)
            rst = rst + resval
        
        if self.concat_out:
            rst = rst.flatten(1)
        else:
            rst = torch.mean(rst, dim=1)
        
        if self.norm is not None:
            rst = self.norm(rst)
        
        if self.activation:
            rst = self.activation(rst)
        
        if get_attention:
            return rst, graph.edata['a']
        else:
            return rst

class GAT(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, out_dim, n_layers, n_heads, n_heads_out,
                 activation, feat_drop, attn_drop, negative_slope, residual, norm,
                 concat_out=False, encoding=False):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.gats = nn.ModuleList()
        self.concat_out = concat_out
        
        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None
        
        if self.n_layers == 1:
            self.gats.append(GATConv(
                n_dim, e_dim, out_dim, n_heads_out, feat_drop, attn_drop, negative_slope,
                last_residual, norm=last_norm, concat_out=self.concat_out
            ))
        else:
            self.gats.append(GATConv(
                n_dim, e_dim, hidden_dim, n_heads, feat_drop, attn_drop, negative_slope,
                residual, create_activation(activation),
                norm=norm, concat_out=self.concat_out
            ))
            for _ in range(1, self.n_layers - 1):
                self.gats.append(GATConv(
                    hidden_dim * self.n_heads, e_dim, hidden_dim, n_heads,
                    feat_drop, attn_drop, negative_slope,
                    residual, create_activation(activation),
                    norm=norm, concat_out=self.concat_out
                ))
            self.gats.append(GATConv(
                hidden_dim * self.n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, last_activation, norm=last_norm, concat_out=self.concat_out
            ))
        self.head = nn.Identity()
    
    def forward(self, g, input_feature, return_hidden=False):
        h = input_feature
        hidden_list = []
        for layer in range(self.n_layers):
            h = self.gats[layer](g, h)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

class GMAEModel(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, mask_rate=0.5, loss_fn="sce", alpha_l=2):
        super(GMAEModel, self).__init__()
        self._mask_rate = mask_rate
        self._output_hidden_size = hidden_dim
        self.recon_loss = nn.BCELoss(reduction='mean')
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.edge_recon_fc.apply(init_weights)
        
        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads
        enc_nhead = n_heads
        
        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim
        
        self.encoder = GAT(
            n_dim=n_dim,
            e_dim=e_dim,
            hidden_dim=enc_num_hidden,
            out_dim=enc_num_hidden,
            n_layers=n_layers,
            n_heads=enc_nhead,
            n_heads_out=enc_nhead,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=True,
        )
        
        self.decoder = GAT(
            n_dim=dec_in_dim,
            e_dim=e_dim,
            hidden_dim=dec_num_hidden,
            out_dim=n_dim,
            n_layers=1,
            n_heads=n_heads,
            n_heads_out=1,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=False,
        )
        
        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim * n_layers, dec_in_dim, bias=False)
        
        from functools import partial
        if loss_fn == "sce":
            self.criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
    
    def embed(self, g):
        x = g.ndata['attr'].to(g.device)
        rep = self.encoder(g, x)
        return rep

def load_entity_level_dataset(base_dir, window_key, use_llm_features=False, global_node_embeddings=None):
    graph_path = f'{base_dir}/processed_data/graphs/graph_{window_key}.pkl'
    
    with open(graph_path, 'rb') as f:
        data = pickle.load(f)
    
    g_nx = nx.node_link_graph(data['graph'], edges='links')
    
    if use_llm_features:
        if 'node_embeddings' in data:
            node_features = data['node_embeddings']
        elif global_node_embeddings is not None and 'node_indices' in data:
            node_indices = data['node_indices']
            node_features = global_node_embeddings[node_indices]
        else:
            raise ValueError(f"LLM features requested but not found in {graph_path}")
    else:
        if 'node_labels' in data:
            node_features = data['node_labels']
        else:
            node_features = None
    
    if 'node_indices' in data:
        node_indices = data['node_indices']
    else:
        node_indices = sorted(g_nx.nodes())
    
    edge_order = data.get('edge_order', None)
    if edge_order is None:
        edge_order = list(g_nx.edges())
    
    return g_nx, node_features, node_indices, edge_order

def transform_graph(g, node_feature_dim, edge_feature_dim, node_features=None, use_llm_features=False):
    for node in g.nodes():
        if 'type' not in g.nodes[node]:
            g.nodes[node]['type'] = 0
    
    for src, dst in g.edges():
        if 'type' not in g.edges[src, dst]:
            edge_data = g.edges[src, dst]
            if 'type' not in edge_data:
                g.edges[src, dst]['type'] = 0
    
    try:
        g_dgl = dgl.from_networkx(g, node_attrs=['type'], edge_attrs=['type'])
    except Exception as e:
        g_dgl = dgl.from_networkx(g)
        if g_dgl.number_of_nodes() > 0:
            g_dgl.ndata['type'] = torch.zeros(g_dgl.number_of_nodes(), dtype=torch.long)
        if g_dgl.number_of_edges() > 0:
            g_dgl.edata['type'] = torch.zeros(g_dgl.number_of_edges(), dtype=torch.long)
    
    if use_llm_features and node_features is not None:
        g_dgl.ndata["attr"] = torch.tensor(node_features, dtype=torch.float32)
    else:
        if 'type' in g_dgl.ndata:
            g_dgl.ndata["attr"] = F.one_hot(g_dgl.ndata["type"].view(-1).long(), num_classes=node_feature_dim).float()
        else:
            g_dgl.ndata["attr"] = torch.zeros((g_dgl.number_of_nodes(), node_feature_dim), dtype=torch.float32)
    
    if 'type' in g_dgl.edata:
        g_dgl.edata["attr"] = F.one_hot(g_dgl.edata["type"].view(-1).long(), num_classes=edge_feature_dim).float()
    else:
        g_dgl.edata["attr"] = torch.zeros((g_dgl.number_of_edges(), edge_feature_dim), dtype=torch.float32)
    
    return g_dgl

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
    
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'rb') as f:
                edge_to_log_mapping = pickle.load(f)
        except Exception as e:
            pass
    
    return edge_to_log_mapping

def aggregate_node_scores_to_logs(node_scores, node_indices, edge_to_log_mapping, graph_edges):
    log_scores = defaultdict(list)
    
    node_to_edges = defaultdict(list)
    for edge_idx, (src, dst) in enumerate(graph_edges):
        node_to_edges[src].append(edge_idx)
        node_to_edges[dst].append(edge_idx)
    
    for node_idx, score in node_scores.items():
        if node_idx in node_to_edges:
            for edge_idx in node_to_edges[node_idx]:
                if edge_idx in edge_to_log_mapping:
                    log_key = edge_to_log_mapping[edge_idx]
                    log_scores[log_key].append(score)
    
    log_scores_final = {}
    log_node_counts = {}
    extracted_logs = set()
    
    for log_key, scores in log_scores.items():
        log_node_counts[log_key] = len(scores)
        log_scores_final[log_key] = max(scores)
        extracted_logs.add(log_key)
    
    return log_scores_final, log_node_counts, extracted_logs

def compute_attack_detection_precision(scores, attack_label_dict):
    if not attack_label_dict:
        return None
    
    if isinstance(scores, list):
        scores = np.array(scores)
    
    num_nodes = len(scores)
    if num_nodes == 0:
        return None
    
    attack_indices = {}
    index_to_attacks = {}
    y_global = np.zeros(num_nodes, dtype=np.int8)
    
    for attack_name, labels in attack_label_dict.items():
        if labels is None or len(labels) != num_nodes:
            continue
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

def evaluate_entity_level_using_knn(x_train, x_test, y_test, n_neighbors=10):
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train_norm = (x_train - x_train_mean) / (x_train_std + 1e-6)
    x_test_norm = (x_test - x_train_mean) / (x_train_std + 1e-6)
    
    x_train_norm = x_train_norm.astype(np.float32)
    x_test_norm = x_test_norm.astype(np.float32)
    
    if FAISS_AVAILABLE:
        try:
            embedding_dim = x_train_norm.shape[1]
            
            index = faiss.IndexFlatL2(embedding_dim)
            
            if FAISS_GPU:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            
            index.add(x_train_norm)
            
            import random
            idx = list(range(x_train_norm.shape[0]))
            random.shuffle(idx)
            sample_size = min(50000, x_train_norm.shape[0])
            distances_train, _ = index.search(x_train_norm[idx][:sample_size], n_neighbors)
            distances_train = np.sqrt(distances_train)
            mean_distance = distances_train.mean()
            del distances_train
            
            distances, _ = index.search(x_test_norm, n_neighbors)
            distances = np.sqrt(distances)
            distances = distances.mean(axis=1)
            
            del index
        except Exception as e:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
            nbrs.fit(x_train_norm)
            
            import random
            idx = list(range(x_train_norm.shape[0]))
            random.shuffle(idx)
            distances_train, _ = nbrs.kneighbors(x_train_norm[idx][:min(50000, x_train_norm.shape[0])], n_neighbors=n_neighbors)
            mean_distance = distances_train.mean()
            del distances_train
            
            distances, _ = nbrs.kneighbors(x_test_norm, n_neighbors=n_neighbors)
            distances = distances.mean(axis=1)
            del nbrs
    else:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        nbrs.fit(x_train_norm)
        
        import random
        idx = list(range(x_train_norm.shape[0]))
        random.shuffle(idx)
        distances_train, _ = nbrs.kneighbors(x_train_norm[idx][:min(50000, x_train_norm.shape[0])], n_neighbors=n_neighbors)
        mean_distance = distances_train.mean()
        del distances_train
        
        distances, _ = nbrs.kneighbors(x_test_norm, n_neighbors=n_neighbors)
        distances = distances.mean(axis=1)
        del nbrs
    
    score = distances / mean_distance
    
    if y_test is not None and len(y_test) > 0 and np.sum(y_test) > 0:
        auc_roc = roc_auc_score(y_test, score)
        auc_pr = average_precision_score(y_test, score)
        prec, rec, threshold = precision_recall_curve(y_test, score)
        f1 = 2 * prec * rec / (rec + prec + 1e-9)
        best_idx = np.argmax(f1)
        
        return auc_roc, auc_pr, score, (prec, rec, threshold, f1, best_idx)
    else:
        return None, None, score, None

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
            'extraction_rate': 0.0,
            'best_f1': 0.0,
            'best_threshold': 0.0,
            'best_precision': 0.0,
            'best_recall': 0.0
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
            'num_malicious': int(sum(y_true)),
            'best_f1': 0.0,
            'best_threshold': 0.0,
            'best_precision': 0.0,
            'best_recall': 0.0
        }
    
    try:
        auc_roc = roc_auc_score(y_true, y_scores)
    except:
        auc_roc = 0.0
    
    try:
        auc_pr = average_precision_score(y_true, y_scores)
    except:
        auc_pr = 0.0
    
    best_f1 = 0.0
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    try:
        prec, rec, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = np.argmax(f1_scores)
        best_f1 = float(f1_scores[best_idx])
        best_precision = float(prec[best_idx])
        best_recall = float(rec[best_idx])
        if best_idx < len(thresholds):
            best_threshold = float(thresholds[best_idx])
        else:
            best_threshold = float(thresholds[-1]) if len(thresholds) > 0 else 0.0
    except Exception as e:
        pass
    
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
            scores_list = y_scores.tolist()
            adp_result = compute_attack_detection_precision(scores_list, attack_label_dict)
            adp = float(adp_result) if adp_result is not None else 0.0
        except Exception as e:
            adp = 0.0
    
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
        'num_malicious': int(sum(y_true)),
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'best_precision': best_precision,
        'best_recall': best_recall
    }

def display_results_table(results_list, overall_adp, f1_metrics=None):
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
    
    if f1_metrics is not None:
        best_f1 = f1_metrics.get('best_f1', 0.0)
        best_threshold = f1_metrics.get('best_threshold', 0.0)
        best_precision = f1_metrics.get('best_precision', 0.0)
        best_recall = f1_metrics.get('best_recall', 0.0)
        print(f"\nBest F1: {best_f1:.4f} | Threshold: {best_threshold:.4f} | Precision: {best_precision:.4f} | Recall: {best_recall:.4f}")

def load_model_for_eval(model_path, model_config_path, device):
    if not os.path.exists(model_path):
        return None
    
    if os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            config = json.load(f)
    else:
        model_dir = os.path.dirname(model_path)
        hyperparams_path = os.path.join(model_dir, 'hyperparams.json')
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, 'r') as f:
                config = json.load(f)
        else:
            config = {
                'n_dim': 128,
                'e_dim': 150,
                'hidden_dim': 64,
                'n_layers': 3,
                'n_heads': 4,
                'activation': 'prelu',
                'feat_drop': 0.1,
                'negative_slope': 0.2,
                'residual': True,
                'mask_rate': 0.5,
                'norm': 'BatchNorm',
                'loss_fn': 'sce',
                'alpha_l': 3
            }
    
    model = GMAEModel(
        n_dim=config['n_dim'],
        e_dim=config['e_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        activation=config['activation'],
        feat_drop=config['feat_drop'],
        negative_slope=config['negative_slope'],
        residual=config['residual'],
        mask_rate=config['mask_rate'],
        norm=config['norm'],
        loss_fn=config.get('loss_fn', 'sce'),
        alpha_l=config.get('alpha_l', 3)
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, config

def evaluate_baseline_mode(base_dir, labels_dir):
    with open(f'{base_dir}/processed_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    use_llm_features = metadata.get('use_llm_features', False)
    n_dim = metadata['node_feature_dim']
    e_dim = metadata['edge_feature_dim']
    
    with open(f'{base_dir}/processed_data/train_windows.json', 'r') as f:
        train_windows = json.load(f)
    with open(f'{base_dir}/processed_data/test_windows.json', 'r') as f:
        test_windows = json.load(f)
    
    with open(f'{base_dir}/processed_data/window_metadata.json', 'r') as f:
        window_metadata = json.load(f)
    
    with open(f'{base_dir}/processed_data/node_mappings/idx2node.pkl', 'rb') as f:
        idx2node = pickle.load(f)
    
    global_node_embeddings = None
    if use_llm_features:
        emb_path = f'{base_dir}/processed_data/node_features/node_embeddings.npy'
        if os.path.exists(emb_path):
            global_node_embeddings = np.load(emb_path)
    
    model_config_path = f'{base_dir}/models/model_config.json'
    if os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        num_hidden = model_config.get('hidden_dim', 64)
        num_layers = model_config.get('n_layers', 3)
        negative_slope = model_config.get('negative_slope', 0.2)
        mask_rate = model_config.get('mask_rate', 0.5)
        alpha_l = model_config.get('alpha_l', 3)
    else:
        num_hidden = 64
        num_layers = 3
        negative_slope = 0.2
        mask_rate = 0.5
        alpha_l = 3
    
    model = GMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=negative_slope,
        residual=True,
        mask_rate=mask_rate,
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=alpha_l
    )
    
    model_path = f'{base_dir}/models/checkpoint_epoch_50.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    
    x_train = []
    with torch.no_grad():
        for window_key in tqdm(train_windows, desc="Train embeddings"):
            try:
                g_nx, node_features, _, _ = load_entity_level_dataset(
                    base_dir, window_key, use_llm_features, global_node_embeddings
                )
                
                if g_nx.number_of_nodes() == 0 or g_nx.number_of_edges() == 0:
                    continue
                
                g = transform_graph(g_nx, n_dim, e_dim, node_features, use_llm_features).to(device)
                embeddings = model.embed(g).cpu().numpy()
                x_train.append(embeddings)
                del g
            except Exception as e:
                continue
    
    x_train = np.concatenate(x_train, axis=0)
    
    test_windows_by_dataset = {}
    for window_key in test_windows:
        meta = window_metadata.get(window_key, {})
        dataset = meta.get('dataset', 'unknown')
        if dataset not in test_windows_by_dataset:
            test_windows_by_dataset[dataset] = []
        test_windows_by_dataset[dataset].append(window_key)
    
    overall_adp_global = 0.0
    
    global_all_log_scores = {}
    global_all_extracted_logs = set()
    global_all_log_level_labels = {}
    global_all_total_logs = {}
    global_all_attack_types = set()
    global_node_scores_dict = {}
    
    all_test_embeddings = []
    all_test_node_indices = []
    all_test_graphs_data = []
    
    for dataset, dataset_windows in test_windows_by_dataset.items():
        x_test_dataset = []
        test_node_indices_dataset = []
        test_graphs_data_dataset = []
        
        with torch.no_grad():
            for window_key in tqdm(dataset_windows, desc=f"{dataset} embeddings", leave=False):
                try:
                    g_nx, node_features, window_node_indices, edge_order = load_entity_level_dataset(
                        base_dir, window_key, use_llm_features, global_node_embeddings
                    )
                    
                    if g_nx.number_of_nodes() == 0 or g_nx.number_of_edges() == 0:
                        continue
                    
                    g = transform_graph(g_nx, n_dim, e_dim, node_features, use_llm_features).to(device)
                    embeddings = model.embed(g).cpu().numpy()
                    
                    for node_idx in window_node_indices:
                        test_node_indices_dataset.append(node_idx)
                    
                    x_test_dataset.append(embeddings)
                    
                    edges = edge_order if edge_order is not None else list(g_nx.edges())
                    test_graphs_data_dataset.append({
                        'window_key': window_key,
                        'edges': edges,
                        'node_indices': window_node_indices,
                        'dataset': dataset
                    })
                    
                    del g
                except Exception as e:
                    continue
        
        if len(x_test_dataset) > 0:
            all_test_embeddings.append(np.concatenate(x_test_dataset, axis=0))
            all_test_node_indices.extend(test_node_indices_dataset)
            all_test_graphs_data.extend(test_graphs_data_dataset)
    
    if len(all_test_embeddings) > 0:
        x_test_all = np.concatenate(all_test_embeddings, axis=0)
        _, _, node_scores_knn_all, _ = evaluate_entity_level_using_knn(
            x_train, x_test_all, None, n_neighbors=10
        )
        
        for i, node_idx in enumerate(all_test_node_indices):
            if node_idx not in global_node_scores_dict:
                global_node_scores_dict[node_idx] = []
            global_node_scores_dict[node_idx].append(node_scores_knn_all[i])
        
        for node_idx in global_node_scores_dict:
            global_node_scores_dict[node_idx] = max(global_node_scores_dict[node_idx])
    
    for graph_data in all_test_graphs_data:
        window_key = graph_data['window_key']
        edges = graph_data['edges']
        window_node_indices = graph_data['node_indices']
        dataset = graph_data['dataset']
        
        meta = window_metadata.get(window_key, {})
        dataset_from_meta = meta.get('dataset', dataset)
        attack_type = ATTACK_TYPE_MAPPING.get(dataset_from_meta, dataset_from_meta)
        
        parts = window_key.split('_')
        if len(parts) == 4:
            date1 = parts[0]
            time1 = parts[1].replace('-', ':')
            date2 = parts[2]
            time2 = parts[3].replace('-', ':')
            timestamp = f"{date1} {time1}_{date2} {time2}"
        else:
            timestamp = window_key.replace('_', ' ').replace('-', ':')
        
        edge_to_log_mapping = load_edge_to_log_mapping(base_dir, window_key, 'test')
        log_level_labels, total_logs, _ = load_log_level_ground_truth(
            dataset_from_meta, timestamp
        )
        
        window_prefix = f"{dataset_from_meta}_{timestamp}"
        for (log_idx, log_type), label in log_level_labels.items():
            unique_key = (window_prefix, log_idx, log_type)
            global_all_log_level_labels[unique_key] = label
            if label and label != False:
                if isinstance(label, str):
                    global_all_attack_types.add(label)
                elif isinstance(label, list):
                    global_all_attack_types.update(label)
        
        for log_type, count in total_logs.items():
            if log_type not in global_all_total_logs:
                global_all_total_logs[log_type] = 0
            global_all_total_logs[log_type] += count
        
        log_scores, _, extracted_logs = aggregate_node_scores_to_logs(
            global_node_scores_dict, window_node_indices, edge_to_log_mapping, edges
        )
        
        for (log_idx, log_type), score in log_scores.items():
            unique_key = (window_prefix, log_idx, log_type)
            if unique_key not in global_all_log_scores or score > global_all_log_scores[unique_key]:
                global_all_log_scores[unique_key] = score
        
        for (log_idx, log_type) in extracted_logs:
            unique_key = (window_prefix, log_idx, log_type)
            global_all_extracted_logs.add(unique_key)
    
    if len(global_all_attack_types) == 0:
        global_all_attack_types = set(ATTACK_TYPE_MAPPING.values())
    
    log_metrics_global = evaluate_log_level(
        global_all_log_scores, global_all_log_level_labels, global_all_total_logs,
        global_all_extracted_logs, sorted(global_all_attack_types)
    )
    
    overall_adp_global = float(log_metrics_global.get('adp', 0.0) or 0.0)
    f1_metrics_global = None
    
    results_list = []
    
    for dataset, dataset_windows in test_windows_by_dataset.items():
        attack_type = ATTACK_TYPE_MAPPING.get(dataset, dataset)
        
        x_test = []
        test_node_indices = []
        test_graphs_data = []
        
        with torch.no_grad():
            for window_key in tqdm(dataset_windows, desc=f"{dataset} embeddings"):
                try:
                    g_nx, node_features, window_node_indices, edge_order = load_entity_level_dataset(
                        base_dir, window_key, use_llm_features, global_node_embeddings
                    )
                    g = transform_graph(g_nx, n_dim, e_dim, node_features, use_llm_features).to(device)
                    embeddings = model.embed(g).cpu().numpy()
                    
                    for node_idx in window_node_indices:
                        test_node_indices.append(node_idx)
                    
                    x_test.append(embeddings)
                    
                    edges = edge_order if edge_order is not None else list(g_nx.edges())
                    test_graphs_data.append({
                        'window_key': window_key,
                        'edges': edges,
                        'node_indices': window_node_indices
                    })
                    
                    del g
                except Exception as e:
                    continue
        
        if len(x_test) == 0:
            continue
        
        x_test = np.concatenate(x_test, axis=0)
        
        _, _, node_scores_knn, _ = evaluate_entity_level_using_knn(
            x_train, x_test, None, n_neighbors=10
        )
        
        node_scores_dict = {}
        for i, node_idx in enumerate(test_node_indices):
            node_scores_dict[node_idx] = node_scores_knn[i]
        
        all_log_scores = {}
        all_extracted_logs = set()
        all_log_level_labels = {}
        all_total_logs = {}
        
        for graph_data in test_graphs_data:
            window_key = graph_data['window_key']
            edges = graph_data['edges']
            window_node_indices = graph_data['node_indices']
            
            meta = window_metadata.get(window_key, {})
            dataset_from_meta = meta.get('dataset', dataset)
            
            parts = window_key.split('_')
            if len(parts) == 4:
                date1 = parts[0]
                time1 = parts[1].replace('-', ':')
                date2 = parts[2]
                time2 = parts[3].replace('-', ':')
                timestamp = f'{date1} {time1}_{date2} {time2}'
            else:
                timestamp = window_key.replace('_', ' ').replace('-', ':')
            
            formatted_window_name = window_key
            edge_to_log_mapping = load_edge_to_log_mapping(base_dir, formatted_window_name, 'test')
            
            log_level_labels, total_logs, _ = load_log_level_ground_truth(
                dataset_from_meta, timestamp
            )
            
            window_prefix = f"{dataset_from_meta}_{timestamp}"
            for (log_idx, log_type), label in log_level_labels.items():
                unique_key = (window_prefix, log_idx, log_type)
                all_log_level_labels[unique_key] = label
            
            for log_type, count in total_logs.items():
                if log_type not in all_total_logs:
                    all_total_logs[log_type] = 0
                all_total_logs[log_type] += count
            
            log_scores, _, extracted_logs = aggregate_node_scores_to_logs(
                node_scores_dict, window_node_indices, edge_to_log_mapping, edges
            )
            
            for (log_idx, log_type), score in log_scores.items():
                unique_key = (window_prefix, log_idx, log_type)
                if unique_key not in all_log_scores or score > all_log_scores[unique_key]:
                    all_log_scores[unique_key] = score
            
            for (log_idx, log_type) in extracted_logs:
                unique_key = (window_prefix, log_idx, log_type)
                all_extracted_logs.add(unique_key)
        
        all_attack_types = set()
        for label in all_log_level_labels.values():
            if label and label != False:
                if isinstance(label, str):
                    all_attack_types.add(label)
                elif isinstance(label, list):
                    all_attack_types.update(label)
        
        if len(all_attack_types) == 0:
            all_attack_types = [attack_type]
        
        log_metrics = evaluate_log_level(
            all_log_scores, all_log_level_labels, all_total_logs,
            all_extracted_logs, sorted(all_attack_types)
        )
        
        overall_auc_roc = float(log_metrics.get('auc_roc', 0.0) or 0.0)
        overall_auc_pr = float(log_metrics.get('auc_pr', 0.0) or 0.0)
        overall_adp = overall_adp_global
        
        result_entry = {
            'dataset': dataset,
            'attack_type': attack_type,
            'AUC_ROC': overall_auc_roc,
            'AUC_PR': overall_auc_pr,
            'ADP': overall_adp
        }
        results_list.append(result_entry)
    
    if len(results_list) > 0:
        return results_list
    else:
        return None

def evaluate_llmlabel_mode(base_dir, labels_dir, folder_name=""):
    with open(f'{base_dir}/processed_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    use_llm_features = metadata.get('use_llm_features', False)
    n_dim = metadata['node_feature_dim']
    e_dim = metadata['edge_feature_dim']
    
    with open(f'{base_dir}/processed_data/train_windows.json', 'r') as f:
        train_windows = json.load(f)
    with open(f'{base_dir}/processed_data/test_windows.json', 'r') as f:
        test_windows = json.load(f)
    
    with open(f'{base_dir}/processed_data/window_metadata.json', 'r') as f:
        window_metadata = json.load(f)
    
    with open(f'{base_dir}/processed_data/node_mappings/idx2node.pkl', 'rb') as f:
        idx2node = pickle.load(f)
    
    global_node_embeddings = None
    if use_llm_features:
        emb_path = f'{base_dir}/processed_data/node_features/node_embeddings.npy'
        if os.path.exists(emb_path):
            global_node_embeddings = np.load(emb_path)
    
    model_dir = f'{base_dir}/models/model_10/'
    model_path = f'{model_dir}checkpoint_epoch_80.pt'
    model_config_path = f'{model_dir}model_config.json'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    model, model_config = load_model_for_eval(model_path, model_config_path, device)
    if model is None:
        raise RuntimeError("Failed to load model")
    
    model.eval()
    
    x_train = []
    with torch.no_grad():
        for window_key in tqdm(train_windows, desc="Train embeddings"):
            try:
                g_nx, node_features, _, _ = load_entity_level_dataset(
                    base_dir, window_key, use_llm_features, global_node_embeddings
                )
                
                if g_nx.number_of_nodes() == 0 or g_nx.number_of_edges() == 0:
                    continue
                
                g = transform_graph(g_nx, n_dim, e_dim, node_features, use_llm_features).to(device)
                embeddings = model.embed(g).cpu().numpy()
                x_train.append(embeddings)
                del g
            except Exception as e:
                continue
    
    if len(x_train) == 0:
        raise RuntimeError("No training embeddings generated")
    
    x_train = np.concatenate(x_train, axis=0)
    
    test_windows_by_dataset = {}
    for window_key in test_windows:
        meta = window_metadata.get(window_key, {})
        dataset = meta.get('dataset', 'unknown')
        if dataset not in test_windows_by_dataset:
            test_windows_by_dataset[dataset] = []
        test_windows_by_dataset[dataset].append(window_key)
    
    overall_adp_global = 0.0
    
    if True:
        global_all_log_scores = {}
        global_all_extracted_logs = set()
        global_all_log_level_labels = {}
        global_all_total_logs = {}
        global_all_attack_types = set()
        global_node_scores_dict = {}
        
        all_test_embeddings = []
        all_test_node_indices = []
        all_test_graphs_data = []
        
        for dataset, dataset_windows in test_windows_by_dataset.items():
            x_test_dataset = []
            test_node_indices_dataset = []
            test_graphs_data_dataset = []
            
            with torch.no_grad():
                for window_key in tqdm(dataset_windows, desc=f"{dataset} embeddings", leave=False):
                    try:
                        g_nx, node_features, window_node_indices, edge_order = load_entity_level_dataset(
                            base_dir, window_key, use_llm_features, global_node_embeddings
                        )
                        
                        if g_nx.number_of_nodes() == 0 or g_nx.number_of_edges() == 0:
                            continue
                        
                        g = transform_graph(g_nx, n_dim, e_dim, node_features, use_llm_features).to(device)
                        embeddings = model.embed(g).cpu().numpy()
                        
                        for node_idx in window_node_indices:
                            test_node_indices_dataset.append(node_idx)
                        
                        x_test_dataset.append(embeddings)
                        
                        edges = edge_order if edge_order is not None else list(g_nx.edges())
                        test_graphs_data_dataset.append({
                            'window_key': window_key,
                            'edges': edges,
                            'node_indices': window_node_indices,
                            'dataset': dataset
                        })
                        
                        del g
                    except Exception as e:
                        continue
            
            if len(x_test_dataset) > 0:
                all_test_embeddings.append(np.concatenate(x_test_dataset, axis=0))
                all_test_node_indices.extend(test_node_indices_dataset)
                all_test_graphs_data.extend(test_graphs_data_dataset)
        
        if len(all_test_embeddings) > 0:
            x_test_all = np.concatenate(all_test_embeddings, axis=0)
            _, _, node_scores_knn_all, _ = evaluate_entity_level_using_knn(
                x_train, x_test_all, None, n_neighbors=10
            )
            
            for i, node_idx in enumerate(all_test_node_indices):
                if node_idx not in global_node_scores_dict:
                    global_node_scores_dict[node_idx] = []
                global_node_scores_dict[node_idx].append(node_scores_knn_all[i])
            
            for node_idx in global_node_scores_dict:
                global_node_scores_dict[node_idx] = max(global_node_scores_dict[node_idx])
        
        for graph_data in all_test_graphs_data:
            window_key = graph_data['window_key']
            edges = graph_data['edges']
            window_node_indices = graph_data['node_indices']
            dataset = graph_data['dataset']
            
            meta = window_metadata.get(window_key, {})
            dataset_from_meta = meta.get('dataset', dataset)
            
            parts = window_key.split('_')
            if len(parts) == 4:
                date1 = parts[0]
                time1 = parts[1].replace('-', ':')
                date2 = parts[2]
                time2 = parts[3].replace('-', ':')
                timestamp = f"{date1} {time1}_{date2} {time2}"
            else:
                timestamp = window_key.replace('_', ' ').replace('-', ':')
            
            edge_to_log_mapping = load_edge_to_log_mapping(base_dir, window_key, 'test')
            log_level_labels, total_logs, _ = load_log_level_ground_truth(
                dataset_from_meta, timestamp
            )
            
            window_prefix = f"{dataset_from_meta}_{timestamp}"
            for (log_idx, log_type), label in log_level_labels.items():
                unique_key = (window_prefix, log_idx, log_type)
                global_all_log_level_labels[unique_key] = label
                if label and label != False:
                    if isinstance(label, str):
                        global_all_attack_types.add(label)
                    elif isinstance(label, list):
                        global_all_attack_types.update(label)
            
            for log_type, count in total_logs.items():
                if log_type not in global_all_total_logs:
                    global_all_total_logs[log_type] = 0
                global_all_total_logs[log_type] += count
            
            log_scores, _, extracted_logs = aggregate_node_scores_to_logs(
                global_node_scores_dict, window_node_indices, edge_to_log_mapping, edges
            )
            
            for (log_idx, log_type), score in log_scores.items():
                unique_key = (window_prefix, log_idx, log_type)
                if unique_key not in global_all_log_scores or score > global_all_log_scores[unique_key]:
                    global_all_log_scores[unique_key] = score
            
            for (log_idx, log_type) in extracted_logs:
                unique_key = (window_prefix, log_idx, log_type)
                global_all_extracted_logs.add(unique_key)
        
        if len(global_all_attack_types) == 0:
            global_all_attack_types = set(ATTACK_TYPE_MAPPING.values())
        
        log_metrics_global = evaluate_log_level(
            global_all_log_scores, global_all_log_level_labels, global_all_total_logs,
            global_all_extracted_logs, sorted(global_all_attack_types)
        )
        
        overall_adp_global = float(log_metrics_global.get('adp', 0.0) or 0.0)
        f1_metrics_global = {
            'best_f1': log_metrics_global.get('best_f1', 0.0),
            'best_threshold': log_metrics_global.get('best_threshold', 0.0),
            'best_precision': log_metrics_global.get('best_precision', 0.0),
            'best_recall': log_metrics_global.get('best_recall', 0.0)
        }
    else:
        f1_metrics_global = None
    
    results_list = []
    
    for dataset, dataset_windows in test_windows_by_dataset.items():
        attack_type = ATTACK_TYPE_MAPPING.get(dataset, dataset)
        
        x_test = []
        test_node_indices = []
        test_graphs_data = []
        
        with torch.no_grad():
            for window_key in tqdm(dataset_windows, desc=f"{dataset} embeddings", leave=False):
                try:
                    g_nx, node_features, window_node_indices, edge_order = load_entity_level_dataset(
                        base_dir, window_key, use_llm_features, global_node_embeddings
                    )
                    
                    if g_nx.number_of_nodes() == 0 or g_nx.number_of_edges() == 0:
                        continue
                    
                    g = transform_graph(g_nx, n_dim, e_dim, node_features, use_llm_features).to(device)
                    embeddings = model.embed(g).cpu().numpy()
                    
                    for node_idx in window_node_indices:
                        test_node_indices.append(node_idx)
                    
                    x_test.append(embeddings)
                    
                    edges = edge_order if edge_order is not None else list(g_nx.edges())
                    test_graphs_data.append({
                        'window_key': window_key,
                        'edges': edges,
                        'node_indices': window_node_indices
                    })
                    
                    del g
                except Exception as e:
                    continue
        
        if len(x_test) == 0:
            continue
        
        x_test = np.concatenate(x_test, axis=0)
        
        _, _, node_scores_knn, _ = evaluate_entity_level_using_knn(
            x_train, x_test, None, n_neighbors=10
        )
        
        node_scores_dict = {}
        for i, node_idx in enumerate(test_node_indices):
            node_scores_dict[node_idx] = node_scores_knn[i]
        
        all_log_scores = {}
        all_extracted_logs = set()
        all_log_level_labels = {}
        all_total_logs = {}
        
        for graph_data in test_graphs_data:
            window_key = graph_data['window_key']
            edges = graph_data['edges']
            window_node_indices = graph_data['node_indices']
            
            meta = window_metadata.get(window_key, {})
            dataset_from_meta = meta.get('dataset', dataset)
            
            parts = window_key.split('_')
            if len(parts) == 4:
                date1 = parts[0]
                time1 = parts[1].replace('-', ':')
                date2 = parts[2]
                time2 = parts[3].replace('-', ':')
                timestamp = f"{date1} {time1}_{date2} {time2}"
            else:
                timestamp = window_key.replace('_', ' ').replace('-', ':')
            
            formatted_window_name = window_key
            edge_to_log_mapping = load_edge_to_log_mapping(base_dir, formatted_window_name, 'test')
            
            log_level_labels, total_logs, _ = load_log_level_ground_truth(
                dataset_from_meta, timestamp
            )
            
            window_prefix = f"{dataset_from_meta}_{timestamp}"
            for (log_idx, log_type), label in log_level_labels.items():
                unique_key = (window_prefix, log_idx, log_type)
                all_log_level_labels[unique_key] = label
            
            for log_type, count in total_logs.items():
                if log_type not in all_total_logs:
                    all_total_logs[log_type] = 0
                all_total_logs[log_type] += count
            
            log_scores, _, extracted_logs = aggregate_node_scores_to_logs(
                node_scores_dict, window_node_indices, edge_to_log_mapping, edges
            )
            
            for (log_idx, log_type), score in log_scores.items():
                unique_key = (window_prefix, log_idx, log_type)
                if unique_key not in all_log_scores or score > all_log_scores[unique_key]:
                    all_log_scores[unique_key] = score
            
            for (log_idx, log_type) in extracted_logs:
                unique_key = (window_prefix, log_idx, log_type)
                all_extracted_logs.add(unique_key)
        
        all_attack_types = set()
        for label in all_log_level_labels.values():
            if label and label != False:
                if isinstance(label, str):
                    all_attack_types.add(label)
                elif isinstance(label, list):
                    all_attack_types.update(label)
        
        if len(all_attack_types) == 0:
            all_attack_types = [attack_type]
        
        log_metrics = evaluate_log_level(
            all_log_scores, all_log_level_labels, all_total_logs,
            all_extracted_logs, sorted(all_attack_types)
        )
        
        overall_auc_roc = float(log_metrics.get('auc_roc', 0.0) or 0.0)
        overall_auc_pr = float(log_metrics.get('auc_pr', 0.0) or 0.0)
        overall_adp = overall_adp_global
        
        result_entry = {
            'dataset': dataset,
            'attack_type': attack_type,
            'AUC_ROC': overall_auc_roc,
            'AUC_PR': overall_auc_pr,
            'ADP': overall_adp
        }
        results_list.append(result_entry)
    
    if len(results_list) > 0:
        return results_list
    else:
        return None

def _normalize_llmfets_model(model_name: str) -> str:
    return model_name.lower().replace(":", "_")

def parse_args():
    p = argparse.ArgumentParser(description="MAGIC ATLAS evaluation (baseline + llmlabel)")

    p.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES to set before importing evaluation code (e.g., '0' or '0,1').",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    default_artifacts_root = os.path.join(autoprov_dir, 'BIGDATA', 'MAGIC_artifacts', 'ATLAS_artifacts')
    
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
        "--llmlabel_dir",
        type=str,
        default=None,
        help="Path to llmlabel artifacts (directory that contains processed_data/ and models/).",
    )
    p.add_argument(
        "--embedding",
        type=str,
        default="mpnet",
        choices=["roberta", "mpnet", "minilm", "distilbert"],
        help="Embedding name for llmlabel artifact path construction (when --llmlabel_dir is not provided).",
    )
    p.add_argument(
        "--cee",
        type=str,
        default="gpt-4o",
        help="Candidate Edge Extractor name for llmlabel artifact path construction.",
    )
    p.add_argument(
        "--rule_generator",
        type=str,
        default="llama3_70b",
        help="Rule Generator name for llmlabel artifact path construction.",
    )
    p.add_argument(
        "--llmfets-model",
        dest="llmfets_model",
        type=str,
        default="llama3:70b",
        help="LLM model name used for feature extraction (used as subdirectory name, normalized).",
    )

    default_labels_dir = os.path.join(autoprov_dir, 'BIGDATA', 'ATLAS', 'labels')
    p.add_argument(
        "--labels_dir",
        type=str,
        default=default_labels_dir,
        help="Directory with ATLAS labels (passed through for parity).",
    )

    p.add_argument("--skip_baseline", action="store_true", help="Skip baseline evaluation.")
    p.add_argument("--skip_llmlabel", action="store_true", help="Skip llmlabel evaluation.")

    return p.parse_args()

def main():
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    artifacts_root = args.artifacts_root
    labels_dir = args.labels_dir

    baseline_dir = args.baseline_dir or os.path.join(artifacts_root, "original_atlas_graph")

    if args.llmlabel_dir:
        llmlabel_dir = args.llmlabel_dir
        folder_name = os.path.basename(os.path.dirname(llmlabel_dir))
    else:
        folder_name = f"{args.cee.lower()}_{args.rule_generator.lower()}"
        llmfets_model_norm = _normalize_llmfets_model(args.llmfets_model)
        llmlabel_dir = os.path.join(
            artifacts_root,
            f"rulellm_llmlabel_{args.embedding.lower()}",
            folder_name,
            llmfets_model_norm,
        )

    if not args.skip_baseline:
        if not os.path.exists(baseline_dir):
            raise FileNotFoundError(f"Baseline artifacts directory not found: {baseline_dir}")
        results_baseline = evaluate_baseline_mode(baseline_dir, labels_dir)
        if results_baseline:
            overall_adp = results_baseline[0].get('ADP', 0.0) if results_baseline else 0.0
            print("\nBaseline Model Results:")
            display_results_table(results_baseline, overall_adp)

    if not args.skip_llmlabel:
        if not os.path.exists(llmlabel_dir):
            raise FileNotFoundError(f"LLMLabel artifacts directory not found: {llmlabel_dir}")
        results_llmlabel = evaluate_llmlabel_mode(llmlabel_dir, labels_dir, folder_name=folder_name)
        if results_llmlabel:
            overall_adp = results_llmlabel[0].get('ADP', 0.0) if results_llmlabel else 0.0
            print("\n\nAuto-Prov Results:")
            display_results_table(results_llmlabel, overall_adp)

if __name__ == "__main__":
    main()

