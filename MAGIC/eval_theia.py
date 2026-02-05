#!/usr/bin/env python3

import os
import sys
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))
step_llm_dir = os.path.dirname(os.path.dirname(script_dir))
magic_dir = os.path.join(step_llm_dir, "STEP-LLM", "MAGIC")
sys.path.insert(0, magic_dir)

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
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score
import zipfile

try:
    import faiss
    FAISS_AVAILABLE = True
    FAISS_GPU = faiss.get_num_gpus() > 0
except ImportError:
    FAISS_AVAILABLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

DATASET_DATES = {
    "THEIA": {
        "train_start_date": "2018-04-03",
        "train_end_date": "2018-04-05",
        "test_start_date": "2018-04-09",
        "test_end_date": "2018-04-12"
    }
}

def load_entity_level_dataset(base_dir, time_window):
    graph_path = f'{base_dir}/processed_data/graphs/graph_{time_window}.pkl'
    
    with open(graph_path, 'rb') as f:
        data = pickle.load(f)
    
    g_nx = nx.node_link_graph(data['graph'], edges='links')
    
    return g_nx, data['node_labels']

def transform_graph(g, node_feature_dim, edge_feature_dim, node_embeddings=None):
    g_dgl = dgl.from_networkx(g, node_attrs=['type'], edge_attrs=['type'])
    
    if node_embeddings is not None:
        node_indices = g_dgl.ndata["type"].view(-1).long().cpu().numpy()
        node_features = torch.tensor(node_embeddings[node_indices], dtype=torch.float32)
        g_dgl.ndata["attr"] = node_features
    else:
        g_dgl.ndata["attr"] = F.one_hot(g_dgl.ndata["type"].view(-1).long(), num_classes=node_feature_dim).float()
    
    g_dgl.edata["attr"] = F.one_hot(g_dgl.edata["type"].view(-1).long(), num_classes=edge_feature_dim).float()
    
    return g_dgl

def get_attack_scenarios(dataset):
    script_dir_local = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir_local))
    step_llm_dir = os.path.join(root_dir, "STEP-LLM")
    pids_gt_dir = os.path.join(step_llm_dir, "PIDS_GT")
    
    attacks = {
        "THEIA": [
            ("Firefox_Backdoor_Drakon", os.path.join(pids_gt_dir, "THEIA", "node_Firefox_Backdoor_Drakon_In_Memory.csv"), "2018-04-10"),
            ("Browser_Extension_Drakon", os.path.join(pids_gt_dir, "THEIA", "node_Browser_Extension_Drakon_Dropper.csv"), "2018-04-12")
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

def evaluate_entity_level_using_knn(dataset, x_train, x_test, y_test, save_path=None):
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train_norm = (x_train - x_train_mean) / (x_train_std + 1e-6)
    x_test_norm = (x_test - x_train_mean) / (x_train_std + 1e-6)
    
    x_train_norm = x_train_norm.astype(np.float32)
    x_test_norm = x_test_norm.astype(np.float32)
    
    if dataset.lower() == 'cadets':
        n_neighbors = 200
    else:
        n_neighbors = 10
    
    if save_path and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            mean_distance, distances = pickle.load(f)
        del x_train_norm, x_test_norm
    else:
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
                
                del index, x_train_norm, x_test_norm
                FAISS_SUCCESS = True
                
            except Exception as e:
                FAISS_SUCCESS = False
        else:
            FAISS_SUCCESS = False
        
        if not FAISS_SUCCESS:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
            nbrs.fit(x_train_norm)
            
            idx = list(range(x_train_norm.shape[0]))
            import random
            random.shuffle(idx)
            distances_train, _ = nbrs.kneighbors(x_train_norm[idx][:min(50000, x_train_norm.shape[0])], n_neighbors=n_neighbors)
            del x_train_norm
            mean_distance = distances_train.mean()
            del distances_train
            distances, _ = nbrs.kneighbors(x_test_norm, n_neighbors=n_neighbors)
            del x_test_norm
            distances = distances.mean(axis=1)
            del nbrs
    
    score = distances / mean_distance
    del distances
    
    if y_test is not None and len(y_test) > 0:
        auc_roc = roc_auc_score(y_test, score)
        auc_pr = average_precision_score(y_test, score)
        return auc_roc, auc_pr, score
    else:
        return None, None, score

def compute_adp_pidsmaker_style(scores, nodes, node2attacks, labels):
    from collections import defaultdict
    
    if len(scores) == 0 or len(nodes) == 0:
        return 0.0
    
    scores = np.array(scores)
    nodes = np.array(nodes)
    labels = np.array(labels)
    
    all_attacks = set()
    for attacks in node2attacks.values():
        all_attacks.update(attacks)
    total_attacks = len(all_attacks)
    
    if total_attacks == 0:
        return 0.0
    
    sorted_indices = np.argsort(scores)[::-1]
    sorted_nodes = [nodes[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    
    detected_attacks = set()
    detected_attacks_percentages = [0]
    precisions = [0]
    
    tp = 0
    fp = 0
    
    for i, node in enumerate(sorted_nodes):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        
        if node in node2attacks:
            detected_attacks.update(node2attacks[node])
        
        precision = tp / (tp + fp)
        detected_attacks_percentage = (len(detected_attacks) / total_attacks) * 100
        
        precisions.append(precision)
        detected_attacks_percentages.append(detected_attacks_percentage)
    
    precision_to_attacks = defaultdict(float)
    for precision, detected_percentage in zip(precisions, detected_attacks_percentages):
        precision_to_attacks[precision] = max(precision_to_attacks[precision], detected_percentage)
    
    unique_precisions = []
    max_detected_attacks_percentages = []
    for precision in sorted(precision_to_attacks.keys()):
        unique_precisions.append(precision)
        max_detected_attacks_percentages.append(precision_to_attacks[precision])
    
    area_under_curve = np.trapz(max_detected_attacks_percentages, unique_precisions) / 100
    
    return float(area_under_curve)

def unzip_if_needed(file_path):
    if os.path.exists(file_path):
        return file_path, False
    
    zip_path = f"{file_path}.zip"
    if os.path.exists(zip_path):
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))
        return file_path, True
    
    return None, False

def load_node_embeddings(embedding_dir, unzipped_files=None):
    single_file = os.path.join(embedding_dir, 'node_embeddings.npy')
    if os.path.exists(single_file):
        return np.load(single_file)
    
    single_zip = f"{single_file}.zip"
    if os.path.exists(single_zip):
        print(f"Unzipping {single_zip}...")
        with zipfile.ZipFile(single_zip, 'r') as zip_ref:
            zip_ref.extractall(embedding_dir)
        embeddings = np.load(single_file)
        try:
            if os.path.exists(single_file):
                os.remove(single_file)
                print(f"Deleted unzipped file: {single_file}")
        except Exception as e:
            print(f"Warning: Could not delete {single_file}: {e}")
        return embeddings
    
    part1_path = os.path.join(embedding_dir, 'node_embeddings_part1.npy')
    part2_path = os.path.join(embedding_dir, 'node_embeddings_part2.npy')
    
    part1_actual, part1_unzipped = unzip_if_needed(part1_path)
    part2_actual, part2_unzipped = unzip_if_needed(part2_path)
    
    if part1_actual and part2_actual:
        if os.path.exists(part1_actual) and os.path.exists(part2_actual):
            print("Loading node embeddings from part files...")
            part1 = np.load(part1_actual)
            part2 = np.load(part2_actual)
            print(f"Concatenating parts: {part1.shape} + {part2.shape}...")
            embeddings = np.concatenate([part1, part2], axis=0)
            print(f"Loaded embeddings shape: {embeddings.shape}")
            
            if part1_unzipped:
                try:
                    if os.path.exists(part1_actual):
                        os.remove(part1_actual)
                        print(f"Deleted unzipped file: {part1_actual}")
                except Exception as e:
                    print(f"Warning: Could not delete {part1_actual}: {e}")
            
            if part2_unzipped:
                try:
                    if os.path.exists(part2_actual):
                        os.remove(part2_actual)
                        print(f"Deleted unzipped file: {part2_actual}")
                except Exception as e:
                    print(f"Warning: Could not delete {part2_actual}: {e}")
            
            return embeddings
        else:
            print(f"Warning: Expected embedding files not found after unzipping attempt")
    
    return None

def evaluate_model(base_dir, model_path, model_config, node_embeddings, dataset_dates, dataset, unzipped_files=None):
    try:
        metadata_path = f'{base_dir}/processed_data/metadata.json'
        if not os.path.exists(metadata_path):
            return pd.DataFrame()
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        n_dim = metadata['node_feature_dim']
        e_dim = metadata['edge_feature_dim']
        
        if not os.path.exists(model_path):
            return pd.DataFrame()
        
        model = GMAEModel(
            n_dim=n_dim,
            e_dim=e_dim,
            hidden_dim=model_config['num_hidden'],
            n_layers=model_config['num_layers'],
            n_heads=model_config['n_heads'],
            activation=model_config['activation'],
            feat_drop=model_config['feat_drop'],
            negative_slope=model_config['negative_slope'],
            residual=model_config['residual'],
            mask_rate=model_config['mask_rate'],
            norm=model_config['norm'],
            loss_fn='sce',
            alpha_l=model_config['alpha_l']
        )
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model = model.to(device)
        model.eval()
    except Exception:
        return pd.DataFrame()
    
    try:
        train_start_date = dataset_dates["train_start_date"]
        train_end_date = dataset_dates["train_end_date"]
        test_start_date = dataset_dates["test_start_date"]
        test_end_date = dataset_dates["test_end_date"]
        
        start_date = datetime.strptime(train_start_date, "%Y-%m-%d")
        end_date = datetime.strptime(train_end_date, "%Y-%m-%d")
        train_windows = []
        current_date = start_date
        while current_date <= end_date:
            for hour in range(24):
                window_key = f"{current_date.strftime('%Y-%m-%d')}_{hour:02d}"
                graph_path = f'{base_dir}/processed_data/graphs/graph_{window_key}.pkl'
                if os.path.exists(graph_path):
                    train_windows.append(window_key)
            current_date += timedelta(days=1)
        
        start_date = datetime.strptime(test_start_date, "%Y-%m-%d")
        end_date = datetime.strptime(test_end_date, "%Y-%m-%d")
        test_windows = []
        current_date = start_date
        while current_date <= end_date:
            for hour in range(24):
                window_key = f"{current_date.strftime('%Y-%m-%d')}_{hour:02d}"
                graph_path = f'{base_dir}/processed_data/graphs/graph_{window_key}.pkl'
                if os.path.exists(graph_path):
                    test_windows.append(window_key)
            current_date += timedelta(days=1)
        
        x_train = []
        with torch.no_grad():
            for window in tqdm(train_windows, desc="Train embeddings"):
                try:
                    g_nx, _ = load_entity_level_dataset(base_dir, window)
                    g = transform_graph(g_nx, n_dim, e_dim, node_embeddings).to(device)
                    embeddings = model.embed(g).cpu().numpy()
                    x_train.append(embeddings)
                    del g
                except Exception:
                    continue
        
        if not x_train:
            return pd.DataFrame()
        x_train = np.concatenate(x_train, axis=0)
        
        x_test = []
        with torch.no_grad():
            for window in tqdm(test_windows, desc="Test embeddings"):
                try:
                    g_nx, _ = load_entity_level_dataset(base_dir, window)
                    g = transform_graph(g_nx, n_dim, e_dim, node_embeddings).to(device)
                    embeddings = model.embed(g).cpu().numpy()
                    x_test.append(embeddings)
                    del g
                except Exception:
                    continue
        
        if not x_test:
            return pd.DataFrame()
        x_test = np.concatenate(x_test, axis=0)
        
        attack_scenarios = get_attack_scenarios(dataset)
        if not attack_scenarios:
            return pd.DataFrame()
        
        mapping_path = f'{base_dir}/processed_data/uuid_to_node_idx.json'
        actual_path, was_unzipped = unzip_if_needed(mapping_path)
        if actual_path is None or not os.path.exists(actual_path):
            return pd.DataFrame()
        
        if was_unzipped and unzipped_files is not None:
            unzipped_files.append(actual_path)
        
        with open(actual_path, 'r') as f:
            uuid_to_node_idx = json.load(f)
        
        node_idx_to_embed_positions = {}
        current_pos = 0
        for window in test_windows:
            try:
                g_nx, _ = load_entity_level_dataset(base_dir, window)
                for node_idx in sorted(g_nx.nodes()):
                    if node_idx not in node_idx_to_embed_positions:
                        node_idx_to_embed_positions[node_idx] = []
                    node_idx_to_embed_positions[node_idx].append((current_pos, window))
                    current_pos += 1
            except Exception:
                continue
        
        dummy_labels = np.zeros(x_test.shape[0])
        _, _, global_scores = evaluate_entity_level_using_knn(
            dataset, x_train, x_test, dummy_labels, save_path=None
        )
        
        results_list = []
        all_node_labels = {}
        node2attacks = {}
        
        for attack_name, csv_path, attack_test_date in attack_scenarios:
            if not os.path.exists(csv_path):
                continue
            
            GT_mal_uuids = load_malicious_ids_from_csv(csv_path)
            GT_mal_node_indices = set()
            for uuid in GT_mal_uuids:
                if uuid in uuid_to_node_idx:
                    node_idx = uuid_to_node_idx[uuid]
                    GT_mal_node_indices.add(node_idx)
            
            GT_mal_in_test = GT_mal_node_indices.intersection(node_idx_to_embed_positions.keys())
            
            y_test = np.zeros(x_test.shape[0])
            for node_idx in GT_mal_in_test:
                embed_positions_with_windows = node_idx_to_embed_positions[node_idx]
                for embed_pos, window_key in embed_positions_with_windows:
                    if window_key.startswith(attack_test_date):
                        y_test[embed_pos] = 1
            
            if int(y_test.sum()) == 0:
                continue
            
            auc_roc = roc_auc_score(y_test, global_scores)
            auc_pr = average_precision_score(y_test, global_scores)
            
            results_list.append({
                'attack_name': attack_name,
                'test_date': attack_test_date,
                'AUC_ROC': float(auc_roc),
                'AUC_PR': float(auc_pr),
                'malicious_nodes': int(y_test.sum())
            })
            
            for node_idx in node_idx_to_embed_positions.keys():
                embed_positions_with_windows = node_idx_to_embed_positions[node_idx]
                is_malicious_for_attack = False
                for embed_pos, window_key in embed_positions_with_windows:
                    if window_key.startswith(attack_test_date) and y_test[embed_pos] == 1:
                        is_malicious_for_attack = True
                        break
                
                if is_malicious_for_attack:
                    if node_idx not in node2attacks:
                        node2attacks[node_idx] = set()
                    node2attacks[node_idx].add(attack_name)
                    all_node_labels[node_idx] = 1
        
        all_unique_nodes = sorted(node_idx_to_embed_positions.keys())
        node_scores = []
        node_labels = []
        node_ids = []
        
        for node_idx in all_unique_nodes:
            embed_positions_with_windows = node_idx_to_embed_positions[node_idx]
            node_score_list = [global_scores[embed_pos] for embed_pos, _ in embed_positions_with_windows]
            max_score = max(node_score_list)
            node_scores.append(max_score)
            node_ids.append(node_idx)
            node_labels.append(all_node_labels.get(node_idx, 0))
        
        if node_scores:
            adp_value = compute_adp_pidsmaker_style(
                np.array(node_scores),
                np.array(node_ids),
                node2attacks,
                np.array(node_labels)
            )
            
            for result in results_list:
                result['ADP'] = float(adp_value)
        
        return pd.DataFrame(results_list)
    except Exception:
        return pd.DataFrame()

def main():
    dataset = "THEIA"
    dataset_dates = DATASET_DATES[dataset]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    artifacts_root = os.path.join(autoprov_dir, 'BIGDATA', 'MAGIC_artifacts')
    
    unzipped_files = []
    
    try:
        base_dir_baseline = os.path.join(artifacts_root, dataset)
        model_path_baseline = os.path.join(base_dir_baseline, 'models', 'checkpoint_final.pt')
        
        baseline_config = {
            'num_hidden': 64,
            'num_layers': 3,
            'n_heads': 4,
            'mask_rate': 0.5,
            'feat_drop': 0.1,
            'activation': 'prelu',
            'negative_slope': 0.2,
            'residual': True,
            'norm': 'BatchNorm',
            'alpha_l': 3
        }
        
        node_embeddings_baseline = None
        
        if not os.path.exists(base_dir_baseline):
            print(f"Warning: Baseline artifacts directory not found: {base_dir_baseline}")
            print("Skipping baseline evaluation.")
            results_baseline = pd.DataFrame()
        else:
            results_baseline = evaluate_model(
                base_dir_baseline, model_path_baseline, baseline_config,
                node_embeddings_baseline, dataset_dates, dataset, unzipped_files
            )
        
        base_dir_model15 = os.path.join(artifacts_root, f'{dataset}_rulellm_llmlabel_mpnet')
        model_dir_model15 = os.path.join(base_dir_model15, 'model_15')
        model_path_model15 = os.path.join(model_dir_model15, 'checkpoint_epoch_100.pt')
        
        model15_config = {
            'num_hidden': 64,
            'num_layers': 3,
            'n_heads': 4,
            'mask_rate': 0.5,
            'feat_drop': 0.3,
            'activation': 'prelu',
            'negative_slope': 0.2,
            'residual': True,
            'norm': 'BatchNorm',
            'alpha_l': 3
        }
        
        if not os.path.exists(base_dir_model15):
            print(f"Warning: AutoProv artifacts directory not found: {base_dir_model15}")
            print("Skipping AutoProv evaluation.")
            results_model15 = pd.DataFrame()
        else:
            embedding_dir = os.path.join(base_dir_model15, 'processed_data')
            node_embeddings_model15 = load_node_embeddings(embedding_dir, unzipped_files)
            
            results_model15 = evaluate_model(
                base_dir_model15, model_path_model15, model15_config,
                node_embeddings_model15, dataset_dates, dataset, unzipped_files
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
        
        if not results_model15.empty:
            display_cols = ['attack_name', 'AUC_ROC', 'AUC_PR', 'ADP']
            model15_display = results_model15[display_cols].copy()
            model15_display['AUC_ROC'] = model15_display['AUC_ROC'].round(3)
            model15_display['AUC_PR'] = model15_display['AUC_PR'].round(3)
            model15_display['ADP'] = model15_display['ADP'].round(3)
            print("\n\nAuto-Prov Results:")
            print(model15_display.to_string(index=False))
        else:
            print("\nNo AutoProv results to display.")
    
    finally:
        for file_path in unzipped_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Cleaned up unzipped file: {file_path}")
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")

if __name__ == "__main__":
    main()

