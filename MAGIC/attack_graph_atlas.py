#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
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
from collections import defaultdict
from pathlib import Path

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
            from functools import partial
            def sce_loss(x, y, alpha=3):
                x = F.normalize(x, p=2, dim=-1)
                y = F.normalize(y, p=2, dim=-1)
                loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
                loss = loss.mean()
                return loss
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
    
    return score


def load_edge_timestamps_from_csv(csv_path, edge_order):
    edge_timestamps = {}
    
    if not os.path.exists(csv_path) or not edge_order:
        return edge_timestamps
    
    try:
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for edge_counter, row in enumerate(reader):
                if edge_counter < len(edge_order):
                    timestamp = row.get('timestamp', '').strip()
                    if timestamp:
                        edge_timestamps[edge_counter] = timestamp
    except Exception as e:
        pass
    
    return edge_timestamps


def build_global_graph_from_preprocessed(base_dir, test_windows, window_metadata, original_graph_dir=None):
    G = nx.MultiDiGraph()
    
    for window_key in tqdm(test_windows, desc="Building global graph", leave=False):
        graph_path = f'{base_dir}/processed_data/graphs/graph_{window_key}.pkl'
        
        if not os.path.exists(graph_path):
            continue
        
        try:
            with open(graph_path, 'rb') as f:
                data = pickle.load(f)
            
            g_nx = nx.node_link_graph(data['graph'], edges='links')
            edge_order = data.get('edge_order', [])
            
            edge_timestamps = {}
            if original_graph_dir and edge_order:
                parts = window_key.split('_')
                if len(parts) == 4:
                    original_window_name = f"{parts[0]} {parts[1].replace('-', ':')}_{parts[2]} {parts[3].replace('-', ':')}"
                else:
                    original_window_name = window_key.replace('_', ' ').replace('-', ':')
                
                csv_path = os.path.join(original_graph_dir, 'test', original_window_name, 'graph.csv')
                if not os.path.exists(csv_path):
                    csv_path = os.path.join(original_graph_dir, 'train', original_window_name, 'graph.csv')
                
                if os.path.exists(csv_path):
                    edge_timestamps = load_edge_timestamps_from_csv(csv_path, edge_order)
            
            for edge_counter, (src_idx, dst_idx) in enumerate(edge_order):
                if not g_nx.has_edge(src_idx, dst_idx):
                    continue
                
                edge_data = g_nx[src_idx][dst_idx]
                
                if isinstance(g_nx, nx.MultiDiGraph):
                    edge_attrs = list(edge_data.values())[0].copy() if edge_data else {}
                else:
                    edge_attrs = edge_data.copy()
                
                if not G.has_node(src_idx):
                    G.add_node(src_idx)
                if not G.has_node(dst_idx):
                    G.add_node(dst_idx)
                
                if 'type' not in edge_attrs:
                    edge_attrs['type'] = 0
                
                if edge_counter in edge_timestamps:
                    edge_attrs['timestamp'] = edge_timestamps[edge_counter]
                
                G.add_edge(src_idx, dst_idx, key=edge_counter, **edge_attrs)
        except Exception as e:
            continue
    
    return G


def expand_n_hop_with_anomalies(G, seed_node, n_hops, anomalous_nodes_set, correlated_nodes):
    if seed_node not in G:
        return set()
    
    subgraph_nodes = {seed_node}
    visited_anomalies = {seed_node}
    
    connected_nodes = set()
    for pred in G.predecessors(seed_node):
        if pred != seed_node:
            connected_nodes.add(pred)
    for succ in G.successors(seed_node):
        if succ != seed_node:
            connected_nodes.add(succ)
    
    if len(connected_nodes) == 0:
        return subgraph_nodes
    
    anomalous_connected = connected_nodes.intersection(anomalous_nodes_set)
    anomalous_connected = anomalous_connected - correlated_nodes
    subgraph_nodes.update(anomalous_connected)
    visited_anomalies.update(anomalous_connected)
    
    if n_hops >= 1:
        for conn_node in connected_nodes:
            neighbor_nodes = set()
            for pred in G.predecessors(conn_node):
                neighbor_nodes.add(pred)
            for succ in G.successors(conn_node):
                neighbor_nodes.add(succ)
            
            anomalies_1hop = neighbor_nodes.intersection(anomalous_nodes_set)
            anomalies_1hop = anomalies_1hop - visited_anomalies
            anomalies_1hop = anomalies_1hop - correlated_nodes
            
            if len(anomalies_1hop) > 0:
                subgraph_nodes.update(anomalies_1hop)
                subgraph_nodes.add(conn_node)
                visited_anomalies.update(anomalies_1hop)
    
    return subgraph_nodes


def partition_subgraph(subgraph, max_edges):
    G_undirected = subgraph.to_undirected()
    
    try:
        communities = nx.community.louvain_communities(G_undirected, resolution=1.0, seed=0)
    except Exception as e:
        return [subgraph]
    
    subgraphs_list = []
    for community in communities:
        if len(community) >= 3:
            comm_subgraph = subgraph.subgraph(community).copy()
            if comm_subgraph.number_of_edges() > 2:
                subgraphs_list.append(comm_subgraph)
    
    if len(subgraphs_list) == 0:
        return [subgraph]
    
    final_subgraphs = []
    for sg in subgraphs_list:
        if sg.number_of_edges() > max_edges:
            all_edges = list(sg.edges(keys=True, data=True))
            if len(all_edges) > max_edges:
                import random
                sampled_edges = random.sample(all_edges, max_edges)
                sampled_subgraph = nx.MultiDiGraph()
                for u, v, k, data in sampled_edges:
                    sampled_subgraph.add_edge(u, v, key=k, **data)
                final_subgraphs.append(sampled_subgraph)
            else:
                final_subgraphs.append(sg)
        else:
            final_subgraphs.append(sg)
    
    return final_subgraphs


def remove_duplicate_subgraphs(subgraphs):
    if len(subgraphs) <= 1:
        return subgraphs
    
    unique_subgraphs = []
    seen_signatures = set()
    
    for subgraph in subgraphs:
        edges = sorted([(u, v, data.get('type', 0)) 
                       for u, v, data in subgraph.edges(data=True)])
        signature = tuple(edges)
        
        if signature not in seen_signatures:
            unique_subgraphs.append(subgraph)
            seen_signatures.add(signature)
    
    return unique_subgraphs


def rank_subgraphs_by_anomaly(subgraphs, anomalous_nodes_scored):
    score_dict = dict(anomalous_nodes_scored)
    
    ranked = []
    for subgraph in subgraphs:
        node_scores = [score_dict.get(node, 0.0) for node in subgraph.nodes()]
        if node_scores:
            avg_score = np.mean(node_scores)
            max_score = np.max(node_scores)
            combined_score = 0.7 * avg_score + 0.3 * max_score
        else:
            combined_score = 0.0
        
        subgraph.graph['anomaly_score'] = combined_score
        ranked.append((subgraph, combined_score))
    
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    return [sg for sg, score in ranked]


def extract_attack_subgraphs(G, detected_node_ids, scores_dict, args, idx2node):
    detected_node_ids_filtered = {
        node_id for node_id in detected_node_ids 
        if node_id in scores_dict
    }
    
    if len(detected_node_ids_filtered) == 0:
        return []
    
    anomalous_nodes_scored = [
        (node_id, scores_dict[node_id])
        for node_id in detected_node_ids_filtered
    ]
    
    anomalous_nodes_scored.sort(key=lambda x: x[1], reverse=True)
    
    seed_nodes = [node_id for node_id, score in anomalous_nodes_scored[:args.top_k_seeds]]
    anomalous_nodes_set = set([node_id for node_id, score in anomalous_nodes_scored])
    
    subgraphs = []
    correlated_nodes = set()
    
    for i, seed_node in enumerate(seed_nodes):
        if seed_node not in G:
            continue
        
        if seed_node in correlated_nodes:
            continue
        
        subgraph_nodes = expand_n_hop_with_anomalies(
            G, seed_node, args.n_hops,
            anomalous_nodes_set, correlated_nodes
        )
        
        if len(subgraph_nodes) < args.min_nodes_subgraph:
            continue
        
        subgraph = G.subgraph(subgraph_nodes).copy()
        
        if subgraph.number_of_edges() > args.max_edges_subgraph:
            subgraphs.extend(partition_subgraph(subgraph, args.max_edges_subgraph))
        else:
            subgraphs.append(subgraph)
        
        correlated_nodes.update(subgraph_nodes)
    
    subgraphs = remove_duplicate_subgraphs(subgraphs)
    
    subgraphs_ranked = rank_subgraphs_by_anomaly(subgraphs, anomalous_nodes_scored)
    
    final_subgraphs = subgraphs_ranked[:args.top_k_subgraphs]
    
    return final_subgraphs


def save_attack_subgraphs(subgraphs, output_dir, dataset, mode, attack_name, idx2node, id_to_edge_type=None, atlas_entity_data=None):
    attack_dir = os.path.join(output_dir, dataset, mode, attack_name)
    os.makedirs(attack_dir, exist_ok=True)
    
    summary_data = []
    enrich_edges = (atlas_entity_data is not None)
    
    for i, subgraph in enumerate(subgraphs):
        node_id_to_name = {}
        for node_id in subgraph.nodes():
            node_name = idx2node.get(node_id, f"node_{node_id}")
            node_id_to_name[node_id] = node_name
        
        subgraph_named = nx.MultiDiGraph()
        
        for node_id in subgraph.nodes():
            node_name = node_id_to_name[node_id]
            subgraph_named.add_node(node_name)
        
        for u_id, v_id, key, edge_data in subgraph.edges(keys=True, data=True):
            u_name = node_id_to_name[u_id]
            v_name = node_id_to_name[v_id]
            
            edge_data_copy = edge_data.copy()
            if 'type' in edge_data_copy and id_to_edge_type is not None:
                type_id = edge_data_copy['type']
                type_name = id_to_edge_type.get(type_id, f"type_{type_id}")
                edge_data_copy['type'] = type_name
            
            subgraph_named.add_edge(u_name, v_name, key=key, **edge_data_copy)
        
        subgraph_named.graph.update(subgraph.graph)
        
        subgraph_file = os.path.join(attack_dir, f'subgraph_{i}.json')
        data = nx.node_link_data(subgraph_named)
        with open(subgraph_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        enriched_edge_count = 0
        
        if enrich_edges:
            enriched_dir = os.path.join(attack_dir, f'subgraph_{i}_enriched')
            os.makedirs(enriched_dir, exist_ok=True)
            
            for u_id, v_id, key, edge_data in subgraph.edges(keys=True, data=True):
                u_name = node_id_to_name[u_id]
                v_name = node_id_to_name[v_id]
                
                source_info = get_atlas_entity_info(u_name, atlas_entity_data)
                target_info = get_atlas_entity_info(v_name, atlas_entity_data)
                
                edge_type_name = 'unknown'
                if 'type' in edge_data and id_to_edge_type is not None:
                    type_id = edge_data['type']
                    edge_type_name = id_to_edge_type.get(type_id, f"type_{type_id}")
                
                edge_info = {
                    'source_node_name': u_name,
                    'target_node_name': v_name,
                    'source_enames': source_info['enames'],
                    'target_enames': target_info['enames'],
                    'source_type': source_info['type'],
                    'target_type': target_info['type'],
                    'source_functionality': source_info['functionality'],
                    'target_functionality': target_info['functionality'],
                    'edge_type': edge_type_name,
                    'timestamp': edge_data.get('timestamp', None)
                }
                
                u_name_safe = u_name[:8].replace('/', '_').replace('\\', '_').replace(':', '_')
                v_name_safe = v_name[:8].replace('/', '_').replace('\\', '_').replace(':', '_')
                edge_filename = f'edge_{u_name_safe}_to_{v_name_safe}_{key}.json'
                edge_file = os.path.join(enriched_dir, edge_filename)
                with open(edge_file, 'w') as f:
                    json.dump(edge_info, f, indent=2)
                
                enriched_edge_count += 1
        
        summary_entry = {
            'subgraph_id': i,
            'num_nodes': subgraph.number_of_nodes(),
            'num_edges': subgraph.number_of_edges(),
            'anomaly_score': subgraph.graph.get('anomaly_score', 0.0)
        }
        
        if enrich_edges:
            summary_entry['enriched_edges'] = enriched_edge_count
        
        summary_data.append(summary_entry)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(attack_dir, 'summary.csv'), index=False)


def load_atlas_entity_data(behavioral_profiles_path, llm_model='llama3_70b'):
    model_normalized = llm_model.lower().replace(':', '_')
    
    timeoh_path = Path(behavioral_profiles_path) / model_normalized / 'timeoh'
    
    data = {
        'typed_enames': {},
        'untyped_enames': {},
        'typed_types': {},
        'untyped_types': {},
        'typed_functionality': {},
        'untyped_functionality': {}
    }
    
    if not timeoh_path.exists():
        return data
    
    try:
        typed_enames_file = timeoh_path / 'typed_nodes_enames_atlas.json'
        if typed_enames_file.exists():
            with open(typed_enames_file, 'r') as f:
                data['typed_enames'] = json.load(f)
    except Exception as e:
        pass
    
    try:
        untyped_enames_file = timeoh_path / 'untyped_nodes_enames_atlas.json'
        if untyped_enames_file.exists():
            with open(untyped_enames_file, 'r') as f:
                data['untyped_enames'] = json.load(f)
    except Exception as e:
        pass
    
    try:
        typed_types_file = timeoh_path / 'typed_nodes_atlas.json'
        if typed_types_file.exists():
            with open(typed_types_file, 'r') as f:
                data['typed_types'] = json.load(f)
    except Exception as e:
        pass
    
    try:
        untyped_types_file = timeoh_path / 'untype2type_nodes_atlas.json'
        if untyped_types_file.exists():
            with open(untyped_types_file, 'r') as f:
                data['untyped_types'] = json.load(f)
    except Exception as e:
        pass
    
    try:
        typed_functionality_file = timeoh_path / 'typed_nodes_functionality_atlas.json'
        if typed_functionality_file.exists():
            with open(typed_functionality_file, 'r') as f:
                data['typed_functionality'] = json.load(f)
    except Exception as e:
        pass
    
    try:
        untyped_functionality_file = timeoh_path / 'untype2type_nodes_functionality_atlas.json'
        if untyped_functionality_file.exists():
            with open(untyped_functionality_file, 'r') as f:
                data['untyped_functionality'] = json.load(f)
    except Exception as e:
        pass
    
    return data


def get_atlas_entity_info(node_name, atlas_entity_data):
    enames = []
    if node_name in atlas_entity_data['typed_enames']:
        enames = atlas_entity_data['typed_enames'][node_name]
    elif node_name in atlas_entity_data['untyped_enames']:
        enames = atlas_entity_data['untyped_enames'][node_name]
    else:
        enames = ['unknown']
    
    entity_type = None
    if node_name in atlas_entity_data['typed_types']:
        entity_type = atlas_entity_data['typed_types'][node_name]
    elif node_name in atlas_entity_data['untyped_types']:
        entity_type = atlas_entity_data['untyped_types'][node_name]
    else:
        entity_type = 'unknown'
    
    functionality = None
    if node_name in atlas_entity_data['typed_functionality']:
        functionality = atlas_entity_data['typed_functionality'][node_name]
    elif node_name in atlas_entity_data['untyped_functionality']:
        functionality = atlas_entity_data['untyped_functionality'][node_name]
    else:
        functionality = 'N/A'
    
    return {'enames': enames, 'type': entity_type, 'functionality': functionality}


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


def extract_attack_graphs(base_dir, labels_dir, folder_name="", threshold=8.9218,
                          top_k_seeds=15, n_hops=1, max_edges_subgraph=5000,
                          min_nodes_subgraph=3, top_k_subgraphs=10, output_dir="./attack_graphs",
                          behavioral_profiles_path=None, llm_model='llama3_70b', original_graph_dir=None):
    with open(f'{base_dir}/processed_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    use_llm_features = metadata.get('use_llm_features', False)
    if not use_llm_features:
        raise ValueError("This script only works with --llmlabel mode (use_llm_features must be True)")
    
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
    
    id_to_edge_type = None
    edge_type_dict_path = f'{base_dir}/processed_data/node_mappings/edge_type_dict.pkl'
    if os.path.exists(edge_type_dict_path):
        with open(edge_type_dict_path, 'rb') as f:
            edge_type_dict = pickle.load(f)
        id_to_edge_type = {v: k for k, v in edge_type_dict.items()}
    
    atlas_entity_data = None
    if behavioral_profiles_path and os.path.exists(behavioral_profiles_path):
        atlas_entity_data = load_atlas_entity_data(behavioral_profiles_path, llm_model)
    
    global_node_embeddings = None
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
    
    G_global = build_global_graph_from_preprocessed(base_dir, test_windows, window_metadata, original_graph_dir)
    
    if G_global.number_of_nodes() == 0 or G_global.number_of_edges() == 0:
        raise RuntimeError("Global graph is empty")
    
    x_train = []
    failed_windows = []
    with torch.no_grad():
        for window_key in tqdm(train_windows, desc="Train embeddings"):
            try:
                g_nx, node_features, _, _ = load_entity_level_dataset(
                    base_dir, window_key, use_llm_features, global_node_embeddings
                )
                
                if g_nx.number_of_nodes() == 0 or g_nx.number_of_edges() == 0:
                    failed_windows.append((window_key, "empty graph"))
                    continue
                
                g = transform_graph(g_nx, n_dim, e_dim, node_features, use_llm_features).to(device)
                embeddings = model.embed(g).cpu().numpy()
                x_train.append(embeddings)
                del g
            except Exception as e:
                failed_windows.append((window_key, str(e)))
                continue
    
    if len(x_train) == 0:
        error_msg = "No training embeddings generated"
        if failed_windows:
            error_msg += f"\nFailed windows:\n"
            for window_key, reason in failed_windows[:10]:
                error_msg += f"  {window_key}: {reason}\n"
            if len(failed_windows) > 10:
                error_msg += f"  ... and {len(failed_windows) - 10} more\n"
        raise RuntimeError(error_msg)
    
    x_train = np.concatenate(x_train, axis=0)
    
    test_windows_by_dataset = {}
    for window_key in test_windows:
        meta = window_metadata.get(window_key, {})
        dataset = meta.get('dataset', 'unknown')
        if dataset not in test_windows_by_dataset:
            test_windows_by_dataset[dataset] = []
        test_windows_by_dataset[dataset].append(window_key)
    
    all_test_embeddings = []
    all_test_node_indices = []
    
    for dataset, dataset_windows in test_windows_by_dataset.items():
        with torch.no_grad():
            for window_key in tqdm(dataset_windows, desc=f"{dataset} embeddings", leave=False):
                try:
                    g_nx, node_features, window_node_indices, _ = load_entity_level_dataset(
                        base_dir, window_key, use_llm_features, global_node_embeddings
                    )
                    
                    if g_nx.number_of_nodes() == 0 or g_nx.number_of_edges() == 0:
                        continue
                    
                    g = transform_graph(g_nx, n_dim, e_dim, node_features, use_llm_features).to(device)
                    embeddings = model.embed(g).cpu().numpy()
                    
                    for node_idx in window_node_indices:
                        all_test_node_indices.append(node_idx)
                    
                    all_test_embeddings.append(embeddings)
                    del g
                except Exception as e:
                    continue
    
    if len(all_test_embeddings) == 0:
        raise RuntimeError("No test embeddings generated")
    
    x_test_all = np.concatenate(all_test_embeddings, axis=0)
    
    node_scores_knn = evaluate_entity_level_using_knn(x_train, x_test_all, None, n_neighbors=10)
    
    global_node_scores_dict = {}
    for i, node_idx in enumerate(all_test_node_indices):
        if node_idx not in global_node_scores_dict:
            global_node_scores_dict[node_idx] = []
        global_node_scores_dict[node_idx].append(node_scores_knn[i])
    
    for node_idx in global_node_scores_dict:
        global_node_scores_dict[node_idx] = max(global_node_scores_dict[node_idx])
    
    detected_node_ids = {
        node_id for node_id, score in global_node_scores_dict.items()
        if score >= threshold
    }
    
    class Args:
        def __init__(self):
            self.top_k_seeds = top_k_seeds
            self.n_hops = n_hops
            self.max_edges_subgraph = max_edges_subgraph
            self.min_nodes_subgraph = min_nodes_subgraph
            self.top_k_subgraphs = top_k_subgraphs
    
    args = Args()
    
    embedding_type = metadata.get('embedding_type', 'mpnet')
    mode = f"llmlabel_{embedding_type}"
    
    for dataset, dataset_windows in test_windows_by_dataset.items():
        attack_name = ATTACK_TYPE_MAPPING.get(dataset, dataset)
        
        dataset_node_ids = set()
        for window_key in dataset_windows:
            graph_path = f'{base_dir}/processed_data/graphs/graph_{window_key}.pkl'
            if os.path.exists(graph_path):
                try:
                    with open(graph_path, 'rb') as f:
                        data = pickle.load(f)
                    if 'node_indices' in data:
                        dataset_node_ids.update(data['node_indices'])
                except:
                    continue
        
        dataset_detected_nodes = detected_node_ids.intersection(dataset_node_ids)
        
        if len(dataset_detected_nodes) == 0:
            continue
        
        dataset_scores_dict = {
            node_id: global_node_scores_dict[node_id]
            for node_id in dataset_detected_nodes
        }
        
        try:
            subgraphs = extract_attack_subgraphs(
                G_global,
                dataset_detected_nodes,
                dataset_scores_dict,
                args,
                idx2node
            )
            
            if subgraphs:
                if folder_name:
                    output_path = os.path.join(output_dir, folder_name)
                else:
                    output_path = output_dir
                
                save_attack_subgraphs(
                    subgraphs,
                    output_path,
                    dataset,
                    mode,
                    attack_name,
                    idx2node,
                    id_to_edge_type,
                    atlas_entity_data
                )
        except Exception as e:
            continue


def parse_args():
    parser = argparse.ArgumentParser(description="MAGIC Attack Graph Extraction for ATLAS (--llmlabel only)")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    default_labels_dir = os.path.join(autoprov_dir, 'BIGDATA', 'ATLAS', 'labels')
    
    parser.add_argument("--artifacts_dir", type=str, default=None,
                       help="Directory with preprocessed ATLAS artifacts (for --llmlabel). If not provided, will auto-construct from --embedding, --cee, --rule_generator")
    parser.add_argument("--labels_dir", type=str, default=default_labels_dir,
                       help=f"Directory with malicious labels (S1-S4). Default: {default_labels_dir}")
    parser.add_argument("--embedding", type=str, default='mpnet',
                       choices=['roberta', 'mpnet', 'minilm', 'distilbert'],
                       help="Embedding model name")
    parser.add_argument("--cee", type=str, default='gpt-4o',
                       help="Candidate Edge Extractor name (e.g., 'gpt-4o')")
    parser.add_argument("--rule_generator", type=str, default='llama3_70b',
                       help="Rule Generator name (e.g., 'llama3_70b')")
    parser.add_argument("--threshold", type=float, default=8.9218,
                       help="Anomaly score threshold for node detection (default: 8.9218)")
    parser.add_argument("--top_k_seeds", type=int, default=15,
                       help="Top-K anomalous nodes as seeds (default: 15)")
    parser.add_argument("--n_hops", type=int, default=1,
                       help="Number of hops for subgraph expansion (default: 1)")
    parser.add_argument("--max_edges_subgraph", type=int, default=5000,
                       help="Maximum edges per subgraph (default: 5000)")
    parser.add_argument("--min_nodes_subgraph", type=int, default=3,
                       help="Minimum nodes per subgraph (default: 3)")
    parser.add_argument("--top_k_subgraphs", type=int, default=10,
                       help="Top-K subgraphs to save (default: 10)")
    parser.add_argument("--output_dir", type=str, default="./attack_graphs",
                       help="Output directory for attack subgraphs (default: ./attack_graphs)")
    default_behavioral_profiles = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ename-processing-ATLAS', 'behavioral-profiles')
    )
    parser.add_argument("--behavioral_profiles_path", type=str, 
                       default=default_behavioral_profiles,
                       help="Path to behavioral-profiles directory for entity enrichment (default: ../ename-processing-ATLAS/behavioral-profiles)")
    parser.add_argument("--llm_model", type=str, default='llama3_70b',
                       help="LLM model name for entity data (default: llama3_70b)")
    parser.add_argument("--llmfets-model", type=str, default="llama3:70b",
                       help="LLM model name used for feature extraction (default: llama3:70b, e.g., gpt-4o)")
    parser.add_argument("--original_graph_dir", type=str, default=None,
                       help="Optional path to original graph directory (autoprov_atlas_graph/{cee}_{rg}/) for loading timestamps from graph.csv files")
    
    args = parser.parse_args()
    
    if args.artifacts_dir is None:
        if not args.embedding or not args.cee or not args.rule_generator:
            parser.error("If --artifacts_dir is not provided, --embedding, --cee, and --rule_generator must be provided")
    
    return args


def main():
    args = parse_args()
    
    if args.artifacts_dir:
        base_dir = args.artifacts_dir
        folder_name = os.path.basename(base_dir)
        if folder_name.startswith('rulellm_llmlabel_'):
            folder_name = ""
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        autoprov_dir = os.path.dirname(script_dir)
        artifacts_root = os.path.join(autoprov_dir, 'BIGDATA', 'MAGIC_artifacts', 'ATLAS_artifacts')
        
        llmfets_model_normalized = args.llmfets_model.lower().replace(':', '_')
        base_artifacts_dir = os.path.join(artifacts_root, f"rulellm_llmlabel_{args.embedding.lower()}")
        folder_name = f"{args.cee.lower()}_{args.rule_generator.lower()}"
        base_dir = os.path.join(base_artifacts_dir, folder_name, llmfets_model_normalized)
    
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Artifacts directory not found: {base_dir}")
    
    metadata_path = f'{base_dir}/processed_data/metadata.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if not metadata.get('use_llm_features', False):
            raise ValueError(f"Artifacts directory {base_dir} is not in --llmlabel mode (use_llm_features=False)")
    else:
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    original_graph_dir = args.original_graph_dir
    if original_graph_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, '..', '..', 'STEP-LLM', 'rule_generator', 'ATLAS', 'ablation', 'autoprov_atlas_graph', folder_name)
        potential_path = os.path.normpath(potential_path)
        if os.path.exists(potential_path) and os.path.isdir(potential_path):
            original_graph_dir = potential_path
    
    extract_attack_graphs(
        base_dir=base_dir,
        labels_dir=args.labels_dir,
        folder_name=folder_name,
        threshold=args.threshold,
        top_k_seeds=args.top_k_seeds,
        n_hops=args.n_hops,
        max_edges_subgraph=args.max_edges_subgraph,
        min_nodes_subgraph=args.min_nodes_subgraph,
        top_k_subgraphs=args.top_k_subgraphs,
        output_dir=args.output_dir,
        behavioral_profiles_path=args.behavioral_profiles_path,
        llm_model=args.llm_model,
        original_graph_dir=original_graph_dir
    )


if __name__ == "__main__":
    main()

