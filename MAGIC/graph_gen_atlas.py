#!/usr/bin/env python3

import os
import argparse
import pickle
import json
import glob
import zipfile
import csv
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict

ATTACK_TYPE_MAPPING = {
    'S1': 'Strategic web compromise',
    'S2': 'Malvertising dominate',
    'S3': 'Spam campaign',
    'S4': 'Pony campaign'
}

NODE_TYPES = ['process', 'file', 'IP_Address', 'connection', 'domain_name']

EDGE_TYPES = ['connect', 'read', 'resolve', 'web_request']

PCA_DIM = 128

def initialize_text_embedder(embedding_type):
    from sentence_transformers import SentenceTransformer
    
    model_map = {
        "mpnet": 'sentence-transformers/all-mpnet-base-v2',
        "minilm": 'sentence-transformers/all-MiniLM-L6-v2',
        "roberta": 'roberta-base',
        "distilbert": 'sentence-transformers/all-distilroberta-v1'
    }
    
    model_name = model_map.get(embedding_type.lower(), model_map["roberta"])
    model = SentenceTransformer(model_name)
    return model

def load_atlas_llm_embeddings(node_names, dataset, embedding_type, feature_type,
                               pca_embedding_path, embedding_path, pca_dim=128,
                               use_pca=True, llmfets_model=None):
    from sklearn.decomposition import IncrementalPCA
    
    model_normalized = None
    if llmfets_model:
        model_normalized = llmfets_model.lower().replace(':', '_')
    
    if use_pca:
        if dataset.lower() == "atlas" and model_normalized:
            pca_file = os.path.join(pca_embedding_path, dataset.lower(), model_normalized, embedding_type.lower(),
                                    f"{feature_type}_pca{pca_dim}_all.pkl")
            pca_model_file = os.path.join(pca_embedding_path, dataset.lower(), model_normalized, embedding_type.lower(),
                                          f"{feature_type}_pca{pca_dim}_model.pkl")
        else:
            pca_file = os.path.join(pca_embedding_path, dataset.lower(), embedding_type.lower(),
                                    f"{feature_type}_pca{pca_dim}_all.pkl")
            pca_model_file = os.path.join(pca_embedding_path, dataset.lower(), embedding_type.lower(),
                                          f"{feature_type}_pca{pca_dim}_model.pkl")
        
        if os.path.exists(pca_file) and os.path.exists(pca_model_file):
            with open(pca_file, 'rb') as f:
                precomputed_pca_embeddings = pickle.load(f)
            with open(pca_model_file, 'rb') as f:
                pca_model = pickle.load(f)
            
            embedding_dim = pca_dim
            
            embedding_model = initialize_text_embedder(embedding_type)
            
            node2embedding = {}
            missing_count = 0
            
            for node_name in tqdm(node_names, desc="  Aligning embeddings", leave=False):
                if node_name in precomputed_pca_embeddings:
                    node2embedding[node_name] = precomputed_pca_embeddings[node_name]
                else:
                    raw_emb = embedding_model.encode(node_name)
                    pca_emb = pca_model.transform(raw_emb.reshape(1, -1))[0]
                    node2embedding[node_name] = pca_emb
                    missing_count += 1
            
            return node2embedding, embedding_dim, pca_model
    if dataset.lower() == "atlas" and model_normalized:
        raw_embedding_file = os.path.join(embedding_path, dataset.lower(), model_normalized,
                                         f"{embedding_type.lower()}_{feature_type}.pkl")
    else:
        raw_embedding_file = os.path.join(embedding_path,
                                         f"{embedding_type.lower()}_{dataset.lower()}_{feature_type}.pkl")
    
    if use_pca:
        if dataset.lower() == "atlas" and model_normalized:
            pca_file = os.path.join(pca_embedding_path, dataset.lower(), model_normalized, embedding_type.lower(),
                                    f"{feature_type}_pca{pca_dim}_all.pkl")
        else:
            pca_file = os.path.join(pca_embedding_path, dataset.lower(), embedding_type.lower(),
                                    f"{feature_type}_pca{pca_dim}_all.pkl")
    else:
        pca_file = "N/A"
    
    if not os.path.exists(raw_embedding_file):
        raise FileNotFoundError(f"Neither PCA embeddings nor raw embeddings found.\n"
                               f"  Tried PCA: {pca_file}\n"
                               f"  Tried raw: {raw_embedding_file}")
    
    with open(raw_embedding_file, 'rb') as f:
        precomputed_raw_embeddings = pickle.load(f)
    
    embedding_model = initialize_text_embedder(embedding_type)
    
    if use_pca:
        first_key = next(iter(precomputed_raw_embeddings.keys()))
        original_dim = precomputed_raw_embeddings[first_key].shape[0]
        
        all_emb_values = np.array(list(precomputed_raw_embeddings.values()))
        ipca = IncrementalPCA(n_components=pca_dim, batch_size=10000)
        
        pca_batch_size = 10000
        num_batches = (all_emb_values.shape[0] + pca_batch_size - 1) // pca_batch_size
        for i in tqdm(range(num_batches), desc="  Fitting PCA", leave=False):
            start_idx = i * pca_batch_size
            end_idx = min((i + 1) * pca_batch_size, all_emb_values.shape[0])
            ipca.partial_fit(all_emb_values[start_idx:end_idx])
        
        transformed_values = ipca.transform(all_emb_values)
        pca_model = ipca
        embedding_dim = pca_dim
        
        node_keys = list(precomputed_raw_embeddings.keys())
        precomputed_pca_embeddings = {node_keys[i]: transformed_values[i] for i in range(len(node_keys))}
    else:
        precomputed_pca_embeddings = precomputed_raw_embeddings
        pca_model = None
        first_key = next(iter(precomputed_raw_embeddings.keys()))
        embedding_dim = precomputed_raw_embeddings[first_key].shape[0]
    
    node2embedding = {}
    missing_count = 0
    
    for node_name in tqdm(node_names, desc="  Aligning embeddings", leave=False):
        if node_name in precomputed_pca_embeddings:
            node2embedding[node_name] = precomputed_pca_embeddings[node_name]
        else:
            raw_emb = embedding_model.encode(node_name)
            if use_pca and pca_model is not None:
                emb = pca_model.transform(raw_emb.reshape(1, -1))[0]
            else:
                emb = raw_emb
            node2embedding[node_name] = emb
            missing_count += 1
    
    return node2embedding, embedding_dim, pca_model

def parse_graph_txt(txt_path, rulellm_format=False):
    edges = []
    nodes = set()
    
    known_edge_types = ['resolve', 'web_request', 'read', 'connect']
    
    with open(txt_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            if rulellm_format:
                if ' A: [' in line:
                    parts = line.split(' A: [', 1)
                    if len(parts) == 2:
                        src = parts[0].strip()
                        right_part = parts[1]
                        
                        if '] ' in right_part:
                            action_and_dest = right_part.split('] ', 1)
                            action_part = action_and_dest[0].strip()
                            dst = action_and_dest[1].strip()
                        elif right_part.endswith(']'):
                            action_part = right_part[:-1].strip()
                            dst = ''
                        elif ']' in right_part:
                            bracket_idx = right_part.index(']')
                            action_part = right_part[:bracket_idx].strip()
                            dst = right_part[bracket_idx + 1:].strip()
                        else:
                            continue
                        
                        if src and action_part and dst:
                            edge_type = action_part
                            edges.append((src, edge_type, dst, line_num))
                            nodes.add(src)
                            nodes.add(dst)
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            edge_type = None
            src = None
            dst = None
            
            for i, part in enumerate(parts):
                if part in known_edge_types:
                    edge_type = part
                    src = ' '.join(parts[:i])
                    dst = ' '.join(parts[i+1:])
                    break
            
            if edge_type is None and len(parts) >= 3:
                if len(parts) == 3:
                    src = parts[0]
                    edge_type = parts[1]
                    dst = parts[2]
                else:
                    src = parts[0]
                    dst = parts[-1]
                    edge_type = ' '.join(parts[1:-1])
            
            if edge_type and src and dst:
                edges.append((src, edge_type, dst, line_num))
                nodes.add(src)
                nodes.add(dst)
    
    return edges, nodes

def parse_graph_csv(csv_path):
    edge_to_log_mapping = {}
    
    if not os.path.exists(csv_path):
        return edge_to_log_mapping
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for edge_counter, row in enumerate(reader):
                log_idx = row.get('log_idx', '')
                log_type = row.get('log_type', '')
                
                if log_idx and log_type:
                    try:
                        log_idx_int = int(log_idx)
                        edge_to_log_mapping[edge_counter] = (log_idx_int, log_type)
                    except ValueError:
                        pass
    except Exception as e:
        pass
    
    return edge_to_log_mapping

def load_atlas_graph(window_path):
    graph_path = os.path.join(window_path, 'graph.pkl')
    metadata_path = os.path.join(window_path, 'window_metadata.json')
    
    with open(graph_path, 'rb') as f:
        g = pickle.load(f)
    
    window_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            window_metadata = json.load(f)
    
    return g, window_metadata

def get_window_key(window_name):
    return window_name.replace(' ', '_').replace(':', '-')

def create_type_mappings():
    node_type_dict = {ntype: idx for idx, ntype in enumerate(NODE_TYPES)}
    edge_type_dict = {etype: idx for idx, etype in enumerate(EDGE_TYPES)}
    
    return node_type_dict, edge_type_dict

def create_dynamic_edge_type_mapping(edge_types):
    sorted_types = sorted(edge_types)
    return {etype: idx for idx, etype in enumerate(sorted_types)}

def transform_atlas_graph(g, node_type_dict, edge_type_dict, global_node2idx):
    g_magic = nx.DiGraph()
    
    local_nodes = set()
    
    for src, dst, edge_data in g.edges(data=True):
        src_idx = global_node2idx.get(src)
        dst_idx = global_node2idx.get(dst)
        
        if src_idx is None or dst_idx is None:
            continue
        
        edge_type_name = edge_data.get('type', 'connect')
        edge_type_id = edge_type_dict.get(edge_type_name, 0)
        
        if not g_magic.has_edge(src_idx, dst_idx):
            g_magic.add_edge(src_idx, dst_idx, type=edge_type_id)
        
        local_nodes.add(src_idx)
        local_nodes.add(dst_idx)
    
    for node_name, node_data in g.nodes(data=True):
        node_idx = global_node2idx.get(node_name)
        if node_idx is None:
            continue
        
        node_type_name = node_data.get('type', 'process')
        node_type_id = node_type_dict.get(node_type_name, 0)
        
        if g_magic.has_node(node_idx):
            g_magic.nodes[node_idx]['type'] = node_type_id
    
    local_node_labels = []
    for node_idx in sorted(g_magic.nodes()):
        node_type_id = g_magic.nodes[node_idx].get('type', 0)
        local_node_labels.append(node_type_id)
    
    return g_magic, local_node_labels

def transform_rulellm_graph(edges, edge_type_dict, global_node2idx):
    g_magic = nx.DiGraph()
    
    for src_name, edge_type, dst_name, _ in edges:
        src_idx = global_node2idx.get(src_name)
        dst_idx = global_node2idx.get(dst_name)
        
        if src_idx is None or dst_idx is None:
            continue
        
        edge_type_id = edge_type_dict.get(edge_type, 0)
        
        if not g_magic.has_edge(src_idx, dst_idx):
            g_magic.add_edge(src_idx, dst_idx, type=edge_type_id)
            if not g_magic.has_node(src_idx):
                g_magic.add_node(src_idx, type=0)
            if not g_magic.has_node(dst_idx):
                g_magic.add_node(dst_idx, type=0)
    
    local_node_indices = sorted(g_magic.nodes())
    
    return g_magic, local_node_indices

def load_malicious_labels(labels_dir):
    malicious_labels = {}
    
    for dataset in ['S1', 'S2', 'S3', 'S4']:
        label_file = os.path.join(labels_dir, dataset, 'malicious_labels.txt')
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                names = set()
                for line in f:
                    name = line.strip()
                    if name:
                        names.add(name.lower())
                malicious_labels[dataset] = names
        else:
            malicious_labels[dataset] = set()
    
    return malicious_labels

def unzip_if_needed(zip_path, extract_to):
    if zip_path.endswith('.zip'):
        folder_name = os.path.splitext(os.path.basename(zip_path))[0]
        extracted_path = os.path.join(extract_to, folder_name)
        
        if os.path.exists(extracted_path) and os.path.isdir(extracted_path):
            if os.path.exists(os.path.join(extracted_path, 'train')) and os.path.exists(os.path.join(extracted_path, 'test')):
                return extracted_path
        
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return extracted_path
    else:
        return zip_path

def get_available_folders(base_dir):
    if not os.path.exists(base_dir):
        return []
    
    available_folders = []
    items = os.listdir(base_dir)
    
    for item in items:
        item_path = os.path.join(base_dir, item)
        
        if item.startswith('.'):
            continue
        
        if item.endswith('.zip'):
            folder_path = unzip_if_needed(item_path, base_dir)
            if folder_path and os.path.isdir(folder_path):
                if os.path.exists(os.path.join(folder_path, 'train')) and os.path.exists(os.path.join(folder_path, 'test')):
                    available_folders.append(os.path.basename(folder_path))
        elif os.path.isdir(item_path):
            if os.path.exists(os.path.join(item_path, 'train')) and os.path.exists(os.path.join(item_path, 'test')):
                available_folders.append(item)
    
    return sorted(available_folders)

def preprocess_atlas_baseline(args):
    atlas_graph_dir = args.atlas_graph_dir
    output_dir = args.output_dir
    labels_dir = args.labels_dir
    
    train_dir = os.path.join(atlas_graph_dir, 'train')
    test_dir = os.path.join(atlas_graph_dir, 'test')
    
    os.makedirs(f'{output_dir}/processed_data/graphs', exist_ok=True)
    os.makedirs(f'{output_dir}/processed_data/node_mappings', exist_ok=True)
    os.makedirs(f'{output_dir}/train/edge_to_log_mapping', exist_ok=True)
    os.makedirs(f'{output_dir}/test/edge_to_log_mapping', exist_ok=True)
    os.makedirs(f'{output_dir}/models', exist_ok=True)
    os.makedirs(f'{output_dir}/results', exist_ok=True)
    
    all_nodes = set()
    node_info = {}
    edge_types_found = set()
    node_types_found = set()
    
    all_windows = []
    
    if os.path.exists(train_dir):
        for window_name in sorted(os.listdir(train_dir)):
            window_path = os.path.join(train_dir, window_name)
            if os.path.isdir(window_path) and os.path.exists(os.path.join(window_path, 'graph.pkl')):
                all_windows.append(('train', window_name, window_path))
    
    if os.path.exists(test_dir):
        for window_name in sorted(os.listdir(test_dir)):
            window_path = os.path.join(test_dir, window_name)
            if os.path.isdir(window_path) and os.path.exists(os.path.join(window_path, 'graph.pkl')):
                all_windows.append(('test', window_name, window_path))
    
    for split, window_name, window_path in tqdm(all_windows, desc="  Scanning windows"):
        g, _ = load_atlas_graph(window_path)
        
        for node_name, node_data in g.nodes(data=True):
            all_nodes.add(node_name)
            node_type = node_data.get('type', 'process')
            node_info[node_name] = node_type
            node_types_found.add(node_type)
        
        for _, _, edge_data in g.edges(data=True):
            edge_type = edge_data.get('type', 'connect')
            edge_types_found.add(edge_type)
    
    node_type_dict, edge_type_dict = create_type_mappings()
    
    for ntype in node_types_found:
        if ntype not in node_type_dict:
            node_type_dict[ntype] = len(node_type_dict)
    
    for etype in edge_types_found:
        if etype not in edge_type_dict:
            edge_type_dict[etype] = len(edge_type_dict)
    
    sorted_nodes = sorted(all_nodes)
    global_node2idx = {node_name: idx for idx, node_name in enumerate(sorted_nodes)}
    global_idx2node = {idx: node_name for node_name, idx in global_node2idx.items()}
    
    node_idx_to_type = {}
    for node_name, idx in global_node2idx.items():
        node_type_name = node_info.get(node_name, 'process')
        node_idx_to_type[idx] = node_type_dict.get(node_type_name, 0)
    
    train_windows_list = []
    test_windows_list = []
    window_metadata_dict = {}
    
    for split, window_name, window_path in tqdm(all_windows, desc="  Processing windows"):
        g, metadata = load_atlas_graph(window_path)
        
        g_magic, local_node_labels = transform_atlas_graph(
            g, node_type_dict, edge_type_dict, global_node2idx
        )
        
        window_key = get_window_key(window_name)
        
        graph_json = nx.node_link_data(g_magic, edges="links")
        
        save_data = {
            'graph': graph_json,
            'node_labels': local_node_labels,
            'malicious_nodes': [],
            'num_nodes': g_magic.number_of_nodes(),
            'num_edges': g_magic.number_of_edges()
        }
        
        save_path = f'{output_dir}/processed_data/graphs/graph_{window_key}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        if g_magic.number_of_nodes() > 0 and g_magic.number_of_edges() > 0:
            if split == 'train':
                train_windows_list.append(window_key)
            else:
                test_windows_list.append(window_key)
        
        window_metadata_dict[window_key] = {
            'original_name': window_name,
            'split': split,
            'dataset': metadata.get('dataset', ''),
            'attack_type': metadata.get('attack_type', ''),
            'num_nodes': g_magic.number_of_nodes(),
            'num_edges': g_magic.number_of_edges()
        }
        
        mapping_src_path = os.path.join(window_path, 'edge_to_log_mapping.pkl')
        if os.path.exists(mapping_src_path):
            try:
                with open(mapping_src_path, 'rb') as f:
                    edge_to_log_mapping = pickle.load(f)
                if edge_to_log_mapping:
                    mapping_path = f'{output_dir}/{split}/edge_to_log_mapping/mapping_{window_key}.pkl'
                    with open(mapping_path, 'wb') as f:
                        pickle.dump(edge_to_log_mapping, f)
            except Exception as e:
                pass
    
    with open(f'{output_dir}/processed_data/node_mappings/node2idx.pkl', 'wb') as f:
        pickle.dump(global_node2idx, f)
    
    with open(f'{output_dir}/processed_data/node_mappings/idx2node.pkl', 'wb') as f:
        pickle.dump(global_idx2node, f)
    
    with open(f'{output_dir}/processed_data/node_mappings/node_idx_to_type.pkl', 'wb') as f:
        pickle.dump(node_idx_to_type, f)
    
    node2idx_json = {str(k): v for k, v in global_node2idx.items()}
    with open(f'{output_dir}/processed_data/node2idx.json', 'w') as f:
        json.dump(node2idx_json, f, indent=2)
    
    with open(f'{output_dir}/processed_data/node_mappings/node_type_dict.pkl', 'wb') as f:
        pickle.dump(node_type_dict, f)
    
    with open(f'{output_dir}/processed_data/node_mappings/edge_type_dict.pkl', 'wb') as f:
        pickle.dump(edge_type_dict, f)
    
    with open(f'{output_dir}/processed_data/train_windows.json', 'w') as f:
        json.dump(train_windows_list, f, indent=2)
    
    with open(f'{output_dir}/processed_data/test_windows.json', 'w') as f:
        json.dump(test_windows_list, f, indent=2)
    
    with open(f'{output_dir}/processed_data/window_metadata.json', 'w') as f:
        json.dump(window_metadata_dict, f, indent=2)
    
    malicious_labels = load_malicious_labels(labels_dir)
    with open(f'{output_dir}/processed_data/malicious_labels.pkl', 'wb') as f:
        pickle.dump(malicious_labels, f)
    
    metadata = {
        'dataset': 'ATLAS',
        'mode': 'original_atlas_graph',
        'use_llm_features': False,
        'node_feature_dim': len(node_type_dict),
        'edge_feature_dim': len(edge_type_dict),
        'num_train_windows': len(train_windows_list),
        'num_test_windows': len(test_windows_list),
        'total_nodes': len(global_node2idx),
        'node_type_dict': node_type_dict,
        'edge_type_dict': edge_type_dict,
        'attack_type_mapping': ATTACK_TYPE_MAPPING,
        'detection_mode': 'entity_level'
    }
    
    with open(f'{output_dir}/processed_data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def preprocess_atlas_autoprov(args, input_graph_dir, output_dir, folder_name):
    train_dir = os.path.join(input_graph_dir, 'train')
    test_dir = os.path.join(input_graph_dir, 'test')
    labels_dir = args.labels_dir
    
    os.makedirs(f'{output_dir}/processed_data/graphs', exist_ok=True)
    os.makedirs(f'{output_dir}/processed_data/node_mappings', exist_ok=True)
    os.makedirs(f'{output_dir}/processed_data/node_features', exist_ok=True)
    os.makedirs(f'{output_dir}/train/edge_to_log_mapping', exist_ok=True)
    os.makedirs(f'{output_dir}/test/edge_to_log_mapping', exist_ok=True)
    os.makedirs(f'{output_dir}/models', exist_ok=True)
    os.makedirs(f'{output_dir}/results', exist_ok=True)
    
    all_nodes = set()
    all_edge_types = set()
    all_windows = []
    
    if os.path.exists(train_dir):
        for window_name in sorted(os.listdir(train_dir)):
            window_path = os.path.join(train_dir, window_name)
            txt_path = os.path.join(window_path, 'graph.txt')
            if os.path.isdir(window_path) and os.path.exists(txt_path):
                all_windows.append(('train', window_name, window_path))
    
    if os.path.exists(test_dir):
        for window_name in sorted(os.listdir(test_dir)):
            window_path = os.path.join(test_dir, window_name)
            txt_path = os.path.join(window_path, 'graph.txt')
            if os.path.isdir(window_path) and os.path.exists(txt_path):
                all_windows.append(('test', window_name, window_path))
    
    for split, window_name, window_path in tqdm(all_windows, desc="  Scanning windows"):
        txt_path = os.path.join(window_path, 'graph.txt')
        edges, nodes = parse_graph_txt(txt_path, rulellm_format=True)
        all_nodes.update(nodes)
        for src, edge_type, dst, _ in edges:
            all_edge_types.add(edge_type)
    
    edge_type_dict = create_dynamic_edge_type_mapping(all_edge_types)
    
    sorted_nodes = sorted(all_nodes)
    global_node2idx = {node_name: idx for idx, node_name in enumerate(sorted_nodes)}
    global_idx2node = {idx: node_name for node_name, idx in global_node2idx.items()}
    
    llm_embeddings, embedding_dim, pca_model = load_atlas_llm_embeddings(
        all_nodes, 
        'atlas', 
        args.embedding, 
        'type',
        args.pca_embedding_path,
        args.embedding_path,
        PCA_DIM,
        True,
        args.llmfets_model
    )
    
    node_embeddings = np.zeros((len(global_node2idx), embedding_dim), dtype=np.float32)
    for node_name, idx in global_node2idx.items():
        if node_name in llm_embeddings:
            node_embeddings[idx] = llm_embeddings[node_name]
    
    with open(f'{output_dir}/processed_data/node_features/node_embeddings.pkl', 'wb') as f:
        pickle.dump(node_embeddings, f)
    
    np.save(f'{output_dir}/processed_data/node_features/node_embeddings.npy', node_embeddings)
    
    train_windows_list = []
    test_windows_list = []
    window_metadata_dict = {}
    
    for split, window_name, window_path in tqdm(all_windows, desc="  Processing windows"):
        txt_path = os.path.join(window_path, 'graph.txt')
        metadata_path = os.path.join(window_path, 'window_metadata.json')
        
        edges, _ = parse_graph_txt(txt_path, rulellm_format=True)
        
        window_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                window_metadata = json.load(f)
        
        g_magic, local_node_indices = transform_rulellm_graph(edges, edge_type_dict, global_node2idx)
        
        window_key = get_window_key(window_name)
        
        edge_order = []
        for src_name, edge_type, dst_name, _ in edges:
            src_idx = global_node2idx.get(src_name)
            dst_idx = global_node2idx.get(dst_name)
            if src_idx is not None and dst_idx is not None:
                edge_order.append((src_idx, dst_idx))
        
        graph_json = nx.node_link_data(g_magic, edges="links")
        
        save_data = {
            'graph': graph_json,
            'node_indices': local_node_indices,
            'edge_order': edge_order,
            'malicious_nodes': [],
            'num_nodes': g_magic.number_of_nodes(),
            'num_edges': g_magic.number_of_edges()
        }
        
        local_embeddings = []
        for node_idx in local_node_indices:
            node_name = global_idx2node[node_idx]
            if node_name in llm_embeddings:
                local_embeddings.append(llm_embeddings[node_name])
            else:
                local_embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
        save_data['node_embeddings'] = np.array(local_embeddings, dtype=np.float32)
        
        save_path = f'{output_dir}/processed_data/graphs/graph_{window_key}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        if g_magic.number_of_nodes() > 0 and g_magic.number_of_edges() > 0:
            if split == 'train':
                train_windows_list.append(window_key)
            else:
                test_windows_list.append(window_key)
        
        window_metadata_dict[window_key] = {
            'original_name': window_name,
            'split': split,
            'dataset': window_metadata.get('dataset', ''),
            'attack_type': window_metadata.get('attack_type', ''),
            'num_nodes': g_magic.number_of_nodes(),
            'num_edges': g_magic.number_of_edges()
        }
        
        csv_path = os.path.join(window_path, 'graph.csv')
        edge_to_log_mapping = parse_graph_csv(csv_path)
        if edge_to_log_mapping:
            mapping_path = f'{output_dir}/{split}/edge_to_log_mapping/mapping_{window_key}.pkl'
            with open(mapping_path, 'wb') as f:
                pickle.dump(edge_to_log_mapping, f)
        else:
            mapping_src_path = os.path.join(window_path, 'edge_to_log_mapping.pkl')
            if os.path.exists(mapping_src_path):
                try:
                    with open(mapping_src_path, 'rb') as f:
                        edge_to_log_mapping = pickle.load(f)
                    if edge_to_log_mapping:
                        mapping_path = f'{output_dir}/{split}/edge_to_log_mapping/mapping_{window_key}.pkl'
                        with open(mapping_path, 'wb') as f:
                            pickle.dump(edge_to_log_mapping, f)
                except Exception as e:
                    pass
    
    with open(f'{output_dir}/processed_data/node_mappings/node2idx.pkl', 'wb') as f:
        pickle.dump(global_node2idx, f)
    
    with open(f'{output_dir}/processed_data/node_mappings/idx2node.pkl', 'wb') as f:
        pickle.dump(global_idx2node, f)
    
    node2idx_json = {str(k): v for k, v in global_node2idx.items()}
    with open(f'{output_dir}/processed_data/node2idx.json', 'w') as f:
        json.dump(node2idx_json, f, indent=2)
    
    with open(f'{output_dir}/processed_data/node_mappings/edge_type_dict.pkl', 'wb') as f:
        pickle.dump(edge_type_dict, f)
    
    with open(f'{output_dir}/processed_data/train_windows.json', 'w') as f:
        json.dump(train_windows_list, f, indent=2)
    
    with open(f'{output_dir}/processed_data/test_windows.json', 'w') as f:
        json.dump(test_windows_list, f, indent=2)
    
    with open(f'{output_dir}/processed_data/window_metadata.json', 'w') as f:
        json.dump(window_metadata_dict, f, indent=2)
    
    malicious_labels = load_malicious_labels(labels_dir)
    with open(f'{output_dir}/processed_data/malicious_labels.pkl', 'wb') as f:
        pickle.dump(malicious_labels, f)
    
    metadata = {
        'dataset': 'ATLAS',
        'mode': 'rulellm',
        'folder': folder_name,
        'use_llm_features': True,
        'embedding_type': args.embedding,
        'pca_applied': True,
        'pca_dim': PCA_DIM,
        'node_feature_dim': embedding_dim,
        'edge_feature_dim': len(edge_type_dict),
        'num_train_windows': len(train_windows_list),
        'num_test_windows': len(test_windows_list),
        'total_nodes': len(global_node2idx),
        'edge_type_dict': edge_type_dict,
        'attack_type_mapping': ATTACK_TYPE_MAPPING,
        'detection_mode': 'entity_level'
    }
    
    with open(f'{output_dir}/processed_data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="MAGIC Graph Generation for ATLAS")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--baseline", action='store_true',
                           help="Baseline mode: Original ATLAS graphs (graph.pkl with node types)")
    mode_group.add_argument("--autoprov", action='store_true',
                           help="AutoProv mode: RuleLLM graphs with LLM embeddings (graph.txt format)")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    
    parser.add_argument("--atlas_graph_dir", type=str, 
                        default=None,
                        help="Directory with ATLAS original graphs (for --baseline). Default: ../rule_generator/ATLAS/original_atlas_graph")
    parser.add_argument("--output_dir", type=str,
                        default=None,
                        help="Directory to save preprocessed artifacts (auto-set based on mode)")
    default_labels_dir = os.path.join(autoprov_dir, 'BIGDATA', 'ATLAS', 'labels')
    parser.add_argument("--labels_dir", type=str,
                        default=default_labels_dir,
                        help=f"Directory with malicious labels (S1-S4). Default: {default_labels_dir}")
    
    parser.add_argument("--cee", type=str, default=None,
                        help="Candidate Edge Extractor name (e.g., 'gpt-4o', 'llama3_70b'). Only used when --autoprov is True.")
    parser.add_argument("--rule_generator", type=str, default=None,
                        help="Rule Generator name (e.g., 'llama3_70b', 'qwen2_72b'). Only used when --autoprov is True.")
    
    parser.add_argument("--embedding", type=str, default="mpnet",
                        choices=['roberta', 'mpnet', 'minilm', 'distilbert'],
                        help="Embedding model to use (for --autoprov). Default: mpnet")
    parser.add_argument("--llmfets-model", type=str, default="llama3:70b",
                        help="LLM model name used for feature extraction (default: llama3:70b, e.g., gpt-4o)")
    default_pca_embedding_path = os.path.join(autoprov_dir, 'BIGDATA', 'llmfets-pca-embedding')
    parser.add_argument("--pca_embedding_path", type=str, 
                        default=default_pca_embedding_path,
                        help=f"Path to pre-computed PCA embeddings directory. Default: {default_pca_embedding_path}")
    default_embedding_path = os.path.join(autoprov_dir, 'BIGDATA', 'llmfets-embedding')
    parser.add_argument("--embedding_path", type=str, 
                        default=default_embedding_path,
                        help=f"Path to raw embeddings directory for fallback. Default: {default_embedding_path}")
    
    args = parser.parse_args()
    
    if args.autoprov:
        if (args.cee is not None) != (args.rule_generator is not None):
            parser.error("Both --cee and --rule_generator must be provided together, or neither (to process all folders).")
    
    if args.baseline and args.atlas_graph_dir is None:
        args.atlas_graph_dir = "../rule_generator/ATLAS/original_atlas_graph"
    
    return args

def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    artifacts_dir = os.path.join(autoprov_dir, 'BIGDATA', 'MAGIC_artifacts', 'ATLAS_artifacts')
    
    if args.baseline:
        if args.output_dir is None:
            args.output_dir = os.path.join(artifacts_dir, 'original_atlas_graph')
        
        os.makedirs(args.output_dir, exist_ok=True)
        preprocess_atlas_baseline(args)
    
    elif args.autoprov:
        base_input_dir = "../rule_generator/ATLAS/ablation/autoprov_atlas_graph"
        
        if (args.cee is not None) != (args.rule_generator is not None):
            return
        
        if args.cee and args.rule_generator:
            folder_name = f"{args.cee.lower()}_{args.rule_generator.lower()}"
            input_graph_dir = os.path.join(base_input_dir, folder_name)
            
            if not os.path.exists(input_graph_dir):
                zip_path = os.path.join(base_input_dir, f"{folder_name}.zip")
                if os.path.exists(zip_path):
                    input_graph_dir = unzip_if_needed(zip_path, base_input_dir)
                else:
                    return
            
            folders_to_process = [(folder_name, input_graph_dir)]
        else:
            available_folders = get_available_folders(base_input_dir)
            
            if not available_folders:
                return
            
            folders_to_process = []
            for folder_name in available_folders:
                folder_path = os.path.join(base_input_dir, folder_name)
                folders_to_process.append((folder_name, folder_path))
        
        for folder_name, input_graph_dir in folders_to_process:
            if args.output_dir is None:
                llmfets_model_normalized = args.llmfets_model.lower().replace(':', '_')
                output_dir = os.path.join(artifacts_dir, f"rulellm_llmlabel_{args.embedding.lower()}", folder_name, llmfets_model_normalized)
            else:
                output_dir = os.path.join(args.output_dir, folder_name)
            
            os.makedirs(output_dir, exist_ok=True)
            preprocess_atlas_autoprov(args, input_graph_dir, output_dir, folder_name)

if __name__ == "__main__":
    main()
