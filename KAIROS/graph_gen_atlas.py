#!/usr/bin/env python3

import os
import sys
import argparse
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import json
import zipfile
import csv
import re
from tqdm import tqdm
from torch_geometric.data import TemporalData
from datetime import datetime
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import IncrementalPCA

def path2higlist(p):
    l = []
    spl = p.strip().split('/')
    for i in spl:
        if len(l) != 0:
            l.append(l[-1] + '/' + i)
        else:
            l.append(i)
    return l

def ip2higlist(p):
    l = []
    spl = p.strip().split('.')
    for i in spl:
        if len(l) != 0:
            l.append(l[-1] + '.' + i)
        else:
            l.append(i)
    return l

def list2str(l):
    return ''.join(l)

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
            
            if edge_type is None:
                if len(parts) >= 3:
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

def normalize_atlas_entity_name(name):
    if not name:
        return name
    
    if name in ['-', '-_0']:
        return name
    
    if re.match(r'^[A-Za-z]+_\d+$', name):
        return re.sub(r'_\d+$', '', name).lower()
    
    if name.startswith('connection_'):
        return name
    
    name = re.sub(r'_(\d+)$', '', name)
    name = name.replace('\\', '/')
    while '//' in name:
        name = name.replace('//', '/')
    
    name = name.lstrip('/')
    name = name.lower()
    
    name = re.sub(r'/system32/', '/system/', name)
    name = re.sub(r'/syswow64/', '/system/', name)
    name = re.sub(r'^system32/', 'system/', name)
    
    name = re.sub(r'harddiskvolume\d+', 'harddiskvolume', name)
    
    name = name.replace('program_files', 'program files')
    name = name.replace('common_files', 'common files')
    name = name.replace('mozilla_firefox', 'mozilla firefox')
    name = name.replace('microsoft_office', 'microsoft office')
    name = name.replace('vmware_tools', 'vmware tools')
    name = name.replace('windows_sidebar', 'windows sidebar')
    name = name.replace('microsoft_shared', 'microsoft shared')
    
    name = re.sub(r'python\d+', 'python', name)
    name = re.sub(r'v\d+(\.\d+)+', 'v.', name)
    name = re.sub(r'tcl\d+(\.\d+)?', 'tcl.', name)
    
    name = re.sub(r'x86_', 'x_', name)
    name = re.sub(r'x64_', 'x_', name)
    name = re.sub(r'amd64_', 'x_', name)
    
    name = re.sub(r'[0-9a-f]{32}', 'bfade...', name)
    name = re.sub(r'_[0-9a-f]{16}', 'bfade...', name)
    name = re.sub(r'_\d+\.\d+\.\d+\.\d+_', '..._', name)
    
    return name

def initialize_text_embedder(embedding_type):
    from sentence_transformers import SentenceTransformer
    
    model_map = {
        "mpnet": 'sentence-transformers/all-mpnet-base-v2',
        "minilm": 'sentence-transformers/all-MiniLM-L6-v2',
        "roberta": 'roberta-base',
        "distilbert": 'sentence-transformers/all-distilroberta-v1'
    }
    
    model_name = model_map.get(embedding_type.lower(), model_map["roberta"])
    print(f"  Initializing embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model

def load_atlas_llm_embeddings(node_names, dataset, embedding_type, feature_type,
                               pca_embedding_path, embedding_path, pca_dim=128,
                               use_pca=True, llmfets_model=None):
    if llmfets_model:
        llmfets_model_normalized = llmfets_model.lower().replace(':', '_')
    else:
        llmfets_model_normalized = 'llama3_70b'
    
    if use_pca:
        pca_file = os.path.join(pca_embedding_path, dataset.lower(), llmfets_model_normalized, embedding_type.lower(),
                                f"{feature_type}_pca{pca_dim}_all.pkl")
        pca_model_file = os.path.join(pca_embedding_path, dataset.lower(), llmfets_model_normalized, embedding_type.lower(),
                                      f"{feature_type}_pca{pca_dim}_model.pkl")
        
        if os.path.exists(pca_file) and os.path.exists(pca_model_file):
            print(f"  Loading PCA embeddings from: {pca_file}")
            with open(pca_file, 'rb') as f:
                precomputed_pca_embeddings = pickle.load(f)
            with open(pca_model_file, 'rb') as f:
                pca_model = pickle.load(f)
            
            print(f"  Loaded {len(precomputed_pca_embeddings)} PCA embeddings")
            embedding_dim = pca_dim
            
            embedding_model = initialize_text_embedder(embedding_type)
            
            node2embedding = {}
            direct_match_count = 0
            normalized_match_count = 0
            fallback_count = 0
            
            for node_name in tqdm(node_names, desc="  Aligning embeddings", leave=False):
                if node_name in precomputed_pca_embeddings:
                    node2embedding[node_name] = precomputed_pca_embeddings[node_name]
                    direct_match_count += 1
                else:
                    normalized_name = normalize_atlas_entity_name(node_name)
                    if normalized_name in precomputed_pca_embeddings:
                        node2embedding[node_name] = precomputed_pca_embeddings[normalized_name]
                        normalized_match_count += 1
                    else:
                        raw_emb = embedding_model.encode(node_name)
                        pca_emb = pca_model.transform(raw_emb.reshape(1, -1))[0]
                        node2embedding[node_name] = pca_emb
                        fallback_count += 1
            
            total_matched = direct_match_count + normalized_match_count
            print(f"\n  ┌───────────────────────────────────────────────────────────┐")
            print(f"  │ LLM Embedding Statistics                                  │")
            print(f"  ├───────────────────────────────────────────────────────────┤")
            print(f"  │ Total entities:        {len(node_names):>10}                      │")
            print(f"  │ Direct match:          {direct_match_count:>10} ({direct_match_count/len(node_names)*100:5.1f}%)             │")
            print(f"  │ Normalized match:      {normalized_match_count:>10} ({normalized_match_count/len(node_names)*100:5.1f}%)             │")
            print(f"  │ Total pre-computed:    {total_matched:>10} ({total_matched/len(node_names)*100:5.1f}%)             │")
            print(f"  │ Fallback (on-fly):     {fallback_count:>10} ({fallback_count/len(node_names)*100:5.1f}%)             │")
            print(f"  └───────────────────────────────────────────────────────────┘\n")
            
            if fallback_count > 0:
                print(f"  ⚠️  {fallback_count} entities used fallback encoding (SentenceTransformer on entity name)")
            if normalized_match_count > 0:
                print(f"  ✓  {normalized_match_count} entities matched via name normalization")
            
            return node2embedding, embedding_dim, pca_model
    
    print(f"  PCA embeddings not found, loading raw embeddings...")
    raw_embedding_file = os.path.join(embedding_path, dataset.lower(), llmfets_model_normalized,
                                     f"{embedding_type.lower()}_{feature_type}.pkl")
    
    pca_file = os.path.join(pca_embedding_path, dataset.lower(), llmfets_model_normalized, embedding_type.lower(),
                            f"{feature_type}_pca{pca_dim}_all.pkl")
    
    if not os.path.exists(raw_embedding_file):
        raise FileNotFoundError(f"Neither PCA embeddings nor raw embeddings found.\n"
                               f"  Tried PCA: {pca_file}\n"
                               f"  Tried raw: {raw_embedding_file}")
    
    print(f"  Loading raw embeddings from: {raw_embedding_file}")
    with open(raw_embedding_file, 'rb') as f:
        precomputed_raw_embeddings = pickle.load(f)
    
    print(f"  Loaded {len(precomputed_raw_embeddings)} raw embeddings")
    embedding_model = initialize_text_embedder(embedding_type)
    
    if use_pca:
        print(f"  Applying PCA reduction ({precomputed_raw_embeddings[list(precomputed_raw_embeddings.keys())[0]].shape[0]}D → {pca_dim}D)...")
        
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
        embedding_dim = precomputed_raw_embeddings[list(precomputed_raw_embeddings.keys())[0]].shape[0]
    
    node2embedding = {}
    direct_match_count = 0
    normalized_match_count = 0
    fallback_count = 0
    
    for node_name in tqdm(node_names, desc="  Aligning embeddings", leave=False):
        if node_name in precomputed_pca_embeddings:
            node2embedding[node_name] = precomputed_pca_embeddings[node_name]
            direct_match_count += 1
        else:
            normalized_name = normalize_atlas_entity_name(node_name)
            if normalized_name in precomputed_pca_embeddings:
                node2embedding[node_name] = precomputed_pca_embeddings[normalized_name]
                normalized_match_count += 1
            else:
                raw_emb = embedding_model.encode(node_name)
                if use_pca and pca_model is not None:
                    emb = pca_model.transform(raw_emb.reshape(1, -1))[0]
                else:
                    emb = raw_emb
                node2embedding[node_name] = emb
                fallback_count += 1
    
    total_matched = direct_match_count + normalized_match_count
    print(f"\n  ┌───────────────────────────────────────────────────────────┐")
    print(f"  │ LLM Embedding Statistics                                  │")
    print(f"  ├───────────────────────────────────────────────────────────┤")
    print(f"  │ Total entities:        {len(node_names):>10}                      │")
    print(f"  │ Direct match:          {direct_match_count:>10} ({direct_match_count/len(node_names)*100:5.1f}%)             │")
    print(f"  │ Normalized match:      {normalized_match_count:>10} ({normalized_match_count/len(node_names)*100:5.1f}%)             │")
    print(f"  │ Total pre-computed:    {total_matched:>10} ({total_matched/len(node_names)*100:5.1f}%)             │")
    print(f"  │ Fallback (on-fly):     {fallback_count:>10} ({fallback_count/len(node_names)*100:5.1f}%)             │")
    print(f"  └───────────────────────────────────────────────────────────┘\n")
    
    if fallback_count > 0:
        print(f"  ⚠️  {fallback_count} entities used fallback encoding (SentenceTransformer on entity name)")
    if normalized_match_count > 0:
        print(f"  ✓  {normalized_match_count} entities matched via name normalization")
    
    return node2embedding, embedding_dim, pca_model

def generate_node_features_llm(node_names, llm_embeddings):
    sorted_node_names = sorted(node_names)
    node2id = {name: idx for idx, name in enumerate(sorted_node_names)}
    
    embedding_dim = next(iter(llm_embeddings.values())).shape[0]
    node2higvec = []
    
    for node_name in sorted_node_names:
        if node_name in llm_embeddings:
            node2higvec.append(llm_embeddings[node_name])
        else:
            print(f"    Warning: Node {node_name} not in llm_embeddings, using zero vector")
            node2higvec.append(np.zeros(embedding_dim, dtype=np.float32))
    
    node2higvec = np.array(node2higvec).astype(np.float32)
    
    return node2higvec, node2id

def generate_node_features(node_names):
    FH_string = FeatureHasher(n_features=16, input_type="string")
    
    node2id = {name: idx for idx, name in enumerate(sorted(node_names))}
    node2higvec = []
    
    print(f"  Generating features for {len(node_names)} nodes...")
    
    for node_name in tqdm(sorted(node_names), desc="  Hashing features"):
        if '.' in node_name and not node_name.startswith('http'):
            higlist = path2higlist(node_name)
        elif '/' in node_name or '\\' in node_name:
            higlist = path2higlist(node_name)
        elif '_' in node_name:
            higlist = [p for p in node_name.split('_') if p]
        else:
            higlist = [node_name]
        
        vec = FH_string.transform([list2str(higlist)]).toarray()
        node2higvec.append(vec)
    
    node2higvec = np.array(node2higvec).reshape([-1, 16])
    
    return node2higvec, node2id

def generate_edge_type_features_dynamic(edge_types):
    num_types = len(edge_types)
    rel2id = {}
    
    for i, edge_type in enumerate(edge_types, start=1):
        rel2id[i] = edge_type
        rel2id[edge_type] = i
    
    relvec = torch.nn.functional.one_hot(torch.arange(0, num_types), num_classes=num_types)
    
    rel2vec = {}
    for i, edge_type in enumerate(edge_types):
        rel2vec[edge_type] = relvec[i]
    
    return rel2id, rel2vec

def zip_file(file_path):
    if not os.path.exists(file_path):
        return
    
    zip_path = file_path + '.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, os.path.basename(file_path))
    
    os.remove(file_path)

def process_original_atlas_graph_with_zipping(original_atlas_graph_dir, output_dir, rulellm_format=False,
                                              use_llmlabel=False, embedding_type=None, pca_dim=128,
                                              use_pca=True, pca_embedding_path=None, embedding_path=None, llmfets_model=None):
    train_dir = os.path.join(original_atlas_graph_dir, 'train')
    test_dir = os.path.join(original_atlas_graph_dir, 'test')
    
    if not os.path.exists(train_dir):
        print(f"  ERROR: Train directory not found: {train_dir}")
        return
    if not os.path.exists(test_dir):
        print(f"  ERROR: Test directory not found: {test_dir}")
        return
    
    experiment_dir = output_dir
    train_graphs_dir = os.path.join(experiment_dir, 'train', 'temporal_graphs')
    test_graphs_dir = os.path.join(experiment_dir, 'test', 'temporal_graphs')
    train_labels_dir = os.path.join(experiment_dir, 'train', 'edge_labels')
    test_labels_dir = os.path.join(experiment_dir, 'test', 'edge_labels')
    train_attack_types_dir = os.path.join(experiment_dir, 'train', 'edge_attack_types')
    test_attack_types_dir = os.path.join(experiment_dir, 'test', 'edge_attack_types')
    train_metadata_dir = os.path.join(experiment_dir, 'train', 'window_metadata')
    test_metadata_dir = os.path.join(experiment_dir, 'test', 'window_metadata')
    train_log_mapping_dir = os.path.join(experiment_dir, 'train', 'edge_to_log_mapping')
    test_log_mapping_dir = os.path.join(experiment_dir, 'test', 'edge_to_log_mapping')
    processed_dir = os.path.join(experiment_dir, 'processed_data', 'node_mappings')
    
    os.makedirs(train_graphs_dir, exist_ok=True)
    os.makedirs(test_graphs_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    os.makedirs(train_attack_types_dir, exist_ok=True)
    os.makedirs(test_attack_types_dir, exist_ok=True)
    os.makedirs(train_metadata_dir, exist_ok=True)
    os.makedirs(test_metadata_dir, exist_ok=True)
    os.makedirs(train_log_mapping_dir, exist_ok=True)
    os.makedirs(test_log_mapping_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"\n  Step 1: Collecting nodes and edge types from train and test...")
    all_nodes = set()
    all_edge_types = set()
    
    train_windows = sorted([d for d in os.listdir(train_dir) 
                           if os.path.isdir(os.path.join(train_dir, d))])
    print(f"    Train windows: {len(train_windows)}")
    for time_window in tqdm(train_windows, desc="      Scanning train", leave=False):
        txt_path = os.path.join(train_dir, time_window, 'graph.txt')
        if os.path.exists(txt_path):
            edges, nodes = parse_graph_txt(txt_path, rulellm_format=rulellm_format)
            all_nodes.update(nodes)
            for src, edge_type, dst, _ in edges:
                all_edge_types.add(edge_type)
    
    test_windows = sorted([d for d in os.listdir(test_dir) 
                          if os.path.isdir(os.path.join(test_dir, d))])
    print(f"    Test windows: {len(test_windows)}")
    for time_window in tqdm(test_windows, desc="      Scanning test", leave=False):
        txt_path = os.path.join(test_dir, time_window, 'graph.txt')
        if os.path.exists(txt_path):
            edges, nodes = parse_graph_txt(txt_path, rulellm_format=rulellm_format)
            all_nodes.update(nodes)
            for src, edge_type, dst, _ in edges:
                all_edge_types.add(edge_type)
    
    print(f"  Total unique nodes: {len(all_nodes)}")
    print(f"  Total unique edge types: {len(all_edge_types)}")
    
    if rulellm_format and len(all_edge_types) > 100:
        print(f"\n  WARNING: Very high number of edge types ({len(all_edge_types)})")
        print(f"     This is expected if LLM-generated actions are diverse.")
    
    print(f"\n  Step 2: Generating node features...")
    if use_llmlabel:
        print(f"  Using LLM embeddings ({embedding_type}, feature_type=type)...")
        llm_embeddings, embedding_dim, pca_model = load_atlas_llm_embeddings(
            all_nodes, 'atlas', embedding_type, 'type',
            pca_embedding_path, embedding_path, pca_dim, use_pca, llmfets_model
        )
        node2higvec, node2id = generate_node_features_llm(sorted(all_nodes), llm_embeddings)
        print(f"  Generated LLM features: shape {node2higvec.shape} (embedding_dim={embedding_dim})")
    else:
        node2higvec, node2id = generate_node_features(all_nodes)
        print(f"  Generated FeatureHasher features: shape {node2higvec.shape}")
    
    nodeid2msg = {}
    for node_name, node_id in node2id.items():
        nodeid2msg[node_id] = node_name
        nodeid2msg[node_name] = node_id
    
    rel2id, rel2vec = generate_edge_type_features_dynamic(sorted(all_edge_types))
    
    print(f"\n  Step 3: Processing train windows...")
    for time_window in tqdm(train_windows, desc="    Train"):
        window_dir = os.path.join(train_dir, time_window)
        txt_path = os.path.join(window_dir, 'graph.txt')
        labels_path = os.path.join(window_dir, 'malicious_labels.pkl')
        metadata_path = os.path.join(window_dir, 'window_metadata.json')
        
        if not os.path.exists(txt_path):
            continue
        
        edges, _ = parse_graph_txt(txt_path, rulellm_format=rulellm_format)
        
        enhanced_labels = {}
        if os.path.exists(labels_path):
            with open(labels_path, 'rb') as f:
                enhanced_labels = pickle.load(f)
        
        window_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                window_metadata = json.load(f)
        
        src_list = []
        dst_list = []
        time_list = []
        msg_list = []
        edge_label_list = []
        edge_attack_list = []
        
        try:
            start_time_str = time_window.split('_')[0]
            base_timestamp = int(datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1e9)
        except:
            base_timestamp = 0
        
        for src_name, edge_type, dst_name, line_num in edges:
            if edge_type not in rel2vec:
                continue
            
            src_idx = node2id[src_name]
            dst_idx = node2id[dst_name]
            
            src_list.append(src_idx)
            dst_list.append(dst_idx)
            time_list.append(base_timestamp + line_num)
            
            msg = torch.cat([
                torch.from_numpy(node2higvec[src_idx]),
                rel2vec[edge_type],
                torch.from_numpy(node2higvec[dst_idx])
            ])
            msg_list.append(msg)
            
            label_value = enhanced_labels.get(line_num, False)
            edge_label_list.append(label_value)
            if label_value == False:
                edge_attack_list.append(None)
            else:
                edge_attack_list.append(label_value)
        
        if len(src_list) == 0:
            continue
        
        temporal_graph = TemporalData()
        temporal_graph.src = torch.tensor(src_list, dtype=torch.long)
        temporal_graph.dst = torch.tensor(dst_list, dtype=torch.long)
        temporal_graph.t = torch.tensor(time_list, dtype=torch.long)
        temporal_graph.msg = torch.vstack(msg_list).to(torch.float)
        
        window_name = time_window.replace(' ', '_').replace(':', '-')
        graph_path = os.path.join(train_graphs_dir, f'graph_{window_name}.pt')
        torch.save(temporal_graph, graph_path)
        zip_file(graph_path)
        
        label_path = os.path.join(train_labels_dir, f'labels_{window_name}.pkl')
        with open(label_path, 'wb') as f:
            pickle.dump(edge_label_list, f)
        zip_file(label_path)
        
        attack_path = os.path.join(train_attack_types_dir, f'attacks_{window_name}.pkl')
        with open(attack_path, 'wb') as f:
            pickle.dump(edge_attack_list, f)
        zip_file(attack_path)
        
        if window_metadata:
            metadata_path_out = os.path.join(train_metadata_dir, f'metadata_{window_name}.json')
            with open(metadata_path_out, 'w') as f:
                json.dump(window_metadata, f, indent=2)
            zip_file(metadata_path_out)
        
        if rulellm_format:
            csv_path = os.path.join(window_dir, 'graph.csv')
            edge_to_log_mapping = parse_graph_csv(csv_path)
            if edge_to_log_mapping:
                mapping_path = os.path.join(train_log_mapping_dir, f'mapping_{window_name}.pkl')
                with open(mapping_path, 'wb') as f:
                    pickle.dump(edge_to_log_mapping, f)
                zip_file(mapping_path)
        else:
            mapping_src_path = os.path.join(window_dir, 'edge_to_log_mapping.pkl')
            if os.path.exists(mapping_src_path):
                try:
                    with open(mapping_src_path, 'rb') as f:
                        edge_to_log_mapping = pickle.load(f)
                    if edge_to_log_mapping:
                        mapping_path = os.path.join(train_log_mapping_dir, f'mapping_{window_name}.pkl')
                        with open(mapping_path, 'wb') as f:
                            pickle.dump(edge_to_log_mapping, f)
                        zip_file(mapping_path)
                except Exception as e:
                    pass
    
    print(f"\n  Step 4: Processing test windows...")
    for time_window in tqdm(test_windows, desc="    Test"):
        window_dir = os.path.join(test_dir, time_window)
        txt_path = os.path.join(window_dir, 'graph.txt')
        labels_path = os.path.join(window_dir, 'malicious_labels.pkl')
        metadata_path = os.path.join(window_dir, 'window_metadata.json')
        
        if not os.path.exists(txt_path):
            continue
        
        edges, _ = parse_graph_txt(txt_path, rulellm_format=rulellm_format)
        
        enhanced_labels = {}
        if os.path.exists(labels_path):
            with open(labels_path, 'rb') as f:
                enhanced_labels = pickle.load(f)
        
        window_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                window_metadata = json.load(f)
        
        src_list = []
        dst_list = []
        time_list = []
        msg_list = []
        edge_label_list = []
        edge_attack_list = []
        
        try:
            start_time_str = time_window.split('_')[0]
            base_timestamp = int(datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1e9)
        except:
            base_timestamp = 0
        
        for src_name, edge_type, dst_name, line_num in edges:
            if edge_type not in rel2vec:
                continue
            
            src_idx = node2id[src_name]
            dst_idx = node2id[dst_name]
            
            src_list.append(src_idx)
            dst_list.append(dst_idx)
            time_list.append(base_timestamp + line_num)
            
            msg = torch.cat([
                torch.from_numpy(node2higvec[src_idx]),
                rel2vec[edge_type],
                torch.from_numpy(node2higvec[dst_idx])
            ])
            msg_list.append(msg)
            
            label_value = enhanced_labels.get(line_num, False)
            edge_label_list.append(label_value)
            if label_value == False:
                edge_attack_list.append(None)
            else:
                edge_attack_list.append(label_value)
        
        if len(src_list) == 0:
            continue
        
        temporal_graph = TemporalData()
        temporal_graph.src = torch.tensor(src_list, dtype=torch.long)
        temporal_graph.dst = torch.tensor(dst_list, dtype=torch.long)
        temporal_graph.t = torch.tensor(time_list, dtype=torch.long)
        temporal_graph.msg = torch.vstack(msg_list).to(torch.float)
        
        window_name = time_window.replace(' ', '_').replace(':', '-')
        graph_path = os.path.join(test_graphs_dir, f'graph_{window_name}.pt')
        torch.save(temporal_graph, graph_path)
        zip_file(graph_path)
        
        label_path = os.path.join(test_labels_dir, f'labels_{window_name}.pkl')
        with open(label_path, 'wb') as f:
            pickle.dump(edge_label_list, f)
        zip_file(label_path)
        
        attack_path = os.path.join(test_attack_types_dir, f'attacks_{window_name}.pkl')
        with open(attack_path, 'wb') as f:
            pickle.dump(edge_attack_list, f)
        zip_file(attack_path)
        
        if window_metadata:
            metadata_path_out = os.path.join(test_metadata_dir, f'metadata_{window_name}.json')
            with open(metadata_path_out, 'w') as f:
                json.dump(window_metadata, f, indent=2)
            zip_file(metadata_path_out)
        
        if rulellm_format:
            csv_path = os.path.join(window_dir, 'graph.csv')
            edge_to_log_mapping = parse_graph_csv(csv_path)
            if edge_to_log_mapping:
                mapping_path = os.path.join(test_log_mapping_dir, f'mapping_{window_name}.pkl')
                with open(mapping_path, 'wb') as f:
                    pickle.dump(edge_to_log_mapping, f)
                zip_file(mapping_path)
        else:
            mapping_src_path = os.path.join(window_dir, 'edge_to_log_mapping.pkl')
            if os.path.exists(mapping_src_path):
                try:
                    with open(mapping_src_path, 'rb') as f:
                        edge_to_log_mapping = pickle.load(f)
                    if edge_to_log_mapping:
                        mapping_path = os.path.join(test_log_mapping_dir, f'mapping_{window_name}.pkl')
                        with open(mapping_path, 'wb') as f:
                            pickle.dump(edge_to_log_mapping, f)
                        zip_file(mapping_path)
                except Exception as e:
                    pass
    
    print(f"\n  Step 5: Saving unified node mappings...")
    nodeid2msg_path = os.path.join(processed_dir, 'nodeid2msg.pkl')
    with open(nodeid2msg_path, 'wb') as f:
        pickle.dump(nodeid2msg, f)
    zip_file(nodeid2msg_path)
    
    features_dir = os.path.join(experiment_dir, 'processed_data', 'node_features')
    os.makedirs(features_dir, exist_ok=True)
    
    node2higvec_path = os.path.join(features_dir, 'node2higvec.pt')
    torch.save(torch.from_numpy(node2higvec), node2higvec_path)
    zip_file(node2higvec_path)
    
    rel2id_path = os.path.join(features_dir, 'rel2id.pkl')
    with open(rel2id_path, 'wb') as f:
        pickle.dump(rel2id, f)
    zip_file(rel2id_path)
    
    rel2vec_path = os.path.join(features_dir, 'rel2vec.pt')
    torch.save(rel2vec, rel2vec_path)
    zip_file(rel2vec_path)
    
    from sklearn.preprocessing import LabelEncoder
    edge_label_encoder = LabelEncoder()
    edge_type_names = sorted([et for et in all_edge_types])
    edge_label_encoder.fit(edge_type_names)
    edge_label_encoder_path = os.path.join(features_dir, 'edge_label_encoder.pkl')
    with open(edge_label_encoder_path, 'wb') as f:
        pickle.dump(edge_label_encoder, f)
    zip_file(edge_label_encoder_path)
    
    print(f"  Saved edge type mappings: {len(all_edge_types)} unique edge types")
    
    embedding_dim_used = node2higvec.shape[1]
    if use_llmlabel:
        metadata_file = os.path.join(experiment_dir, 'processed_data', 'llm_embedding_metadata.json')
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        llm_metadata = {
            'use_llmlabel': True,
            'embedding_type': embedding_type,
            'feature_type': 'type',
            'pca_applied': use_pca,
            'pca_dim': pca_dim if use_pca else None,
            'embedding_dim': embedding_dim_used,
            'pca_embedding_path': pca_embedding_path,
            'embedding_path': embedding_path
        }
        with open(metadata_file, 'w') as f:
            json.dump(llm_metadata, f, indent=2)
        zip_file(metadata_file)
    
    print(f"\n  Original ATLAS Graph processing complete!")
    print(f"     Total nodes: {len(all_nodes)}")
    print(f"     Total edge types: {len(all_edge_types)}")
    print(f"     Train windows: {len(train_windows)}")
    print(f"     Test windows: {len(test_windows)}")
    if use_llmlabel:
        print(f"     Feature mode: LLM Type ({embedding_type}, {embedding_dim_used}D)")
    else:
        print(f"     Feature mode: FeatureHasher (16D)")
    print(f"     Artifacts saved to: {experiment_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="ATLAS Graph Generation")
    parser.add_argument("--baseline", action='store_true',
                        help="Generate baseline graphs (original ATLAS, no LLM)")
    parser.add_argument("--autoprov", action='store_true',
                        help="Generate autoprov graphs (RuleLLM with LLM embeddings)")
    parser.add_argument("--original_atlas_graph_dir", type=str, default=None,
                        help="Path to original_atlas_graph directory")
    parser.add_argument("--cee", type=str, default=None,
                        help="Candidate Edge Extractor name (e.g., 'gpt-4o', 'llama3_70b'). Required for --autoprov")
    parser.add_argument("--rule_generator", type=str, default=None,
                        help="Rule Generator name (e.g., 'llama3_70b', 'qwen2_72b'). Required for --autoprov")
    parser.add_argument("--embedding", type=str, default='mpnet',
                        choices=['roberta', 'mpnet', 'minilm', 'distilbert'],
                        help="Embedding model to use (for --autoprov). Default: mpnet")
    parser.add_argument("--pca_dim", type=int, default=128,
                        help="PCA dimensionality for embedding reduction (default: 128, for --autoprov)")
    parser.add_argument("--no_pca", action='store_true',
                        help="Disable PCA dimensionality reduction (for --autoprov)")
    parser.add_argument("--pca_embedding_path", type=str, default=None,
                        help="Path to pre-computed PCA embeddings directory")
    parser.add_argument("--embedding_path", type=str, default=None,
                        help="Path to raw embeddings directory for fallback")
    parser.add_argument("--llmfets-model", type=str, default="llama3:70b",
                        help="LLM model name used for feature extraction (default: llama3:70b)")
    args = parser.parse_args()
    
    if not args.baseline and not args.autoprov:
        parser.error("Must specify either --baseline or --autoprov")
    
    if args.baseline and args.autoprov:
        parser.error("--baseline and --autoprov are mutually exclusive")
    
    if args.autoprov:
        if not args.cee or not args.rule_generator:
            parser.error("--autoprov requires both --cee and --rule_generator")
    
    return args

def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    artifacts_root = os.path.join(autoprov_dir, 'KAIROS', 'ATLAS_artifacts')
    
    if args.baseline:
        if args.original_atlas_graph_dir is None:
            input_graph_dir = os.path.join(autoprov_dir, 'rule_generator', 'ATLAS', 'original_atlas_graph')
        else:
            input_graph_dir = args.original_atlas_graph_dir
        
        output_dir = os.path.join(artifacts_root, 'original_atlas_graph')
        
        print("="*70)
        print("ATLAS GRAPH GENERATION - BASELINE")
        print("="*70)
        print(f"Input:  {input_graph_dir}")
        print(f"Output: {output_dir}")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        process_original_atlas_graph_with_zipping(
            original_atlas_graph_dir=input_graph_dir,
            output_dir=output_dir,
            rulellm_format=False,
            use_llmlabel=False,
            embedding_type=None,
            pca_dim=128,
            use_pca=True,
            pca_embedding_path=None,
            embedding_path=None,
            llmfets_model=None
        )
    
    elif args.autoprov:
        base_input_dir = os.path.join(autoprov_dir, 'rule_generator', 'ATLAS', 'ablation', 'autoprov_atlas_graph')
        folder_name = f"{args.cee.lower()}_{args.rule_generator.lower()}"
        input_graph_dir = os.path.join(base_input_dir, folder_name)
        
        if not os.path.exists(input_graph_dir):
            print(f"ERROR: Folder not found: {input_graph_dir}")
            return
        
        llmfets_model_normalized = args.llmfets_model.lower().replace(':', '_')
        
        if args.pca_embedding_path is None:
            pca_embedding_path = os.path.join(autoprov_dir, 'BIGDATA', 'llmfets-pca-embedding')
        else:
            pca_embedding_path = args.pca_embedding_path
        
        if args.embedding_path is None:
            embedding_path = os.path.join(autoprov_dir, 'BIGDATA', 'llmfets-embedding')
        else:
            embedding_path = args.embedding_path
        
        output_dir = os.path.join(artifacts_root, f'rulellm_llmlabel_{args.embedding.lower()}', folder_name, llmfets_model_normalized)
        
        print("="*70)
        print("ATLAS GRAPH GENERATION - AUTOPROV")
        print("="*70)
        print(f"Mode: RuleLLM with LLM Embeddings")
        print(f"LLM Embeddings: {args.embedding} (PCA: {not args.no_pca}, dim: {args.pca_dim if not args.no_pca else 'full'})")
        print(f"Folder: {folder_name}")
        print(f"Input:  {input_graph_dir}")
        print(f"Output: {output_dir}")
        print("="*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        process_original_atlas_graph_with_zipping(
            original_atlas_graph_dir=input_graph_dir,
            output_dir=output_dir,
            rulellm_format=True,
            use_llmlabel=True,
            embedding_type=args.embedding,
            pca_dim=args.pca_dim,
            use_pca=not args.no_pca,
            pca_embedding_path=pca_embedding_path,
            embedding_path=embedding_path,
            llmfets_model=args.llmfets_model
        )
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()

