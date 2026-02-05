#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import pickle
import glob
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path
from sklearn.decomposition import IncrementalPCA

script_dir = os.path.dirname(os.path.abspath(__file__))
flash_dir = os.path.join(os.path.dirname(script_dir), 'FLASH')
sys.path.insert(0, flash_dir)

from stepllm_utils.vtype_res import getvtypes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute PCA-reduced LLM feature embeddings (matching MAGIC logic) - ATLAS version"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["theia", "fivedirections", "atlas"],
        help="Dataset name"
    )
    
    parser.add_argument(
        "--llmfets-model",
        type=str,
        default=None,
        help="LLM model name used for feature extraction (required for ATLAS, e.g., llama3:70b, gpt-4o)"
    )
    
    parser.add_argument(
        "--embedding",
        type=str,
        required=True,
        choices=["all", "mpnet", "minilm", "roberta", "distilbert"],
        help="Embedding type ('all' will process all embedding types)"
    )
    
    parser.add_argument(
        "--feature_type",
        type=str,
        required=True,
        choices=["all", "type", "functionality"],
        help="Feature type ('all' will process both type and functionality)"
    )
    
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=128,
        help="PCA dimensionality (default: 128)"
    )
    
    parser.add_argument(
        "--train_start_date",
        type=str,
        default=None,
        help="Training start date (YYYY-MM-DD), e.g., 2018-04-03"
    )
    
    parser.add_argument(
        "--train_end_date",
        type=str,
        default=None,
        help="Training end date (YYYY-MM-DD), e.g., 2018-04-05"
    )
    
    parser.add_argument(
        "--test_start_date",
        type=str,
        default=None,
        help="Test start date (YYYY-MM-DD), e.g., 2018-04-09"
    )
    
    parser.add_argument(
        "--test_end_date",
        type=str,
        default=None,
        help="Test end date (YYYY-MM-DD), e.g., 2018-04-12"
    )
    
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="../BIGDATA/llmfets-embedding/",
        help="Path to raw embeddings directory (default: ../BIGDATA/llmfets-embedding/)"
    )
    
    parser.add_argument(
        "--saving_path",
        type=str,
        default="../BIGDATA/llmfets-pca-embedding/",
        help="Output directory for PCA embeddings (default: ../BIGDATA/llmfets-pca-embedding/)"
    )
    
    parser.add_argument(
        "--extracted_graph_path",
        type=str,
        default=None,
        help="Path to extracted provenance graphs (required for non-ATLAS datasets, not needed for ATLAS)"
    )
    
    args = parser.parse_args()
    
    if args.dataset.lower() == "atlas":
        if not args.llmfets_model:
            parser.error("--llmfets-model is required for ATLAS dataset")
    else:
        if not all([args.train_start_date, args.train_end_date, args.test_start_date, args.test_end_date]):
            parser.error("Date arguments (train_start_date, train_end_date, test_start_date, test_end_date) are required for non-ATLAS datasets")
        if not args.extracted_graph_path:
            parser.error("--extracted_graph_path is required for non-ATLAS datasets")
    
    return args


def load_vtype_mapping(dataset):
    mapping_file = os.path.join(flash_dir, f"llmgeneratedvtypegroup_{dataset.lower()}.pkl")
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Vertex type mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'rb') as f:
        vtype_mapping = pickle.load(f)
    
    return vtype_mapping


def load_edge_type_validation(dataset):
    validation_file = os.path.join(script_dir, f"edge_type_validation_{dataset.lower()}.json")
    
    if not os.path.exists(validation_file):
        return {}
    
    with open(validation_file, 'r') as f:
        edge_validation = json.load(f)
    
    return edge_validation


def load_dependencies(dataset, extracted_graph_path):
    vtype_combinations, id_labels = getvtypes(dataset.lower(), extracted_graph_path)
    vtype_mapping = load_vtype_mapping(dataset)
    edge_validation = load_edge_type_validation(dataset)
    
    return vtype_combinations, id_labels, vtype_mapping, edge_validation


def fill_missing_timestamps(df):
    if df['timestamp'].isna().sum() == 0:
        return df
    
    df['timestamp'] = df['timestamp'].bfill()
    df['timestamp'] = df['timestamp'].ffill()
    
    return df


def load_csv_data(timestamp_dir):
    csv_files = glob.glob(os.path.join(timestamp_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {timestamp_dir}")
    
    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)
    df = fill_missing_timestamps(df)
    
    return df


def process_csv_nodes_and_edges(file_path, id_labels, start_date_str, end_date_str, 
                                  vtype_combinations, vtype_mapping, edge_validation):
    nodes = {}
    edges = []
    
    def is_no_label_action(action):
        if action == "NO LABEL":
            return True
        if action in edge_validation and edge_validation[action] == "INVALID":
            return True
        return False
    
    if start_date_str and end_date_str:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    else:
        start_date = None
        end_date = None
    
    timestamp_dirs = glob.glob(file_path)
    
    for timestamp_dir in tqdm(timestamp_dirs, desc="Processing CSV directories", leave=False):
        if start_date and end_date:
            base_name = timestamp_dir.split('/')[-2]
            file_date_str = base_name.split(' ')[0]
            try:
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                if not (start_date <= file_date <= end_date):
                    continue
            except:
                continue
        
        try:
            df = load_csv_data(timestamp_dir)
        except Exception as e:
            continue
        
        df = df[df['action'].notna()]
        if df.empty:
            continue
        
        def parse_list_fast(s):
            if not isinstance(s, str) or s == '[]' or not s.strip():
                return []
            s = s.strip('[]').replace("'", "").replace('"', '')
            return [x.strip() for x in s.split(',') if x.strip()]
        
        def create_compound_vtype(val_str):
            vals = parse_list_fast(val_str)
            if vals:
                unique_elements = sorted(set([v.lower() for v in vals]))
                return '+'.join(unique_elements)
            return None
        
        def safe_eval_longest_ename(val_str):
            vals = parse_list_fast(val_str)
            meaningless = {'datum', 'N/A', '},'}
            clean = [v for v in vals if v not in meaningless and len(v) > 1]
            return max(clean, key=len) if clean else ''
        
        df['src_vtype'] = df['source_vtypes'].apply(create_compound_vtype)
        df['dst_vtype'] = df['dest_vtypes'].apply(create_compound_vtype)
        df['src_ename'] = df['source_enames'].apply(safe_eval_longest_ename)
        df['dst_ename'] = df['dest_enames'].apply(safe_eval_longest_ename)
        df['is_no_label'] = df['action'].apply(is_no_label_action)
        
        no_label_df = df[df['is_no_label']]
        for row in no_label_df[['source_id', 'dest_id', 'src_vtype', 'dst_vtype', 'src_ename', 'dst_ename']].itertuples(index=False, name=None):
            src_id, dst_id, src_vtype, dst_vtype, src_ename, dst_ename = row
            
            if src_id not in id_labels or dst_id not in id_labels:
                continue
            
            src_vtype_final = src_vtype if src_vtype else id_labels.get(src_id, 'unknown')
            dst_vtype_final = dst_vtype if dst_vtype else id_labels.get(dst_id, 'unknown')
            
            nodes[src_id] = {src_vtype_final: src_ename}
            nodes[dst_id] = {dst_vtype_final: dst_ename}
        
        valid_df = df[~df['is_no_label']]
        for row in valid_df[['source_id', 'dest_id', 'action', 'timestamp', 'src_vtype', 'dst_vtype', 'src_ename', 'dst_ename']].itertuples(index=False, name=None):
            src_id, dst_id, action, timestamp, src_vtype, dst_vtype, src_ename, dst_ename = row
            
            if src_id not in id_labels or dst_id not in id_labels:
                continue
            
            if not isinstance(action, str):
                continue
            
            src_vtype_final = src_vtype if src_vtype else id_labels.get(src_id, 'unknown')
            dst_vtype_final = dst_vtype if dst_vtype else id_labels.get(dst_id, 'unknown')
            
            if src_id not in nodes:
                nodes[src_id] = {src_vtype_final: src_ename}
            if dst_id not in nodes:
                nodes[dst_id] = {dst_vtype_final: dst_ename}
            
            edges.append((src_id, dst_id, action, timestamp))
    
    return nodes, edges


def split_train_test_nodes(nodes_dict, edges_list, train_start_date, train_end_date, 
                            test_start_date, test_end_date):
    train_nodes_dict = {}
    test_nodes_dict = {}
    
    train_start = datetime.strptime(train_start_date, "%Y-%m-%d")
    train_end = datetime.strptime(train_end_date, "%Y-%m-%d")
    test_start = datetime.strptime(test_start_date, "%Y-%m-%d")
    test_end = datetime.strptime(test_end_date, "%Y-%m-%d")
    
    for src_uuid, dst_uuid, edge_type, timestamp in tqdm(edges_list, desc="Splitting train/test nodes", leave=False):
        dt = datetime.fromtimestamp(int(timestamp) / 1e9)
        edge_date = datetime.strptime(dt.strftime("%Y-%m-%d"), "%Y-%m-%d")
        
        if train_start <= edge_date <= train_end:
            if src_uuid in nodes_dict:
                train_nodes_dict[src_uuid] = nodes_dict[src_uuid]
            if dst_uuid in nodes_dict:
                train_nodes_dict[dst_uuid] = nodes_dict[dst_uuid]
        elif test_start <= edge_date <= test_end:
            if src_uuid in nodes_dict:
                test_nodes_dict[src_uuid] = nodes_dict[src_uuid]
            if dst_uuid in nodes_dict:
                test_nodes_dict[dst_uuid] = nodes_dict[dst_uuid]
    
    overlap_nodes = set(train_nodes_dict.keys()) & set(test_nodes_dict.keys())
    for uuid in overlap_nodes:
        del test_nodes_dict[uuid]
    
    return train_nodes_dict, test_nodes_dict, len(overlap_nodes)


def load_precomputed_embeddings(dataset, embedding_type, feature_type, embedding_path, llmfets_model=None):
    if dataset.lower() == "atlas" and llmfets_model:
        model_normalized = llmfets_model.lower().replace(':', '_')
        embedding_file = os.path.join(
            embedding_path,
            dataset.lower(),
            model_normalized,
            f"{embedding_type.lower()}_{feature_type}.pkl"
        )
    else:
        embedding_file = os.path.join(
            embedding_path,
            f"{embedding_type.lower()}_{dataset.lower()}_{feature_type}.pkl"
        )
    
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Pre-computed embeddings not found: {embedding_file}")
    
    with open(embedding_file, 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    return embeddings_dict


def initialize_text_embedder(embedding_type):
    from sentence_transformers import SentenceTransformer
    
    model_map = {
        "mpnet": 'sentence-transformers/all-mpnet-base-v2',
        "minilm": 'sentence-transformers/all-MiniLM-L6-v2',
        "roberta": 'roberta-base',
        "distilbert": 'sentence-transformers/all-distilroberta-v1'
    }
    
    model = SentenceTransformer(model_map[embedding_type])
    
    return model


def generate_embeddings_with_fallback(nodes_dict, precomputed_embeddings, 
                                       vtype_mapping, embedding_model, mode="train"):
    node_ids = list(nodes_dict.keys())
    total_nodes = len(node_ids)
    
    batch_size = 50000
    num_batches = (total_nodes + batch_size - 1) // batch_size
    
    all_embeddings = []
    missing_count = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_nodes)
        batch_node_ids = node_ids[start_idx:end_idx]
        
        batch_embeddings = []
        for node_id in tqdm(batch_node_ids, 
                           desc=f"Generating {mode} embeddings (batch {batch_idx+1}/{num_batches})",
                           leave=False):
            if node_id in precomputed_embeddings:
                embedding = precomputed_embeddings[node_id]
            else:
                vtype = list(nodes_dict[node_id].keys())[0]
                semantic_group = vtype_mapping.get(vtype, vtype)
                embedding = embedding_model.encode(semantic_group)
                missing_count += 1
            
            batch_embeddings.append(embedding)
        
        batch_array = np.vstack(batch_embeddings)
        all_embeddings.append(batch_array)
    
    embeddings_array = np.vstack(all_embeddings)
    
    return embeddings_array, node_ids, missing_count


def apply_incremental_pca(train_embeddings, test_embeddings, pca_dim=128):
    original_dim = train_embeddings.shape[1]
    ipca = IncrementalPCA(n_components=pca_dim, batch_size=10000)
    
    pca_batch_size = 10000
    num_pca_batches = (train_embeddings.shape[0] + pca_batch_size - 1) // pca_batch_size
    
    for i in tqdm(range(num_pca_batches), desc=f"Fitting PCA ({original_dim}→{pca_dim}D)", leave=False):
        start_idx = i * pca_batch_size
        end_idx = min((i + 1) * pca_batch_size, train_embeddings.shape[0])
        batch = train_embeddings[start_idx:end_idx]
        ipca.partial_fit(batch)
    
    train_result_batches = []
    for i in tqdm(range(num_pca_batches), desc="Transforming train", leave=False):
        start_idx = i * pca_batch_size
        end_idx = min((i + 1) * pca_batch_size, train_embeddings.shape[0])
        batch = train_embeddings[start_idx:end_idx]
        train_result_batches.append(ipca.transform(batch))
    
    train_pca = np.vstack(train_result_batches)
    
    if test_embeddings.shape[0] > 0:
        num_test_batches = (test_embeddings.shape[0] + pca_batch_size - 1) // pca_batch_size
        test_result_batches = []
        
        for i in tqdm(range(num_test_batches), desc="Transforming test", leave=False):
            start_idx = i * pca_batch_size
            end_idx = min((i + 1) * pca_batch_size, test_embeddings.shape[0])
            batch = test_embeddings[start_idx:end_idx]
            test_result_batches.append(ipca.transform(batch))
        
        test_pca = np.vstack(test_result_batches)
    else:
        test_pca = np.array([]).reshape(0, pca_dim)
    
    explained_var = ipca.explained_variance_ratio_.sum()
    
    return train_pca, test_pca, ipca, explained_var


def save_pca_embeddings(train_pca, train_node_ids, test_pca, test_node_ids,
                        pca_model, metadata, output_path, dataset, embedding_type, 
                        feature_type, pca_dim, llmfets_model=None):
    dataset_dir = os.path.join(output_path, dataset.lower())
    
    if dataset.lower() == "atlas" and llmfets_model:
        model_normalized = llmfets_model.lower().replace(':', '_')
        model_dir = os.path.join(dataset_dir, model_normalized)
        embedding_dir = os.path.join(model_dir, embedding_type.lower())
    else:
        embedding_dir = os.path.join(dataset_dir, embedding_type.lower())
    
    os.makedirs(embedding_dir, exist_ok=True)
    
    file_prefix = f"{feature_type}_pca{pca_dim}"
    
    train_dict = {node_id: train_pca[i] for i, node_id in enumerate(train_node_ids)}
    test_dict = {node_id: test_pca[i] for i, node_id in enumerate(test_node_ids)}
    all_dict = {**train_dict, **test_dict}
    
    with open(os.path.join(embedding_dir, f"{file_prefix}_train.pkl"), 'wb') as f:
        pickle.dump(train_dict, f)
    
    with open(os.path.join(embedding_dir, f"{file_prefix}_test.pkl"), 'wb') as f:
        pickle.dump(test_dict, f)
    
    with open(os.path.join(embedding_dir, f"{file_prefix}_all.pkl"), 'wb') as f:
        pickle.dump(all_dict, f)
    
    with open(os.path.join(embedding_dir, f"{file_prefix}_model.pkl"), 'wb') as f:
        pickle.dump(pca_model, f)
    
    with open(os.path.join(embedding_dir, f"{file_prefix}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def process_single_embedding_atlas(dataset, embedding_type, feature_type, pca_dim,
                                   embedding_path, saving_path, llmfets_model=None):
    precomputed_embeddings = load_precomputed_embeddings(
        dataset, embedding_type, feature_type, embedding_path, llmfets_model
    )
    
    if not precomputed_embeddings:
        return False
    
    node_ids = list(precomputed_embeddings.keys())
    embeddings_list = [precomputed_embeddings[node_id] for node_id in node_ids]
    embeddings_array = np.vstack(embeddings_list)
    
    original_dim = embeddings_array.shape[1]
    ipca = IncrementalPCA(n_components=pca_dim, batch_size=10000)
    
    pca_batch_size = 10000
    num_pca_batches = (embeddings_array.shape[0] + pca_batch_size - 1) // pca_batch_size
    
    for i in tqdm(range(num_pca_batches), desc=f"Fitting PCA ({original_dim}→{pca_dim}D)"):
        start_idx = i * pca_batch_size
        end_idx = min((i + 1) * pca_batch_size, embeddings_array.shape[0])
        batch = embeddings_array[start_idx:end_idx]
        ipca.partial_fit(batch)
    
    result_batches = []
    for i in tqdm(range(num_pca_batches), desc="Transforming embeddings"):
        start_idx = i * pca_batch_size
        end_idx = min((i + 1) * pca_batch_size, embeddings_array.shape[0])
        batch = embeddings_array[start_idx:end_idx]
        result_batches.append(ipca.transform(batch))
    
    pca_embeddings = np.vstack(result_batches)
    explained_variance = ipca.explained_variance_ratio_.sum()
    
    train_pca = pca_embeddings
    train_node_ids = node_ids
    test_pca = np.array([]).reshape(0, pca_dim)
    test_node_ids = []
    
    metadata = {
        "dataset": dataset,
        "embedding_type": embedding_type,
        "feature_type": feature_type,
        "pca_dim": pca_dim,
        "original_dim": int(original_dim),
        "explained_variance": float(explained_variance),
        "train_nodes": len(train_node_ids),
        "test_nodes": 0,
        "overlap_nodes": 0,
        "train_date_range": None,
        "test_date_range": None,
        "fallback_embeddings_train": 0,
        "fallback_embeddings_test": 0,
        "total_csv_nodes": len(node_ids),
        "total_csv_edges": 0,
        "note": "ATLAS: PCA applied directly to all embeddings without CSV processing"
    }
    
    save_pca_embeddings(
        train_pca, train_node_ids, test_pca, test_node_ids,
        ipca, metadata, saving_path, dataset, embedding_type, 
        feature_type, pca_dim, llmfets_model
    )
    
    return True


def process_single_embedding(dataset, embedding_type, feature_type, pca_dim, 
                              train_start_date, train_end_date, test_start_date, test_end_date,
                              embedding_path, saving_path, extracted_graph_path):
    vtype_combinations, id_labels, vtype_mapping, edge_validation = load_dependencies(
        dataset, extracted_graph_path
    )
    
    precomputed_embeddings = load_precomputed_embeddings(
        dataset, embedding_type, feature_type, embedding_path, None
    )
    
    embedding_model = initialize_text_embedder(embedding_type)
    
    csv_dataset_path = f'{extracted_graph_path}/{dataset.upper()}/*/'
    start_date_for_processing = train_start_date
    end_date_for_processing = test_end_date
    
    nodes_dict, edges_list = process_csv_nodes_and_edges(
        csv_dataset_path, id_labels, start_date_for_processing, end_date_for_processing,
        vtype_combinations, vtype_mapping, edge_validation
    )
    
    train_nodes_dict, test_nodes_dict, overlap_count = split_train_test_nodes(
        nodes_dict, edges_list, train_start_date, train_end_date, 
        test_start_date, test_end_date
    )
    
    train_embeddings, train_node_ids, train_missing = generate_embeddings_with_fallback(
        train_nodes_dict, precomputed_embeddings, vtype_mapping, embedding_model, mode="train"
    )
    
    test_embeddings, test_node_ids, test_missing = generate_embeddings_with_fallback(
        test_nodes_dict, precomputed_embeddings, vtype_mapping, embedding_model, mode="test"
    )
    
    train_pca, test_pca, pca_model, explained_variance = apply_incremental_pca(
        train_embeddings, test_embeddings, pca_dim
    )
    
    metadata = {
        "dataset": dataset,
        "embedding_type": embedding_type,
        "feature_type": feature_type,
        "pca_dim": pca_dim,
        "original_dim": int(train_embeddings.shape[1]),
        "explained_variance": float(explained_variance),
        "train_nodes": len(train_node_ids),
        "test_nodes": len(test_node_ids),
        "overlap_nodes": overlap_count,
        "train_date_range": [train_start_date, train_end_date],
        "test_date_range": [test_start_date, test_end_date],
        "fallback_embeddings_train": train_missing,
        "fallback_embeddings_test": test_missing,
        "total_csv_nodes": len(nodes_dict),
        "total_csv_edges": len(edges_list)
    }
    
    save_pca_embeddings(
        train_pca, train_node_ids, test_pca, test_node_ids,
        pca_model, metadata, saving_path, dataset, embedding_type, 
        feature_type, pca_dim, None
    )
    
    return True


def main():
    args = parse_args()
    
    if args.embedding == 'all':
        embedding_types = ['roberta', 'mpnet', 'minilm', 'distilbert']
    else:
        embedding_types = [args.embedding]
    
    if args.feature_type == 'all':
        feature_types = ['type', 'functionality']
    else:
        feature_types = [args.feature_type]
    
    total_count = len(embedding_types) * len(feature_types)
    
    with tqdm(total=total_count, desc="Processing combinations", position=0) as pbar:
        for embedding_type in embedding_types:
            for feature_type in feature_types:
                pbar.set_description(f"Processing {embedding_type}/{feature_type}")
                
                if args.dataset.lower() == "atlas":
                    process_single_embedding_atlas(
                        args.dataset, embedding_type, feature_type, args.pca_dim,
                        args.embedding_path, args.saving_path, args.llmfets_model
                    )
                else:
                    process_single_embedding(
                        args.dataset, embedding_type, feature_type, args.pca_dim,
                        args.train_start_date, args.train_end_date,
                        args.test_start_date, args.test_end_date,
                        args.embedding_path, args.saving_path, args.extracted_graph_path
                    )
                
                pbar.update(1)


if __name__ == "__main__":
    main()

