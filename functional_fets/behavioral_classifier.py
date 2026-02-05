#!/usr/bin/env python3

import argparse
import json
import numpy as np
import os
import re
import sys
import pickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from load_sparse_profiles import load_typed_dataset, load_untyped_dataset
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_memory_usage():
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_gb = process.memory_info().rss / 1024**3
        return f"{mem_gb:.2f} GB"
    return "N/A"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify untyped nodes using 1-NN on behavioral profiles"
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
        "--n-components",
        type=int,
        default=128,
        help="Number of PCA components (default: 128)"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use causal profiles (load from causal/ directory)"
    )
    
    parser.add_argument(
        "--timeoh",
        action="store_true",
        help="Use per-timestamp one-hop profiles (load from timeoh/ directory)"
    )
    
    parser.add_argument(
        "--embedding",
        type=str,
        default="roberta",
        choices=["mpnet", "minilm", "roberta", "distilbert"],
        help="Embedding type for ename similarity (default: roberta)"
    )
    
    args = parser.parse_args()
    
    if args.dataset.lower() == "atlas" and not args.llmfets_model:
        parser.error("--llmfets-model is required for ATLAS dataset")
    
    return args


def _remove_numerics(text: str, special_chars: str = '-_') -> str:
    if not text:
        return ''
    if text.isdigit():
        return ''
    text = re.sub(r'\d+$', '', text)
    escaped_chars = re.escape(special_chars)
    text = re.sub(f'[{escaped_chars}]\\d+', '', text)
    text = re.sub(r'\d+', '', text)
    return text if text else ''


def _normalize_path(path: str) -> str:
    normalized = ""
    i = 0
    while i < len(path):
        if path[i] == '\\':
            normalized += '/'
            while i+1 < len(path) and path[i+1] == '\\':
                i += 1
        else:
            normalized += path[i]
        i += 1
    while '//' in normalized:
        normalized = normalized.replace('//', '/')
    return normalized


def _is_file(path: str) -> bool:
    norm_path = _normalize_path(path)
    _, ext = os.path.splitext(norm_path)
    return bool(ext)


def _process_filename_and_extension(filename: str) -> str:
    if '.' in filename:
        name, ext = filename.split('.', 1)
        processed_name = _remove_numerics(name)
        return f"{processed_name}.{ext}"
    else:
        return _remove_numerics(filename)


def normalize_ename(ename: str) -> str:
    if not ename:
        return ename
    
    normalized = re.sub(r'\\+', r'\\', ename)
    normalized = normalized.lower()
    normalized = _normalize_path(normalized)
    
    if _is_file(normalized):
        dirname = os.path.dirname(normalized)
        basename = os.path.basename(normalized)
        
        if dirname:
            dir_segments = dirname.split('/')
            processed_dir_segments = []
            for segment in dir_segments:
                if segment:
                    cleaned = _remove_numerics(segment)
                    if cleaned:
                        processed_dir_segments.append(cleaned)
            processed_dir = '/'.join(processed_dir_segments)
        else:
            processed_dir = ''
        
        processed_filename = _process_filename_and_extension(basename)
        
        if processed_dir:
            processed_path = f"{processed_dir}/{processed_filename}"
        else:
            processed_path = processed_filename
        
        return processed_path
    else:
        segments = normalized.split('/')
        processed_segments = []
        for segment in segments:
            if segment:
                cleaned = _remove_numerics(segment)
                if cleaned:
                    processed_segments.append(cleaned)
        processed_path = '/'.join(processed_segments)
        return processed_path


def break_tie_with_ename_similarity(
    query_enames,
    candidate_indices,
    labels_typed,
    node_ids_typed,
    typed_nodes_enames,
    fets_dict,
    ename_embeddings_cache
):
    if len(candidate_indices) == 1:
        return candidate_indices[0], []
    
    ename_pool = []
    ename_to_node_idx = {}
    
    for idx in candidate_indices:
        node_id = node_ids_typed[idx]
        node_enames = typed_nodes_enames.get(node_id, [])
        
        for ename in node_enames:
            ename_pool.append(ename)
            ename_to_node_idx[ename] = idx
    
    if not ename_pool:
        return candidate_indices[0], []
    
    query_results = []
    
    for query_ename in query_enames:
        if query_ename not in ename_embeddings_cache:
            continue
        
        query_embedding = ename_embeddings_cache[query_ename]
        
        similarities = []
        for pool_ename in ename_pool:
            if pool_ename not in ename_embeddings_cache:
                continue
            
            pool_embedding = ename_embeddings_cache[pool_ename]
            
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            pool_norm = pool_embedding / (np.linalg.norm(pool_embedding) + 1e-8)
            sim = np.dot(query_norm, pool_norm)
            
            similarities.append((pool_ename, sim))
        
        if not similarities:
            continue
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        for pool_ename, sim in similarities:
            normalized_ename = normalize_ename(pool_ename)
            if normalized_ename in fets_dict:
                ename_type = fets_dict[normalized_ename].get("Type", "")
                if ename_type and ename_type.upper() not in ["NO LABEL", "NO_LABEL"]:
                    query_results.append({
                        'query_ename': query_ename,
                        'closest_ename': pool_ename,
                        'similarity': float(sim),
                        'type': ename_type.lower(),
                        'candidate_idx': int(ename_to_node_idx[pool_ename])
                    })
                    break
    
    if not query_results:
        return candidate_indices[0], []
    
    best_result = max(query_results, key=lambda x: x['similarity'])
    
    return best_result['candidate_idx'], query_results


def main():
    args = parse_args()
    
    is_atlas = args.dataset.lower() == "atlas"
    if is_atlas and args.llmfets_model:
        model_normalized = args.llmfets_model.lower().replace(':', '_')
    else:
        model_normalized = None
    
    if args.causal:
        profile_type = "CAUSAL"
    elif args.timeoh:
        profile_type = "PER-TIMESTAMP ONE-HOP"
    else:
        profile_type = "AGGREGATED ONE-HOP"
    
    X_typed, node_ids_typed, labels_typed, pattern_cols = load_typed_dataset(
        args.dataset, 
        llmfets_model=args.llmfets_model,
        causal=args.causal, 
        timeoh=args.timeoh
    )
    X_untyped, node_ids_untyped, pattern_cols_untyped = load_untyped_dataset(
        args.dataset,
        llmfets_model=args.llmfets_model,
        causal=args.causal,
        timeoh=args.timeoh
    )
    
    if X_untyped.shape[0] == 0:
        return
    if not np.array_equal(pattern_cols, pattern_cols_untyped):
        return
    
    labels_typed = np.array(labels_typed)
    unique_labels = np.unique(labels_typed)
    
    sample_indices = []
    
    for label in tqdm(unique_labels, desc="Stratified sampling", ncols=100, position=0):
        label_indices = np.where(labels_typed == label)[0]
        n_samples_for_label = min(1000, len(label_indices))
        sampled = np.random.RandomState(args.random_state + hash(label) % 10000).choice(
            label_indices, n_samples_for_label, replace=False)
        sample_indices.extend(sampled)
    
    sample_indices = np.array(sample_indices)
    X_sample = X_typed[sample_indices]
    
    svd = TruncatedSVD(
        n_components=min(args.n_components, X_typed.shape[1] - 1),
        n_iter=3,
        random_state=args.random_state
    )
    
    pbar_svd = tqdm(total=3, desc="Applying PCA/SVD", ncols=100, position=0)
    
    pbar_svd.set_postfix_str(f"Fitting on {len(sample_indices):,} samples")
    svd.fit(X_sample)
    pbar_svd.update(1)
    
    pbar_svd.set_postfix_str(f"Transforming typed ({X_typed.shape[0]:,} nodes)")
    X_typed_pca = svd.transform(X_typed)
    pbar_svd.update(1)
    
    pbar_svd.set_postfix_str(f"Transforming untyped ({X_untyped.shape[0]:,} nodes)")
    X_untyped_pca = svd.transform(X_untyped)
    pbar_svd.update(1)
    
    explained_var = svd.explained_variance_ratio_.sum()
    pbar_svd.set_postfix_str(f"✓ Variance: {explained_var:.1%}")
    pbar_svd.close()
    
    X_typed_norm = normalize(X_typed_pca, norm='l2', axis=1)
    X_untyped_norm = normalize(X_untyped_pca, norm='l2', axis=1)
    if is_atlas and model_normalized:
        if args.causal:
            base_dir_temp = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized, "causal")
        elif args.timeoh:
            base_dir_temp = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized, "timeoh")
        else:
            base_dir_temp = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized)
    else:
        if args.causal:
            base_dir_temp = os.path.join(_SCRIPT_DIR, "behavioral-profiles", "causal")
        elif args.timeoh:
            base_dir_temp = os.path.join(_SCRIPT_DIR, "behavioral-profiles", "timeoh")
        else:
            base_dir_temp = os.path.join(_SCRIPT_DIR, "behavioral-profiles")
    
    typed_enames_file = os.path.join(base_dir_temp, f'typed_nodes_enames_{args.dataset}.json')
    with open(typed_enames_file, 'r') as f:
        typed_nodes_enames = json.load(f)
    untyped_enames_file = os.path.join(base_dir_temp, f'untyped_nodes_enames_{args.dataset}.json')
    with open(untyped_enames_file, 'r') as f:
        untyped_nodes_enames = json.load(f)
    if is_atlas and args.llmfets_model:
        fets_path = os.path.join(_SCRIPT_DIR, "llm-fets", args.llmfets_model, f'ename_fets_{args.dataset}.json')
    else:
        fets_path = os.path.join(_SCRIPT_DIR, "llm-fets", f'ename_fets_{args.dataset}.json')
    
    try:
        with open(fets_path, 'r', encoding='utf-8') as f:
            content = f.read()
            content = content.replace(',}', '}').replace(',]', ']')
            fets_dict = json.loads(content)
    except:
        fets_dict = {}
    if is_atlas and model_normalized:
        cache_file = os.path.join(_SCRIPT_DIR, "ename-embeddings", model_normalized, f'enameemb_{args.dataset}_{args.embedding}.pkl')
    else:
        cache_file = os.path.join(_SCRIPT_DIR, "ename-embeddings", f'enameemb_{args.dataset}_{args.embedding}.pkl')
    
    if not os.path.exists(cache_file):
        sys.exit(1)
    with open(cache_file, 'rb') as f:
        ename_embeddings_cache = pickle.load(f)
    predictions = {}
    tie_breaks = 0
    ename_match_details = {}
    
    for i in tqdm(range(X_untyped_norm.shape[0]), desc="Classifying", ncols=100, position=0):
        untyped_id = node_ids_untyped[i]
        
        similarities = X_untyped_norm[i] @ X_typed_norm.T
        
        if hasattr(similarities, 'toarray'):
            similarities = similarities.toarray().flatten()
        
        max_sim = similarities.max()
        candidates = np.where(np.isclose(similarities, max_sim, rtol=1e-9))[0]
        
        query_results = []
        if len(candidates) > 1 and max_sim >= 0.99:
            query_enames = untyped_nodes_enames.get(untyped_id, [])
            
            if query_enames:
                best_idx, query_results = break_tie_with_ename_similarity(
                    query_enames,
                    candidates,
                    labels_typed,
                    node_ids_typed,
                    typed_nodes_enames,
                    fets_dict,
                    ename_embeddings_cache
                )
                tie_breaks += 1
            else:
                best_idx = candidates[0]
        else:
            best_idx = candidates[0]
        
        predicted_type = labels_typed[best_idx]
        similarity_score = float(similarities[best_idx])
        nearest_neighbor_id = node_ids_typed[best_idx]
        
        predictions[untyped_id] = {
            'predicted_type': predicted_type,
            'similarity': similarity_score,
            'nearest_neighbor': nearest_neighbor_id,
            'n_candidates': len(candidates)
        }
        
        if query_results:
            ename_match_details[untyped_id] = query_results
    
    if is_atlas and model_normalized:
        if args.causal:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized, "causal")
        elif args.timeoh:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized, "timeoh")
        else:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized)
    else:
        if args.causal:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", "causal")
        elif args.timeoh:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", "timeoh")
        else:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles")
    
    typed_func_file = os.path.join(base_dir, f'typed_nodes_functionality_{args.dataset}.json')
    with open(typed_func_file, 'r') as f:
        typed_functionality = json.load(f)
    untype2type = {}
    untype2type_functionality = {}
    untype2type_meta = {}
    
    for node_id in tqdm(predictions.keys(), desc="Building outputs", ncols=100, position=0):
        nearest_neighbor_id = predictions[node_id]['nearest_neighbor']
        predicted_type = predictions[node_id]['predicted_type']
        
        functionality = typed_functionality.get(nearest_neighbor_id, "")
        neighbor_enames = typed_nodes_enames.get(nearest_neighbor_id, [])
        
        untype2type[node_id] = predicted_type
        untype2type_functionality[node_id] = functionality
        
        meta_entry = {
            'predicted_type': predicted_type,
            'similarity': predictions[node_id]['similarity'],
            'n_candidates': predictions[node_id]['n_candidates'],
            'nearest_neighbor_id': nearest_neighbor_id,
            'nearest_neighbor_enames': neighbor_enames,
            'query_node_enames': untyped_nodes_enames.get(node_id, []),
            'functionality': functionality,
            'inherited_from': nearest_neighbor_id
        }
        
        if node_id in ename_match_details:
            meta_entry['ename_matches'] = ename_match_details[node_id]
        
        untype2type_meta[node_id] = meta_entry
    
    os.makedirs(base_dir, exist_ok=True)
    
    untype2type_file = os.path.join(base_dir, f'untype2type_nodes_{args.dataset}.json')
    with open(untype2type_file, 'w') as f:
        json.dump(untype2type, f, indent=2)
    
    untype2type_func_file = os.path.join(base_dir, f'untype2type_nodes_functionality_{args.dataset}.json')
    with open(untype2type_func_file, 'w') as f:
        json.dump(untype2type_functionality, f, indent=2)
    
    untype2type_meta_file = os.path.join(base_dir, f'untype2type_meta_{args.dataset}.json')
    with open(untype2type_meta_file, 'w') as f:
        json.dump(untype2type_meta, f, indent=2)


if __name__ == "__main__":
    main()

