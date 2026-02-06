#!/usr/bin/env python3

import os
import re
import argparse
import json
import pickle
import numpy as np
import random
from datetime import datetime
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
import glob

from river import cluster


def parse_args():
    parser = argparse.ArgumentParser(description="Continual clustering of ATLAS micro-cluster data")
    parser.add_argument("--embedding", type=str, required=True, help="Embedding type (e.g., mptnet, all-miniLM)")
    parser.add_argument("--method", type=str, default="dbstream", 
                       choices=["dbstream", "denstream"], help="Continual clustering method")
    parser.add_argument("--base_dir", type=str, default="./sample-micro-cluster-atlas", 
                       help="Base directory containing ATLAS micro-cluster data")
    parser.add_argument("--embeddings_dir", type=str, default="../BIGDATA/ATLAS_embeddings/", 
                       help="Base directory containing ATLAS embeddings")
    parser.add_argument("--output_dir", type=str, default="./continual-clusters-atlas", 
                       help="Output directory for continual clustering results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--clustering_threshold", type=float, default=0.6, 
                       help="Clustering threshold for DBSTREAM")
    parser.add_argument("--fading_factor", type=float, default=0.005, 
                       help="Fading factor for DBSTREAM")
    parser.add_argument("--intersection_factor", type=float, default=0.3, 
                       help="Intersection factor for DBSTREAM")
    parser.add_argument("--decaying_factor", type=float, default=0.05, 
                       help="Decaying factor for DenStream")
    parser.add_argument("--epsilon", type=float, default=0.5, 
                       help="Epsilon for DenStream")
    parser.add_argument("--beta", type=float, default=0.5, 
                       help="Beta for DenStream")
    parser.add_argument("--mu", type=float, default=3.0, 
                       help="Mu for DenStream")
    parser.add_argument("--log_type", type=str, choices=["audit", "dns", "firefox"],
                       help="Process only this log type. If not specified, processes all log types.")
    return parser.parse_args()


def parse_seed_filename(filename: str) -> Tuple[str, str, str]:
    pattern = r"seeds_(.+?)_k\d+\.pkl"
    match = re.search(pattern, filename)
    if not match:
        return "", "", ""
    
    core = match.group(1)
    
    parts = core.split('_')
    for i, part in enumerate(parts):
        if part in ['audit', 'dns', 'firefox']:
            scenario = '_'.join(parts[:i])
            log_type = part
            timestamp = '_'.join(parts[i+1:])
            return scenario, log_type, timestamp
    
    return "", "", ""


def find_atlas_seed_files(base_dir: str, embedding: str) -> List[Tuple[str, str, str, str]]:
    search_dir = os.path.join(base_dir, embedding)
    
    if not os.path.exists(search_dir):
        return []
    
    files = []
    
    for log_type in ['audit', 'dns', 'firefox']:
        log_type_dir = os.path.join(search_dir, log_type)
        if not os.path.exists(log_type_dir):
            continue
        
        pattern = os.path.join(log_type_dir, "seeds_*.pkl")
        
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            pattern_match = re.search(r"seeds_(.+?)_k\d+\.pkl", filename)
            if not pattern_match:
                continue
            
            core = pattern_match.group(1)
            
            parts = core.split('_')
            
            timestamp_start_idx = None
            for i, part in enumerate(parts):
                if re.match(r'\d{4}-\d{2}-\d{2}', part):
                    timestamp_start_idx = i
                    break
            
            if timestamp_start_idx is not None:
                scenario = '_'.join(parts[:timestamp_start_idx])
                timestamp = '_'.join(parts[timestamp_start_idx:])
                files.append((scenario, log_type, timestamp, file_path))
    
    files.sort(key=lambda x: x[2])
    return files


def initialize_clustering_model(method: str, **kwargs):
    if method == "dbstream":
        return cluster.DBSTREAM(
            clustering_threshold=kwargs.get('clustering_threshold', 0.6),
            fading_factor=kwargs.get('fading_factor', 0.005),
            intersection_factor=kwargs.get('intersection_factor', 0.3)
        )
    elif method == "denstream":
        return cluster.DenStream(
            decaying_factor=kwargs.get('decaying_factor', 0.05),
            epsilon=kwargs.get('epsilon', 0.5),
            beta=kwargs.get('beta', 0.5),
            mu=kwargs.get('mu', 3.0)
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def load_seed_data(file_path: str) -> List[str]:
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            return data
        else:
            return [str(data)]
    except Exception as e:
        return []


def parse_seed_id(seed_id: str) -> Tuple[str, str, str, int]:
    try:
        last_underscore = seed_id.rfind('_')
        if last_underscore == -1:
            return "", "", "", 0
        
        index = int(seed_id[last_underscore + 1:])
        remainder = seed_id[:last_underscore]
        
        parts = remainder.split('_')
        for i, part in enumerate(parts):
            if part in ['audit', 'dns', 'firefox']:
                scenario = '_'.join(parts[:i])
                log_type = part
                timestamp = '_'.join(parts[i+1:])
                return scenario, log_type, timestamp, index
        
        return "", "", "", 0
    except ValueError:
        return "", "", "", 0


def load_embedding_from_file(embeddings_dir: str, embedding_type: str, scenario: str, log_type: str, timestamp: str, index: int) -> np.ndarray:
    try:
        if '_' in scenario and scenario.split('_')[0].startswith('M'):
            scenario_parts = scenario.split('_')
            base_scenario = scenario_parts[0]
            host = scenario_parts[1]
            embeddings_path = os.path.join(embeddings_dir, base_scenario, host, log_type, timestamp, "embeddings.pkl")
        else:
            embeddings_path = os.path.join(embeddings_dir, scenario, log_type, timestamp, "embeddings.pkl")
        
        if not os.path.exists(embeddings_path):
            return None
        
        with open(embeddings_path, 'rb') as f:
            embeddings_list = pickle.load(f)
        
        if isinstance(embeddings_list, list) and 0 <= index < len(embeddings_list):
            embedding = np.array(embeddings_list[index], dtype=np.float32)
            return embedding
        else:
            return None
            
    except Exception as e:
        return None


def compute_embeddings_from_seeds(seed_data: List[str], embeddings_dir: str, 
                                  embedding_type: str) -> np.ndarray:
    embeddings = []
    failed_loads = 0
    embedding_dim = None
    
    for seed_id in tqdm(seed_data, desc="Loading embeddings", leave=False):
        scenario, log_type, timestamp, index = parse_seed_id(seed_id)
        
        if not scenario or not log_type or not timestamp:
            failed_loads += 1
            continue
        
        embedding_vector = load_embedding_from_file(embeddings_dir, embedding_type, scenario, log_type, timestamp, index)
        
        if embedding_vector is not None:
            if embedding_dim is None:
                embedding_dim = embedding_vector.shape[0]
            
            if embedding_vector.shape[0] != embedding_dim:
                continue
            
            embeddings.append(embedding_vector)
        else:
            failed_loads += 1
    
    if not embeddings:
        return np.array([])
    
    embeddings_array = np.array(embeddings)
    
    return embeddings_array


def prepare_embedding_for_river(embedding):
    return {f'dim_{j}': float(val) for j, val in enumerate(embedding)}


def perform_initial_clustering(embeddings: np.ndarray, clustering_model) -> Dict[int, List[int]]:
    clusters: Dict[int, List[int]] = {}
    
    for i, embedding in enumerate(tqdm(embeddings, desc="Initial clustering", leave=False)):
        x = prepare_embedding_for_river(embedding)
        
        clustering_model.learn_one(x)
        
        pred = clustering_model.predict_one(x)
        label = pred if pred is not None else -1
        
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return clusters


def perform_continual_clustering(embeddings: np.ndarray, clustering_model, 
                               existing_clusters: Set[int]) -> Dict[int, List[int]]:
    clusters: Dict[int, List[int]] = {}
    new_clusters: Set[int] = set()
    
    for i, embedding in enumerate(tqdm(embeddings, desc="Continual clustering", leave=False)):
        x = prepare_embedding_for_river(embedding)
        
        pred = clustering_model.predict_one(x)
        
        clustering_model.learn_one(x)
        
        label = pred if pred is not None else -1
        
        if label not in existing_clusters and label != -1:
            new_clusters.add(label)
            existing_clusters.add(label)
        
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return clusters


def save_clustering_results(output_path: str, clusters: Dict[int, List[int]], 
                          seed_data: List[str], scenario: str, log_type: str, timestamp: str):
    results = {
        "metadata": {
            "scenario": scenario,
            "log_type": log_type,
            "timestamp": timestamp,
            "total_points": len(seed_data),
            "num_clusters": len(clusters),
            "timestamp_saved": datetime.now().isoformat()
        },
        "clusters": {}
    }
    
    for cluster_id, indices in clusters.items():
        cluster_data = [seed_data[i] for i in indices]
        results["clusters"][str(cluster_id)] = cluster_data
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def main():
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    seed_files = find_atlas_seed_files(args.base_dir, args.embedding)
    
    if not seed_files:
        return
    
    seed_files_by_type: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for scenario, log_type, timestamp, file_path in seed_files:
        if args.log_type and log_type != args.log_type:
            continue
        if log_type not in seed_files_by_type:
            seed_files_by_type[log_type] = []
        seed_files_by_type[log_type].append((scenario, log_type, timestamp, file_path))
    
    for log_type_idx, (log_type, log_type_files) in enumerate(sorted(seed_files_by_type.items())):
        clustering_model = initialize_clustering_model(
            args.method,
            clustering_threshold=args.clustering_threshold,
            fading_factor=args.fading_factor,
            decaying_factor=args.decaying_factor,
            intersection_factor=args.intersection_factor,
            epsilon=args.epsilon,
            beta=args.beta,
            mu=args.mu
        )
        
        existing_clusters: Set[int] = set()
        
        for i, (scenario, log_type, timestamp, file_path) in enumerate(tqdm(log_type_files, desc=f"Processing {log_type}", position=0)):
            seed_data = load_seed_data(file_path)
            if not seed_data:
                continue
            
            embeddings = compute_embeddings_from_seeds(seed_data, args.embeddings_dir, args.embedding)
            if embeddings.size == 0:
                continue
                
            if i == 0:
                clusters = perform_initial_clustering(embeddings, clustering_model)
                new_cluster_ids = set(clusters.keys())
                num_new_clusters = len(new_cluster_ids)
            else:
                clusters = perform_continual_clustering(embeddings, clustering_model, existing_clusters)
                new_cluster_ids = clusters.keys() - existing_clusters
                num_new_clusters = len(new_cluster_ids)
            
            existing_clusters.update(clusters.keys())
            
            output_dir = os.path.join(args.output_dir, args.embedding, log_type)
            os.makedirs(output_dir, exist_ok=True)
            
            output_filename = f"continual_clusters_{scenario}_{timestamp}_{args.method}.json"
            output_path = os.path.join(output_dir, output_filename)
            
            save_clustering_results(output_path, clusters, seed_data, scenario, log_type, timestamp)


if __name__ == "__main__":
    main()
