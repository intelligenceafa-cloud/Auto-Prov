#!/usr/bin/env python3

import os
import re
import argparse
import json
import pickle
from glob import glob
from typing import List, Dict, Set, Tuple
from tqdm import tqdm
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from ATLAS continual clustering results")
    parser.add_argument("--embedding", type=str, required=True, help="Embedding type (e.g., mptnet, all-miniLM)")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to take from each cluster")
    parser.add_argument("--continual_clusters_dir", type=str, default="./continual-clusters-atlas", 
                       help="Directory containing continual clustering results")
    parser.add_argument("--original_data_dir", type=str, default="../BIGDATA/ATLAS/", 
                       help="Directory containing original ATLAS log data")
    parser.add_argument("--candidates_dir", type=str, default="./candidates-atlas", 
                       help="Directory to save candidate samples")
    parser.add_argument("--log_type", type=str, choices=["audit", "dns", "firefox"],
                       help="Process only this log type. If not specified, processes all log types.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


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


def find_continual_cluster_files(continual_clusters_dir: str, embedding: str) -> List[Tuple[str, str, str, str]]:
    search_dir = os.path.join(continual_clusters_dir, embedding)
    
    if not os.path.exists(search_dir):
        return []
    
    files = []
    
    for log_type in ['audit', 'dns', 'firefox']:
        log_type_dir = os.path.join(search_dir, log_type)
        if not os.path.exists(log_type_dir):
            continue
        
        pattern = os.path.join(log_type_dir, "continual_clusters_*.json")
        
        for file_path in glob(pattern):
            filename = os.path.basename(file_path)
            
            clean_name = filename.replace('continual_clusters_', '').replace('.json', '')
            
            for method in ['_dbstream', '_denstream']:
                if clean_name.endswith(method):
                    clean_name = clean_name[:-len(method)]
                    break
            
            parts = clean_name.split('_')
            
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


def sample_from_clusters(cluster_items: List[str], n_samples: int, cluster_id: str) -> List[str]:
    if len(cluster_items) <= n_samples:
        return cluster_items
    else:
        return random.sample(cluster_items, n_samples)


def process_timestamps_and_sample(continual_clusters_dir: str, embedding: str, n_samples: int, log_type_filter: str = None) -> Dict[str, List[str]]:
    continual_cluster_files = find_continual_cluster_files(continual_clusters_dir, embedding)
    
    if not continual_cluster_files:
        return {}
    
    files_by_type: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for scenario, log_type, timestamp, file_path in continual_cluster_files:
        if log_type_filter and log_type != log_type_filter:
            continue
        if log_type not in files_by_type:
            files_by_type[log_type] = []
        files_by_type[log_type].append((scenario, log_type, timestamp, file_path))
    
    candidates_by_type: Dict[str, List[str]] = {}
    
    for log_type, log_type_files in sorted(files_by_type.items()):
        existing_clusters: Set[str] = set()
        log_type_candidates: List[str] = []
        
        for i, (scenario, log_type_name, timestamp, file_path) in enumerate(log_type_files):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            clusters = data.get('clusters', {})
            
            if i == 0:
                for cluster_id, cluster_items in clusters.items():
                    if cluster_id == "-1":
                        continue
                    
                    samples = sample_from_clusters(cluster_items, n_samples, cluster_id)
                    log_type_candidates.extend(samples)
                    existing_clusters.add(cluster_id)
            else:
                new_clusters = set(clusters.keys()) - existing_clusters
                if new_clusters:
                    for cluster_id in new_clusters:
                        if cluster_id == "-1":
                            continue
                        
                        cluster_items = clusters[cluster_id]
                        samples = sample_from_clusters(cluster_items, n_samples, cluster_id)
                        log_type_candidates.extend(samples)
                        existing_clusters.add(cluster_id)
            
            existing_clusters.update(clusters.keys())
        
        candidates_by_type[log_type] = log_type_candidates
    
    return candidates_by_type


def load_original_log_data(candidate_ids: List[str], original_data_dir: str) -> List[str]:
    original_logs = []
    failed_loads = 0
    
    for i, candidate_id in enumerate(tqdm(candidate_ids, desc="Loading log data")):
        try:
            scenario, log_type, timestamp, index = parse_seed_id(candidate_id)
            
            if not scenario or not log_type or not timestamp:
                failed_loads += 1
                continue
            
            if '_' in scenario and scenario.split('_')[0].startswith('M'):
                scenario_parts = scenario.split('_')
                base_scenario = scenario_parts[0]
                host = scenario_parts[1]
                log_path = os.path.join(original_data_dir, base_scenario, host, log_type, timestamp, f"{log_type}.pkl")
            else:
                log_path = os.path.join(original_data_dir, scenario, log_type, timestamp, f"{log_type}.pkl")
            
            if not os.path.exists(log_path):
                failed_loads += 1
                continue
            
            with open(log_path, 'rb') as f:
                log_data = pickle.load(f)
            
            if isinstance(log_data, list) and 0 <= index < len(log_data):
                original_logs.append(log_data[index])
            else:
                failed_loads += 1
                
        except Exception as e:
            failed_loads += 1
    
    return original_logs


def save_candidates_by_type(candidates_dir: str, embedding: str, 
                            candidates_by_type: Dict[str, List[str]], 
                            original_logs_by_type: Dict[str, List[str]]):
    for log_type in sorted(candidates_by_type.keys()):
        output_dir = os.path.join(candidates_dir, embedding, log_type)
        os.makedirs(output_dir, exist_ok=True)
        
        candidate_ids = candidates_by_type[log_type]
        original_logs = original_logs_by_type[log_type]
        
        candidate_ids_path = os.path.join(output_dir, "candidate_ids.pkl")
        with open(candidate_ids_path, 'wb') as f:
            pickle.dump(candidate_ids, f)
        
        candidate_logs_path = os.path.join(output_dir, "candidate.pkl")
        with open(candidate_logs_path, 'wb') as f:
            pickle.dump(original_logs, f)


def main():
    args = parse_args()
    
    random.seed(args.seed)
    
    candidates_by_type = process_timestamps_and_sample(
        args.continual_clusters_dir, 
        args.embedding, 
        args.n_samples,
        args.log_type
    )
    
    if not candidates_by_type:
        return
    
    original_logs_by_type: Dict[str, List[str]] = {}
    
    for log_type, candidate_ids in sorted(candidates_by_type.items()):
        original_logs = load_original_log_data(candidate_ids, args.original_data_dir)
        original_logs_by_type[log_type] = original_logs
    
    save_candidates_by_type(args.candidates_dir, args.embedding, candidates_by_type, original_logs_by_type)


if __name__ == "__main__":
    main()
