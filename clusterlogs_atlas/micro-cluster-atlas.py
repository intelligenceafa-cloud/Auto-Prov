#!/usr/bin/env python3

import os
import re
import argparse
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Find k most dissimilar seeds per hour from ATLAS embeddings")
    parser.add_argument("--embedding", type=str, required=True, help="Embedding type (e.g., mptnet, all-miniLM)")
    parser.add_argument("--k", type=int, required=True, help="Number of most dissimilar seeds to select")
    parser.add_argument("--embeddings_dir", type=str, default="../BIGDATA/ATLAS_embeddings/", 
                       help="Base directory containing ATLAS embeddings")
    parser.add_argument("--original_data_dir", type=str, default="../BIGDATA/ATLAS/", 
                       help="Base directory containing original ATLAS log data")
    parser.add_argument("--output_dir", type=str, default="./sample-micro-cluster-atlas", 
                       help="Output directory for seeds")
    parser.add_argument("--fps_subset", type=int, default=0, 
                       help="If >0, sample this many points for FPS seed selection for speed")
    parser.add_argument("--test", action="store_true", 
                       help="Test mode: process only 1 timestamp, don't save results")
    parser.add_argument("--scenario", type=str, 
                       help="Specific scenario to process (e.g., S1, M1). If not specified, processes all scenarios.")
    parser.add_argument("--log_type", type=str, choices=["audit", "dns", "firefox"],
                       help="Process only this log type. If not specified, processes all log types.")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing seed files instead of skipping them")
    parser.add_argument("--verbose", action="store_true", 
                       help="Display detailed log samples for selected seeds")
    return parser.parse_args()


def parse_timestamp_folder(folder_name: str) -> Tuple[str, str, str]:
    pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d{2}):\d{2}:\d{2}'
    match = re.search(pattern, folder_name)
    if match:
        date = match.group(1)
        hour = match.group(2)
        return date, hour, folder_name
    return "", "", ""


def find_atlas_embedding_folders(embeddings_dir: str, scenario_filter: str = None, log_type_filter: str = None) -> List[Tuple[str, str, str, str]]:
    folders = []
    
    if not os.path.exists(embeddings_dir):
        return folders
    
    for scenario_name in sorted(os.listdir(embeddings_dir)):
        scenario_path = os.path.join(embeddings_dir, scenario_name)
        
        if not os.path.isdir(scenario_path) or scenario_name.startswith('.'):
            continue
        
        if scenario_filter and scenario_name.upper() != scenario_filter.upper():
            continue
        
        if scenario_name.startswith('S'):
            log_types = [log_type_filter] if log_type_filter else ['audit', 'dns', 'firefox']
            for log_type in log_types:
                log_type_path = os.path.join(scenario_path, log_type)
                if not os.path.exists(log_type_path):
                    continue
                
                for timestamp_folder in sorted(os.listdir(log_type_path)):
                    timestamp_path = os.path.join(log_type_path, timestamp_folder)
                    if not os.path.isdir(timestamp_path):
                        continue
                    
                    embeddings_file = os.path.join(timestamp_path, "embeddings.pkl")
                    if os.path.exists(embeddings_file):
                        folders.append((scenario_name, log_type, timestamp_folder, embeddings_file))
        
        elif scenario_name.startswith('M'):
            for host_name in sorted(os.listdir(scenario_path)):
                host_path = os.path.join(scenario_path, host_name)
                if not os.path.isdir(host_path) or not host_name.startswith('h'):
                    continue
                
                log_types = [log_type_filter] if log_type_filter else ['audit', 'dns', 'firefox']
                for log_type in log_types:
                    log_type_path = os.path.join(host_path, log_type)
                    if not os.path.exists(log_type_path):
                        continue
                    
                    for timestamp_folder in sorted(os.listdir(log_type_path)):
                        timestamp_path = os.path.join(log_type_path, timestamp_folder)
                        if not os.path.isdir(timestamp_path):
                            continue
                        
                        embeddings_file = os.path.join(timestamp_path, "embeddings.pkl")
                        if os.path.exists(embeddings_file):
                            combined_scenario = f"{scenario_name}_{host_name}"
                            folders.append((combined_scenario, log_type, timestamp_folder, embeddings_file))
    
    return folders


def load_embeddings_from_pkl(embeddings_path: str) -> Tuple[np.ndarray, int]:
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_list = pickle.load(f)
        
        if not isinstance(embeddings_list, list) or len(embeddings_list) == 0:
            return np.zeros((0, 0), dtype=np.float32), 0
        
        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        
        return embeddings, len(embeddings_list)
        
    except Exception as e:
        return np.zeros((0, 0), dtype=np.float32), 0


def greedy_fps_cpu(X: np.ndarray, k: int) -> List[int]:
    n = X.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    seeds: List[int] = [0]
    dists = 1.0 - (X @ X[0].reshape(-1, 1)).ravel()
    for _ in tqdm(range(1, k), desc="FPS", leave=False):
        next_seed = int(np.argmax(dists))
        seeds.append(next_seed)
        s = X[next_seed]
        new_d = 1.0 - (X @ s)
        dists = np.minimum(dists, new_d)
    return seeds


def create_seed_id(scenario: str, log_type: str, timestamp: str, index: int) -> str:
    return f"{scenario}_{log_type}_{timestamp}_{index}"


def parse_seed_id(seed_id: str) -> Tuple[str, str, str, int]:
    try:
        last_underscore = seed_id.rfind('_')
        if last_underscore == -1:
            return "", "", "", 0
        
        index = int(seed_id[last_underscore + 1:])
        remainder = seed_id[:last_underscore]
        
        parts = remainder.split('_')
        
        if len(parts) >= 3:
            for i, part in enumerate(parts):
                if part in ['audit', 'dns', 'firefox']:
                    scenario = '_'.join(parts[:i])
                    log_type = part
                    timestamp = '_'.join(parts[i+1:])
                    return scenario, log_type, timestamp, index
        
        return "", "", "", 0
    except ValueError:
        return "", "", "", 0


def read_original_log_data(original_data_dir: str, scenario: str, log_type: str, timestamp: str, index: int):
    try:
        if '_' in scenario and scenario.split('_')[0].startswith('M'):
            scenario_parts = scenario.split('_')
            base_scenario = scenario_parts[0]
            host = scenario_parts[1]
            log_path = os.path.join(original_data_dir, base_scenario, host, log_type, timestamp, f"{log_type}.pkl")
        else:
            log_path = os.path.join(original_data_dir, scenario, log_type, timestamp, f"{log_type}.pkl")
        
        if not os.path.exists(log_path):
            return f"ERROR: {log_type}.pkl not found at {log_path}"
        
        with open(log_path, 'rb') as f:
            logs = pickle.load(f)
        
        if isinstance(logs, list):
            if 0 <= index < len(logs):
                return logs[index]
            else:
                return f"ERROR: Index {index} out of range (0-{len(logs)-1})"
        else:
            return f"ERROR: logs does not contain a list, got {type(logs)}"
            
    except Exception as e:
        return f"ERROR reading data: {str(e)}"


def display_seed_samples(seeds: List[str], original_data_dir: str):
    for seed_idx, seed_id in enumerate(seeds):
        scenario, log_type, timestamp, index = parse_seed_id(seed_id)
        log_data = read_original_log_data(original_data_dir, scenario, log_type, timestamp, index)


def save_seeds_to_pickle(seeds: List[str], output_path: str, scenario: str, log_type: str, timestamp: str, k: int):
    with open(output_path, 'wb') as f:
        pickle.dump(seeds, f)


def main():
    args = parse_args()
    
    embedding_folders = find_atlas_embedding_folders(args.embeddings_dir, args.scenario, args.log_type)
    
    if not embedding_folders:
        return
    
    if args.test:
        embedding_folders = embedding_folders[:1]
    
    processed_count = 0
    skipped_count = 0
    
    for scenario, log_type, timestamp, embeddings_path in tqdm(embedding_folders, desc="Processing timestamps"):
        date, hour, _ = parse_timestamp_folder(timestamp)
        
        output_dir = os.path.join(args.output_dir, args.embedding, log_type)
        if not args.test:
            os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"seeds_{scenario}_{timestamp}_k{args.k}.pkl"
        output_path = os.path.join(output_dir, output_filename) if not args.test else None
        
        if output_path and os.path.exists(output_path) and not args.overwrite:
            skipped_count += 1
            continue
        
        embeddings, num_embeddings = load_embeddings_from_pkl(embeddings_path)
        
        if embeddings.shape[0] == 0:
            continue
        
        if args.fps_subset and args.fps_subset > 0 and embeddings.shape[0] > args.fps_subset:
            rng = np.random.default_rng(42)
            subset_idx = rng.choice(embeddings.shape[0], size=args.fps_subset, replace=False)
            X_subset = embeddings[subset_idx]
            seeds_subset = greedy_fps_cpu(X_subset, args.k)
            seeds_indices = [int(subset_idx[s]) for s in seeds_subset]
        else:
            seeds_indices = greedy_fps_cpu(embeddings, args.k)
        
        seeds = [create_seed_id(scenario, log_type, timestamp, idx) for idx in seeds_indices]
        
        if args.verbose:
            display_seed_samples(seeds, args.original_data_dir)
        
        if not args.test and output_path:
            save_seeds_to_pickle(seeds, output_path, scenario, log_type, timestamp, args.k)
        
        processed_count += 1


if __name__ == "__main__":
    main()
