#!/usr/bin/env python3

import os
import csv
import pickle
import json
import shutil
import re
import argparse
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path

MERGED_DIR = "./merged"
OUTPUT_BASE_DIR = "./autoprov_atlas_graph"
DATASETS = ['S1', 'S2', 'S3', 'S4']

ATTACK_TYPES = {
    'S1': 'Strategic web compromise',
    'S2': 'Malvertising dominate',
    'S3': 'Spam campaign',
    'S4': 'Pony campaign'
}

def clean_node_name(name):
    if not name:
        return ''
    return str(name).replace(' ', '_').replace('\t', '_').strip()

def normalize_action(action):
    if not action:
        return 'unknown'
    
    action = str(action).lower().strip()
    
    action = re.sub(r'\s+', ' ', action)
    
    if ' a handle to an object was ' in action or ' the handle to an object was ' in action:
        match = re.search(r'was\s+(\w+)', action)
        if match:
            verb = match.group(1)
            if 'handle' in action:
                action = f'handle_{verb}'
                return re.sub(r'[^a-z0-9_-]', '_', action).strip('_') or 'unknown'
    
    patterns = [
        (r'the handle to an object was closed', 'handle_closed'),
        (r'a handle to an object was requested', 'handle_requested'),
        (r'a handle to an object was (\w+)', r'handle_\1'),
        (r'the handle to an object was (\w+)', r'handle_\1'),
        (r'an attempt was made to access an object', 'access_attempt'),
        (r'an attempt was made to access', 'access_attempt'),
        (r'an object was deleted', 'object_deleted'),
        (r'accesses readdata \(or listdirectory\)', 'read_data'),
        (r'accesses readdata', 'read_data'),
        (r'readdata \(or listdirectory\)', 'read_data'),
        (r'readdata', 'read_data'),
        (r'accesses read_control', 'read_control'),
        (r'accesses synchronize', 'synchronize'),
        (r'accesses readattributes', 'read_attributes'),
        (r'accesses execute/traverse', 'execute'),
        (r'accesses execute', 'execute'),
        (r'^http$', 'http'),
        (r'^connect$', 'connect'),
        (r'^outbound$', 'outbound'),
        (r'^inbound$', 'inbound'),
        (r'^transport$', 'transport'),
        (r'^main thread$', 'main_thread'),
        (r'^accesses\s+(.+)$', r'\1'),
        (r'.*access\s+mask.*', 'access_mask'),
        (r'.*access\s+reasons.*', 'access_reasons'),
    ]
    
    for pattern, replacement in patterns:
        if re.search(pattern, action, re.IGNORECASE):
            action = re.sub(pattern, replacement, action, flags=re.IGNORECASE)
            break
    
    if len(action) > 50:
        words = action.split()
        filler_words = {'an', 'a', 'the', 'to', 'was', 'were', 'is', 'are', 'of', 'in', 'on', 'at', 'for', 'and', 'or', 'object', 'handle'}
        important_words = [w for w in words if w not in filler_words and len(w) > 2]
        if important_words:
            action = '_'.join(important_words[:5])
    
    action = re.sub(r'[^a-z0-9_-]', '_', action)
    action = re.sub(r'_+', '_', action)
    action = action.strip('_')
    
    if len(action) > 50:
        action = action[:50].rstrip('_')
    
    action = re.sub(r'[^a-z0-9_-]', '_', action)
    action = re.sub(r'_+', '_', action)
    action = action.strip('_')
    
    if not action:
        return 'unknown'
    
    return action

def get_node_name(row, prefix='source'):
    name_col = f'{prefix}_name'
    node_name = row.get(name_col, '').strip() if row.get(name_col) else ''
    
    if node_name:
        return clean_node_name(node_name)
    else:
        return ''

def parse_timestamp(timestamp_str, window_name):
    timestamp_str = timestamp_str.strip()
    
    try:
        if '.' in timestamp_str:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
        else:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        return dt.timestamp()
    except:
        pass
    
    try:
        if '_' in window_name:
            start_part = window_name.split('_')[0]
            date_part = start_part.split()[0]
        else:
            parts = window_name.split()
            if parts:
                date_part = parts[0]
            else:
                return 0.0
        
        time_str = timestamp_str.upper()
        if 'AM' in time_str or 'PM' in time_str:
            time_only = time_str.replace('AM', '').replace('PM', '').strip()
            h, m, s = time_only.split(':')
            h = int(h)
            m = int(m)
            s = int(s.split()[0] if ' ' in s else s)
            
            if 'PM' in time_str and h != 12:
                h += 12
            elif 'AM' in time_str and h == 12:
                h = 0
            
            dt = datetime.strptime(date_part, '%Y-%m-%d')
            dt = dt.replace(hour=h, minute=m, second=s)
            return dt.timestamp()
        else:
            h, m, s = timestamp_str.split(':')
            dt = datetime.strptime(date_part, '%Y-%m-%d')
            dt = dt.replace(hour=int(h), minute=int(m), second=int(s.split('.')[0]))
            return dt.timestamp()
    except Exception as e:
        return 0.0

def parse_csv_window(csv_path, dataset, window_name):
    edges = []
    
    if not os.path.exists(csv_path):
        return edges
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        skipped_count = 0
        for row in reader:
            source = get_node_name(row, 'source')
            dest = get_node_name(row, 'dest')
            
            if not source or not dest:
                skipped_count += 1
                continue
            
            edge_type_raw = row.get('action', 'unknown').strip()
            if not edge_type_raw:
                edge_type_raw = 'unknown'
            
            edge_type = normalize_action(edge_type_raw)
            
            timestamp_str = row.get('timestamp', '').strip()
            timestamp = parse_timestamp(timestamp_str, window_name)
            
            label = int(row.get('label', '0').strip())
            
            log_type = row.get('log_type', '').strip()
            
            original_row = row.copy()
            if 'source_id' not in original_row:
                original_row['source_id'] = ''
            if 'dest_id' not in original_row:
                original_row['dest_id'] = ''
            
            edges.append({
                'source': source,
                'dest': dest,
                'edge_type': edge_type,
                'timestamp': timestamp,
                'label': label,
                'log_type': log_type,
                'original_row': original_row
            })
    
    return edges

def create_graph_files(edges, output_dir, dataset, window_name, cee_model_name, edge_type_frequencies=None):
    os.makedirs(output_dir, exist_ok=True)
    
    edges_sorted = sorted(edges, key=lambda x: x['timestamp'])
    
    if edge_type_frequencies is not None:
        original_count = len(edges_sorted)
        edges_sorted = [e for e in edges_sorted 
                       if edge_type_frequencies.get(e['edge_type'], 0) >= 1000]
        filtered_count = original_count - len(edges_sorted)
    
    graph_lines = []
    malicious_labels = {}
    attack_type = ATTACK_TYPES.get(dataset, 'Unknown')
    
    csv_rows = []
    
    for edge_counter, edge in enumerate(edges_sorted):
        line = f"{edge['source']} A: [{edge['edge_type']}] {edge['dest']}"
        graph_lines.append(line)
        
        if edge['label'] == 1:
            malicious_labels[edge_counter] = attack_type
        else:
            malicious_labels[edge_counter] = False
        
        csv_row = {
            'log_idx': edge['original_row'].get('log_idx', ''),
            'log_type': edge['original_row'].get('log_type', ''),
            'source_id': edge['original_row'].get('source_id', ''),
            'source_name': edge['original_row'].get('source_name', ''),
            'dest_id': edge['original_row'].get('dest_id', ''),
            'dest_name': edge['original_row'].get('dest_name', ''),
            'action': edge['original_row'].get('action', ''),
            'timestamp': edge['original_row'].get('timestamp', ''),
            'label': edge['original_row'].get('label', '0')
        }
        csv_rows.append(csv_row)
    
    graph_txt_path = os.path.join(output_dir, 'graph.txt')
    with open(graph_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(graph_lines))
    
    graph_csv_path = os.path.join(output_dir, 'graph.csv')
    csv_fieldnames = ['log_idx', 'log_type', 'source_id', 'source_name', 'dest_id', 'dest_name', 'action', 'timestamp', 'label']
    with open(graph_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    labels_path = os.path.join(output_dir, 'malicious_labels.pkl')
    with open(labels_path, 'wb') as f:
        pickle.dump(malicious_labels, f)
    
    malicious_count = sum(1 for e in edges_sorted if e['label'] == 1)
    is_test = malicious_count > 0
    
    metadata = {
        'cee_model_name': cee_model_name,
        'dataset': dataset,
        'datasets': [dataset],
        'attack_type': attack_type,
        'attack_types': [attack_type],
        'time_window': window_name,
        'is_test': is_test,
        'total_edges': len(edges_sorted),
        'malicious_edges': malicious_count,
        'benign_edges': sum(1 for e in edges_sorted if e['label'] == 0),
        'source_path': f"{MERGED_DIR}/{cee_model_name}/{dataset}/{window_name}/merged_graph.csv",
        'is_merged': False
    }
    
    metadata_path = os.path.join(output_dir, 'window_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def merge_graphs_and_labels(existing_graph_path, existing_graph_csv_path, existing_labels_path, 
                             new_graph_lines, new_graph_csv_rows, new_labels, dataset):
    existing_graph_lines = []
    if os.path.exists(existing_graph_path):
        with open(existing_graph_path, 'r', encoding='utf-8') as f:
            existing_graph_lines = [line.strip() for line in f if line.strip()]
    
    existing_graph_csv_rows = []
    if os.path.exists(existing_graph_csv_path):
        with open(existing_graph_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_graph_csv_rows = list(reader)
    
    existing_labels = {}
    if os.path.exists(existing_labels_path):
        with open(existing_labels_path, 'rb') as f:
            existing_labels = pickle.load(f)
    
    existing_edge_to_idx = {}
    for idx, line in enumerate(existing_graph_lines):
        existing_edge_to_idx[line] = idx
    
    merged_graph_lines = existing_graph_lines.copy()
    merged_graph_csv_rows = existing_graph_csv_rows.copy()
    merged_labels = {}
    
    for idx, label_val in existing_labels.items():
        merged_labels[idx] = label_val
    
    for new_idx_key, new_label_val in new_labels.items():
        new_line = new_graph_lines[new_idx_key]
        new_csv_row = new_graph_csv_rows[new_idx_key]
        
        if new_line in existing_edge_to_idx:
            existing_idx = existing_edge_to_idx[new_line]
            existing_val = existing_labels.get(existing_idx, False)
            
            if existing_val != False and new_label_val != False:
                new_edge_idx = len(merged_graph_lines)
                merged_graph_lines.append(new_line)
                merged_graph_csv_rows.append(new_csv_row)
                merged_labels[new_edge_idx] = new_label_val
            elif existing_val == False and new_label_val != False:
                merged_labels[existing_idx] = new_label_val
            elif existing_val != False and new_label_val == False:
                merged_labels[existing_idx] = existing_val
        else:
            new_edge_idx = len(merged_graph_lines)
            merged_graph_lines.append(new_line)
            merged_graph_csv_rows.append(new_csv_row)
            merged_labels[new_edge_idx] = new_label_val
    
    return merged_graph_lines, merged_graph_csv_rows, merged_labels

def merge_window_files(new_dir, existing_dir, dataset):
    new_graph_path = os.path.join(new_dir, 'graph.txt')
    new_graph_csv_path = os.path.join(new_dir, 'graph.csv')
    new_labels_path = os.path.join(new_dir, 'malicious_labels.pkl')
    existing_graph_path = os.path.join(existing_dir, 'graph.txt')
    existing_graph_csv_path = os.path.join(existing_dir, 'graph.csv')
    existing_labels_path = os.path.join(existing_dir, 'malicious_labels.pkl')
    
    new_graph_lines = []
    if os.path.exists(new_graph_path):
        with open(new_graph_path, 'r', encoding='utf-8') as f:
            new_graph_lines = [line.strip() for line in f if line.strip()]
    
    new_graph_csv_rows = []
    if os.path.exists(new_graph_csv_path):
        with open(new_graph_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            new_graph_csv_rows = list(reader)
    
    new_labels = {}
    if os.path.exists(new_labels_path):
        with open(new_labels_path, 'rb') as f:
            new_labels = pickle.load(f)
    
    merged_graph_lines, merged_graph_csv_rows, merged_labels = merge_graphs_and_labels(
        existing_graph_path, existing_graph_csv_path, existing_labels_path,
        new_graph_lines, new_graph_csv_rows, new_labels, dataset
    )
    
    with open(existing_graph_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(merged_graph_lines))
    
    csv_fieldnames = ['log_idx', 'log_type', 'source_id', 'source_name', 'dest_id', 'dest_name', 'action', 'timestamp', 'label']
    with open(existing_graph_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(merged_graph_csv_rows)
    
    with open(existing_labels_path, 'wb') as f:
        pickle.dump(merged_labels, f)
    
    metadata_path = os.path.join(existing_dir, 'window_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    if 'datasets' not in metadata:
        old_dataset = metadata.get('dataset', 'Unknown')
        old_attack = metadata.get('attack_type', 'Unknown')
        metadata['datasets'] = [old_dataset] if old_dataset != 'Unknown' else []
        metadata['attack_types'] = [old_attack] if old_attack != 'Unknown' else []
    
    if dataset not in metadata['datasets']:
        metadata['datasets'].append(dataset)
    
    current_attack_type = ATTACK_TYPES.get(dataset, 'Unknown')
    if current_attack_type not in metadata['attack_types']:
        metadata['attack_types'].append(current_attack_type)
    
    metadata['total_edges'] = len(merged_graph_lines)
    metadata['malicious_edges'] = sum(1 for v in merged_labels.values() if v != False)
    metadata['benign_edges'] = len(merged_graph_lines) - metadata['malicious_edges']
    metadata['is_merged'] = True
    metadata['is_test'] = metadata['malicious_edges'] > 0
    
    if 'dataset' not in metadata or metadata['dataset'] == 'Unknown':
        metadata['dataset'] = metadata['datasets'][0] if metadata['datasets'] else 'Unknown'
    if 'attack_type' not in metadata or metadata['attack_type'] == 'Unknown':
        metadata['attack_type'] = metadata['attack_types'][0] if metadata['attack_types'] else 'Unknown'
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def collect_edge_type_frequencies(cee_model_name, merged_dir):
    edge_type_counts = Counter()
    total_windows = 0
    
    for dataset in DATASETS:
        dataset_dir = os.path.join(merged_dir, cee_model_name, dataset)
        
        if not os.path.exists(dataset_dir):
            continue
        
        time_windows = sorted([d for d in os.listdir(dataset_dir) 
                              if os.path.isdir(os.path.join(dataset_dir, d))])
        
        for window_name in time_windows:
            csv_path = os.path.join(dataset_dir, window_name, 'merged_graph.csv')
            
            if not os.path.exists(csv_path):
                continue
            
            total_windows += 1
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    source = get_node_name(row, 'source')
                    dest = get_node_name(row, 'dest')
                    
                    if not source or not dest:
                        continue
                    
                    edge_type_raw = row.get('action', 'unknown').strip()
                    if not edge_type_raw:
                        edge_type_raw = 'unknown'
                    
                    edge_type = normalize_action(edge_type_raw)
                    edge_type_counts[edge_type] += 1
    
    single_occurrence = sum(1 for count in edge_type_counts.values() if count == 1)
    less_than_1000 = sum(1 for count in edge_type_counts.values() if count < 1000)
    at_least_1000 = sum(1 for count in edge_type_counts.values() if count >= 1000)
    
    return edge_type_counts

def process_dataset(cee_model_name, dataset, merged_dir, output_base_dir, edge_type_frequencies=None):
    dataset_dir = os.path.join(merged_dir, cee_model_name, dataset)
    
    if not os.path.exists(dataset_dir):
        return []
    
    time_windows = sorted([d for d in os.listdir(dataset_dir) 
                          if os.path.isdir(os.path.join(dataset_dir, d))])
    
    processed_windows = []
    
    for window_name in time_windows:
        csv_path = os.path.join(dataset_dir, window_name, 'merged_graph.csv')
        
        if not os.path.exists(csv_path):
            continue
        
        edges = parse_csv_window(csv_path, dataset, window_name)
        
        if not edges:
            continue
        
        total_parsed = len(edges)
        
        edges = [e for e in edges if e['source'] and e['dest']]
        filtered_count = total_parsed - len(edges)
        
        has_malicious = any(e['label'] == 1 for e in edges)
        malicious_count = sum(1 for e in edges if e['label'] == 1)
        
        if has_malicious:
            output_dir = os.path.join(output_base_dir, cee_model_name, 'test', window_name)
            is_test = True
        else:
            output_dir = os.path.join(output_base_dir, cee_model_name, 'train', window_name)
            is_test = False
        
        window_exists = os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, 'malicious_labels.pkl'))
        is_merge = window_exists
        
        if is_merge:
            temp_dir = os.path.join(output_base_dir, 'temp', window_name)
            create_graph_files(edges, temp_dir, dataset, window_name, cee_model_name, edge_type_frequencies)
            
            merge_window_files(temp_dir, output_dir, dataset)
            
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            metadata = create_graph_files(edges, output_dir, dataset, window_name, cee_model_name, edge_type_frequencies)
        
        processed_windows.append({
            'window': window_name,
            'is_test': is_test,
            'malicious_count': malicious_count,
            'total_edges': len(edges)
        })
        
    
    return processed_windows

def find_cee_model_folders(merged_dir):
    if not os.path.exists(merged_dir):
        return []
    
    cee_model_folders = []
    for item in os.listdir(merged_dir):
        item_path = os.path.join(merged_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            has_datasets = any(os.path.isdir(os.path.join(item_path, d)) and d in DATASETS 
                             for d in os.listdir(item_path))
            if has_datasets:
                cee_model_folders.append(item)
    
    return sorted(cee_model_folders)

def process_cee_model(cee_model_name, merged_dir, output_base_dir, edge_type_frequencies=None):
    stats = {
        'total_windows': 0,
        'benign_windows': 0,
        'malicious_windows': 0,
        'by_dataset': defaultdict(lambda: {'benign': 0, 'malicious': 0})
    }
    
    for dataset in DATASETS:
        processed = process_dataset(cee_model_name, dataset, merged_dir, output_base_dir, edge_type_frequencies)
        
        for window_info in processed:
            stats['total_windows'] += 1
            if window_info['is_test']:
                stats['malicious_windows'] += 1
                stats['by_dataset'][dataset]['malicious'] += 1
            else:
                stats['benign_windows'] += 1
                stats['by_dataset'][dataset]['benign'] += 1
    
    return stats

def create_autoprov_atlas_graph(merged_dir=None, output_base_dir=None, cee_models=None, cee_model=None, model_name=None):
    if merged_dir is None:
        merged_dir = MERGED_DIR
    if output_base_dir is None:
        output_base_dir = OUTPUT_BASE_DIR
    
    if cee_models is not None:
        pass
    elif cee_model is not None and model_name is not None:
        sanitized_cee = sanitize_cee_name(cee_model.lower())
        sanitized_model = sanitize_cee_name(model_name.lower())
        combined_name = f"{sanitized_cee}_{sanitized_model}"
        cee_models = [combined_name]
    elif cee_model is not None or model_name is not None:
        return
    else:
        cee_models = find_cee_model_folders(merged_dir)
        if not cee_models:
            return
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    all_stats = {}
    
    for cee_model_name in cee_models:
        edge_type_frequencies = collect_edge_type_frequencies(cee_model_name, merged_dir)
        
        stats = process_cee_model(cee_model_name, merged_dir, output_base_dir, edge_type_frequencies)
        all_stats[cee_model_name] = stats
        
        for dataset in DATASETS:
            dataset_stats = stats['by_dataset'][dataset]
            total = dataset_stats['benign'] + dataset_stats['malicious']
        
        summary_data = {
            'cee_model_name': cee_model_name,
            'total_windows': stats['total_windows'],
            'benign_windows': stats['benign_windows'],
            'malicious_windows': stats['malicious_windows'],
            'attack_types': ATTACK_TYPES,
            'by_dataset': {}
        }
        
        for dataset in DATASETS:
            dataset_stats = stats['by_dataset'][dataset]
            summary_data['by_dataset'][dataset] = {
                'attack_type': ATTACK_TYPES.get(dataset, 'Unknown'),
                'benign': dataset_stats['benign'],
                'malicious': dataset_stats['malicious']
            }
        
        summary_path = os.path.join(output_base_dir, cee_model_name, 'train_test_split_summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
    total_windows = sum(stats['total_windows'] for stats in all_stats.values())
    total_benign = sum(stats['benign_windows'] for stats in all_stats.values())
    total_malicious = sum(stats['malicious_windows'] for stats in all_stats.values())

def sanitize_cee_name(name):
    if not name:
        return name
    sanitized = re.sub(r'[:/\\]', '_', str(name))
    return sanitized

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create autoprov_atlas_graph from merged CSV files (Ablation version)"
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default=MERGED_DIR,
        help=f"Path to merged directory (default: {MERGED_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_BASE_DIR,
        help=f"Path to output directory (default: {OUTPUT_BASE_DIR})"
    )
    parser.add_argument(
        "--cee-models",
        type=str,
        nargs='+',
        default=None,
        help="Specific CEE/model folders to process (default: auto-detect all). Overrides --cee-model and --model-name."
    )
    parser.add_argument(
        "--cee-model",
        type=str,
        default=None,
        help="Candidate Edge Extractor model name (e.g., 'gpt-3.5-turbo', 'gpt-4o'). Must be used with --model-name."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Rule generator model name (e.g., 'llama3:70b', 'qwen2:72b'). Must be used with --cee-model."
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.cee_models is not None and (args.cee_model is not None or args.model_name is not None):
        exit(1)
    
    create_autoprov_atlas_graph(
        merged_dir=args.merged_dir,
        output_base_dir=args.output_dir,
        cee_models=args.cee_models,
        cee_model=args.cee_model,
        model_name=args.model_name
    )
