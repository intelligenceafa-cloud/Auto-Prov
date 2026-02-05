#!/usr/bin/env python3

import os
import pickle
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional
import shutil
from datetime import datetime
import re


DEFAULT_EXTRACTED_GRAPH_DIR = "./Extracted_Graph/ATLAS"
DEFAULT_ATLAS_DIR = "../../../BIGDATA/ATLAS"
DEFAULT_LABELS_DIR = "../../../BIGDATA/ATLAS/labels"
DEFAULT_OUTPUT_DIR = "./Extracted_Graph/"

DATASETS = ['S1', 'S2', 'S3', 'S4']
LOG_TYPES = ['audit', 'dns', 'firefox']


def load_malicious_labels(dataset: str, labels_dir: str) -> List[str]:
    label_file = os.path.join(labels_dir, dataset, 'malicious_labels.txt')
    
    if not os.path.exists(label_file):
        return []
    
    with open(label_file, 'r') as f:
        labels = [line.strip().lower() for line in f if line.strip()]
    
    return labels


def is_matched(string: str, malicious_labels: List[str]) -> bool:
    if not string or not isinstance(string, str):
        return False
    
    string_lower = str(string).lower()
    for label in malicious_labels:
        if label in string_lower:
            return True
    return False


def hex_to_decimal(hex_str: str) -> str:
    if not hex_str or not isinstance(hex_str, str):
        return hex_str
    
    hex_str = hex_str.strip()
    
    if hex_str.startswith('0x') or hex_str.startswith('0X'):
        try:
            decimal_value = int(hex_str, 16)
            return str(decimal_value)
        except ValueError:
            return hex_str
    
    return hex_str


def normalize_timestamp(timestamp_str: str, window_timestamp: str, log_type: str) -> str:
    if not timestamp_str or pd.isna(timestamp_str):
        try:
            window_start = window_timestamp.split('_')[0]
            return f"{window_start}.000000"
        except:
            return ""
    
    timestamp_str = str(timestamp_str).strip()
    
    if re.match(r'^\d{4}-\d{2}-\d{2}', timestamp_str):
        try:
            for fmt in [
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
            ]:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    continue
            return timestamp_str
        except:
            return timestamp_str
    
    if log_type == 'audit' and ('AM' in timestamp_str or 'PM' in timestamp_str):
        try:
            window_start = window_timestamp.split('_')[0]
            window_date = window_start.split(' ')[0]
            
            time_str = timestamp_str.strip()
            
            for fmt in ['%I:%M:%S %p', '%I:%M %p', '%I:%M:%S.%f %p']:
                try:
                    time_obj = datetime.strptime(time_str, fmt).time()
                    dt = datetime.strptime(f"{window_date} {time_obj.strftime('%H:%M:%S')}", '%Y-%m-%d %H:%M:%S')
                    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    continue
            
            return f"{window_start}.000000"
        except Exception as e:
            try:
                window_start = window_timestamp.split('_')[0]
                return f"{window_start}.000000"
            except:
                return ""
    
    try:
        window_start = window_timestamp.split('_')[0]
        return f"{window_start}.000000"
    except:
        return timestamp_str


def label_edge(row: pd.Series, log_type: str, malicious_labels: List[str]) -> int:
    source = row.get('source_name', '')
    dest = row.get('dest_name', '')
    
    if is_matched(source, malicious_labels) or is_matched(dest, malicious_labels):
        return 1
    
    return 0


def load_and_label_csv(
    csv_path: str,
    log_type: str,
    dataset: str,
    timestamp: str,
    malicious_labels: List[str],
    atlas_dir: str
) -> Optional[pd.DataFrame]:
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return None
        
        if log_type == 'audit':
            if 'source_id' in df.columns:
                df['source_id'] = df['source_id'].astype(str).apply(hex_to_decimal)
            if 'dest_id' in df.columns:
                df['dest_id'] = df['dest_id'].astype(str).apply(hex_to_decimal)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].apply(
                lambda x: normalize_timestamp(x, timestamp, log_type)
            )
        
        df['log_type'] = log_type
        
        df['label'] = df.apply(
            lambda row: label_edge(row, log_type, malicious_labels),
            axis=1
        )
        
        return df
    
    except Exception as e:
        return None


def find_all_timestamps(dataset: str, extracted_graph_dir: str) -> Set[str]:
    timestamps = set()
    
    for log_type in LOG_TYPES:
        log_type_dir = os.path.join(extracted_graph_dir, dataset, log_type)
        
        if not os.path.exists(log_type_dir):
            continue
        
        for item in os.listdir(log_type_dir):
            item_path = os.path.join(log_type_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                csv_path = os.path.join(item_path, 'provgraph.csv')
                if os.path.exists(csv_path):
                    timestamps.add(item)
    
    return timestamps


def merge_duplicate_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    group_cols = ['log_idx']
    
    if 'source_id' in df.columns:
        group_cols.append('source_id')
    if 'source_name' in df.columns:
        group_cols.append('source_name')
    
    if 'dest_id' in df.columns:
        group_cols.append('dest_id')
    if 'dest_name' in df.columns:
        group_cols.append('dest_name')
    
    for col in group_cols:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('')
    
    agg_dict = {
        'action': lambda x: ' '.join(str(a).strip() for a in x if pd.notna(a) and str(a).strip()),
    }
    
    if 'label' in df.columns:
        agg_dict['label'] = lambda x: 1 if any(x == 1) else 0
    
    other_cols = [col for col in df.columns if col not in group_cols and col not in ['action', 'label']]
    for col in other_cols:
        agg_dict[col] = 'first'
    
    merged_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
    
    return merged_df


def merge_and_save_timestamp(
    dataset: str,
    timestamp: str,
    extracted_graph_dir: str,
    malicious_labels: List[str],
    output_dir: str
) -> bool:
    all_dfs = []
    
    for log_type in LOG_TYPES:
        csv_path = os.path.join(
            extracted_graph_dir,
            dataset,
            log_type,
            timestamp,
            'provgraph.csv'
        )
        
        df = load_and_label_csv(
            csv_path,
            log_type,
            dataset,
            timestamp,
            malicious_labels,
            None
        )
        
        if df is not None and not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        return False
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    merged_df = merge_duplicate_edges(merged_df)
    
    if 'timestamp' in merged_df.columns:
        merged_df['timestamp'] = merged_df.apply(
            lambda row: normalize_timestamp(row['timestamp'], timestamp, row.get('log_type', '')),
            axis=1
        )
        
        def safe_to_datetime(ts_str):
            if not ts_str or pd.isna(ts_str):
                return pd.Timestamp.min
            try:
                for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']:
                    try:
                        dt = datetime.strptime(str(ts_str), fmt)
                        return pd.Timestamp(dt)
                    except ValueError:
                        continue
                return pd.to_datetime(str(ts_str), errors='coerce') or pd.Timestamp.min
            except:
                return pd.Timestamp.min
        
        merged_df['_timestamp_sort'] = merged_df['timestamp'].apply(safe_to_datetime)
        merged_df = merged_df.sort_values('_timestamp_sort').drop('_timestamp_sort', axis=1)
        merged_df = merged_df.reset_index(drop=True)
    
    has_source_id = 'source_id' in merged_df.columns
    has_dest_id = 'dest_id' in merged_df.columns
    
    if has_source_id and has_dest_id:
        column_order = [
            'log_idx',
            'log_type',
            'source_id',
            'source_name',
            'dest_id',
            'dest_name',
            'action',
            'timestamp',
            'label'
        ]
    else:
        column_order = [
            'log_idx',
            'log_type',
            'source_name',
            'dest_name',
            'action',
            'timestamp',
            'label'
        ]
    
    for col in column_order:
        if col not in merged_df.columns:
            merged_df[col] = ''
    
    merged_df = merged_df[column_order]
    
    output_path = os.path.join(output_dir, dataset, timestamp)
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, 'merged_graph.csv')
    merged_df.to_csv(output_file, index=False)
    
    total_edges = len(merged_df)
    malicious_edges = merged_df['label'].sum()
    benign_edges = total_edges - malicious_edges
    
    return True


def process_dataset(
    dataset: str,
    extracted_graph_dir: str,
    labels_dir: str,
    output_dir: str
):
    malicious_labels = load_malicious_labels(dataset, labels_dir)
    
    timestamps = find_all_timestamps(dataset, extracted_graph_dir)
    
    if not timestamps:
        return
    
    successful = 0
    for timestamp in sorted(timestamps):
        if merge_and_save_timestamp(
            dataset,
            timestamp,
            extracted_graph_dir,
            malicious_labels,
            output_dir
        ):
            successful += 1


def zip_dataset_directories(output_dir: str, datasets: List[str]):
    for dataset in datasets:
        dataset_dir = os.path.join(output_dir, dataset)
        
        if not os.path.exists(dataset_dir):
            continue
        
        if not os.path.isdir(dataset_dir):
            continue
        
        zip_path = os.path.join(output_dir, f"{dataset}.zip")
        
        try:
            shutil.make_archive(
                base_name=os.path.join(output_dir, dataset),
                format='zip',
                root_dir=output_dir,
                base_dir=dataset
            )
            
        except Exception as e:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Merge and label CSV graphs from apply_rules.py output"
    )
    parser.add_argument(
        "--extracted-graph-dir",
        type=str,
        default=DEFAULT_EXTRACTED_GRAPH_DIR,
        help=f"Directory containing extracted CSV graphs (default: {DEFAULT_EXTRACTED_GRAPH_DIR})"
    )
    parser.add_argument(
        "--atlas-dir",
        type=str,
        default=DEFAULT_ATLAS_DIR,
        help=f"Directory containing ATLAS datasets (default: {DEFAULT_ATLAS_DIR})"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=DEFAULT_LABELS_DIR,
        help=f"Directory containing malicious labels (default: {DEFAULT_LABELS_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for merged graphs (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=DATASETS,
        choices=DATASETS,
        help=f"Datasets to process (default: all: {DATASETS})"
    )
    
    args = parser.parse_args()
    
    extracted_graph_dir = os.path.abspath(args.extracted_graph_dir)
    labels_dir = os.path.abspath(args.labels_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    for dataset in args.datasets:
        process_dataset(
            dataset,
            extracted_graph_dir,
            labels_dir,
            output_dir
        )
    
    zip_dataset_directories(output_dir, args.datasets)


if __name__ == "__main__":
    main()
