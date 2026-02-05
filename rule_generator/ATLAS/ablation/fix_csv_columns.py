#!/usr/bin/env python3

import os
import csv
import argparse
from pathlib import Path


def fix_csv_file(csv_path):
    try:
        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            if not fieldnames:
                return False
            
            if 'source_id' not in fieldnames or 'dest_id' not in fieldnames:
                return False
            
            for row in reader:
                rows.append(row)
        
        if not rows:
            return False
        
        source_name_empty = True
        dest_name_empty = True
        
        if 'source_name' in fieldnames:
            for row in rows:
                if row.get('source_name', '').strip():
                    source_name_empty = False
                    break
        
        if 'dest_name' in fieldnames:
            for row in rows:
                if row.get('dest_name', '').strip():
                    dest_name_empty = False
                    break
        
        new_rows = []
        for row in rows:
            new_row = {}
            new_row['log_idx'] = row.get('log_idx', '')
            
            if source_name_empty and 'source_id' in row:
                new_row['source_name'] = row.get('source_id', '')
            elif 'source_name' in row and row.get('source_name', '').strip():
                new_row['source_name'] = row.get('source_name', '')
            else:
                new_row['source_name'] = row.get('source_id', '')
            
            if dest_name_empty and 'dest_id' in row:
                new_row['dest_name'] = row.get('dest_id', '')
            elif 'dest_name' in row and row.get('dest_name', '').strip():
                new_row['dest_name'] = row.get('dest_name', '')
            else:
                new_row['dest_name'] = row.get('dest_id', '')
            
            new_row['action'] = row.get('action', '')
            new_row['timestamp'] = row.get('timestamp', '')
            
            new_rows.append(new_row)
        
        new_fieldnames = ['log_idx', 'source_name', 'dest_name', 'action', 'timestamp']
        
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        
        return True
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


def fix_dataset(dataset, extracted_graph_dir):
    fixed_count = 0
    total_count = 0
    
    for log_type in ['dns', 'firefox']:
        log_type_dir = os.path.join(extracted_graph_dir, dataset, log_type)
        
        if not os.path.exists(log_type_dir):
            continue
        
        for item in os.listdir(log_type_dir):
            item_path = os.path.join(log_type_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                csv_path = os.path.join(item_path, 'provgraph.csv')
                if os.path.exists(csv_path):
                    total_count += 1
                    if fix_csv_file(csv_path):
                        fixed_count += 1
    
    return fixed_count, total_count


def main():
    parser = argparse.ArgumentParser(
        description="Fix CSV column names for DNS and Firefox logs"
    )
    parser.add_argument(
        "--extracted-graph-dir",
        type=str,
        required=True,
        help="Directory containing extracted CSV graphs"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Datasets to process"
    )
    
    args = parser.parse_args()
    
    extracted_graph_dir = os.path.abspath(args.extracted_graph_dir)
    
    total_fixed = 0
    total_files = 0
    
    for dataset in args.datasets:
        fixed, total = fix_dataset(dataset, extracted_graph_dir)
        total_fixed += fixed
        total_files += total


if __name__ == "__main__":
    main()
