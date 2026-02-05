#!/usr/bin/env python3

import json
import numpy as np
import os
from scipy.sparse import load_npz

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_typed_dataset(dataset_name, llmfets_model=None, causal=False, timeoh=False):
    is_atlas = dataset_name.lower() == "atlas"
    
    if is_atlas and not llmfets_model:
        raise ValueError("llmfets_model is required for ATLAS dataset")
    
    if llmfets_model:
        model_normalized = llmfets_model.lower().replace(':', '_')
    else:
        model_normalized = None
    
    if is_atlas and model_normalized:
        if causal:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized, "causal")
            prefix = "Causal"
        elif timeoh:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized, "timeoh")
            prefix = "TimeOH"
        else:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized)
            prefix = ""
    else:
        if causal:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", "causal")
            prefix = "Causal"
        elif timeoh:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", "timeoh")
            prefix = "TimeOH"
        else:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles")
            prefix = ""
    
    matrix_file = os.path.join(base_dir, f"{prefix}behavioral_dataset_typed_{dataset_name}.npz")
    X = load_npz(matrix_file)
    
    metadata_file = os.path.join(base_dir, f"{prefix}behavioral_dataset_typed_{dataset_name}_metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    node_ids = metadata['node_ids']
    labels = metadata['labels']
    patterns = metadata['pattern_columns']
    return X, node_ids, labels, patterns


def load_untyped_dataset(dataset_name, llmfets_model=None, causal=False, timeoh=False):
    is_atlas = dataset_name.lower() == "atlas"
    
    if is_atlas and not llmfets_model:
        raise ValueError("llmfets_model is required for ATLAS dataset")
    
    if llmfets_model:
        model_normalized = llmfets_model.lower().replace(':', '_')
    else:
        model_normalized = None
    
    if is_atlas and model_normalized:
        if causal:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized, "causal")
            prefix = "Causal"
        elif timeoh:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized, "timeoh")
            prefix = "TimeOH"
        else:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", model_normalized)
            prefix = ""
    else:
        if causal:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", "causal")
            prefix = "Causal"
        elif timeoh:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles", "timeoh")
            prefix = "TimeOH"
        else:
            base_dir = os.path.join(_SCRIPT_DIR, "behavioral-profiles")
            prefix = ""
    
    matrix_file = os.path.join(base_dir, f"{prefix}behavioral_dataset_untyped_{dataset_name}.npz")
    X = load_npz(matrix_file)
    
    metadata_file = os.path.join(base_dir, f"{prefix}behavioral_dataset_untyped_{dataset_name}_metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    node_ids = metadata['node_ids']
    patterns = metadata['pattern_columns']
    return X, node_ids, patterns


def convert_to_dataframe(X, node_ids, patterns, labels=None, max_rows=None):
    import pandas as pd
    
    if max_rows:
        X = X[:max_rows]
        node_ids = node_ids[:max_rows]
    if labels:
        labels = labels[:max_rows]
    X_dense = X.toarray()
    df = pd.DataFrame(X_dense, columns=patterns)
    df.insert(0, 'node_id', node_ids)
    
    if labels:
        df['label'] = labels
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['theia', 'fivedirections', 'atlas'])
    parser.add_argument('--llmfets-model', type=str, help='LLM model name (required for ATLAS)')
    parser.add_argument('--causal', action='store_true', help='Load from causal/ directory')
    parser.add_argument('--timeoh', action='store_true', help='Load from timeoh/ directory')
    parser.add_argument('--type', default='typed', choices=['typed', 'untyped'])
    parser.add_argument('--to-csv', action='store_true', help='Convert to CSV (WARNING: memory intensive!)')
    parser.add_argument('--max-rows', type=int, help='Only export first N rows to CSV')
    args = parser.parse_args()
    
    if args.dataset.lower() == 'atlas' and not args.llmfets_model:
        parser.error("--llmfets-model is required for ATLAS dataset")
    
    if args.type == 'typed':
        X, node_ids, labels, patterns = load_typed_dataset(
            args.dataset, 
            llmfets_model=args.llmfets_model,
            causal=args.causal, 
            timeoh=args.timeoh
        )
        
        if args.to_csv:
            df = convert_to_dataframe(X, node_ids, patterns, labels, args.max_rows)
            output_file = f"behavioral_dataset_typed_{args.dataset}_sample.csv" if args.max_rows else f"behavioral_dataset_typed_{args.dataset}.csv"
            df.to_csv(output_file, index=False)
    else:
        X, node_ids, patterns = load_untyped_dataset(
            args.dataset,
            llmfets_model=args.llmfets_model,
            causal=args.causal,
            timeoh=args.timeoh
        )
        
        if args.to_csv:
            df = convert_to_dataframe(X, node_ids, patterns, max_rows=args.max_rows)
            output_file = f"behavioral_dataset_untyped_{args.dataset}_sample.csv" if args.max_rows else f"behavioral_dataset_untyped_{args.dataset}.csv"
            df.to_csv(output_file, index=False)

