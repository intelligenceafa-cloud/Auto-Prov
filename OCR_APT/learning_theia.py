#!/usr/bin/env python3

import os
import sys
import argparse
import pickle
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from graph_learning import (
    build_detector,
    create_base_config,
    get_model_file_name,
    get_mapping_file_name,
    detect_per_vtype_files,
    load_vtype_data,
    split_data_by_vtype,
    prepare_pyg_data,
    get_unique_vtypes,
    train_model_simple
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model_8(base_dir, hypersearch_root, train_data, features_train, edge_types, 
                  vtypes_from_files=None, vtype_counts_from_files=None, 
                  embedding='mpnet', device='cuda', max_epochs=50):
    model_id = 8
    config = {
        'model_id': 8,
        'name': 'with_dropout',
        'hid_dim': 32,
        'num_layers': 3,
        'dropout': 0.1,
        'lr': 0.005,
        'beta': 0.5,
        'contamination': 0.001,
        'warmup': 10,
        'eps': 0.1,
        'epoch': max_epochs
    }
    
    if vtypes_from_files is not None:
        vtypes, vtype_counts = vtypes_from_files, vtype_counts_from_files
        use_per_vtype_files = True
    else:
        vtypes, vtype_counts = get_unique_vtypes(train_data)
        use_per_vtype_files = False
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL 8 (with_dropout) - Epochs: {max_epochs}")
    print(f"{'='*80}")
    print(f"Node Types: {len(vtypes)} ({', '.join([f'{v}: {vtype_counts[v]:,}' for v in vtypes[:5]])}...)")
    print(f"{'='*80}\n")
    
    os.makedirs(hypersearch_root, exist_ok=True)
    
    for vtype in vtypes:
        if vtype is not None:
            model_name = f"model_{model_id}_{vtype}"
            model_dir = os.path.join(hypersearch_root, model_name)
        else:
            model_name = f"model_{model_id}"
            model_dir = os.path.join(hypersearch_root, model_name)
        
        print(f"\n{'='*60}")
        print(f"Model {model_id} ({config['name']}) - VType: {vtype}")
        print(f"{'='*60}")
        
        if use_per_vtype_files:
            print(f"  Loading per-vtype data...", end=' ', flush=True)
            vtype_train_data, vtype_features_train, _ = load_vtype_data(base_dir, vtype)
            print(f"✓ {len(vtype_train_data['nodes'])} nodes, {len(vtype_train_data['edges'])} edges")
        else:
            vtype_train_data, vtype_features_train, num_nodes = split_data_by_vtype(
                train_data, features_train, edge_types, vtype
            )
            print(f"  Filtered to {num_nodes:,} nodes of type '{vtype}'")
        
        os.makedirs(model_dir, exist_ok=True)
        
        config_with_vtype = config.copy()
        config_with_vtype['vtype'] = vtype
        config_with_vtype['model'] = 'ocrgcn'
        hyperparams_path = os.path.join(model_dir, 'hyperparams.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(config_with_vtype, f, indent=2)
        
        data_train_vtype, edge_type_mapping_vtype, node_id_to_idx_vtype = prepare_pyg_data(
            vtype_train_data, vtype_features_train, edge_types
        )
        
        if data_train_vtype.num_nodes == 0:
            print(f"  Skipping - no nodes of type '{vtype}'")
            continue
        
        print(f"  PyG graph: {data_train_vtype.num_nodes:,} nodes, {data_train_vtype.num_edges:,} edges")
        
        train_model_simple(
            'ocrgcn',
            dict(config),
            data_train_vtype.x,
            data_train_vtype.edge_index,
            data_train_vtype.edge_type,
            max_epochs,
            model_dir,
            device,
            len(edge_type_mapping_vtype),
            target_mask=getattr(data_train_vtype, 'target_mask', None)
        )
        
        del data_train_vtype
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"Model 8 training complete!")
    print(f"{'='*80}\n")

def parse_args():
    parser = argparse.ArgumentParser(description='OCR_APT Graph Learning for THEIA')
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--baseline', action='store_true',
                           help='Train baseline model (single model, no hyperparameter search)')
    mode_group.add_argument('--autoprov', action='store_true',
                           help='Train model 8 only (autoprov mode, similar to --rulellm --llmlabel)')
    
    parser.add_argument('--embedding', type=str, default='mpnet',
                       choices=['mpnet', 'minilm', 'roberta', 'distilbert'],
                       help='Embedding model to use for autoprov mode (default: mpnet)')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    parser.add_argument('--gpus', type=str, default='',
                       help='Comma-separated GPU ids (e.g. 0,1,2); default empty = use all available')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.gpus and args.gpus.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    artifacts_root = os.path.join(autoprov_dir, 'BIGDATA', 'OCR_APT_artifacts')
    
    dataset = 'theia'
    
    if args.baseline:
        base_dir = os.path.join(artifacts_root, dataset)
        model_root = os.path.join(base_dir, 'ocrgcn')
        output_dir = os.path.join(model_root, 'original')
        
        print(f"\n{'='*80}")
        print(f"OCR_APT Graph Learning - THEIA (BASELINE)")
        print(f"Mode: Baseline (Regex mode, OCR-APT features)")
        print(f"Device: {args.device}")
        print(f"{'='*80}\n")
        
        vtypes_from_files, vtype_counts_from_files = detect_per_vtype_files(base_dir)
        
        if vtypes_from_files is not None:
            use_per_vtype_files = True
            print(f"✓ Detected per-vtype data files ({len(vtypes_from_files)} vtypes)")
            print(f"  {', '.join([f'{v}: {vtype_counts_from_files[v]:,}' for v in vtypes_from_files[:5]])}...")
            
            with open(os.path.join(base_dir, 'edge_types.pkl'), 'rb') as f:
                edge_types = pickle.load(f)
            
            train_data = None
            features_train = None
            vtypes = vtypes_from_files
            vtype_counts = vtype_counts_from_files
        else:
            use_per_vtype_files = False
            print("Loading training data...", end=' ', flush=True)
            
            train_data_zip_part1 = os.path.join(base_dir, 'train_data_part1.pkl.zip')
            train_data_zip_part2 = os.path.join(base_dir, 'train_data_part2.pkl.zip')
            train_data_path = os.path.join(base_dir, 'train_data.pkl')
            
            features_train_zip_part1 = os.path.join(base_dir, 'features_train_part1.pkl.zip')
            features_train_zip_part2 = os.path.join(base_dir, 'features_train_part2.pkl.zip')
            features_train_path = os.path.join(base_dir, 'features_train.pkl')
            
            import zipfile
            import pandas as pd
            
            if os.path.exists(train_data_zip_part1) and os.path.exists(train_data_zip_part2):
                with zipfile.ZipFile(train_data_zip_part1, 'r') as zf:
                    with zf.open('train_data.pkl', 'r') as f:
                        train_data_part1 = pickle.load(f)
                with zipfile.ZipFile(train_data_zip_part2, 'r') as zf:
                    with zf.open('train_data.pkl', 'r') as f:
                        train_data_part2 = pickle.load(f)
                train_data = {
                    'nodes': {},
                    'edges': [],
                    'graph': None
                }
                train_data['nodes'].update(train_data_part1['nodes'])
                train_data['nodes'].update(train_data_part2['nodes'])
                train_data['edges'].extend(train_data_part1['edges'])
                train_data['edges'].extend(train_data_part2['edges'])
            elif os.path.exists(train_data_path):
                with open(train_data_path, 'rb') as f:
                    train_data = pickle.load(f)
            else:
                raise FileNotFoundError(f"Training data not found in {base_dir}")
            
            if os.path.exists(features_train_zip_part1) and os.path.exists(features_train_zip_part2):
                with zipfile.ZipFile(features_train_zip_part1, 'r') as zf:
                    with zf.open('features_train.pkl', 'r') as f:
                        features_train_part1 = pickle.load(f)
                with zipfile.ZipFile(features_train_zip_part2, 'r') as zf:
                    with zf.open('features_train.pkl', 'r') as f:
                        features_train_part2 = pickle.load(f)
                features_train = pd.concat([features_train_part1, features_train_part2], ignore_index=True)
            elif os.path.exists(features_train_path):
                with open(features_train_path, 'rb') as f:
                    features_train = pickle.load(f)
            else:
                raise FileNotFoundError(f"Features train not found in {base_dir}")
            
            with open(os.path.join(base_dir, 'edge_types.pkl'), 'rb') as f:
                edge_types = pickle.load(f)
            
            print(f"✓ {len(train_data['nodes'])} nodes, {len(train_data['edges'])} edges, {len(edge_types)} edge_types")
            
            vtypes, vtype_counts = get_unique_vtypes(train_data)
        
        os.makedirs(output_dir, exist_ok=True)
        
        max_epochs = args.epochs if args.epochs is not None else 100
        
        print(f"\n{'='*80}")
        print(f"TRAINING SEPARATE MODELS PER VTYPE")
        print(f"{'='*80}")
        print(f"Node Types: {len(vtypes)} ({', '.join([f'{v}: {vtype_counts[v]:,}' for v in vtypes])})")
        print(f"Total Models: {len(vtypes)}")
        print(f"Epochs: {max_epochs}")
        print(f"{'='*80}\n")
        
        for vtype in vtypes:
            print(f"\n{'='*60}")
            print(f"Training model for VType: {vtype}")
            print(f"{'='*60}")
            
            if use_per_vtype_files:
                print(f"  Loading per-vtype data...", end=' ', flush=True)
                vtype_train_data, vtype_features_train, _ = load_vtype_data(base_dir, vtype)
                print(f"✓ {len(vtype_train_data['nodes'])} nodes, {len(vtype_train_data['edges'])} edges")
            else:
                vtype_train_data, vtype_features_train, num_target_nodes = split_data_by_vtype(
                    train_data, features_train, edge_types, vtype
                )
                print(f"  Target nodes: {num_target_nodes:,} of type '{vtype}' (full graph preserved for message passing)")
            
            model_suffix = f"_{vtype}"
            
            print("  Preparing PyG graph...", end=' ', flush=True)
            data, edge_type_mapping, node_id_to_idx = prepare_pyg_data(
                vtype_train_data, vtype_features_train, edge_types
            )
            
            if data.num_nodes == 0:
                print(f"  ⚠ Skipping - no nodes")
                continue
            
            if data.num_edges == 0:
                print(f"  ⚠ Skipping - no edges (vtype has only isolated nodes)")
                continue
            
            print(f"✓ {data.num_nodes:,} nodes, {data.num_edges:,} edges, {len(edge_type_mapping)} relations")
            
            class Args:
                def __init__(self):
                    self.rulellm = False
                    self.llmlabel = False
                    self.llmfunc = False
                    self.hid_dim = 32
                    self.num_layers = 3
                    self.dropout = 0.0
                    self.lr = 0.005
                    self.epoch = max_epochs
                    self.beta = 0.5
                    self.contamination = 0.001
                    self.warmup = 2
                    self.eps = 0.1
            
            args_obj = Args()
            detector_config = create_base_config(args_obj)
            model_config = detector_config.copy()
            
            print(f"  Initializing OCRGCN...", end=' ', flush=True)
            model = build_detector(
                'ocrgcn',
                detector_config,
                data.x.shape[1],
                len(edge_type_mapping),
                args.device
            )
            print(f"✓ (dim: {data.x.shape[1]}→{detector_config['hid_dim']}, layers: {detector_config['num_layers']})")
            
            target_mask = data.target_mask if hasattr(data, 'target_mask') else None
            model.fit(data.x, data.edge_index, data.edge_type, target_mask=target_mask)
            
            print(f"  Saving model...", end=' ', flush=True)
            model_path = os.path.join(output_dir, get_model_file_name('ocrgcn', model_suffix))
            model.save_model(model_path)
            
            mappings = {
                'edge_type_mapping': edge_type_mapping,
                'node_id_to_idx': node_id_to_idx,
                'vtype': vtype,
                'model': 'ocrgcn',
                'model_config': model_config
            }
            mappings_path = os.path.join(output_dir, get_mapping_file_name('ocrgcn', model_suffix))
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings, f)
            print(f"✓ Saved to {output_dir}/\n")
            
            del model, data
            torch.cuda.empty_cache()
        
        print(f"{'='*80}")
        print(f"Baseline training completed successfully!")
        print(f"{'='*80}\n")
    
    elif args.autoprov:
        embedding = args.embedding.lower()
        base_dir = os.path.join(artifacts_root, f'{dataset}_rulellm_llmlabel_{embedding}')
        model_root = os.path.join(base_dir, 'ocrgcn')
        hypersearch_root = os.path.join(model_root, 'hypersearch_models')
        
        print(f"\n{'='*80}")
        print(f"OCR_APT Graph Learning - THEIA (AUTOPROV)")
        print(f"Mode: RuleLLM + LLM Type Embeddings ({embedding})")
        print(f"Model: 8 (with_dropout) | Epochs: 50")
        print(f"Device: {args.device}")
        print(f"{'='*80}\n")
        
        vtypes_from_files, vtype_counts_from_files = detect_per_vtype_files(base_dir)
        
        if vtypes_from_files is not None:
            use_per_vtype_files = True
            print(f"✓ Detected per-vtype data files ({len(vtypes_from_files)} vtypes)")
            print(f"  {', '.join([f'{v}: {vtype_counts_from_files[v]:,}' for v in vtypes_from_files[:5]])}...")
            
            with open(os.path.join(base_dir, 'edge_types.pkl'), 'rb') as f:
                edge_types = pickle.load(f)
            
            train_data = None
            features_train = None
        else:
            use_per_vtype_files = False
            print("Loading training data...", end=' ', flush=True)
            
            train_data_zip_part1 = os.path.join(base_dir, 'train_data_part1.pkl.zip')
            train_data_zip_part2 = os.path.join(base_dir, 'train_data_part2.pkl.zip')
            train_data_path = os.path.join(base_dir, 'train_data.pkl')
            
            features_train_zip_part1 = os.path.join(base_dir, 'features_train_part1.pkl.zip')
            features_train_zip_part2 = os.path.join(base_dir, 'features_train_part2.pkl.zip')
            features_train_path = os.path.join(base_dir, 'features_train.pkl')
            
            import zipfile
            import pandas as pd
            
            if os.path.exists(train_data_zip_part1) and os.path.exists(train_data_zip_part2):
                with zipfile.ZipFile(train_data_zip_part1, 'r') as zf:
                    with zf.open('train_data.pkl', 'r') as f:
                        train_data_part1 = pickle.load(f)
                with zipfile.ZipFile(train_data_zip_part2, 'r') as zf:
                    with zf.open('train_data.pkl', 'r') as f:
                        train_data_part2 = pickle.load(f)
                train_data = {
                    'nodes': {},
                    'edges': [],
                    'graph': None
                }
                train_data['nodes'].update(train_data_part1['nodes'])
                train_data['nodes'].update(train_data_part2['nodes'])
                train_data['edges'].extend(train_data_part1['edges'])
                train_data['edges'].extend(train_data_part2['edges'])
            elif os.path.exists(train_data_path):
                with open(train_data_path, 'rb') as f:
                    train_data = pickle.load(f)
            else:
                raise FileNotFoundError(f"Training data not found in {base_dir}")
            
            if os.path.exists(features_train_zip_part1) and os.path.exists(features_train_zip_part2):
                with zipfile.ZipFile(features_train_zip_part1, 'r') as zf:
                    with zf.open('features_train.pkl', 'r') as f:
                        features_train_part1 = pickle.load(f)
                with zipfile.ZipFile(features_train_zip_part2, 'r') as zf:
                    with zf.open('features_train.pkl', 'r') as f:
                        features_train_part2 = pickle.load(f)
                features_train = pd.concat([features_train_part1, features_train_part2], ignore_index=True)
            elif os.path.exists(features_train_path):
                with open(features_train_path, 'rb') as f:
                    features_train = pickle.load(f)
            else:
                raise FileNotFoundError(f"Features train not found in {base_dir}")
            
            with open(os.path.join(base_dir, 'edge_types.pkl'), 'rb') as f:
                edge_types = pickle.load(f)
            
            print(f"✓ {len(train_data['nodes'])} nodes, {len(train_data['edges'])} edges, {len(edge_types)} edge_types")
        
        max_epochs = args.epochs if args.epochs is not None else 50
        
        train_model_8(
            base_dir, hypersearch_root,
            train_data, features_train, edge_types,
            vtypes_from_files, vtype_counts_from_files,
            embedding=embedding,
            device=args.device,
            max_epochs=max_epochs
        )
        
        print(f"{'='*80}")
        print(f"AutoProv training completed successfully!")
        print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

