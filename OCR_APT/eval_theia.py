#!/usr/bin/env python3

import os
import sys
import warnings
from contextlib import redirect_stdout
from io import StringIO
import zipfile

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import torch
import pickle
import json
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from graph_learning import (
    build_detector,
    create_base_config,
    get_model_file_name,
    get_mapping_file_name,
    compute_attack_detection_precision,
    prepare_pyg_data
)
from gnn_inference import (
    get_attack_scenarios,
    load_malicious_ids_from_csv,
    build_node_to_dates_mapping,
    prepare_test_data,
    compute_metrics,
    compute_metrics_hybrid,
    run_inference_per_vtype
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_DATES = {
    "THEIA": {
        "train_start_date": "2018-04-03",
        "train_end_date": "2018-04-05",
        "test_start_date": "2018-04-09",
        "test_end_date": "2018-04-12"
    }
}

def load_from_zip_or_pkl(zip_path_part1, zip_path_part2, pkl_path, internal_name):
    if os.path.exists(zip_path_part1) and os.path.exists(zip_path_part2):
        data_part1 = None
        data_part2 = None
        
        with zipfile.ZipFile(zip_path_part1, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                data_part1 = pickle.load(f)
        
        with zipfile.ZipFile(zip_path_part2, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                data_part2 = pickle.load(f)
        
        if isinstance(data_part1, dict) and 'nodes' in data_part1:
            combined_data = {
                'nodes': {},
                'edges': [],
                'graph': None
            }
            combined_data['nodes'].update(data_part1['nodes'])
            combined_data['nodes'].update(data_part2['nodes'])
            combined_data['edges'].extend(data_part1['edges'])
            combined_data['edges'].extend(data_part2['edges'])
            return combined_data
        elif isinstance(data_part1, pd.DataFrame):
            return pd.concat([data_part1, data_part2], ignore_index=True)
        else:
            return data_part1, data_part2
    elif os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Neither zip files ({zip_path_part1}, {zip_path_part2}) nor pkl file ({pkl_path}) found")

def load_vtype_from_zip_or_pkl(base_dir, vtype_safe, file_prefix):
    zip_path_part1 = os.path.join(base_dir, f'{file_prefix}_{vtype_safe}_part1.pkl.zip')
    zip_path_part2 = os.path.join(base_dir, f'{file_prefix}_{vtype_safe}_part2.pkl.zip')
    pkl_path = os.path.join(base_dir, f'{file_prefix}_{vtype_safe}.pkl')
    internal_name = f'{file_prefix}_{vtype_safe}.pkl'
    
    return load_from_zip_or_pkl(zip_path_part1, zip_path_part2, pkl_path, internal_name)

def detect_per_vtype_test_files(base_dir):
    vtypes_list_file = os.path.join(base_dir, 'vtypes_list.pkl')
    
    if os.path.exists(vtypes_list_file):
        with open(vtypes_list_file, 'rb') as f:
            vtype_info = pickle.load(f)
        return vtype_info['vtypes'], vtype_info['vtype_counts']
    
    test_data_pkl = os.path.join(base_dir, 'test_data.pkl')
    test_data_zip_part1 = os.path.join(base_dir, 'test_data_part1.pkl.zip')
    test_data_zip_part2 = os.path.join(base_dir, 'test_data_part2.pkl.zip')
    
    if os.path.exists(test_data_pkl) or (os.path.exists(test_data_zip_part1) and os.path.exists(test_data_zip_part2)):
        return None, None
    
    print(f"No data files found in {base_dir}")
    return None, None

def load_vtype_test_data(base_dir, vtype):
    vtype_safe = vtype.replace('/', '_').replace('+', '_')
    
    test_file_zip_part1 = os.path.join(base_dir, f'test_data_{vtype_safe}_part1.pkl.zip')
    test_file_zip_part2 = os.path.join(base_dir, f'test_data_{vtype_safe}_part2.pkl.zip')
    test_file = os.path.join(base_dir, f'test_data_{vtype_safe}.pkl')
    
    features_file_zip_part1 = os.path.join(base_dir, f'features_test_{vtype_safe}_part1.pkl.zip')
    features_file_zip_part2 = os.path.join(base_dir, f'features_test_{vtype_safe}_part2.pkl.zip')
    features_file = os.path.join(base_dir, f'features_test_{vtype_safe}.pkl')
    
    if os.path.exists(test_file_zip_part1) and os.path.exists(test_file_zip_part2):
        internal_name = f'test_data_{vtype_safe}.pkl'
        test_data_part1 = None
        test_data_part2 = None
        
        with zipfile.ZipFile(test_file_zip_part1, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                test_data_part1 = pickle.load(f)
        
        with zipfile.ZipFile(test_file_zip_part2, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                test_data_part2 = pickle.load(f)
        
        test_data = {
            'nodes': {},
            'edges': [],
            'graph': None
        }
        test_data['nodes'].update(test_data_part1['nodes'])
        test_data['nodes'].update(test_data_part2['nodes'])
        test_data['edges'].extend(test_data_part1['edges'])
        test_data['edges'].extend(test_data_part2['edges'])
    elif os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
    else:
        raise FileNotFoundError(f"Test data file not found for vtype {vtype}")
    
    if os.path.exists(features_file_zip_part1) and os.path.exists(features_file_zip_part2):
        internal_name = f'features_test_{vtype_safe}.pkl'
        features_part1 = None
        features_part2 = None
        
        with zipfile.ZipFile(features_file_zip_part1, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                features_part1 = pickle.load(f)
        
        with zipfile.ZipFile(features_file_zip_part2, 'r') as zf:
            with zf.open(internal_name, 'r') as f:
                features_part2 = pickle.load(f)
        
        features_test = pd.concat([features_part1, features_part2], ignore_index=True)
    elif os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            features_test = pickle.load(f)
    else:
        raise FileNotFoundError(f"Features file not found for vtype {vtype}")
    
    return test_data, features_test

def evaluate_ocrgcn_model(base_dir, model_path, model_config, dataset_dates, dataset, embedding=None, use_per_vtype_files=False, is_hypersearch=False):
    try:
        if not os.path.exists(base_dir):
            return pd.DataFrame()
        
        import traceback
        
        vtypes_from_files, vtype_counts_from_files = detect_per_vtype_test_files(base_dir)
        
        if vtypes_from_files is not None:
            use_per_vtype_files = True
            
            with open(os.path.join(base_dir, 'edge_types.pkl'), 'rb') as f:
                edge_types = pickle.load(f)
            
            test_data = {'nodes': {}, 'edges': []}
            all_test_features = []
            
            for vtype in vtypes_from_files:
                vtype_safe = vtype.replace('/', '_').replace('+', '_')
                vtype_test_data = load_vtype_from_zip_or_pkl(base_dir, vtype_safe, 'test_data')
                vtype_features_test = load_vtype_from_zip_or_pkl(base_dir, vtype_safe, 'features_test')
                test_data['nodes'].update(vtype_test_data['nodes'])
                test_data['edges'].extend(vtype_test_data['edges'])
                all_test_features.append(vtype_features_test)
            
            features_test = pd.concat(all_test_features, ignore_index=True)
        else:
            use_per_vtype_files = False
            
            test_data_zip_part1 = os.path.join(base_dir, 'test_data_part1.pkl.zip')
            test_data_zip_part2 = os.path.join(base_dir, 'test_data_part2.pkl.zip')
            test_data_path = os.path.join(base_dir, 'test_data.pkl')
            
            features_test_zip_part1 = os.path.join(base_dir, 'features_test_part1.pkl.zip')
            features_test_zip_part2 = os.path.join(base_dir, 'features_test_part2.pkl.zip')
            features_test_path = os.path.join(base_dir, 'features_test.pkl')
            
            edge_types_path = os.path.join(base_dir, 'edge_types.pkl')
            
            if not os.path.exists(edge_types_path):
                return pd.DataFrame()
            
            if not ((os.path.exists(test_data_zip_part1) and os.path.exists(test_data_zip_part2)) or os.path.exists(test_data_path)):
                return pd.DataFrame()
            
            if not ((os.path.exists(features_test_zip_part1) and os.path.exists(features_test_zip_part2)) or os.path.exists(features_test_path)):
                return pd.DataFrame()
            
            test_data = load_from_zip_or_pkl(test_data_zip_part1, test_data_zip_part2, test_data_path, 'test_data.pkl')
            features_test = load_from_zip_or_pkl(features_test_zip_part1, features_test_zip_part2, features_test_path, 'features_test.pkl')
            
            with open(edge_types_path, 'rb') as f:
                edge_types = pickle.load(f)
        
        if is_hypersearch:
            model_root = os.path.join(base_dir, 'ocrgcn')
            hypersearch_root = os.path.join(model_root, 'hypersearch_models')
            
            model_8_dirs = glob.glob(os.path.join(hypersearch_root, 'model_8*'))
            has_per_vtype_models = any('_' in os.path.basename(d) for d in model_8_dirs)
            
            if has_per_vtype_models:
                vtypes_for_models = []
                for d in model_8_dirs:
                    basename = os.path.basename(d)
                    if basename.startswith('model_8_'):
                        vtype = basename.replace('model_8_', '')
                        checkpoint_path = os.path.join(d, 'checkpoint_epoch_50.pth')
                        if os.path.exists(checkpoint_path):
                            vtypes_for_models.append(vtype)
                
                if not vtypes_for_models:
                    return pd.DataFrame()
                
                checkpoint = None
                model_config_loaded = model_config.copy()
            else:
                if not os.path.exists(model_path):
                    return pd.DataFrame()
                vtypes_for_models = [None]
                checkpoint = torch.load(model_path, map_location=device)
                if 'config' in checkpoint:
                    model_config_loaded = checkpoint['config'].copy()
                else:
                    model_config_loaded = model_config.copy()
            
            edge_type_mapping = {etype: idx for idx, etype in enumerate(sorted(edge_types))}
            node_id_to_idx = {}
            trained_model_name = 'ocrgcn'
        else:
            has_per_vtype_models = False
            vtypes_for_models = None
            
            model_root = os.path.join(base_dir, 'ocrgcn')
            original_model_dir = os.path.join(model_root, 'original')
            
            from gnn_inference import detect_vtype_models
            vtypes = detect_vtype_models(original_model_dir, 'ocrgcn', legacy_dir=base_dir)
            
            if not vtypes:
                return pd.DataFrame()
            
            trained_model_name = 'ocrgcn'
            model_config_loaded = model_config.copy()
            
            if vtypes[0] is None:
                mapping_path = os.path.join(original_model_dir, get_mapping_file_name('ocrgcn', ''))
                if not os.path.exists(mapping_path):
                    legacy_mapping_path = os.path.join(base_dir, 'mappings.pkl')
                    if os.path.exists(legacy_mapping_path):
                        mapping_path = legacy_mapping_path
                if not os.path.exists(mapping_path):
                    return pd.DataFrame()
                
                with open(mapping_path, 'rb') as f:
                    mappings = pickle.load(f)
                
                edge_type_mapping = mappings['edge_type_mapping']
                node_id_to_idx = mappings['node_id_to_idx']
                trained_model_name = mappings.get('model', 'ocrgcn')
                model_config_loaded = mappings.get('model_config', model_config_loaded)
            else:
                edge_type_mapping = {etype: idx for idx, etype in enumerate(sorted(edge_types))}
                node_id_to_idx = {}
        
        hash_to_uuid_path = os.path.join(base_dir, 'hash_to_uuid.json')
        if os.path.exists(hash_to_uuid_path):
            with open(hash_to_uuid_path, 'r') as f:
                hash_to_uuid = json.load(f)
            uuid_to_hash = {v: k for k, v in hash_to_uuid.items()}
        else:
            uuid_to_hash = None
        
        node_to_dates = build_node_to_dates_mapping(test_data)
        
        if is_hypersearch and has_per_vtype_models and vtypes_for_models and vtypes_for_models[0] is not None:
            from graph_learning import prepare_pyg_data
            all_scores = {}
            all_node_ids_list = []
            
            for vtype in vtypes_for_models:
                vtype_safe = vtype.replace('/', '_').replace('+', '_')
                vtype_test_data = load_vtype_from_zip_or_pkl(base_dir, vtype_safe, 'test_data')
                vtype_features_test = load_vtype_from_zip_or_pkl(base_dir, vtype_safe, 'features_test')
                
                vtype_model_path = os.path.join(hypersearch_root, f'model_8_{vtype}', 'checkpoint_epoch_50.pth')
                if not os.path.exists(vtype_model_path):
                    continue
                
                data_vtype, _, node_ids_vtype = prepare_pyg_data(
                    vtype_test_data, vtype_features_test, edge_types
                )
                
                vtype_checkpoint = torch.load(vtype_model_path, map_location=device)
                if 'config' in vtype_checkpoint:
                    vtype_config = vtype_checkpoint['config'].copy()
                else:
                    vtype_config = model_config_loaded.copy()
                
                model = build_detector(
                    trained_model_name,
                    vtype_config,
                    data_vtype.x.shape[1],
                    len(edge_type_mapping),
                    device
                )
                model.load_model(vtype_model_path)
                
                target_mask = data_vtype.target_mask if hasattr(data_vtype, 'target_mask') else None
                scores_vtype = model.predict(data_vtype.x, data_vtype.edge_index, data_vtype.edge_type, target_mask=target_mask)
                
                for idx, node_id in enumerate(node_ids_vtype):
                    all_scores[node_id] = scores_vtype[idx]
                    all_node_ids_list.append(node_id)
                
                del model, data_vtype
                torch.cuda.empty_cache()
            
            node_ids = sorted(set(all_node_ids_list))
            scores = np.array([all_scores.get(nid, 0.0) for nid in node_ids])
        elif is_hypersearch:
            all_malicious = set()
            data, _, node_ids = prepare_test_data(
                test_data, features_test, edge_types,
                edge_type_mapping, node_id_to_idx, all_malicious
            )
            
            model = build_detector(
                trained_model_name,
                model_config_loaded,
                data.x.shape[1],
                len(edge_type_mapping),
                device
            )
            model.load_model(model_path)
            
            target_mask = data.target_mask if hasattr(data, 'target_mask') else None
            scores = model.predict(data.x, data.edge_index, data.edge_type, target_mask=target_mask)
        else:
            from gnn_inference import detect_vtype_models
            model_root = os.path.join(base_dir, 'ocrgcn')
            original_model_dir = os.path.join(model_root, 'original')
            vtypes = detect_vtype_models(original_model_dir, 'ocrgcn', legacy_dir=base_dir)
            
            if vtypes[0] is not None:
                class Args:
                    def __init__(self):
                        self.model = trained_model_name
                        self.device = device
                        self.contamination = model_config_loaded.get('contamination', 0.001)
                        self.rulellm = False
                        self.llmlabel = False
                        self.llmfunc = False
                        self.hid_dim = model_config_loaded.get('hid_dim', 32)
                        self.num_layers = model_config_loaded.get('num_layers', 3)
                        self.dropout = model_config_loaded.get('dropout', 0.0)
                        self.lr = model_config_loaded.get('lr', 0.005)
                        self.epoch = model_config_loaded.get('epoch', 100)
                        self.beta = model_config_loaded.get('beta', 0.5)
                        self.warmup = model_config_loaded.get('warmup', 2)
                        self.eps = model_config_loaded.get('eps', 0.1)
                
                with redirect_stdout(StringIO()):
                    scores, node_ids = run_inference_per_vtype(
                        base_dir, edge_types, original_model_dir, base_dir, vtypes, 
                        Args(), trained_model_name, use_per_vtype_files
                    )
                
                if test_data is None or 'nodes' not in test_data or not test_data['nodes']:
                    test_data = {'nodes': {}, 'edges': []}
                    if use_per_vtype_files:
                        for vtype_load in vtypes:
                            if vtype_load is not None:
                                vtype_safe = vtype_load.replace('/', '_').replace('+', '_')
                                vtype_test_data = load_vtype_from_zip_or_pkl(base_dir, vtype_safe, 'test_data')
                            else:
                                vtype_test_data = load_from_zip_or_pkl(
                                    os.path.join(base_dir, 'test_data_part1.pkl.zip'),
                                    os.path.join(base_dir, 'test_data_part2.pkl.zip'),
                                    os.path.join(base_dir, 'test_data.pkl'),
                                    'test_data.pkl'
                                )
                            test_data['nodes'].update(vtype_test_data['nodes'])
                            test_data['edges'].extend(vtype_test_data['edges'])
            else:
                all_malicious = set()
                data, _, node_ids = prepare_test_data(
                    test_data, features_test, edge_types,
                    edge_type_mapping, node_id_to_idx, all_malicious
                )
                
                model = build_detector(
                    trained_model_name,
                    model_config_loaded,
                    data.x.shape[1],
                    len(edge_type_mapping),
                    device
                )
                
                if not os.path.exists(model_path):
                    legacy_model_path = os.path.join(base_dir, 'ocrgcn_model.pth')
                    if os.path.exists(legacy_model_path):
                        model_path = legacy_model_path
                
                if not os.path.exists(model_path):
                    return pd.DataFrame()
                
                model.load_model(model_path)
                
                target_mask = data.target_mask if hasattr(data, 'target_mask') else None
                scores = model.predict(data.x, data.edge_index, data.edge_type, target_mask=target_mask)
        
        attack_scenarios = get_attack_scenarios(dataset, '../PIDS_GT')
        if not attack_scenarios:
            return pd.DataFrame()
        
        node_id_to_test_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        results_list = []
        
        for attack_name, csv_path, attack_test_date in attack_scenarios:
            if not os.path.exists(csv_path):
                continue
            
            GT_mal_uuids = load_malicious_ids_from_csv(csv_path)
            
            if uuid_to_hash:
                GT_mal_hashes = set()
                for uuid in GT_mal_uuids:
                    if uuid in uuid_to_hash:
                        GT_mal_hashes.add(uuid_to_hash[uuid])
                GT_mal_in_test = GT_mal_hashes.intersection(set(node_ids))
            else:
                GT_mal_in_test = GT_mal_uuids.intersection(set(node_ids))
            
            labels = np.zeros(len(node_ids), dtype=int)
            malicious_count = 0
            
            for node_uuid in GT_mal_in_test:
                if node_uuid in node_to_dates and attack_test_date in node_to_dates[node_uuid]:
                    test_idx = node_id_to_test_idx[node_uuid]
                    labels[test_idx] = 1
                    malicious_count += 1
            
            if malicious_count == 0:
                continue
            
            if is_hypersearch and has_per_vtype_models and vtypes_for_models and vtypes_for_models[0] is not None:
                metrics = compute_metrics_hybrid(labels, scores, node_ids, test_data, contamination=model_config_loaded.get('contamination', 0.001))
            else:
                contamination_value = model_config_loaded.get('contamination', 0.001)
                metrics = compute_metrics(labels, scores, contamination=contamination_value)
            
            adp_value = compute_attack_detection_precision(scores, {attack_name: labels})
            metrics['adp'] = adp_value if adp_value is not None else 0.0
            
            results_list.append({
                'attack_name': attack_name,
                'test_date': attack_test_date,
                'AUC_ROC': float(metrics['auc_roc']),
                'AUC_PR': float(metrics['auc_pr']),
                'ADP': float(metrics['adp']),
                'malicious_nodes': int(malicious_count)
            })
        
        return pd.DataFrame(results_list)
    except Exception as e:
        import traceback
        print(f"Error in evaluate_ocrgcn_model for {base_dir}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def main():
    dataset = "THEIA"
    dataset_dates = DATASET_DATES[dataset]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    input_dir = os.path.join(autoprov_dir, 'BIGDATA', 'OCR_APT_artifacts')
    
    try:
        base_dir_baseline = os.path.join(input_dir, dataset.lower())
        model_path_baseline = os.path.join(base_dir_baseline, 'ocrgcn', 'original', 'ocrgcn_model.pth')
        
        baseline_config = {
            'hid_dim': 32,
            'num_layers': 3,
            'dropout': 0.0,
            'lr': 0.005,
            'epoch': 100,
            'beta': 0.5,
            'contamination': 0.001,
            'warmup': 2,
            'eps': 0.1
        }
        
        if not os.path.exists(base_dir_baseline):
            print(f"Warning: Baseline artifacts directory not found: {base_dir_baseline}")
            print("Skipping baseline evaluation.")
            results_baseline = pd.DataFrame()
        else:
            results_baseline = evaluate_ocrgcn_model(
                base_dir_baseline, model_path_baseline, baseline_config,
                dataset_dates, dataset, embedding=None, use_per_vtype_files=False
            )
        
        embedding = 'mpnet'
        base_dir_model8 = os.path.join(input_dir, f'{dataset.lower()}_rulellm_llmlabel_{embedding}')
        model_dir_model8 = os.path.join(base_dir_model8, 'ocrgcn', 'hypersearch_models', 'model_8')
        model_path_model8 = os.path.join(model_dir_model8, 'checkpoint_epoch_50.pth')
        
        model8_config = {
            'hid_dim': 32,
            'num_layers': 3,
            'dropout': 0.1,
            'lr': 0.005,
            'epoch': 50,
            'beta': 0.5,
            'contamination': 0.001,
            'warmup': 10,
            'eps': 0.1
        }
        
        if not os.path.exists(base_dir_model8):
            print(f"Warning: AutoProv artifacts directory not found: {base_dir_model8}")
            print("Skipping AutoProv evaluation.")
            results_model8 = pd.DataFrame()
        else:
            results_model8 = evaluate_ocrgcn_model(
                base_dir_model8, model_path_model8, model8_config,
                dataset_dates, dataset, embedding=embedding, use_per_vtype_files=True, is_hypersearch=True
            )
        
        if not results_baseline.empty:
            display_cols = ['attack_name', 'AUC_ROC', 'AUC_PR', 'ADP']
            baseline_display = results_baseline[display_cols].copy()
            baseline_display['AUC_ROC'] = baseline_display['AUC_ROC'].round(3)
            baseline_display['AUC_PR'] = baseline_display['AUC_PR'].round(3)
            baseline_display['ADP'] = baseline_display['ADP'].round(3)
            print("\nBaseline Model Results:")
            print(baseline_display.to_string(index=False))
        else:
            print("\nNo baseline results to display.")
        
        if not results_model8.empty:
            display_cols = ['attack_name', 'AUC_ROC', 'AUC_PR', 'ADP']
            model8_display = results_model8[display_cols].copy()
            model8_display['AUC_ROC'] = model8_display['AUC_ROC'].round(3)
            model8_display['AUC_PR'] = model8_display['AUC_PR'].round(3)
            model8_display['ADP'] = model8_display['ADP'].round(3)
            print("\n\nAuto-Prov Results:")
            print(model8_display.to_string(index=False))
        else:
            print("\nNo AutoProv results to display.")
    
    except Exception as e:
        import traceback
        print(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

