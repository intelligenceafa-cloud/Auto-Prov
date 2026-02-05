#!/usr/bin/env python3

import os
import json
import argparse
import ast
import re
import numpy as np
from pathlib import Path
from glob import glob
from collections import defaultdict
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import save_npz, hstack, csr_matrix
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract behavioral profiles from graph data"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="../rule_generator/ATLAS/ablation/autoprov_atlas_graph",
        help="Path to the directory containing timestamped graph CSV files (or zip files for ATLAS)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["theia", "fivedirections", "atlas"],
        help="Dataset name (theia, fivedirections, or atlas)"
    )
    
    parser.add_argument(
        "--llmfets-model",
        type=str,
        required=True,
        help="LLM model name used for feature extraction (e.g., llama3:70b, gpt-4o)"
    )
    
    parser.add_argument(
        "--cee",
        type=str,
        default=None,
        help="CEE model name (required for ATLAS dataset, e.g., llama3:70b, gpt-4o)"
    )
    
    parser.add_argument(
        "--rule-generator",
        type=str,
        default=None,
        help="Rule generator model name (required for ATLAS dataset, e.g., llama3:70b, qwen2:72b)"
    )
    
    parser.add_argument(
        "--script-dir",
        type=str,
        default="./",
        help="Directory where this script and reference files are located"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default='2018-04-03',
        help="Start date to filter CSV files (format: YYYY-MM-DD). Not used for ATLAS dataset."
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default='2018-04-12',
        help="End date to filter CSV files (format: YYYY-MM-DD). Not used for ATLAS dataset."
    )
    
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Build causal profiles with full forward path traversal (includes all downstream nodes)"
    )
    
    parser.add_argument(
        "--timeoh",
        action="store_true",
        help="Build one-hop profiles per-timestamp (temporal isolation for one-hop patterns)"
    )
    
    return parser.parse_args()


_VALID_EXTENSIONS = None


def _load_valid_extensions(extension_list_path=None):
    global _VALID_EXTENSIONS
    
    if _VALID_EXTENSIONS is not None:
        return _VALID_EXTENSIONS
    
    if extension_list_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        extension_list_path = os.path.join(script_dir, 'extensionlist', 'Filename extension list')
    
    valid_extensions = set()
    
    try:
        with open(extension_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                ext = line.strip()
                if ext and ext.startswith('.'):
                    valid_extensions.add(ext.lower())
                    valid_extensions.add(ext)
    except FileNotFoundError:
        pass
    _VALID_EXTENSIONS = valid_extensions
    return valid_extensions


def _is_valid_extension(extension):
    valid_extensions = _load_valid_extensions()
    
    if not extension:
        return False
    
    if not extension.startswith('.'):
        extension = '.' + extension
    
    return extension in valid_extensions or extension.lower() in valid_extensions


def _remove_invalid_extension(filename):
    if '.' not in filename:
        return filename
    
    name, ext = os.path.splitext(filename)
    
    if len(ext) < 6:
        return filename
    
    if _is_valid_extension(ext):
        return filename
    else:
        return name


def load_reference_files(script_dir: str, dataset: str, llmfets_model: str) -> Tuple[Dict, Dict, Dict]:
    dataset_lower = dataset.lower()
    
    validity_path = os.path.join(
        script_dir, 
        "file_classification_results",
        llmfets_model,
        f"ename_validity_{dataset_lower}.json"
    )
    with open(validity_path, 'r') as f:
        validity_dict = json.load(f)
    
    fets_path = os.path.join(
        script_dir,
        "llm-fets",
        llmfets_model,
        f"ename_fets_{dataset_lower}.json"
    )

    try:
        with open(fets_path, 'r', encoding='utf-8') as f:
            content = f.read()
            content = content.replace(',}', '}').replace(',]', ']')
            fets_dict = json.loads(content)
    except json.JSONDecodeError:
        fets_dict = {}
        with open(fets_path, 'r', encoding='utf-8') as f:
            try:
                lines = f.readlines()
                partial_json = ''.join(lines[:int(len(lines) * 0.9)])
                if not partial_json.strip().endswith('}'):
                    partial_json = partial_json.rsplit(',', 1)[0] + '\n}'
                fets_dict = json.loads(partial_json)
            except:
                fets_dict = {}
    
    edge_type_path = os.path.join(
        script_dir,
        f"edge_type_validation_{dataset_lower}.json"
    )
    
    try:
        with open(edge_type_path, 'r') as f:
            edge_type_dict = json.load(f)
    except FileNotFoundError:
        edge_type_dict = {}
    
    return validity_dict, fets_dict, edge_type_dict


def parse_ename_list(ename_str: str) -> List[str]:
    if not ename_str or pd.isna(ename_str):
        return []
    try:
        parsed = ast.literal_eval(ename_str)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return []
    except (ValueError, SyntaxError):
        return []

def is_ip_address(ename: str) -> bool:
    if not ename or pd.isna(ename):
        return False
    
    ename_str = str(ename).strip()
    if not ename_str:
        return False
    
    if ename_str.startswith('[') and ']:' in ename_str:
        ip_part = ename_str.split(']:')[0][1:]
    elif ':' in ename_str and not ename_str.startswith('http'):
        parts = ename_str.rsplit(':', 1)
        if len(parts) == 2 and parts[1].isdigit():
            ip_part = parts[0]
        else:
            ip_part = ename_str
    else:
        ip_part = ename_str
    
    ipv4_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    if ipv4_pattern.match(ip_part):
        try:
            octets = ip_part.split('.')
            if all(0 <= int(octet) <= 255 for octet in octets):
                return True
        except ValueError:
            pass
    
    if '::' in ip_part or (ip_part.count(':') >= 2 and all(c in '0123456789abcdefABCDEF:' for c in ip_part)):
        if re.match(r'^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$', ip_part) or '::' in ip_part:
            return True
    
    alphanumeric_chars = sum(1 for c in ename_str if c.isalnum() or c in '.:[]')
    letter_chars = sum(1 for c in ename_str if c.isalpha())
    if alphanumeric_chars > 0:
        letter_ratio = letter_chars / len(ename_str) if len(ename_str) > 0 else 0
        if letter_ratio < 0.2 and ('.' in ename_str or ':' in ename_str):
            digit_dot_colon_chars = sum(1 for c in ename_str if c.isdigit() or c in '.:')
            if digit_dot_colon_chars / len(ename_str) > 0.7:
                return True
    
    return False


def parse_atlas_ename(ename_str: str) -> List[str]:
    if not ename_str or pd.isna(ename_str) or ename_str == '':
        return []
    
    if is_ip_address(ename_str):
        return []
    
    return [str(ename_str)]


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
        basename_no_invalid_ext = _remove_invalid_extension(basename)
        
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
        
        processed_filename = _process_filename_and_extension(basename_no_invalid_ext)
        
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


def filter_csv_files_by_date(
    csv_files: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[str]:
    if not start_date and not end_date:
        return csv_files
    
    filtered_files = []
    
    for csv_file in csv_files:
        folder_name = os.path.basename(os.path.dirname(csv_file))
        
        try:
            if '_' in folder_name:
                timestamp_str = folder_name.split('_')[0]
                file_date = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                file_date_only = file_date.date()
                
                if start_date:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
                    if file_date_only < start_dt:
                        continue
                
                if end_date:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
                    if file_date_only > end_dt:
                        continue
                
                filtered_files.append(csv_file)
        except (ValueError, IndexError):
            continue
    
    return filtered_files


def process_node_enames(enames: List[str], validity_dict: Dict, fets_dict: Dict) -> Tuple[Optional[str], Optional[str], bool]:
    if not enames:
        return None, None, True
    
    types = []
    functionalities = []
    filtered_enames = []
    
    for ename in enames:
        normalized = normalize_ename(ename)
        
        if normalized not in validity_dict:
            continue
        
        classification = validity_dict[normalized]
        
        if classification == "INVALID":
            alpha_count = sum(1 for c in normalized if c.isalpha())
            if alpha_count < 3:
                continue
            
            if '/' not in normalized or '.' not in normalized:
                continue
        
        filtered_enames.append(ename)
        
        if classification != "INVALID" and normalized in fets_dict:
            ename_type = fets_dict[normalized].get("Type", "")
            if ename_type and ename_type.upper() not in ["NO LABEL", "NO_LABEL"]:
                types.append(ename_type.lower())
            
            func = fets_dict[normalized].get("Functionality", "")
            if func:
                functionalities.append(func)
    
    if not filtered_enames:
        return None, None, True
    
    class_label = None
    if types:
        unique_types = list(dict.fromkeys(types))
        class_label = "|".join(unique_types)
    
    functionality = None
    if functionalities:
        unique_funcs = list(dict.fromkeys(functionalities))
        functionality = "|".join(unique_funcs)
    
    return class_label, functionality, False


def build_causal_paths(node_id: str, graph: Dict, id_to_class: Dict, max_nodes: int = 1000) -> Set[str]:
    patterns = set()
    visited = {node_id}
    queue = [(node_id, [])]
    nodes_explored = 0
    
    while queue and nodes_explored < max_nodes:
        current, path = queue.pop(0)
        nodes_explored += 1
        
        for neighbor_id, action in graph.get(current, []):
            if neighbor_id not in id_to_class:
                continue
            
            neighbor_type = id_to_class[neighbor_id]
            new_path = path + [(action, neighbor_type)]
            pattern_str = "|".join([f"{act}_{ntype}" for act, ntype in new_path])
            patterns.add(pattern_str)
            
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.append((neighbor_id, new_path))
    
    return patterns


def find_atlas_csv_files(data_path: str, cee: str, rule_generator: str) -> List[str]:
    import zipfile
    
    csv_files = []
    
    cee_normalized = cee.lower().replace(':', '_')
    rule_generator_normalized = rule_generator.lower().replace(':', '_')
    zip_filename = f"{cee_normalized}_{rule_generator_normalized}.zip"
    zip_file_path = os.path.join(data_path, zip_filename)
    
    if not os.path.exists(zip_file_path):
        return []
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for file_path in file_list:
                if ('/train/' in file_path or '/test/' in file_path) and file_path.endswith('/graph.csv'):
                    csv_files.append((zip_file_path, file_path))
    except Exception as e:
        return []
    
    return csv_files


def load_csv_from_zip(zip_file: str, path_in_zip: str) -> pd.DataFrame:
    import zipfile
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            with zip_ref.open(path_in_zip) as csv_file:
                return pd.read_csv(csv_file)
    except Exception as e:
        return pd.DataFrame()


def process_csv_files_and_build_profiles(
    data_path: str,
    dataset: str,
    validity_dict: Dict,
    fets_dict: Dict,
    edge_type_dict: Dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_causal: bool = False,
    use_timeoh: bool = False,
    cee: Optional[str] = None,
    rule_generator: Optional[str] = None
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    is_atlas = dataset.lower() == "atlas"
    
    if is_atlas:
        if not cee or not rule_generator:
            raise ValueError("For ATLAS dataset, --cee and --rule-generator arguments are required")
        
        csv_file_info = find_atlas_csv_files(data_path, cee, rule_generator)
        
        if not csv_file_info:
            return {}, {}, {}, {}, {}, {}
        cee_normalized = cee.lower().replace(':', '_')
        rule_generator_normalized = rule_generator.lower().replace(':', '_')
    else:
        dataset_path = os.path.join(data_path, dataset.upper())
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        csv_pattern = os.path.join(dataset_path, '*', '*.csv')
        csv_files = glob(csv_pattern)
        csv_files = filter_csv_files_by_date(csv_files, start_date, end_date)
        csv_file_info = [(None, f) for f in csv_files]
    
    if not csv_file_info:
        return {}, {}, {}, {}, {}, {}
    
    id_to_class = {}
    id_to_functionality = {}
    typed_node_enames = {}
    behavioral_profiles = defaultdict(set)
    untyped_nodes = {}
    cached_dfs = {}
    
    for csv_file_info_item in tqdm(csv_file_info, desc="Phase 1: Type classification", position=0):
        try:
            zip_file, csv_path = csv_file_info_item
            
            if zip_file:
                df = load_csv_from_zip(zip_file, csv_path)
            else:
                df = pd.read_csv(csv_path)
            
            if df.empty:
                continue
            
            cache_key = f"{zip_file}:{csv_path}" if zip_file else csv_path
            cached_dfs[cache_key] = df
            
            is_atlas_format = is_atlas or ('source_name' in df.columns and 'dest_name' in df.columns)
            
            for row in tqdm(df.itertuples(index=False), total=len(df), 
                           desc=f"  Processing rows", leave=False, position=1):
                
                if is_atlas_format:
                    source_enames = parse_atlas_ename(getattr(row, 'source_name', None))
                    dest_enames = parse_atlas_ename(getattr(row, 'dest_name', None))
                    
                    if source_enames:
                        source_identifier = normalize_ename(source_enames[0])
                        if source_identifier and source_identifier not in id_to_class:
                            class_label, functionality, should_ignore = process_node_enames(
                                source_enames, validity_dict, fets_dict
                            )
                            
                            if class_label:
                                id_to_class[source_identifier] = class_label
                                typed_node_enames[source_identifier] = source_enames
                                if functionality:
                                    id_to_functionality[source_identifier] = functionality
                            elif not should_ignore:
                                if source_identifier not in untyped_nodes:
                                    untyped_nodes[source_identifier] = source_enames
                    
                    if dest_enames:
                        dest_identifier = normalize_ename(dest_enames[0])
                        if dest_identifier and dest_identifier not in id_to_class:
                            class_label, functionality, should_ignore = process_node_enames(
                                dest_enames, validity_dict, fets_dict
                            )
                            
                            if class_label:
                                id_to_class[dest_identifier] = class_label
                                typed_node_enames[dest_identifier] = dest_enames
                                if functionality:
                                    id_to_functionality[dest_identifier] = functionality
                            elif not should_ignore:
                                if dest_identifier not in untyped_nodes:
                                    untyped_nodes[dest_identifier] = dest_enames
                else:
                    source_id = row.source_id
                    source_enames_str = getattr(row, 'source_enames', None)
                    source_enames = parse_ename_list(source_enames_str)
                    
                    if pd.notna(source_id) and source_id not in id_to_class:
                        if source_enames:
                            class_label, functionality, should_ignore = process_node_enames(
                                source_enames, validity_dict, fets_dict
                            )
                            
                            if class_label:
                                id_to_class[source_id] = class_label
                                typed_node_enames[source_id] = source_enames
                                if functionality:
                                    id_to_functionality[source_id] = functionality
                            elif not should_ignore:
                                if source_id not in untyped_nodes:
                                    untyped_nodes[source_id] = source_enames
                    
                    dest_id = row.dest_id
                    dest_enames_str = getattr(row, 'dest_enames', None)
                    dest_enames = parse_ename_list(dest_enames_str)
                    
                    if pd.notna(dest_id) and dest_id not in id_to_class:
                        if dest_enames:
                            class_label, functionality, should_ignore = process_node_enames(
                                dest_enames, validity_dict, fets_dict
                            )
                            
                            if class_label:
                                id_to_class[dest_id] = class_label
                                typed_node_enames[dest_id] = dest_enames
                                if functionality:
                                    id_to_functionality[dest_id] = functionality
                            elif not should_ignore:
                                if dest_id not in untyped_nodes:
                                    untyped_nodes[dest_id] = dest_enames
        except Exception as e:
            continue
    if not id_to_class:
        return {}, {}, {}, {}, {}, {}
    
    id_set = set(id_to_class.keys())
    untyped_set = set(untyped_nodes.keys())
    
    valid_edge_types = set()
    if edge_type_dict:
        valid_edge_types = {edge.lower() for edge, validity in edge_type_dict.items() if validity == "VALID"}
    untyped_profiles = defaultdict(set)
    if use_causal:
        
        for csv_file_info_item in tqdm(csv_file_info, desc="Phase 2: Building causal profiles per timestamp", position=0):
            try:
                zip_file, csv_path = csv_file_info_item
                cache_key = f"{zip_file}:{csv_path}" if zip_file else csv_path
                df = cached_dfs[cache_key]
                is_atlas_format = is_atlas
                
                if valid_edge_types:
                    df_filtered = df[df['action'].str.lower().isin(valid_edge_types)]
                else:
                    df['action_lower'] = df['action'].str.lower()
                    mask = (df['action_lower'].notna()) & \
                           (df['action_lower'] != 'no label') & \
                           (df['action_lower'] != 'no_label')
                    df_filtered = df[mask]
                
                df_filtered = df_filtered.copy()
                df_filtered['action_lower'] = df_filtered['action'].str.lower()
                
                timestamp_graph = defaultdict(list)
                for row in df_filtered.itertuples(index=False):
                    if is_atlas_format:
                        source_name = getattr(row, 'source_name', None)
                        dest_name = getattr(row, 'dest_name', None)
                        if source_name and dest_name:
                            source_enames = parse_atlas_ename(source_name)
                            dest_enames = parse_atlas_ename(dest_name)
                            if source_enames and dest_enames:
                                source_identifier = normalize_ename(source_enames[0])
                                dest_identifier = normalize_ename(dest_enames[0])
                                if source_identifier and dest_identifier:
                                    timestamp_graph[source_identifier].append((dest_identifier, row.action_lower))
                    else:
                        if pd.notna(row.source_id) and pd.notna(row.dest_id):
                            timestamp_graph[row.source_id].append((row.dest_id, row.action_lower))
                
                for node_id in id_set:
                    if node_id in timestamp_graph or any(node_id in edges for edges in timestamp_graph.values()):
                        causal_patterns = build_causal_paths(node_id, timestamp_graph, id_to_class)
                        behavioral_profiles[node_id].update(causal_patterns)
                
                for row in df_filtered.itertuples(index=False):
                    action_lower = row.action_lower
                    
                    if is_atlas_format:
                        source_name = getattr(row, 'source_name', None)
                        dest_name = getattr(row, 'dest_name', None)
                        if source_name and dest_name:
                            source_enames = parse_atlas_ename(source_name)
                            dest_enames = parse_atlas_ename(dest_name)
                            if source_enames and dest_enames:
                                source_identifier = normalize_ename(source_enames[0])
                                dest_identifier = normalize_ename(dest_enames[0])
                                if dest_identifier and dest_identifier in id_set:
                                    if source_identifier and source_identifier in id_to_class:
                                        source_type = id_to_class[source_identifier]
                                        pattern = f"{action_lower}_by_{source_type}"
                                        behavioral_profiles[dest_identifier].add(pattern)
                    else:
                        dest_id = row.dest_id
                        source_id = row.source_id
                        if pd.notna(dest_id) and dest_id in id_set:
                            if pd.notna(source_id) and source_id in id_to_class:
                                source_type = id_to_class[source_id]
                                pattern = f"{action_lower}_by_{source_type}"
                                behavioral_profiles[dest_id].add(pattern)
                
                for node_id in untyped_set:
                    if node_id in timestamp_graph or any(node_id in edges for edges in timestamp_graph.values()):
                        causal_patterns = build_causal_paths(node_id, timestamp_graph, id_to_class)
                        untyped_profiles[node_id].update(causal_patterns)
                
                for row in df_filtered.itertuples(index=False):
                    action_lower = row.action_lower
                    
                    if is_atlas_format:
                        source_name = getattr(row, 'source_name', None)
                        dest_name = getattr(row, 'dest_name', None)
                        if source_name and dest_name:
                            source_enames = parse_atlas_ename(source_name)
                            dest_enames = parse_atlas_ename(dest_name)
                            if source_enames and dest_enames:
                                source_identifier = normalize_ename(source_enames[0])
                                dest_identifier = normalize_ename(dest_enames[0])
                                if dest_identifier and dest_identifier in untyped_set:
                                    if source_identifier and source_identifier in id_to_class:
                                        source_type = id_to_class[source_identifier]
                                        pattern = f"{action_lower}_by_{source_type}"
                                        untyped_profiles[dest_identifier].add(pattern)
                    else:
                        dest_id = row.dest_id
                        source_id = row.source_id
                        if pd.notna(dest_id) and dest_id in untyped_set:
                            if pd.notna(source_id) and source_id in id_to_class:
                                source_type = id_to_class[source_id]
                                pattern = f"{action_lower}_by_{source_type}"
                                untyped_profiles[dest_id].add(pattern)
            except:
                continue
    
    else:
        for csv_file_info_item in tqdm(csv_file_info, desc="Phase 2: Building one-hop profiles", position=0):
            try:
                zip_file, csv_path = csv_file_info_item
                cache_key = f"{zip_file}:{csv_path}" if zip_file else csv_path
                df = cached_dfs[cache_key]
                is_atlas_format = is_atlas
                
                if valid_edge_types:
                    df_filtered = df[df['action'].str.lower().isin(valid_edge_types)]
                else:
                    df['action_lower'] = df['action'].str.lower()
                    mask = (df['action_lower'].notna()) & \
                           (df['action_lower'] != 'no label') & \
                           (df['action_lower'] != 'no_label')
                    df_filtered = df[mask]
                
                df_filtered = df_filtered.copy()
                df_filtered['action_lower'] = df_filtered['action'].str.lower()
                
                for row in tqdm(df_filtered.itertuples(index=False), total=len(df_filtered),
                               desc=f"  Building patterns", leave=False, position=1):
                    action_lower = row.action_lower
                    
                    if is_atlas_format:
                        source_name = getattr(row, 'source_name', None)
                        dest_name = getattr(row, 'dest_name', None)
                        if source_name and dest_name:
                            source_enames = parse_atlas_ename(source_name)
                            dest_enames = parse_atlas_ename(dest_name)
                            if source_enames and dest_enames:
                                source_identifier = normalize_ename(source_enames[0])
                                dest_identifier = normalize_ename(dest_enames[0])
                                
                                if source_identifier and dest_identifier:
                                    if source_identifier in id_set:
                                        if dest_identifier in id_set:
                                            dest_type = id_to_class[dest_identifier]
                                            pattern = f"{action_lower}_{dest_type}"
                                            behavioral_profiles[source_identifier].add(pattern)
                                    
                                    if dest_identifier in id_set:
                                        if source_identifier in id_set:
                                            source_type = id_to_class[source_identifier]
                                            pattern = f"{action_lower}_by_{source_type}"
                                            behavioral_profiles[dest_identifier].add(pattern)
                                    
                                    if source_identifier in untyped_set:
                                        if dest_identifier in id_set:
                                            dest_type = id_to_class[dest_identifier]
                                            pattern = f"{action_lower}_{dest_type}"
                                            untyped_profiles[source_identifier].add(pattern)
                                    
                                    if dest_identifier in untyped_set:
                                        if source_identifier in id_set:
                                            source_type = id_to_class[source_identifier]
                                            pattern = f"{action_lower}_by_{source_type}"
                                            untyped_profiles[dest_identifier].add(pattern)
                    else:
                        source_id = row.source_id
                        dest_id = row.dest_id
                        
                        if pd.notna(source_id) and source_id in id_set:
                            if pd.notna(dest_id) and dest_id in id_set:
                                dest_type = id_to_class[dest_id]
                                pattern = f"{action_lower}_{dest_type}"
                                behavioral_profiles[source_id].add(pattern)
                        
                        if pd.notna(dest_id) and dest_id in id_set:
                            if pd.notna(source_id) and source_id in id_set:
                                source_type = id_to_class[source_id]
                                pattern = f"{action_lower}_by_{source_type}"
                                behavioral_profiles[dest_id].add(pattern)
                        
                        if pd.notna(source_id) and source_id in untyped_set:
                            if pd.notna(dest_id) and dest_id in id_set:
                                dest_type = id_to_class[dest_id]
                                pattern = f"{action_lower}_{dest_type}"
                                untyped_profiles[source_id].add(pattern)
                        
                        if pd.notna(dest_id) and dest_id in untyped_set:
                            if pd.notna(source_id) and source_id in id_set:
                                source_type = id_to_class[source_id]
                                pattern = f"{action_lower}_by_{source_type}"
                                untyped_profiles[dest_id].add(pattern)
            except:
                continue
    
    cached_dfs.clear()
    
    return id_to_class, id_to_functionality, typed_node_enames, untyped_nodes, dict(behavioral_profiles), dict(untyped_profiles)


def create_and_save_datasets(
    typed_profiles: Dict[str, Set[str]],
    untyped_profiles: Dict[str, Set[str]],
    id_to_class: Dict[str, str],
    id_to_functionality: Dict[str, str],
    typed_node_enames: Dict[str, List[str]],
    untyped_nodes: Dict[str, List[str]],
    dataset_name: str,
    llmfets_model: str,
    use_causal: bool = False,
    use_timeoh: bool = False
):
    model_normalized = llmfets_model.lower().replace(':', '_')
    
    if use_causal:
        save_dir = f"./behavioral-profiles/{model_normalized}/causal"
        prefix = "Causal"
    elif use_timeoh:
        save_dir = f"./behavioral-profiles/{model_normalized}/timeoh"
        prefix = "TimeOH"
    else:
        save_dir = f"./behavioral-profiles/{model_normalized}"
        prefix = ""
    
    os.makedirs(save_dir, exist_ok=True)
    
    typed_nodes_file = f"{save_dir}/typed_nodes_{dataset_name}.json"
    with open(typed_nodes_file, 'w') as f:
        json.dump(id_to_class, f, indent=2)
    
    functionality_file = f"{save_dir}/typed_nodes_functionality_{dataset_name}.json"
    with open(functionality_file, 'w') as f:
        json.dump(id_to_functionality, f, indent=2)
    
    typed_enames_file = f"{save_dir}/typed_nodes_enames_{dataset_name}.json"
    with open(typed_enames_file, 'w') as f:
        json.dump(typed_node_enames, f, indent=2)
    
    untyped_nodes_enames_file = f"{save_dir}/untyped_nodes_enames_{dataset_name}.json"
    with open(untyped_nodes_enames_file, 'w') as f:
        json.dump(untyped_nodes, f, indent=2)
    
    typed_ids = set(typed_profiles.keys()) & set(id_to_class.keys())
    typed_data = []
    for node_id in typed_ids:
        typed_data.append({
            'node_id': node_id,
            'features': list(typed_profiles[node_id]),
            'label': id_to_class[node_id]
        })
    
    untyped_ids = set(untyped_profiles.keys()) & set(untyped_nodes.keys())
    untyped_data = []
    for node_id in untyped_ids:
        untyped_data.append({
            'node_id': node_id,
            'features': list(untyped_profiles[node_id])
        })
    
    all_features = [d['features'] for d in typed_data] + [d['features'] for d in untyped_data]
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(all_features)
    pattern_columns = mlb.classes_
    if typed_data:
        df_typed = pd.DataFrame(typed_data)
        X_typed_sparse = mlb.transform(df_typed['features'])
        
        typed_sparse_file = f"{save_dir}/{prefix}behavioral_dataset_typed_{dataset_name}.npz"
        save_npz(typed_sparse_file, X_typed_sparse)
        
        metadata = {
            'node_ids': df_typed['node_id'].tolist(),
            'labels': df_typed['label'].tolist(),
            'pattern_columns': pattern_columns.tolist(),
            'shape': X_typed_sparse.shape,
            'nnz': X_typed_sparse.nnz
        }
        metadata_file = f"{save_dir}/{prefix}behavioral_dataset_typed_{dataset_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    if untyped_data:
        df_untyped = pd.DataFrame(untyped_data)
        X_untyped_sparse = mlb.transform(df_untyped['features'])
        
        untyped_sparse_file = f"{save_dir}/{prefix}behavioral_dataset_untyped_{dataset_name}.npz"
        save_npz(untyped_sparse_file, X_untyped_sparse)
        
        metadata = {
            'node_ids': df_untyped['node_id'].tolist(),
            'pattern_columns': pattern_columns.tolist(),
            'shape': X_untyped_sparse.shape,
            'nnz': X_untyped_sparse.nnz
        }
        metadata_file = f"{save_dir}/{prefix}behavioral_dataset_untyped_{dataset_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    args = parse_args()
    
    if args.dataset.lower() == "atlas":
        if not args.cee or not args.rule_generator:
            raise ValueError("For ATLAS dataset, --cee and --rule-generator arguments are required")
    _load_valid_extensions()
    validity_dict, fets_dict, edge_type_dict = load_reference_files(args.script_dir, args.dataset, args.llmfets_model)
    
    id_to_class, id_to_functionality, typed_node_enames, untyped_nodes, typed_profiles, untyped_profiles = process_csv_files_and_build_profiles(
        args.data_path,
        args.dataset,
        validity_dict,
        fets_dict,
        edge_type_dict,
        args.start_date if args.dataset.lower() != "atlas" else None,
        args.end_date if args.dataset.lower() != "atlas" else None,
        args.causal,
        args.timeoh,
        args.cee,
        args.rule_generator
    )
    
    if not id_to_class:
        return
    create_and_save_datasets(typed_profiles, untyped_profiles, id_to_class, id_to_functionality, typed_node_enames, untyped_nodes, args.dataset, args.llmfets_model, args.causal, args.timeoh)


if __name__ == "__main__":
    main()
