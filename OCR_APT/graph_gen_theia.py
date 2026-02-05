#!/usr/bin/env python3

import os
import sys
import json
import pickle
import glob
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import zipfile
from datetime import datetime, timedelta
import time
import pytz
from collections import defaultdict
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from OCR_APT.ocrapt_utils import (
    stringtomd5, ns_time_to_datetime_US, datetime_to_ns_time_US,
    normalize_edge_features, scale_temporal_features,
    print_graph_stats, get_node_type_distribution, get_edge_type_distribution,
    order_edge_types_ocrapt, validate_graph_data, ensure_dir, get_output_path
)

def parse_args():
    parser = argparse.ArgumentParser(description='OCR_APT Graph Generation for THEIA')
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--baseline', action='store_true',
                           help='Baseline mode: regex mode with OCR-APT behavioral features')
    mode_group.add_argument('--autoprov', action='store_true',
                           help='AutoProv mode: RuleLLM mode with LLM type embeddings')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    
    parser.add_argument('--dataset_path', type=str, default='../BIGDATA/DARPA-E3',
                       help='Base path to raw log dataset directory (for baseline mode)')
    parser.add_argument('--extracted_graph_path', type=str, default='../BIGDATA/ExtractedProvGraph',
                       help='Path to extracted provenance graph CSVs (for autoprov mode)')
    
    parser.add_argument('--embedding', type=str, default='mpnet',
                       choices=['mpnet', 'minilm', 'roberta', 'distilbert'],
                       help='Embedding model to use for autoprov mode (default: mpnet)')
    parser.add_argument('--pca_dim', type=int, default=128,
                       help='PCA dimensionality for embeddings (default: 128)')
    
    parser.add_argument('--gpu', type=str, default='2',
                       help='GPU device ID (default: 2)')
    
    return parser.parse_args()

def load_log_data(timestamp_dir):
    zip_file_path = os.path.join(timestamp_dir, 'logs.pkl.zip')
    pkl_file_path = os.path.join(timestamp_dir, 'logs.pkl')
    
    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open('logs.pkl') as f:
                return pickle.load(f)
    elif os.path.exists(pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Neither {zip_file_path} nor {pkl_file_path} found")

def load_csv_data(timestamp_dir):
    csv_files = glob.glob(os.path.join(timestamp_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {timestamp_dir}")
    
    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)
    
    if 'timestamp' in df.columns:
        missing_before = df['timestamp'].isna().sum()
        if missing_before > 0:
            df['timestamp'] = df['timestamp'].bfill()
            df['timestamp'] = df['timestamp'].ffill()
            missing_after = df['timestamp'].isna().sum()
            if missing_after > 0:
                print(f"  Warning: {missing_after} timestamps still missing after filling")
    
    return df

def process_netflow_nodes(file_path, start_date, end_date):
    netobjset = set()
    netobj2hash = {}
    
    timestamp_dirs = sorted(glob.glob(file_path))
    
    for timestamp_dir in tqdm(timestamp_dirs, desc="Extracting NetFlow nodes"):
        if not is_dir_in_date_range(timestamp_dir, start_date, end_date):
            continue
        
        data = load_log_data(timestamp_dir)
        
        for line in data:
            if '{"datum":{"com.bbn.tc.schema.avro.cdm18.NetFlowObject"' in line:
                res = re.findall(
                    'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),',
                    line
                )
                if res:
                    res = res[0]
                    nodeid = res[0]
                    srcaddr = res[2]
                    srcport = res[3]
                    dstaddr = res[4]
                    dstport = res[5]
                    
                    nodeproperty = srcaddr + "," + srcport + "," + dstaddr + "," + dstport
                    hashstr = stringtomd5(nodeproperty)
                    netobj2hash[nodeid] = [hashstr, nodeproperty]
                    netobj2hash[hashstr] = nodeid
                    netobjset.add(hashstr)
    
    return netobjset, netobj2hash

def process_subject_nodes(file_path, start_date, end_date):
    subjectset = set()
    subject2hash = {}
    
    timestamp_dirs = sorted(glob.glob(file_path))
    
    for timestamp_dir in tqdm(timestamp_dirs, desc="Extracting Subject nodes"):
        if not is_dir_in_date_range(timestamp_dir, start_date, end_date):
            continue
        
        data = load_log_data(timestamp_dir)
        
        for line in data:
            if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line:
                res = re.findall(
                    'Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"properties":{"map":{"tgid":"(.*?)"',
                    line
                )
                if res:
                    res = res[0]
                    
                    path_match = re.findall('"path":"(.*?)"', line)
                    path = path_match[0] if path_match else "null"
                    
                    nodeid = res[0]
                    cmdLine = res[2]
                    tgid = res[4]
                    
                    nodeproperty = cmdLine + "," + tgid + "," + path
                    hashstr = stringtomd5(nodeproperty)
                    subject2hash[nodeid] = [hashstr, cmdLine, tgid, path]
                    subject2hash[hashstr] = nodeid
                    subjectset.add(hashstr)
    
    return subjectset, subject2hash

def process_file_nodes(file_path, start_date, end_date):
    fileset = set()
    file2hash = {}
    
    timestamp_dirs = sorted(glob.glob(file_path))
    
    for timestamp_dir in tqdm(timestamp_dirs, desc="Extracting File nodes"):
        if not is_dir_in_date_range(timestamp_dir, start_date, end_date):
            continue
        
        data = load_log_data(timestamp_dir)
        
        for line in data:
            if '{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' in line:
                res = re.findall('FileObject":{"uuid":"(.*?)"(.*?)"filename":"(.*?)"', line)
                if res:
                    res = res[0]
                    nodeid = res[0]
                    filepath = res[2]
                    nodeproperty = filepath
                    hashstr = stringtomd5(nodeproperty)
                    file2hash[nodeid] = [hashstr, nodeproperty]
                    file2hash[hashstr] = nodeid
                    fileset.add(hashstr)
    
    return fileset, file2hash

def process_regex_edges(file_path, subject2hash, file2hash, netobj2hash, start_date, end_date):
    edges = []
    edge_types_set = set()
    
    timestamp_dirs = sorted(glob.glob(file_path))
    
    for timestamp_dir in tqdm(timestamp_dirs, desc="Processing Events (Regex)"):
        if not is_dir_in_date_range(timestamp_dir, start_date, end_date):
            continue
        
        data = load_log_data(timestamp_dir)
        
        for line in data:
            if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line and "EVENT_FLOWS_TO" not in line:
                time_match = re.findall('"timestampNanos":(.*?),', line)
                subject_match = re.findall('"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"', line)
                object_match = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"', line)
                type_match = re.findall('"type":"(.*?)"', line)
                
                if time_match and subject_match and object_match and type_match:
                    time_ns = int(time_match[0])
                    subjectid = subject_match[0]
                    objectid = object_match[0]
                    relation_type = type_match[0]
                    
                    edge_types_set.add(relation_type)
                    
                    if subjectid in subject2hash:
                        subjectid = subject2hash[subjectid][0]
                    if objectid in subject2hash:
                        objectid = subject2hash[objectid][0]
                    if objectid in file2hash:
                        objectid = file2hash[objectid][0]
                    if objectid in netobj2hash:
                        objectid = netobj2hash[objectid][0]
                    
                    if len(subjectid) == 64 and len(objectid) == 64:
                        if relation_type in ['EVENT_READ', 'EVENT_READ_SOCKET_PARAMS', 'EVENT_RECVFROM', 'EVENT_RECVMSG']:
                            edges.append((objectid, subjectid, relation_type, time_ns))
                        else:
                            edges.append((subjectid, objectid, relation_type, time_ns))
    
    return edges, list(edge_types_set)

def create_nodes_from_regex(subject2hash, file2hash, netobj2hash):
    nodes = {}
    
    for uuid, info in subject2hash.items():
        if len(uuid) == 64:
            nodes[uuid] = {'type': 'Subject', 'attr': info[3]}
    
    for uuid, info in file2hash.items():
        if len(uuid) == 64:
            nodes[uuid] = {'type': 'FileObject', 'attr': info[1]}
    
    for uuid, info in netobj2hash.items():
        if len(uuid) == 64:
            nodes[uuid] = {'type': 'NetFlowObject', 'attr': info[1]}
    
    return nodes

def load_vtype_mapping(vtype_mapping_path, dataset):
    if vtype_mapping_path and os.path.exists(vtype_mapping_path):
        mapping_file = vtype_mapping_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        flash_dir = os.path.join(os.path.dirname(script_dir), 'FLASH')
        mapping_file = os.path.join(flash_dir, f"llmgeneratedvtypegroup_{dataset.lower()}.pkl")
    
    if not os.path.exists(mapping_file):
        return {}
    
    with open(mapping_file, 'rb') as f:
        vtype_mapping = pickle.load(f)
    
    return vtype_mapping

def load_edge_validation(edge_validation_path, dataset):
    if edge_validation_path and os.path.exists(edge_validation_path):
        validation_file = edge_validation_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ename_dir = os.path.join(os.path.dirname(script_dir), 'ename-processing')
        validation_file = os.path.join(ename_dir, f"edge_type_validation_{dataset.lower()}.json")
    
    if not os.path.exists(validation_file):
        return {}
    
    with open(validation_file, 'r') as f:
        edge_validation = json.load(f)
    
    return edge_validation

def process_csv_nodes_and_edges(file_path, start_date, end_date, vtype_mapping, edge_validation):
    nodes = {}
    edges = []
    edge_types_set = set()
    
    def is_no_label_action(action):
        if action == "NO LABEL":
            return True
        if action in edge_validation and edge_validation[action] == "INVALID":
            return True
        return False
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    timestamp_dirs = sorted(glob.glob(file_path))
    
    skipped_dirs = []
    for timestamp_dir in tqdm(timestamp_dirs, desc="Processing CSV directories", leave=False):
        base_name = os.path.basename(timestamp_dir)
        date_parts = base_name.split(' ')
        if not date_parts or not date_parts[0]:
            skipped_dirs.append((base_name, "Empty date part"))
            continue
        file_date_str = date_parts[0]
        
        if not file_date_str or len(file_date_str) < 10:
            skipped_dirs.append((base_name, f"Invalid date format: '{file_date_str}'"))
            continue
        
        file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
        
        if not (start_dt <= file_date <= end_dt):
            continue
        
        df = load_csv_data(timestamp_dir)
        if df.empty:
            continue
        
        df = df[df['action'].notna()]
        if df.empty:
            continue
        
        def parse_list_fast(s):
            if not isinstance(s, str) or s == '[]' or not s.strip():
                return []
            s = s.strip('[]').replace("'", "").replace('"', '')
            return [x.strip() for x in s.split(',') if x.strip()]
        
        def safe_eval_first(val_str):
            vals = parse_list_fast(val_str)
            return vals[0] if vals else 'unknown'
        
        def safe_eval_and_join(val_str):
            vals = parse_list_fast(val_str)
            if not vals:
                return 'unknown'
            unique_elements = sorted(set([v.lower() for v in vals]))
            return '+'.join(unique_elements)
        
        def safe_eval_longest_ename(val_str):
            vals = parse_list_fast(val_str)
            meaningless = {'datum', 'N/A', '},'}
            clean = [v for v in vals if v not in meaningless and len(v) > 1]
            return max(clean, key=len) if clean else ''
        
        df['src_vtype'] = df['source_vtypes'].apply(safe_eval_and_join)
        df['dst_vtype'] = df['dest_vtypes'].apply(safe_eval_and_join)
        df['src_ename'] = df['source_enames'].apply(safe_eval_longest_ename)
        df['dst_ename'] = df['dest_enames'].apply(safe_eval_longest_ename)
        df['is_no_label'] = df['action'].apply(is_no_label_action)
        
        no_label_df = df[df['is_no_label']]
        for row in no_label_df[['source_id', 'dest_id', 'src_vtype', 'dst_vtype', 'src_ename', 'dst_ename']].itertuples(index=False, name=None):
            src_vtype_mapped = vtype_mapping.get(row[2], row[2]) if vtype_mapping else row[2]
            dst_vtype_mapped = vtype_mapping.get(row[3], row[3]) if vtype_mapping else row[3]
            nodes[row[0]] = {'type': src_vtype_mapped, 'attr': row[4]}
            nodes[row[1]] = {'type': dst_vtype_mapped, 'attr': row[5]}
        
        valid_df = df[~df['is_no_label']]
        for row in valid_df[['source_id', 'dest_id', 'action', 'timestamp', 'src_vtype', 'dst_vtype', 'src_ename', 'dst_ename']].itertuples(index=False, name=None):
            src_id, dst_id, action, timestamp, src_vtype, dst_vtype, src_ename, dst_ename = row
            
            if not isinstance(action, str):
                continue
            
            src_vtype_mapped = vtype_mapping.get(src_vtype, src_vtype) if vtype_mapping else src_vtype
            dst_vtype_mapped = vtype_mapping.get(dst_vtype, dst_vtype) if vtype_mapping else dst_vtype
            
            if src_id not in nodes:
                nodes[src_id] = {'type': src_vtype_mapped, 'attr': src_ename}
            if dst_id not in nodes:
                nodes[dst_id] = {'type': dst_vtype_mapped, 'attr': dst_ename}
            
            edges.append((src_id, dst_id, action, timestamp))
            edge_types_set.add(action)
    
    return nodes, edges, list(edge_types_set)

def feature_engineering_ocrapt(nodes, edges, edge_types, dataset):
    all_nodes_uuid = list(nodes.keys())
    
    ordered_edge_features = order_edge_types_ocrapt(dataset, edge_types)
    edge_types_dic = {edge: 0 for edge in ordered_edge_features}
    x_list = {node: edge_types_dic.copy() for node in all_nodes_uuid}
    
    for src_uuid, dst_uuid, edge_type, timestamp in tqdm(edges, desc="Computing edge features", leave=False):
        normalized_type = edge_type.replace("EVENT_", "").lower()
        
        if f"out_{normalized_type}" in x_list[src_uuid]:
            x_list[src_uuid][f"out_{normalized_type}"] += 1
        if f"in_{normalized_type}" in x_list[dst_uuid]:
            x_list[dst_uuid][f"in_{normalized_type}"] += 1
        
        if True:
            if "event_timestamp_lst" not in x_list[src_uuid]:
                x_list[src_uuid]["event_timestamp_lst"] = []
            if "event_timestamp_lst" not in x_list[dst_uuid]:
                x_list[dst_uuid]["event_timestamp_lst"] = []
            
            x_list[src_uuid]["event_timestamp_lst"].append(timestamp)
            x_list[dst_uuid]["event_timestamp_lst"].append(timestamp)
    
    if True:
        second_threshold = 1.0
        
        for node_uuid in tqdm(all_nodes_uuid, desc="Computing temporal features", leave=False):
            if "event_timestamp_lst" not in x_list[node_uuid]:
                x_list[node_uuid]["avg_idle_time"] = 0
                x_list[node_uuid]["max_idle_time"] = 0
                x_list[node_uuid]["min_idle_time"] = 0
                x_list[node_uuid]["cumulative_active_time"] = 0
                x_list[node_uuid]["lifespan"] = 0
                continue
            
            timestamps = x_list[node_uuid]["event_timestamp_lst"]
            
            if len(timestamps) > 1:
                timestamps.sort()
                
                gaps_sec = []
                for i in range(1, len(timestamps)):
                    if isinstance(timestamps[i], int) and timestamps[i] > 1e15:
                        gap = (timestamps[i] - timestamps[i-1]) / 1e9
                    else:
                        gap = timestamps[i] - timestamps[i-1]
                    gaps_sec.append(gap)
                
                gaps_sec = np.array(gaps_sec)
                
                active_gaps = gaps_sec[gaps_sec < second_threshold]
                idle_gaps = gaps_sec[gaps_sec >= second_threshold]
                
                if len(idle_gaps) > 0:
                    x_list[node_uuid]["avg_idle_time"] = int(round(idle_gaps.mean()))
                    x_list[node_uuid]["max_idle_time"] = int(round(idle_gaps.max()))
                    x_list[node_uuid]["min_idle_time"] = int(round(idle_gaps.min()))
                else:
                    x_list[node_uuid]["avg_idle_time"] = 0
                    x_list[node_uuid]["max_idle_time"] = 0
                    x_list[node_uuid]["min_idle_time"] = 0
                
                if len(active_gaps) > 0:
                    x_list[node_uuid]["cumulative_active_time"] = int(round(active_gaps.sum()))
                else:
                    x_list[node_uuid]["cumulative_active_time"] = 0
                
                x_list[node_uuid]["lifespan"] = int(round(gaps_sec.sum()))
            else:
                x_list[node_uuid]["avg_idle_time"] = 0
                x_list[node_uuid]["max_idle_time"] = 0
                x_list[node_uuid]["min_idle_time"] = 0
                x_list[node_uuid]["cumulative_active_time"] = 0
                x_list[node_uuid]["lifespan"] = 0
            
            del x_list[node_uuid]["event_timestamp_lst"]
    
    x_list_df = pd.DataFrame.from_dict(x_list, orient='index')
    x_list_df = x_list_df.loc[:, (x_list_df != 0).any(axis=0)]
    x_list_df = x_list_df.reset_index()
    x_list_df.rename(columns={'index': 'node_uuid'}, inplace=True)
    x_list_df = x_list_df.fillna(0)
    
    edge_type_cols = [col for col in x_list_df.columns 
                     if col.startswith('in_') or col.startswith('out_')]
    temporal_cols = [col for col in x_list_df.columns 
                    if col not in edge_type_cols and col != 'node_uuid']
    
    x_list_df = normalize_edge_features(x_list_df, edge_type_cols)
    x_list_df, scaler = scale_temporal_features(x_list_df, temporal_cols)
    
    return x_list_df, scaler

def split_train_test_by_dates(edges, nodes, train_start, train_end, test_start, test_end):
    from datetime import datetime
    
    train_start_dt = datetime.strptime(train_start, "%Y-%m-%d")
    train_end_dt = datetime.strptime(train_end, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    test_start_dt = datetime.strptime(test_start, "%Y-%m-%d")
    test_end_dt = datetime.strptime(test_end, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    
    train_edges = []
    test_edges = []
    skipped_invalid_timestamps = 0
    
    for src, dst, etype, timestamp in edges:
        if isinstance(timestamp, (int, float)) and timestamp > 1e15:
            timestamp_sec = timestamp / 1e9
            if timestamp_sec < 0 or timestamp_sec > 4e9:
                skipped_invalid_timestamps += 1
                continue
            edge_dt = datetime.fromtimestamp(timestamp_sec)
        elif isinstance(timestamp, (int, float)):
            if timestamp < 0 or timestamp > 4e9:
                skipped_invalid_timestamps += 1
                continue
            edge_dt = datetime.fromtimestamp(timestamp)
        else:
            if isinstance(timestamp, str):
                edge_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            else:
                edge_dt = timestamp
        
        if train_start_dt <= edge_dt <= train_end_dt:
            train_edges.append((src, dst, etype, timestamp))
        elif test_start_dt <= edge_dt <= test_end_dt:
            test_edges.append((src, dst, etype, timestamp))
    
    train_nodes = set()
    for src, dst, _, _ in train_edges:
        train_nodes.add(src)
        train_nodes.add(dst)
    
    test_nodes = set()
    for src, dst, _, _ in test_edges:
        test_nodes.add(src)
        test_nodes.add(dst)
    
    train_nodes_dict = {nid: nodes[nid] for nid in train_nodes if nid in nodes}
    test_nodes_dict = {nid: nodes[nid] for nid in test_nodes if nid in nodes}
    
    return train_nodes_dict, test_nodes_dict, train_edges, test_edges

def construct_networkx_graph(nodes, edges, edge_types):
    G = nx.MultiDiGraph()
    
    for node_id, node_info in nodes.items():
        G.add_node(node_id, **node_info)
    
    for src, dst, etype, timestamp in edges:
        if src in nodes and dst in nodes:
            G.add_edge(src, dst, type=etype, timestamp=timestamp)
    
    return G

def get_unique_vtypes(graph_data):
    nodes = graph_data['nodes']
    vtype_counts = {}
    
    for node_info in nodes.values():
        vtype = node_info.get('type', 'unknown')
        vtype_counts[vtype] = vtype_counts.get(vtype, 0) + 1
    
    vtypes = sorted(vtype_counts.keys(), key=lambda x: vtype_counts[x], reverse=True)
    return vtypes, vtype_counts

def display_final_vtypes(nodes):
    vtype_counts = {}
    for node_info in nodes.values():
        vtype = node_info.get('type', 'unknown')
        vtype_counts[vtype] = vtype_counts.get(vtype, 0) + 1
    
    print(f"\n{'='*80}")
    print(f"FINAL MAPPED VTYPES")
    print(f"{'='*80}")
    print(f"Total unique vtypes: {len(vtype_counts)}")
    print(f"Total nodes: {sum(vtype_counts.values()):,}\n")
    
    sorted_vtypes = sorted(vtype_counts.items(), key=lambda x: x[1], reverse=True)
    
    for vtype, count in sorted_vtypes:
        pct = (count / sum(vtype_counts.values())) * 100
        print(f"  {vtype:40s}: {count:10,} ({pct:5.2f}%)")
    print(f"{'='*80}\n")

def split_and_save_by_vtype(output_dir, dataset, train_data, test_data, features_train, 
                             features_test, edge_types, scaler, hash_to_uuid=None):
    vtypes, vtype_counts = get_unique_vtypes(train_data)
    
    dataset_dir = os.path.join(output_dir, dataset)
    ensure_dir(dataset_dir)
    
    saved_vtypes = []
    saved_vtype_counts = {}
    
    for vtype in tqdm(vtypes, desc="Splitting per vtype"):
        train_vtype_nodes = {nid: ninfo for nid, ninfo in train_data['nodes'].items()
                             if ninfo.get('type') == vtype}
        train_vtype_node_ids = set(train_vtype_nodes.keys())
        
        train_vtype_edges = [(src, dst, etype, ts) for src, dst, etype, ts in train_data['edges']
                             if src in train_vtype_node_ids and dst in train_vtype_node_ids]
        
        test_vtype_nodes = {nid: ninfo for nid, ninfo in test_data['nodes'].items()
                            if ninfo.get('type') == vtype}
        test_vtype_node_ids = set(test_vtype_nodes.keys())
        
        test_vtype_edges = [(src, dst, etype, ts) for src, dst, etype, ts in test_data['edges']
                            if src in test_vtype_node_ids and dst in test_vtype_node_ids]
        
        if not train_vtype_nodes or not train_vtype_edges:
            continue
        
        saved_vtypes.append(vtype)
        saved_vtype_counts[vtype] = vtype_counts[vtype]
        
        train_vtype_data = {
            'nodes': train_vtype_nodes,
            'edges': train_vtype_edges,
            'graph': None
        }
        
        test_vtype_data = {
            'nodes': test_vtype_nodes,
            'edges': test_vtype_edges,
            'graph': None
        }
        
        train_vtype_features = features_train[features_train['node_uuid'].isin(train_vtype_node_ids)].reset_index(drop=True)
        test_vtype_features = features_test[features_test['node_uuid'].isin(test_vtype_node_ids)].reset_index(drop=True)
        
        vtype_safe = vtype.replace('/', '_').replace('+', '_')
        
        num_train_vtype_nodes = len(train_vtype_data['nodes'])
        num_test_vtype_nodes = len(test_vtype_data['nodes'])
        train_vtype_split_idx = math.ceil(num_train_vtype_nodes / 2)
        test_vtype_split_idx = math.ceil(num_test_vtype_nodes / 2)
        
        train_vtype_data_part1, train_vtype_data_part2 = split_data_dict(train_vtype_data, train_vtype_split_idx)
        test_vtype_data_part1, test_vtype_data_part2 = split_data_dict(test_vtype_data, test_vtype_split_idx)
        
        with zipfile.ZipFile(os.path.join(dataset_dir, f'train_data_{vtype_safe}_part1.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
            with zf.open(f'train_data_{vtype_safe}.pkl', 'w') as f:
                pickle.dump(train_vtype_data_part1, f)
        
        with zipfile.ZipFile(os.path.join(dataset_dir, f'train_data_{vtype_safe}_part2.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
            with zf.open(f'train_data_{vtype_safe}.pkl', 'w') as f:
                pickle.dump(train_vtype_data_part2, f)
        
        with zipfile.ZipFile(os.path.join(dataset_dir, f'test_data_{vtype_safe}_part1.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
            with zf.open(f'test_data_{vtype_safe}.pkl', 'w') as f:
                pickle.dump(test_vtype_data_part1, f)
        
        with zipfile.ZipFile(os.path.join(dataset_dir, f'test_data_{vtype_safe}_part2.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
            with zf.open(f'test_data_{vtype_safe}.pkl', 'w') as f:
                pickle.dump(test_vtype_data_part2, f)
        
        num_train_vtype_features = len(train_vtype_features)
        num_test_vtype_features = len(test_vtype_features)
        train_vtype_features_split_idx = math.ceil(num_train_vtype_features / 2)
        test_vtype_features_split_idx = math.ceil(num_test_vtype_features / 2)
        
        train_vtype_features_part1 = train_vtype_features.iloc[:train_vtype_features_split_idx].reset_index(drop=True)
        train_vtype_features_part2 = train_vtype_features.iloc[train_vtype_features_split_idx:].reset_index(drop=True)
        test_vtype_features_part1 = test_vtype_features.iloc[:test_vtype_features_split_idx].reset_index(drop=True)
        test_vtype_features_part2 = test_vtype_features.iloc[test_vtype_features_split_idx:].reset_index(drop=True)
        
        with zipfile.ZipFile(os.path.join(dataset_dir, f'features_train_{vtype_safe}_part1.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
            with zf.open(f'features_train_{vtype_safe}.pkl', 'w') as f:
                pickle.dump(train_vtype_features_part1, f)
        
        with zipfile.ZipFile(os.path.join(dataset_dir, f'features_train_{vtype_safe}_part2.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
            with zf.open(f'features_train_{vtype_safe}.pkl', 'w') as f:
                pickle.dump(train_vtype_features_part2, f)
        
        with zipfile.ZipFile(os.path.join(dataset_dir, f'features_test_{vtype_safe}_part1.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
            with zf.open(f'features_test_{vtype_safe}.pkl', 'w') as f:
                pickle.dump(test_vtype_features_part1, f)
        
        with zipfile.ZipFile(os.path.join(dataset_dir, f'features_test_{vtype_safe}_part2.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
            with zf.open(f'features_test_{vtype_safe}.pkl', 'w') as f:
                pickle.dump(test_vtype_features_part2, f)
    
    vtypes_file = os.path.join(dataset_dir, 'vtypes_list.pkl')
    with open(vtypes_file, 'wb') as f:
        pickle.dump({'vtypes': saved_vtypes, 'vtype_counts': saved_vtype_counts}, f)
    
    with open(os.path.join(dataset_dir, 'edge_types.pkl'), 'wb') as f:
        pickle.dump(edge_types, f)
    
    if scaler is not None:
        with open(os.path.join(dataset_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    
    if hash_to_uuid:
        with open(os.path.join(dataset_dir, 'hash_to_uuid.json'), 'w') as f:
            json.dump(hash_to_uuid, f, indent=2)

def split_data_dict(data_dict, split_idx):
    nodes_list = list(data_dict['nodes'].items())
    nodes_part1 = dict(nodes_list[:split_idx])
    nodes_part2 = dict(nodes_list[split_idx:])
    
    nodes_part1_set = set(nodes_part1.keys())
    nodes_part2_set = set(nodes_part2.keys())
    
    edges_part1 = [(src, dst, etype, ts) for src, dst, etype, ts in data_dict['edges']
                   if src in nodes_part1_set and dst in nodes_part1_set]
    edges_part2 = [(src, dst, etype, ts) for src, dst, etype, ts in data_dict['edges']
                   if src in nodes_part2_set and dst in nodes_part2_set]
    
    data_part1 = {
        'nodes': nodes_part1,
        'edges': edges_part1,
        'graph': None
    }
    data_part2 = {
        'nodes': nodes_part2,
        'edges': edges_part2,
        'graph': None
    }
    
    return data_part1, data_part2

def save_graph_data(output_dir, dataset, train_data, test_data, features_train, features_test, 
                    edge_types, scaler, hash_to_uuid=None):
    dataset_dir = os.path.join(output_dir, dataset)
    ensure_dir(dataset_dir)
    
    num_train_nodes = len(train_data['nodes'])
    num_test_nodes = len(test_data['nodes'])
    train_split_idx = math.ceil(num_train_nodes / 2)
    test_split_idx = math.ceil(num_test_nodes / 2)
    
    train_data_part1, train_data_part2 = split_data_dict(train_data, train_split_idx)
    test_data_part1, test_data_part2 = split_data_dict(test_data, test_split_idx)
    
    with zipfile.ZipFile(os.path.join(dataset_dir, 'train_data_part1.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
        with zf.open('train_data.pkl', 'w') as f:
            pickle.dump(train_data_part1, f)
    
    with zipfile.ZipFile(os.path.join(dataset_dir, 'train_data_part2.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
        with zf.open('train_data.pkl', 'w') as f:
            pickle.dump(train_data_part2, f)
    
    with zipfile.ZipFile(os.path.join(dataset_dir, 'test_data_part1.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
        with zf.open('test_data.pkl', 'w') as f:
            pickle.dump(test_data_part1, f)
    
    with zipfile.ZipFile(os.path.join(dataset_dir, 'test_data_part2.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
        with zf.open('test_data.pkl', 'w') as f:
            pickle.dump(test_data_part2, f)
    
    if hash_to_uuid:
        with open(os.path.join(dataset_dir, 'hash_to_uuid.json'), 'w') as f:
            json.dump(hash_to_uuid, f, indent=2)
    
    num_train_features = len(features_train)
    num_test_features = len(features_test)
    train_features_split_idx = math.ceil(num_train_features / 2)
    test_features_split_idx = math.ceil(num_test_features / 2)
    
    features_train_part1 = features_train.iloc[:train_features_split_idx].reset_index(drop=True)
    features_train_part2 = features_train.iloc[train_features_split_idx:].reset_index(drop=True)
    features_test_part1 = features_test.iloc[:test_features_split_idx].reset_index(drop=True)
    features_test_part2 = features_test.iloc[test_features_split_idx:].reset_index(drop=True)
    
    with zipfile.ZipFile(os.path.join(dataset_dir, 'features_train_part1.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
        with zf.open('features_train.pkl', 'w') as f:
            pickle.dump(features_train_part1, f)
    
    with zipfile.ZipFile(os.path.join(dataset_dir, 'features_train_part2.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
        with zf.open('features_train.pkl', 'w') as f:
            pickle.dump(features_train_part2, f)
    
    with zipfile.ZipFile(os.path.join(dataset_dir, 'features_test_part1.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
        with zf.open('features_test.pkl', 'w') as f:
            pickle.dump(features_test_part1, f)
    
    with zipfile.ZipFile(os.path.join(dataset_dir, 'features_test_part2.pkl.zip'), 'w', zipfile.ZIP_DEFLATED) as zf:
        with zf.open('features_test.pkl', 'w') as f:
            pickle.dump(features_test_part2, f)
    
    with open(os.path.join(dataset_dir, 'edge_types.pkl'), 'wb') as f:
        pickle.dump(edge_types, f)
    
    if scaler is not None:
        with open(os.path.join(dataset_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

def load_pca_embeddings_as_dataframe(dataset, embedding_type, feature_type, pca_dim,
                                     nodes_dict, embedding_path, hash_to_uuid=None):
    from sentence_transformers import SentenceTransformer
    
    print(f"\n[Loading LLM Embeddings]")
    print(f"  Embedding: {embedding_type}, Feature: {feature_type}, Dim: {pca_dim}D")
    
    pca_file = os.path.join(
        embedding_path, dataset.lower(), embedding_type.lower(),
        f"{feature_type}_pca{pca_dim}_all.pkl"
    )
    
    if not os.path.exists(pca_file):
        print(f"\nError: PCA embeddings not found: {pca_file}")
        print(f"\nPlease run llmfet-pca-embedding.py first:")
        print(f"  cd ename-processing")
        print(f"  python llmfet-pca-embedding.py \\")
        print(f"    --dataset {dataset} \\")
        print(f"    --embedding {embedding_type} \\")
        print(f"    --feature_type {feature_type} \\")
        print(f"    --pca_dim {pca_dim} \\")
        print(f"    --train_start_date <YOUR_TRAIN_START> \\")
        print(f"    --train_end_date <YOUR_TRAIN_END> \\")
        print(f"    --test_start_date <YOUR_TEST_START> \\")
        print(f"    --test_end_date <YOUR_TEST_END>")
        sys.exit(1)
    
    with open(pca_file, 'rb') as f:
        pca_embeddings = pickle.load(f)
    
    print(f"  ✓ Loaded {len(pca_embeddings):,} pre-computed PCA embeddings ({pca_dim}D)")
    
    pca_model_file = os.path.join(
        embedding_path, dataset.lower(), embedding_type.lower(),
        f"{feature_type}_pca{pca_dim}_model.pkl"
    )
    
    with open(pca_model_file, 'rb') as f:
        pca_model = pickle.load(f)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flash_dir = os.path.join(os.path.dirname(script_dir), 'FLASH')
    mapping_file = os.path.join(flash_dir, f"llmgeneratedvtypegroup_{dataset.lower()}.pkl")
    
    with open(mapping_file, 'rb') as f:
        vtype_mapping = pickle.load(f)
    
    print(f"  Loading sentence transformer for fallback...")
    model_map = {
        "mpnet": 'sentence-transformers/all-mpnet-base-v2',
        "minilm": 'sentence-transformers/all-MiniLM-L6-v2',
        "roberta": 'roberta-base',
        "distilbert": 'sentence-transformers/all-distilroberta-v1'
    }
    embedding_model = SentenceTransformer(model_map[embedding_type])
    print(f"  ✓ Initialized {embedding_type} sentence transformer")
    
    node_uuids = list(nodes_dict.keys())
    embeddings_list = []
    missing_count = 0
    
    for node_uuid in tqdm(node_uuids, desc="  Creating embedding features"):
        lookup_uuid = node_uuid
        if hash_to_uuid and node_uuid in hash_to_uuid:
            lookup_uuid = hash_to_uuid[node_uuid]

        if lookup_uuid in pca_embeddings:
            emb = pca_embeddings[lookup_uuid]
        else:
            vtype = nodes_dict[node_uuid].get('type', 'unknown')
            semantic_group = vtype_mapping.get(vtype, vtype)
            raw_emb = embedding_model.encode(semantic_group)
            emb = pca_model.transform(raw_emb.reshape(1, -1))[0]
            missing_count += 1
        
        embeddings_list.append(emb)
    
    if missing_count > 0:
        print(f"  Note: {missing_count:,}/{len(node_uuids):,} nodes used fallback embeddings")
    
    embeddings_array = np.vstack(embeddings_list)
    
    col_names = [f'emb_{i}' for i in range(pca_dim)]
    
    features_df = pd.DataFrame(embeddings_array, columns=col_names)
    features_df.insert(0, 'node_uuid', node_uuids)
    
    print(f"  ✓ Created embedding features: {features_df.shape[0]:,} nodes × {pca_dim}D")
    
    return features_df, None

def is_dir_in_date_range(dir_path, start_date, end_date):
    base_name = os.path.basename(dir_path.rstrip('/'))
    parts = base_name.split('_')
    if not parts:
        return True
    
    start_time_str = parts[0]
    date_parts = start_time_str.split(' ')
    if not date_parts:
        return True
    
    dir_date_str = date_parts[0]
    dir_date = datetime.strptime(dir_date_str, "%Y-%m-%d")
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    return start_dt <= dir_date <= end_dt

def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    
    dataset = "theia"
    
    start_date = "2018-04-03"
    end_date = "2018-04-12"
    train_start_date = "2018-04-03"
    train_end_date = "2018-04-05"
    test_start_date = "2018-04-09"
    test_end_date = "2018-04-12"
    
    output_dir = os.path.join(autoprov_dir, 'BIGDATA', 'OCR_APT_artifacts')
    
    if args.baseline:
        rulellm = False
        llmlabel = False
        use_ocr_apt_features = True
        dataset_with_suffix = dataset
        mode_str = 'Baseline (Regex + OCR-APT Features)'
    else:
        rulellm = True
        llmlabel = True
        use_ocr_apt_features = False
        dataset_with_suffix = f"{dataset}_rulellm_llmlabel_{args.embedding}"
        mode_str = f'AutoProv (RuleLLM + LLM Type Embeddings, {args.embedding})'
    
    print(f"\n{'='*80}")
    print(f"OCR_APT Graph Generation - THEIA")
    print(f"Mode: {mode_str}")
    print(f"{'='*80}\n")
    
    if not rulellm:
        dataset_dir_name = dataset.upper()
        full_dataset_path = f'{args.dataset_path}/{dataset_dir_name}/*/'
        
        subject_set, subject2hash = process_subject_nodes(
            full_dataset_path, start_date, end_date
        )
        file_set, file2hash = process_file_nodes(
            full_dataset_path, start_date, end_date
        )
        netobj_set, netobj2hash = process_netflow_nodes(
            full_dataset_path, start_date, end_date
        )
        
        nodes = create_nodes_from_regex(subject2hash, file2hash, netobj2hash)
        
        hash_to_uuid = {}
        for hash_dict in [subject2hash, file2hash, netobj2hash]:
            for key, value in hash_dict.items():
                if len(key) == 64:
                    if isinstance(value, str):
                        hash_to_uuid[key] = value
                elif isinstance(value, list) and len(value) > 0:
                    hash_val = value[0]
                    if len(hash_val) == 64:
                        hash_to_uuid[hash_val] = key
        
        edges, edge_types = process_regex_edges(
            full_dataset_path,
            subject2hash, file2hash, netobj2hash,
            start_date, end_date
        )
    else:
        dataset_dir_name = dataset.upper()
        csv_dataset_path = f'{args.extracted_graph_path}/{dataset_dir_name}/*'
        
        vtype_mapping = load_vtype_mapping(None, dataset)
        edge_validation = load_edge_validation(None, dataset)
        
        nodes, edges, edge_types = process_csv_nodes_and_edges(
            csv_dataset_path, start_date, end_date,
            vtype_mapping, edge_validation
        )
        
        hash_to_uuid = None
    
    validate_graph_data(nodes, edges, edge_types)
    
    display_final_vtypes(nodes)
    
    if llmlabel:
        feature_type = 'type'
        
        embedding_path = os.path.join(autoprov_dir, 'BIGDATA', 'llmfets-pca-embedding')
        
        features_all, scaler = load_pca_embeddings_as_dataframe(
            dataset, args.embedding, feature_type, args.pca_dim,
            nodes, embedding_path, hash_to_uuid=hash_to_uuid
        )
        
        print(f"\n✓ Using LLM {feature_type} embeddings ({args.embedding}, {args.pca_dim}D)")
    else:
        features_all, scaler = feature_engineering_ocrapt(
            nodes, edges, edge_types, dataset
        )
    
    train_nodes_dict, test_nodes_dict, train_edges, test_edges = split_train_test_by_dates(
        edges, nodes, 
        train_start_date, train_end_date,
        test_start_date, test_end_date
    )
    
    train_node_uuids = set(train_nodes_dict.keys())
    test_node_uuids = set(test_nodes_dict.keys())
    features_train = features_all[features_all['node_uuid'].isin(train_node_uuids)].reset_index(drop=True)
    features_test = features_all[features_all['node_uuid'].isin(test_node_uuids)].reset_index(drop=True)
    
    train_data = {
        'nodes': train_nodes_dict,
        'edges': train_edges,
        'graph': construct_networkx_graph(train_nodes_dict, train_edges, edge_types)
    }
    
    test_data = {
        'nodes': test_nodes_dict,
        'edges': test_edges,
        'graph': construct_networkx_graph(test_nodes_dict, test_edges, edge_types)
    }
    
    if rulellm:
        split_and_save_by_vtype(
            output_dir, dataset_with_suffix,
            train_data, test_data,
            features_train, features_test,
            edge_types, scaler, hash_to_uuid
        )
    else:
        save_graph_data(
            output_dir, dataset_with_suffix,
            train_data, test_data,
            features_train, features_test,
            edge_types, scaler, hash_to_uuid
        )
    
    print(f"\n{'='*80}")
    print(f"✓ Graph generation completed: {output_dir}/{dataset_with_suffix}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

