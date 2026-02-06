#!/usr/bin/env python3

import os
import sys
import argparse

def parse_args_early():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids (e.g. 0,1,2); default empty = use all available")
    args, _ = parser.parse_known_args()
    return args.gpus

gpus = parse_args_early()
if gpus and gpus.strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import json
import pickle
import glob
import re
import numpy as np
from tqdm import tqdm
import networkx as nx
import hashlib
import zipfile
from datetime import datetime, timedelta
import time
import pytz
from time import mktime
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FLASH'))
from stepllm_utils.vtype_res import getvtypes

DATASET_DATES = {
    "THEIA": {
        "train_start_date": "2018-04-03",
        "train_end_date": "2018-04-05",
        "test_start_date": "2018-04-09",
        "test_end_date": "2018-04-12",
        "start_date": "2018-04-03",
        "end_date": "2018-04-12"
    }
}

PCA_DIM = 128

def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

def ns_time_to_datetime_US(ns):
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1000000000, tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def datetime_to_ns_time_US(date):
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
    return int(timeStamp)

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

def process_netflow_nodes(file_path):
    netobjset = set()
    netobj2hash = {}
    
    timestamp_dirs = sorted(glob.glob(file_path))
    
    for timestamp_dir in tqdm(timestamp_dirs, desc="Extracting NetFlow nodes"):
        data = load_log_data(timestamp_dir)
        
        for line in data:
            if '{"datum":{"com.bbn.tc.schema.avro.cdm18.NetFlowObject"' in line:
                uuid_matches = re.findall('NetFlowObject":{"uuid":"(.*?)"', line)
                if not uuid_matches:
                    continue
                nodeid = uuid_matches[0]
                
                srcaddr = "null"
                srcport = "null"
                dstaddr = "null"
                dstport = "null"
                
                local_addr_matches = re.findall('"localAddress":"(.*?)"', line)
                if local_addr_matches:
                    srcaddr = local_addr_matches[0]
                
                local_port_matches = re.findall('"localPort":(.*?),', line)
                if local_port_matches:
                    srcport = local_port_matches[0].strip('"')
                
                remote_addr_matches = re.findall('"remoteAddress":"(.*?)"', line)
                if remote_addr_matches:
                    dstaddr = remote_addr_matches[0]
                
                remote_port_matches = re.findall('"remotePort":(.*?),', line)
                if remote_port_matches:
                    dstport = remote_port_matches[0].strip('"')
                
                nodeproperty = srcaddr + "," + srcport + "," + dstaddr + "," + dstport
                hashstr = stringtomd5(nodeproperty)
                netobj2hash[nodeid] = [hashstr, nodeproperty]
                netobj2hash[hashstr] = nodeid
                netobjset.add(hashstr)
    
    return netobjset, netobj2hash

def process_subject_nodes(file_path):
    subjectset = set()
    subject2hash = {}
    
    timestamp_dirs = sorted(glob.glob(file_path))
    
    for timestamp_dir in tqdm(timestamp_dirs, desc="Extracting Subject nodes"):
        data = load_log_data(timestamp_dir)
        
        for line in data:
            if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line:
                uuid_matches = re.findall('Subject":{"uuid":"(.*?)"', line)
                if not uuid_matches:
                    continue
                nodeid = uuid_matches[0]
                
                cmdLine = "null"
                cmdLine_matches = re.findall('"cmdLine":{"string":"(.*?)"', line)
                if cmdLine_matches:
                    cmdLine = cmdLine_matches[0]
                
                tgid = "null"
                tgid_matches = re.findall('"properties":{"map":{"tgid":"(.*?)"', line)
                if tgid_matches:
                    tgid = tgid_matches[0]
                
                path_matches = re.findall('"path":"(.*?)"', line)
                path = path_matches[0] if path_matches else "null"
                
                nodeproperty = cmdLine + "," + tgid + "," + path
                hashstr = stringtomd5(nodeproperty)
                subject2hash[nodeid] = [hashstr, cmdLine, tgid, path]
                subject2hash[hashstr] = nodeid
                subjectset.add(hashstr)
    
    return subjectset, subject2hash

def process_file_nodes(file_path):
    fileset = set()
    file2hash = {}
    
    timestamp_dirs = sorted(glob.glob(file_path))
    
    for timestamp_dir in tqdm(timestamp_dirs, desc="Extracting File nodes"):
        data = load_log_data(timestamp_dir)
        
        for line in data:
            if '{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' in line:
                uuid_matches = re.findall('FileObject":{"uuid":"(.*?)"', line)
                if not uuid_matches:
                    continue
                nodeid = uuid_matches[0]
                
                filepath = "null"
                filename_matches = re.findall('"filename":"(.*?)"', line)
                if filename_matches:
                    filepath = filename_matches[0]
                
                nodeproperty = filepath
                hashstr = stringtomd5(nodeproperty)
                file2hash[nodeid] = [hashstr, nodeproperty]
                file2hash[hashstr] = nodeid
                fileset.add(hashstr)
    
    return fileset, file2hash

def get_vtype_mapping(dataset="theia"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flash_dir = os.path.join(os.path.dirname(script_dir), 'FLASH')
    mapping_file = os.path.join(flash_dir, f"llmgeneratedvtypegroup_{dataset.lower()}.pkl")
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Vertex type mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'rb') as f:
        vtype_mapping = pickle.load(f)
    
    return vtype_mapping

def load_edge_type_validation(dataset="theia"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ename_processing_dir = os.path.join(os.path.dirname(script_dir), 'ename-processing')
    validation_file = os.path.join(ename_processing_dir, f"edge_type_validation_{dataset.lower()}.json")
    
    if not os.path.exists(validation_file):
        return {}
    
    with open(validation_file, 'r') as f:
        edge_validation = json.load(f)
    
    return edge_validation

def load_precomputed_embeddings(dataset, embedding_type, feature_type, embedding_path):
    embedding_file = os.path.join(
        embedding_path,
        f"{embedding_type.lower()}_{dataset.lower()}_{feature_type}.pkl"
    )
    
    if not os.path.exists(embedding_file):
        return {}
    
    with open(embedding_file, 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    return embeddings_dict

def initialize_text_embedder(embedding_type):
    from sentence_transformers import SentenceTransformer
    
    model_map = {
        "mpnet": 'sentence-transformers/all-mpnet-base-v2',
        "minilm": 'sentence-transformers/all-MiniLM-L6-v2',
        "roberta": 'roberta-base',
        "distilbert": 'sentence-transformers/all-distilroberta-v1'
    }
    
    model = SentenceTransformer(model_map[embedding_type])
    return model

def create_node_embeddings_with_pca(nodes_dict, precomputed_embeddings, vtype_mapping, 
                                     embedding_model, is_train, pca_model=None, 
                                     pca_dim=128, use_pca=True):
    from sklearn.decomposition import IncrementalPCA
    
    node_ids = list(nodes_dict.keys())
    total_nodes = len(node_ids)
    
    mode_str = "train" if is_train else "test"
    batch_size = 50000
    num_batches = (total_nodes + batch_size - 1) // batch_size
    
    all_embeddings = []
    missing_count = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_nodes)
        batch_node_ids = node_ids[start_idx:end_idx]
        
        batch_embeddings = []
        for node_id in tqdm(batch_node_ids, 
                           desc=f"Generating {mode_str} embeddings",
                           leave=False):
            if node_id in precomputed_embeddings:
                embedding = precomputed_embeddings[node_id]
            else:
                vtype = list(nodes_dict[node_id].keys())[0]
                semantic_group = vtype_mapping.get(vtype, vtype)
                embedding = embedding_model.encode(semantic_group)
                missing_count += 1
            
            batch_embeddings.append(embedding)
        
        batch_array = np.vstack(batch_embeddings)
        all_embeddings.append(batch_array)
    
    embeddings_array = np.vstack(all_embeddings)
    original_dim = embeddings_array.shape[1]
    
    if use_pca:
        if is_train:
            ipca = IncrementalPCA(n_components=pca_dim, batch_size=10000)
            
            pca_batch_size = 10000
            num_pca_batches = (embeddings_array.shape[0] + pca_batch_size - 1) // pca_batch_size
            
            for i in tqdm(range(num_pca_batches), desc=f"Fitting PCA ({original_dim}→{pca_dim}D)", leave=False):
                start_idx = i * pca_batch_size
                end_idx = min((i + 1) * pca_batch_size, embeddings_array.shape[0])
                batch = embeddings_array[start_idx:end_idx]
                ipca.partial_fit(batch)
            
            result_batches = []
            for i in tqdm(range(num_pca_batches), desc="Transforming train", leave=False):
                start_idx = i * pca_batch_size
                end_idx = min((i + 1) * pca_batch_size, embeddings_array.shape[0])
                batch = embeddings_array[start_idx:end_idx]
                result_batches.append(ipca.transform(batch))
            
            embeddings_pca = np.vstack(result_batches)
            return embeddings_pca, node_ids, ipca
            
        else:
            if pca_model is None:
                raise ValueError("PCA model required for test data but not provided")
            
            pca_batch_size = 10000
            num_pca_batches = (embeddings_array.shape[0] + pca_batch_size - 1) // pca_batch_size
            
            result_batches = []
            for i in tqdm(range(num_pca_batches), desc="Transforming test", leave=False):
                start_idx = i * pca_batch_size
                end_idx = min((i + 1) * pca_batch_size, embeddings_array.shape[0])
                batch = embeddings_array[start_idx:end_idx]
                result_batches.append(pca_model.transform(batch))
            
            embeddings_pca = np.vstack(result_batches)
            return embeddings_pca, node_ids, pca_model
    else:
        return embeddings_array, node_ids, None

def fill_missing_timestamps(df):
    missing_before = df['timestamp'].isna().sum()
    
    if missing_before == 0:
        return df
    
    df['timestamp'] = df['timestamp'].bfill()
    
    df['timestamp'] = df['timestamp'].ffill()
    
    missing_after = df['timestamp'].isna().sum()
    
    return df

def load_csv_data(timestamp_dir):
    csv_files = glob.glob(os.path.join(timestamp_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {timestamp_dir}")
    
    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)
    
    df = fill_missing_timestamps(df)
    
    return df

def process_csv_nodes_and_edges(file_path, id_labels, start_date_str, end_date_str, vtype_combinations, vtype_mapping, edge_validation):
    nodes = {}
    edges = []
    edge_types_set = set()
    nodeid2address = {}
    
    def is_no_label_action(action):
        if action == "NO LABEL":
            return True
        if action in edge_validation and edge_validation[action] == "INVALID":
            return True
        return False
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    timestamp_dirs = glob.glob(file_path)
    
    for timestamp_dir in tqdm(timestamp_dirs, desc="Processing CSV directories", leave=False):
        base_name = timestamp_dir.split('/')[-2]
        file_date_str = base_name.split(' ')[0]
        file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
        
        if not (start_date <= file_date <= end_date):
            continue
        
        df = load_csv_data(timestamp_dir)
        
        df = df[df['action'].notna()]
        if df.empty:
            continue
        
        def parse_list_fast(s):
            if not isinstance(s, str) or s == '[]' or not s.strip():
                return []
            s = s.strip('[]').replace("'", "").replace('"', '')
            return [x.strip() for x in s.split(',') if x.strip()]
        
        def create_compound_vtype(val_str):
            vals = parse_list_fast(val_str)
            if vals:
                unique_elements = sorted(set([v.lower() for v in vals]))
                return '+'.join(unique_elements)
            return None
        
        def safe_eval_longest_ename(val_str):
            vals = parse_list_fast(val_str)
            meaningless = {'datum', 'N/A', '},'}
            clean = [v for v in vals if v not in meaningless and len(v) > 1]
            return max(clean, key=len) if clean else ''
        
        df['src_vtype'] = df['source_vtypes'].apply(create_compound_vtype)
        df['dst_vtype'] = df['dest_vtypes'].apply(create_compound_vtype)
        df['src_ename'] = df['source_enames'].apply(safe_eval_longest_ename)
        df['dst_ename'] = df['dest_enames'].apply(safe_eval_longest_ename)
        df['is_no_label'] = df['action'].apply(is_no_label_action)
        
        log_idx_groups = {}
        no_label_df = df[df['is_no_label']]
        
        for row in no_label_df.itertuples(index=True, name=None):
            log_idx = row[0]
            src_id = row[no_label_df.columns.get_loc('source_id') + 1]
            dst_id = row[no_label_df.columns.get_loc('dest_id') + 1]
            src_vtype = row[no_label_df.columns.get_loc('src_vtype') + 1]
            dst_vtype = row[no_label_df.columns.get_loc('dst_vtype') + 1]
            src_ename = row[no_label_df.columns.get_loc('src_ename') + 1]
            dst_ename = row[no_label_df.columns.get_loc('dst_ename') + 1]
            
            if src_id not in id_labels or dst_id not in id_labels:
                continue
            
            src_vtype_final = src_vtype if src_vtype else id_labels.get(src_id, 'unknown')
            dst_vtype_final = dst_vtype if dst_vtype else id_labels.get(dst_id, 'unknown')
            
            nodes[src_id] = {src_vtype_final: src_ename}
            nodes[dst_id] = {dst_vtype_final: dst_ename}
            
            if log_idx not in log_idx_groups:
                log_idx_groups[log_idx] = []
            
            src_group = vtype_mapping.get(src_vtype_final, 'other').lower()
            dst_group = vtype_mapping.get(dst_vtype_final, 'other').lower()
            
            log_idx_groups[log_idx].append((src_id, src_vtype_final, src_group))
            log_idx_groups[log_idx].append((dst_id, dst_vtype_final, dst_group))
        
        for log_idx, id_info_list in log_idx_groups.items():
            unique_ids = {}
            for node_id, vtype_label, mapped_group in id_info_list:
                unique_ids[node_id] = (vtype_label, mapped_group)
            
            address_ids = []
            non_host_ids = []
            
            for node_id, (vtype_label, mapped_group) in unique_ids.items():
                if 'address' in mapped_group or mapped_group == 'local+remote':
                    address_ids.append(node_id)
                elif 'host' not in mapped_group:
                    non_host_ids.append(node_id)
            
            if address_ids and non_host_ids:
                concatenated_addresses = '|'.join(address_ids)
                for non_host_id in non_host_ids:
                    nodeid2address[non_host_id] = concatenated_addresses
        
        valid_df = df[~df['is_no_label']]
        
        for row in valid_df[['source_id', 'dest_id', 'action', 'timestamp', 'src_vtype', 'dst_vtype', 'src_ename', 'dst_ename']].itertuples(index=False, name=None):
            src_id, dst_id, action, timestamp, src_vtype, dst_vtype, src_ename, dst_ename = row
            
            if src_id not in id_labels or dst_id not in id_labels:
                continue
            
            if not isinstance(action, str):
                continue
            
            src_vtype_final = src_vtype if src_vtype else id_labels.get(src_id, 'unknown')
            dst_vtype_final = dst_vtype if dst_vtype else id_labels.get(dst_id, 'unknown')
            
            if src_id not in nodes:
                nodes[src_id] = {src_vtype_final: src_ename}
            if dst_id not in nodes:
                nodes[dst_id] = {dst_vtype_final: dst_ename}
            
            edges.append((src_id, dst_id, action, timestamp))
            edge_types_set.add(action)
    
    return nodes, edges, list(edge_types_set), nodeid2address

node_type_dict = {}
edge_type_dict = {}
node_type_cnt = 0
edge_type_cnt = 0

def create_node_type_mappings(subject2hash, file2hash, netobj2hash):
    global node_type_dict, node_type_cnt
    
    node_list = {}
    node_index = 0
    
    node_types = ['FileObject', 'Subject', 'NetFlowObject']
    for i, node_type in enumerate(node_types):
        if node_type not in node_type_dict:
            node_type_dict[node_type] = node_type_cnt
            node_type_cnt += 1
    
    for uuid, info in file2hash.items():
        if len(uuid) == 64:
            node_list[uuid] = ["FileObject", info[1], node_index]
            node_index += 1
    
    for uuid, info in subject2hash.items():
        if len(uuid) == 64:
            node_list[uuid] = ["Subject", info[3], node_index]
            node_index += 1
    
    for uuid, info in netobj2hash.items():
        if len(uuid) == 64:
            node_list[uuid] = ["NetFlowObject", info[1], node_index]
            node_index += 1
    
    return node_list

def process_events(file_path, subject2hash, file2hash, netobj2hash, node_list, start_date_str, end_date_str):
    global edge_type_dict, edge_type_cnt
    
    events_by_window = {}
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    timestamp_dirs = sorted(glob.glob(file_path))
    
    skipped_by_date = 0
    
    for timestamp_dir in tqdm(timestamp_dirs, desc="Processing Events"):
        base_name = os.path.basename(timestamp_dir.rstrip('/'))
        start_time_str = base_name.split('_')[0]
        dir_date = datetime.strptime(start_time_str.split(' ')[0], "%Y-%m-%d")
        dir_hour = int(start_time_str.split(' ')[1].split(':')[0])
        
        if not (start_date <= dir_date <= end_date):
            skipped_by_date += 1
            continue
        
        window_key = f"{dir_date.strftime('%Y-%m-%d')}_{dir_hour:02d}"
        
        if window_key not in events_by_window:
            events_by_window[window_key] = []
        
        data = load_log_data(timestamp_dir)
        
        for line in data:
            if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line and "EVENT_FLOWS_TO" not in line:
                timestamp_matches = re.findall('"timestampNanos":(.*?),', line)
                if not timestamp_matches:
                    continue
                time_ns = int(timestamp_matches[0])
                
                subject_matches = re.findall('"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"', line)
                if not subject_matches:
                    continue
                subjectid = subject_matches[0]
                
                object_matches = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"', line)
                if not object_matches:
                    continue
                objectid = object_matches[0]
                
                type_matches = re.findall('"type":"(.*?)"', line)
                if not type_matches:
                    continue
                relation_type = type_matches[0]
                
                if relation_type not in edge_type_dict:
                    edge_type_dict[relation_type] = edge_type_cnt
                    edge_type_cnt += 1
                
                if subjectid in subject2hash.keys():
                    subjectid = subject2hash[subjectid][0]
                if objectid in subject2hash.keys():
                    objectid = subject2hash[objectid][0]
                if objectid in file2hash.keys():
                    objectid = file2hash[objectid][0]
                if objectid in netobj2hash.keys():
                    objectid = netobj2hash[objectid][0]
                
                if len(subjectid) == 64 and len(objectid) == 64:
                    if subjectid in node_list and objectid in node_list:
                        src_idx = node_list[subjectid][2]
                        dst_idx = node_list[objectid][2]
                        
                        if relation_type in ['EVENT_READ', 'EVENT_READ_SOCKET_PARAMS', 'EVENT_RECVFROM', 'EVENT_RECVMSG']:
                            events_by_window[window_key].append([objectid, dst_idx, relation_type, subjectid, src_idx, time_ns])
                        else:
                            events_by_window[window_key].append([subjectid, src_idx, relation_type, objectid, dst_idx, time_ns])
    
    return events_by_window

def build_graphs(events_by_window, node_list, node_type_dict, edge_type_dict, malicious_entities=None):
    graphs_dict = {}
    
    for window_key, events in tqdm(events_by_window.items(), desc="Building graphs"):
        if len(events) == 0:
            continue
        
        g = nx.DiGraph()
        
        window_nodes = set()
        
        for event in events:
            src_hash, src_idx, edge_type, dst_hash, dst_idx, timestamp = event
            
            if not g.has_node(src_idx):
                src_type = node_list[src_hash][0]
                g.add_node(src_idx, type=node_type_dict[src_type])
                window_nodes.add(src_idx)
            
            if not g.has_node(dst_idx):
                dst_type = node_list[dst_hash][0]
                g.add_node(dst_idx, type=node_type_dict[dst_type])
                window_nodes.add(dst_idx)
            
            if not g.has_edge(src_idx, dst_idx):
                g.add_edge(src_idx, dst_idx, type=edge_type_dict[edge_type])
        
        node_labels = []
        for node_idx in sorted(g.nodes()):
            node_type_id = g.nodes[node_idx]['type']
            node_labels.append(node_type_id)
        
        malicious_nodes = []
        if malicious_entities:
            for node_idx in sorted(g.nodes()):
                pass
        
        graphs_dict[window_key] = {
            'graph': g,
            'node_labels': node_labels,
            'malicious_nodes': malicious_nodes,
            'num_nodes': g.number_of_nodes(),
            'num_edges': g.number_of_edges()
        }
    
    return graphs_dict

def apply_vtype_mapping_to_combinations(vtype_combinations, vtype_mapping):
    semantic_groups = set()
    for vtype in vtype_combinations.keys():
        mapped_group = vtype_mapping.get(vtype, vtype)
        semantic_groups.add(mapped_group)
    
    mapped_combinations = {group: idx for idx, group in enumerate(sorted(semantic_groups))}
    
    return mapped_combinations

def create_node2id_mapping_csv(nodes_dict, id_labels, vtype_combinations, vtype_mapping=None):
    node2id = {}
    uuid_to_node_idx = {}
    node_labels = []
    
    node_idx = 0
    for uuid, vtype_ename_dict in tqdm(nodes_dict.items(), desc="Creating node mappings"):
        node2id[uuid] = node_idx
        uuid_to_node_idx[uuid] = node_idx
        
        vtype = list(vtype_ename_dict.keys())[0]
        
        if vtype_mapping:
            mapped_vtype = vtype_mapping.get(vtype, vtype)
        else:
            mapped_vtype = vtype
        
        vtype_id = vtype_combinations.get(mapped_vtype, 0)
        node_labels.append(vtype_id)
        
        node_idx += 1
    
    return node2id, uuid_to_node_idx, node_labels

def generate_edge_type_features_auto(edge_types_list):
    edge_label_encoder = LabelEncoder()
    edge_label_encoder.fit(edge_types_list)
    
    rel2id = {edge_type: idx for idx, edge_type in enumerate(edge_label_encoder.classes_)}
    
    return rel2id, edge_label_encoder

def build_graphs_csv(edges_list, node2id, rel2id, node_labels_list, start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    events_by_window = {}
    
    for src_uuid, dst_uuid, edge_type, timestamp in tqdm(edges_list, desc="Grouping edges by window"):
        try:
            dt = datetime.fromtimestamp(int(timestamp) / 1e9)
            date_str = dt.strftime("%Y-%m-%d")
            hour = dt.hour
            window_key = f"{date_str}_{hour:02d}"
            
            event_date = datetime.strptime(date_str, "%Y-%m-%d")
            if not (start_date <= event_date <= end_date):
                continue
            
            if window_key not in events_by_window:
                events_by_window[window_key] = []
            
            if src_uuid in node2id and dst_uuid in node2id:
                src_idx = node2id[src_uuid]
                dst_idx = node2id[dst_uuid]
                edge_type_id = rel2id.get(edge_type, 0)
                
                events_by_window[window_key].append((src_idx, dst_idx, edge_type_id))
        except:
            continue
    
    graphs_dict = {}
    
    for window_key, events in tqdm(events_by_window.items(), desc="Building graphs", leave=False):
        if len(events) == 0:
            continue
        
        g = nx.DiGraph()
        
        for src_idx, dst_idx, edge_type_id in events:
            if not g.has_node(src_idx):
                g.add_node(src_idx, type=node_labels_list[src_idx])
            if not g.has_node(dst_idx):
                g.add_node(dst_idx, type=node_labels_list[dst_idx])
            
            if not g.has_edge(src_idx, dst_idx):
                g.add_edge(src_idx, dst_idx, type=edge_type_id)
        
        window_node_labels = [g.nodes[node_idx]['type'] for node_idx in sorted(g.nodes())]
        
        graphs_dict[window_key] = {
            'graph': g,
            'node_labels': window_node_labels,
            'malicious_nodes': [],
            'num_nodes': g.number_of_nodes(),
            'num_edges': g.number_of_edges()
        }
    
    return graphs_dict

def split_and_zip_embeddings(embeddings_path, base_dir):
    embeddings = np.load(embeddings_path, mmap_mode='r')
    total_rows = embeddings.shape[0]
    split_point = total_rows // 2
    
    part1_path = os.path.join(base_dir, 'processed_data', 'node_embeddings_part1.npy')
    part1 = embeddings[:split_point]
    np.save(part1_path, part1)
    del part1
    
    part2_path = os.path.join(base_dir, 'processed_data', 'node_embeddings_part2.npy')
    part2 = embeddings[split_point:]
    np.save(part2_path, part2)
    del part2
    del embeddings
    
    with zipfile.ZipFile(f'{part1_path}.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(part1_path, os.path.basename(part1_path))
    os.remove(part1_path)
    
    with zipfile.ZipFile(f'{part2_path}.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(part2_path, os.path.basename(part2_path))
    os.remove(part2_path)
    
    if os.path.exists(embeddings_path):
        os.remove(embeddings_path)

def zip_large_file(file_path):
    if not os.path.exists(file_path):
        return None
    
    zip_path = f"{file_path}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, os.path.basename(file_path))
    
    os.remove(file_path)
    
    return zip_path

def parse_args():
    parser = argparse.ArgumentParser(description="MAGIC Graph Generation for THEIA")
    parser.add_argument("--baseline", action="store_true",
                        help="Baseline mode: raw log processing (no embeddings)")
    parser.add_argument("--autoprov", action="store_true",
                        help="AutoProv mode: CSV processing with LLM embeddings (mpnet)")
    parser.add_argument("--dataset_path", type=str, default="../BIGDATA/DARPA-E3/",
                        help="Base path to dataset directory (for --baseline mode)")
    parser.add_argument("--autoprov_graph_path", type=str, default="../BIGDATA/ExtractedProvGraph/",
                        help="Path to extracted provenance graphs (for --autoprov mode)")
    parser.add_argument("--embedding", type=str, default="mpnet",
                        choices=["mpnet", "roberta", "distilbert"],
                        help="Embedding model to use (for --autoprov mode, default: mpnet)")
    parser.add_argument("--embedding_path", type=str, default=None,
                        help="Path to pre-computed embeddings directory (for --autoprov mode). If not provided, uses AutoProv/BIGDATA/llmfets-embedding")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.baseline and args.autoprov:
        return
    
    if not args.baseline and not args.autoprov:
        return
    
    dataset = "THEIA"
    dataset_dates = DATASET_DATES[dataset]
    
    start_date = dataset_dates.get("start_date")
    end_date = dataset_dates.get("end_date")
    train_start_date = dataset_dates.get("train_start_date")
    train_end_date = dataset_dates.get("train_end_date")
    test_start_date = dataset_dates.get("test_start_date")
    test_end_date = dataset_dates.get("test_end_date")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(os.path.dirname(script_dir), "BIGDATA", "MAGIC_artifacts")
    
    if args.embedding_path is None:
        args.embedding_path = os.path.join(os.path.dirname(script_dir), "BIGDATA", "llmfets-embedding")
    
    if args.baseline:
        base_dir = os.path.join(artifacts_dir, dataset)
        rulellm = False
        llmlabel = False
        embedding = "roberta"
    else:
        embedding = args.embedding
        base_dir = os.path.join(artifacts_dir, f"{dataset}_rulellm_llmlabel_{embedding}")
        rulellm = True
        llmlabel = True
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f'{base_dir}/processed_data/graphs', exist_ok=True)
    os.makedirs(f'{base_dir}/processed_data/node_mappings', exist_ok=True)
    
    if args.baseline:
        full_dataset_path = f'{args.dataset_path}/{dataset}/*/'
        
        netobjset, netobj2hash = process_netflow_nodes(full_dataset_path)
        subjectset, subject2hash = process_subject_nodes(full_dataset_path)
        fileset, file2hash = process_file_nodes(full_dataset_path)
        node_list = create_node_type_mappings(subject2hash, file2hash, netobj2hash)
        
        with open(f'{base_dir}/processed_data/node_mappings/node_list.pkl', 'wb') as f:
            pickle.dump(node_list, f)
        with open(f'{base_dir}/processed_data/node_mappings/node_type_dict.pkl', 'wb') as f:
            pickle.dump(node_type_dict, f)
        
        events_by_window = process_events(
            full_dataset_path, subject2hash, file2hash, netobj2hash, node_list,
            start_date, end_date
        )
        
        with open(f'{base_dir}/processed_data/node_mappings/edge_type_dict.pkl', 'wb') as f:
            pickle.dump(edge_type_dict, f)
        
        graphs_dict = build_graphs(events_by_window, node_list, node_type_dict, edge_type_dict)
        
        for window_key, graph_data in tqdm(graphs_dict.items(), desc="Saving graphs", leave=False):
            graph_json = nx.node_link_data(graph_data['graph'], edges="links")
            
            save_data = {
                'graph': graph_json,
                'node_labels': graph_data['node_labels'],
                'malicious_nodes': graph_data['malicious_nodes'],
                'num_nodes': graph_data['num_nodes'],
                'num_edges': graph_data['num_edges']
            }
            
            save_path = f'{base_dir}/processed_data/graphs/graph_{window_key}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
        
        uuid_to_node_idx = {}
        for hash_val, node_info in node_list.items():
            node_idx = node_info[2]
            uuid = None
            if hash_val in subject2hash:
                result = subject2hash[hash_val]
                if isinstance(result, str):
                    uuid = result
            elif hash_val in file2hash:
                result = file2hash[hash_val]
                if isinstance(result, str):
                    uuid = result
            elif hash_val in netobj2hash:
                result = netobj2hash[hash_val]
                if isinstance(result, str):
                    uuid = result
            
            if uuid:
                uuid_to_node_idx[uuid] = node_idx
        
        mapping_path = f'{base_dir}/processed_data/uuid_to_node_idx.json'
        with open(mapping_path, 'w') as f:
            json.dump(uuid_to_node_idx, f, indent=2)
        
        metadata = {
            'dataset': dataset,
            'mode': 'raw',
            'start_date': start_date,
            'end_date': end_date,
            'node_feature_dim': len(node_type_dict),
            'edge_feature_dim': len(edge_type_dict),
            'num_time_windows': len(graphs_dict),
            'time_window_type': 'hourly',
            'total_nodes': len(node_list),
            'node_type_dict': node_type_dict,
            'edge_type_dict': edge_type_dict,
            'detection_mode': 'entity_level'
        }
        
        with open(f'{base_dir}/processed_data/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    else:
        csv_dataset_path = f'{args.autoprov_graph_path}/{dataset}/*/'
        
        vtype_combinations, id_labels = getvtypes(dataset.lower(), args.autoprov_graph_path)
        vtype_mapping = get_vtype_mapping(dataset.lower())
        edge_validation = load_edge_type_validation(dataset.lower())
        
        start_date_for_processing = train_start_date
        end_date_for_processing = test_end_date
        
        nodes_dict, edges_list, edge_types_list, nodeid2address = process_csv_nodes_and_edges(
            csv_dataset_path, id_labels, start_date_for_processing, end_date_for_processing,
            vtype_combinations, vtype_mapping, edge_validation
        )
        
        mapped_vtype_combinations = apply_vtype_mapping_to_combinations(vtype_combinations, vtype_mapping)
        
        feature_type = 'type'
        
        pca_base_path = "../BIGDATA/llmfets-pca-embedding/"
        pca_emb_file = os.path.join(pca_base_path, dataset.lower(), embedding, 
                                     f"{feature_type}_pca{PCA_DIM}_all.pkl")
        pca_model_file = os.path.join(pca_base_path, dataset.lower(), embedding,
                                       f"{feature_type}_pca{PCA_DIM}_model.pkl")
        
        use_precomputed_pca = os.path.exists(pca_emb_file) and os.path.exists(pca_model_file)
        
        precomputed_embeddings = load_precomputed_embeddings(
            dataset.lower(), embedding, feature_type, args.embedding_path
        )
        embedding_model = initialize_text_embedder(embedding)
        
        train_nodes_dict = {}
        test_nodes_dict = {}
        train_edges_list = []
        test_edges_list = []
        
        train_start = datetime.strptime(train_start_date, "%Y-%m-%d")
        train_end = datetime.strptime(train_end_date, "%Y-%m-%d")
        test_start = datetime.strptime(test_start_date, "%Y-%m-%d")
        test_end = datetime.strptime(test_end_date, "%Y-%m-%d")
        for src_uuid, dst_uuid, edge_type, timestamp in edges_list:
            try:
                dt = datetime.fromtimestamp(int(timestamp) / 1e9)
                edge_date = datetime.strptime(dt.strftime("%Y-%m-%d"), "%Y-%m-%d")
                
                if train_start <= edge_date <= train_end:
                    train_edges_list.append((src_uuid, dst_uuid, edge_type, timestamp))
                    if src_uuid in nodes_dict:
                        train_nodes_dict[src_uuid] = nodes_dict[src_uuid]
                    if dst_uuid in nodes_dict:
                        train_nodes_dict[dst_uuid] = nodes_dict[dst_uuid]
                elif test_start <= edge_date <= test_end:
                    test_edges_list.append((src_uuid, dst_uuid, edge_type, timestamp))
                    if src_uuid in nodes_dict:
                        test_nodes_dict[src_uuid] = nodes_dict[src_uuid]
                    if dst_uuid in nodes_dict:
                        test_nodes_dict[dst_uuid] = nodes_dict[dst_uuid]
            except:
                continue
        
        if use_precomputed_pca:
            with open(pca_emb_file, 'rb') as f:
                precomputed_pca_embeddings = pickle.load(f)
            with open(pca_model_file, 'rb') as f:
                pca_model = pickle.load(f)
            
            all_nodes_dict = {**train_nodes_dict, **test_nodes_dict}
            all_node_ids = list(all_nodes_dict.keys())
            node2id = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
            uuid_to_node_idx = node2id.copy()
            
            all_embeddings = []
            
            for node_id in tqdm(all_node_ids, desc="Creating embeddings", leave=False):
                if node_id in precomputed_pca_embeddings:
                    all_embeddings.append(precomputed_pca_embeddings[node_id])
                else:
                    vtype = list(all_nodes_dict[node_id].keys())[0]
                    semantic_group = vtype_mapping.get(vtype, vtype)
                    raw_emb = embedding_model.encode(semantic_group)
                    pca_emb = pca_model.transform(raw_emb.reshape(1, -1))[0]
                    all_embeddings.append(pca_emb)
            
            all_embeddings = np.vstack(all_embeddings).astype(np.float32)
            embedding_dim = all_embeddings.shape[1]
            use_pca = False
            
        else:
            use_pca = True
            train_embeddings, train_node_ids, pca_model = create_node_embeddings_with_pca(
                train_nodes_dict, precomputed_embeddings, vtype_mapping,
                embedding_model, is_train=True, pca_dim=PCA_DIM, use_pca=use_pca
            )
            
            test_embeddings, test_node_ids, _ = create_node_embeddings_with_pca(
                test_nodes_dict, precomputed_embeddings, vtype_mapping,
                embedding_model, is_train=False, pca_model=pca_model, 
                pca_dim=PCA_DIM, use_pca=use_pca
            )
            
            all_nodes_dict = {**train_nodes_dict, **test_nodes_dict}
            all_node_ids = list(all_nodes_dict.keys())
            node2id = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
            uuid_to_node_idx = node2id.copy()
            
            embedding_dim = train_embeddings.shape[1]
            all_embeddings = np.zeros((len(all_node_ids), embedding_dim), dtype=np.float32)
            
            for i, node_id in enumerate(all_node_ids):
                if node_id in train_node_ids:
                    train_idx = train_node_ids.index(node_id)
                    all_embeddings[i] = train_embeddings[train_idx]
                else:
                    test_idx = test_node_ids.index(node_id)
                    all_embeddings[i] = test_embeddings[test_idx]
        
        rel2id, edge_label_encoder = generate_edge_type_features_auto(edge_types_list)
        
        node_id_to_embedding_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
        
        graphs_dict = build_graphs_csv(
            edges_list, node2id, rel2id, 
            [node_id_to_embedding_idx[nid] for nid in all_node_ids],
            start_date_for_processing, end_date_for_processing
        )
        for window_key, graph_data in tqdm(graphs_dict.items(), desc="Saving graphs", leave=False):
            graph_json = nx.node_link_data(graph_data['graph'], edges="links")
            
            save_data = {
                'graph': graph_json,
                'node_labels': graph_data['node_labels'],
                'malicious_nodes': graph_data['malicious_nodes'],
                'num_nodes': graph_data['num_nodes'],
                'num_edges': graph_data['num_edges']
            }
            
            save_path = f'{base_dir}/processed_data/graphs/graph_{window_key}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
        
        embeddings_path = f'{base_dir}/processed_data/node_embeddings.npy'
        np.save(embeddings_path, all_embeddings)
        
        if pca_model is not None:
            with open(f'{base_dir}/processed_data/pca_model.pkl', 'wb') as f:
                pickle.dump(pca_model, f)
        
        node_id_to_embedding_idx_path = f'{base_dir}/processed_data/node_id_to_embedding_idx.json'
        with open(node_id_to_embedding_idx_path, 'w') as f:
            json.dump(node_id_to_embedding_idx, f, indent=2)
        
        uuid_to_node_idx_path = f'{base_dir}/processed_data/uuid_to_node_idx.json'
        with open(uuid_to_node_idx_path, 'w') as f:
            json.dump(uuid_to_node_idx, f, indent=2)
        
        with open(f'{base_dir}/processed_data/node_mappings/vtype_combinations_raw.pkl', 'wb') as f:
            pickle.dump(vtype_combinations, f)
        with open(f'{base_dir}/processed_data/node_mappings/vtype_combinations_mapped.pkl', 'wb') as f:
            pickle.dump(mapped_vtype_combinations, f)
        with open(f'{base_dir}/processed_data/node_mappings/vtype_mapping.pkl', 'wb') as f:
            pickle.dump(vtype_mapping, f)
        
        id_labels_path = f'{base_dir}/processed_data/node_mappings/id_labels.pkl'
        with open(id_labels_path, 'wb') as f:
            pickle.dump(id_labels, f)
        
        with open(f'{base_dir}/processed_data/node_mappings/rel2id.pkl', 'wb') as f:
            pickle.dump(rel2id, f)
        with open(f'{base_dir}/processed_data/node_mappings/edge_label_encoder.pkl', 'wb') as f:
            pickle.dump(edge_label_encoder, f)
        with open(f'{base_dir}/processed_data/node_mappings/nodeid2address.pkl', 'wb') as f:
            pickle.dump(nodeid2address, f)
        
        metadata = {
            'dataset': dataset,
            'mode': 'rulellm_embedding',
            'train_start_date': train_start_date,
            'train_end_date': train_end_date,
            'test_start_date': test_start_date,
            'test_end_date': test_end_date,
            'node_feature_dim': embedding_dim,
            'edge_feature_dim': len(edge_types_list),
            'num_time_windows': len(graphs_dict),
            'time_window_type': 'hourly',
            'total_nodes': len(node2id),
            'detection_mode': 'entity_level',
            'embedding_type': embedding,
            'feature_type': 'llmlabel',
            'pca_applied': True if (use_precomputed_pca or use_pca) else False,
            'pca_dim': PCA_DIM if (use_precomputed_pca or use_pca) else None,
            'precomputed_pca_used': use_precomputed_pca,
            'llm_semantic_grouping': False,
            'embedding_based_features': True,
            'raw_vtype_count': len(vtype_combinations),
            'semantic_group_count': len(mapped_vtype_combinations)
        }
        
        with open(f'{base_dir}/processed_data/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        split_and_zip_embeddings(embeddings_path, base_dir)
        
        zip_large_file(uuid_to_node_idx_path)
        zip_large_file(node_id_to_embedding_idx_path)
        
        zip_large_file(id_labels_path)

if __name__ == "__main__":
    main()

