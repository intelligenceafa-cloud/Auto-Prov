#!/usr/bin/env python3

import os
import sys
import argparse
import zipfile
import shutil
import json
import pickle
import glob
import re
from pathlib import Path
from types import SimpleNamespace
from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

from stepllm_utils.vtype_res import getvtypes
from gnn import (
    process_nodes_with_gpu,
    load_from_zip,
    save_phrases_chunked,
    load_phrases_generator,
    process_phrases_to_embeddings,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

STEP_LLM_DIR = os.path.dirname(SCRIPT_DIR)
BIGDATA_DIR = os.path.join(STEP_LLM_DIR, "BIGDATA")
DEFAULT_ARTIFACTS_DIR = os.path.join(BIGDATA_DIR, "FLASH_artifacts")

DATASET_DATES = {
    "theia": {
        "train_start_date": "2018-04-03",
        "train_end_date": "2018-04-05",
        "test_start_date": "2018-04-09",
        "test_end_date": "2018-04-12",
    },
}


def get_dataset_dates(dataset: str, train_start_date: str = None, train_end_date: str = None,
                      test_start_date: str = None, test_end_date: str = None):
    dataset_lower = dataset.lower()
    defaults = DATASET_DATES.get(dataset_lower, {})
    return {
        "train_start_date": train_start_date or defaults.get("train_start_date"),
        "train_end_date": train_end_date or defaults.get("train_end_date"),
        "test_start_date": test_start_date or defaults.get("test_start_date"),
        "test_end_date": test_end_date or defaults.get("test_end_date"),
    }


class TextEmbedder:
    def __init__(self, embedding_type="word2vec", vector_size=30):
        self.embedding_type = embedding_type
        self.vector_size = vector_size

        if embedding_type == "word2vec":
            self.model = None
        elif embedding_type == "mpnet":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        elif embedding_type == "minilm":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        elif embedding_type == "roberta":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('roberta-base')
        elif embedding_type == "distilbert":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        elif embedding_type == "fasttext":
            import fasttext
            self.model = fasttext.load_model('cc.en.300.bin')
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

    def train_word2vec(self, hierarchical_lists, logger, saver):
        if self.embedding_type == "word2vec":
            sentences = hierarchical_lists
            self.model = Word2Vec(sentences=sentences, vector_size=self.vector_size, window=5, min_count=1, workers=8, epochs=300, callbacks=[saver, logger])
        elif self.embedding_type == "fasttext" and self.model is None:
            import fasttext
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
                for sentence in hierarchical_lists:
                    f.write(' '.join(sentence) + '\n')
                temp_file = f.name
            self.model = fasttext.train_unsupervised(temp_file, model='skipgram', dim=self.vector_size, epoch=300)
            os.unlink(temp_file)

    def get_embedding(self, text):
        if self.embedding_type == "word2vec":
            words = text.split()
            vectors = [self.model.wv[word] for word in words if word in self.model.wv]
            if not vectors:
                return np.zeros(self.vector_size)
            return np.mean(vectors, axis=0)
        elif self.embedding_type == "fasttext":
            words = text.split()
            if not words:
                return np.zeros(self.vector_size)
            vectors = [self.model.get_word_vector(word) for word in words]
            embedding = np.mean(vectors, axis=0)
            if len(embedding) > self.vector_size:
                embedding = np.mean(embedding.reshape(-1, len(embedding) // self.vector_size), axis=1)
            return embedding[:self.vector_size]
        else:
            return self.model.encode(text)

    def get_embeddings(self, texts, use_pca=False, pca_dim=128, pca_model=None):
        embeddings = np.vstack([self.get_embedding(text) for text in texts])
        if use_pca and self.embedding_type in ['mpnet', 'minilm', 'roberta', 'distilbert']:
            original_dim = embeddings.shape[1]
            if pca_model is None:
                pca = PCA(n_components=pca_dim)
                embeddings = pca.fit_transform(embeddings)
                return embeddings, pca
            else:
                embeddings = pca_model.transform(embeddings)
                return embeddings, pca_model
        return embeddings, None


def split_dataframe_by_hour(df):
    from datetime import datetime
    hourly_dfs = {}
    for _, row in tqdm(df.iterrows(), desc="Splitting by hour", total=len(df)):
        timestamp_val = row['timestamp']
        if isinstance(timestamp_val, str):
            timestamp_ns = int(float(timestamp_val))
        else:
            timestamp_ns = int(timestamp_val)
        dt = datetime.fromtimestamp(timestamp_ns // 1000000000)
        window_key = f"{dt.strftime('%Y-%m-%d')}_{dt.hour:02d}"
        if window_key not in hourly_dfs:
            hourly_dfs[window_key] = []
        hourly_dfs[window_key].append(row.to_dict())
    result = {k: pd.DataFrame(rows) for k, rows in hourly_dfs.items()}
    return result


def load_vtype_semantic_mapping(dataset="theia"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_file = os.path.join(script_dir, f"llmgeneratedvtypegroup_{dataset.lower()}.pkl")
    if os.path.exists(mapping_file):
        with open(mapping_file, 'rb') as f:
            vtype_mapping = pickle.load(f)
        return vtype_mapping
    return None


def remap_labels_to_semantic_groups(labels_dict, vtype_combinations, vtype_mapping):
    id_to_vtype = {v: k for k, v in vtype_combinations.items()}
    group_names_dict = {}
    unmapped_vtypes = set()
    for uuid, label_id in labels_dict.items():
        vtype_string = id_to_vtype.get(label_id)
        if vtype_string and vtype_string in vtype_mapping:
            group_names_dict[uuid] = vtype_mapping[vtype_string]
        else:
            group_names_dict[uuid] = "other"
            if vtype_string:
                unmapped_vtypes.add(vtype_string)
    all_group_names = list(group_names_dict.values())
    label_encoder = LabelEncoder()
    label_encoder.fit(all_group_names)
    remapped_labels = {uuid: label_encoder.transform([group_name])[0] for uuid, group_name in group_names_dict.items()}
    return remapped_labels, label_encoder


def extract_uuid(line):
    pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
    return pattern_uuid.findall(line)


def extract_subject_type(line):
    pattern_type = re.compile(r'type\":\"(.*?)\"')
    return pattern_type.findall(line)


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
    raise FileNotFoundError(f"Neither {zip_file_path} nor {pkl_file_path} found")


def load_unknown_actions_mapping(dataset="theia"):
    unknown_actions_file = f"inter_info/unknown_actions_{dataset.lower()}.txt"
    if not os.path.exists(unknown_actions_file):
        return {}
    action_mapping = {}
    with open(unknown_actions_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r'^\s*\d+\.\s*source_vtypes:', line):
            source_vtypes = line.split('source_vtypes: ')[1].strip()
            if i + 1 < len(lines):
                dest_line = lines[i + 1].strip()
                if dest_line.startswith('dest_vtypes:'):
                    dest_vtypes = dest_line.split('dest_vtypes: ')[1].strip()
                    if i + 2 < len(lines):
                        action_line = lines[i + 2].strip()
                        if action_line.startswith('action:'):
                            action = action_line.split('action: ')[1].strip()
                            action_mapping[(source_vtypes, dest_vtypes)] = action
            i += 3
        else:
            i += 1
    return action_mapping


def load_action_validation(dataset="theia"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    step_llm_dir = os.path.dirname(script_dir)
    validation_file = os.path.join(step_llm_dir, "ename-processing", f"edge_type_validation_{dataset.lower()}.json")
    if os.path.exists(validation_file):
        with open(validation_file, 'r') as f:
            return json.load(f)
    return {}


def is_valid_action(action, validation_dict):
    if action == "NO LABEL":
        return False
    if not validation_dict:
        return True
    return validation_dict.get(action, "VALID") == "VALID"


def load_csv_data(timestamp_dir):
    csv_files = glob.glob(os.path.join(timestamp_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {timestamp_dir}")
    csv_file = csv_files[0]
    chunk_list = []
    for chunk in pd.read_csv(csv_file, chunksize=100000):
        chunk_list.append(chunk)
    df = pd.concat(chunk_list, ignore_index=True)
    return fill_missing_timestamps(df)


def fill_missing_timestamps(df):
    if df['timestamp'].isna().sum() == 0:
        return df
    df = df.copy()
    df['timestamp'] = df['timestamp'].bfill()
    df['timestamp'] = df['timestamp'].ffill()
    return df


def extract_edge_info(line):
    pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_time = re.compile(r'timestampNanos\":(.*?),')
    edge_type = extract_subject_type(line)[0]
    timestamp = pattern_time.findall(line)[0]
    src_id = pattern_src.findall(line)
    if len(src_id) == 0:
        return None, None, None, None, None
    src_id = src_id[0]
    dst_id1 = pattern_dst1.findall(line)
    dst_id2 = pattern_dst2.findall(line)
    dst_id1 = dst_id1[0] if (len(dst_id1) > 0 and dst_id1[0] != 'null') else None
    dst_id2 = dst_id2[0] if (len(dst_id2) > 0 and dst_id2[0] != 'null') else None
    return src_id, edge_type, timestamp, dst_id1, dst_id2


def process_msg(file_path):
    id_nodetype_map = {}
    for timestamp_dir in tqdm(glob.glob(file_path), desc="EXTRACTING MESSAGES"):
        data = load_log_data(timestamp_dir)
        for line in tqdm(data, desc='Processing Logs'):
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line:
                continue
            if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line:
                continue
            if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line:
                continue
            if 'com.bbn.tc.schema.avro.cdm18.Subject' in line:
                old_pattern = 'Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"properties":{"map":{"tgid":"(.*?)"'
                res_list = re.findall(old_pattern, line)
                if res_list:
                    res = res_list[0]
                    path_str_list = re.findall('"path":"(.*?)"', line)
                    path = path_str_list[0] if path_str_list else "null"
                    id_nodetype_map[res[0]] = path
                    continue
            if 'com.bbn.tc.schema.avro.cdm18.FileObject' in line:
                res_list = re.findall('FileObject":{"uuid":"(.*?)"(.*?)"filename":"(.*?)"', line)
                if res_list:
                    res = res_list[0]
                    id_nodetype_map[res[0]] = res[2]
                    continue
    return id_nodetype_map


def process_csv_edges_and_msg(file_path, path_to_save, id_labels, dataset="theia", nolabel=False, train_start_date="2018-04-03", train_end_date="2018-04-05", test_start_date="2018-04-09", test_end_date="2018-04-12", llmfunc=False, llm_func_dict=None, llmlabel=False, llm_label_dict=None, action_validation=None):
    import ast
    from datetime import datetime
    id_nodemsg_map = {}
    main_msg_path = os.path.join(os.path.dirname(path_to_save), "uuid2msg.json")
    unknown_actions = load_unknown_actions_mapping(dataset) if nolabel else {}
    total_dropped_invalid = 0
    llm_func_stats = {'total_uuids': 0, 'enriched': 0, 'not_found': 0}
    if id_labels is None:
        _, id_labels = getvtypes(dataset.lower(), "../BIGDATA/ExtractedProvGraph/")
    csv_dirs = glob.glob(file_path)
    train_start = datetime.strptime(train_start_date, "%Y-%m-%d").date()
    train_end = datetime.strptime(train_end_date, "%Y-%m-%d").date()
    test_start = datetime.strptime(test_start_date, "%Y-%m-%d").date()
    test_end = datetime.strptime(test_end_date, "%Y-%m-%d").date()
    filtered_dirs = [d for d in csv_dirs
                     if train_start <= datetime.strptime(d.split('/')[-2].split(' ')[0], "%Y-%m-%d").date() <= train_end
                     or test_start <= datetime.strptime(d.split('/')[-2].split(' ')[0], "%Y-%m-%d").date() <= test_end]
    dirs_needing_txt = []
    dirs_for_uuid_extraction = []
    for timestamp_dir in filtered_dirs:
        timestamp_name = timestamp_dir.split('/')[-2]
        output_txt_path = os.path.join(path_to_save, f"{timestamp_name}.txt")
        if os.path.exists(output_txt_path):
            uuid_msg_needs_rebuild = not os.path.exists(main_msg_path)
            if not uuid_msg_needs_rebuild and (llmfunc and llm_func_dict) or (llmlabel and llm_label_dict):
                with open(main_msg_path, 'r') as f:
                    uuid_msg_needs_rebuild = len(json.load(f)) == 0
            if uuid_msg_needs_rebuild:
                dirs_for_uuid_extraction.append(timestamp_dir)
        else:
            dirs_needing_txt.append(timestamp_dir)
    dirs_to_process = dirs_needing_txt + dirs_for_uuid_extraction
    if len(dirs_to_process) == 0 and os.path.exists(main_msg_path):
        with open(main_msg_path, 'r') as f:
            return json.load(f)
    for timestamp_dir in tqdm(dirs_to_process, desc="CSV PROCESSING"):
        timestamp_name = timestamp_dir.split('/')[-2]
        df = load_csv_data(timestamp_dir)
        output_txt_path = os.path.join(path_to_save, f"{timestamp_name}.txt")
        txt_file_exists = os.path.exists(output_txt_path)
        fw = open(output_txt_path, 'w') if not txt_file_exists else None
        try:
            for _, row in tqdm(df.iterrows(), desc='Processing CSV data'):
                source_id, dest_id = row['source_id'], row['dest_id']
                action, timestamp = row['action'], row['timestamp']
                if source_id not in id_labels or dest_id not in id_labels:
                    continue
                source_vtypes = ast.literal_eval(row['source_vtypes']) if isinstance(row['source_vtypes'], str) else []
                dest_vtypes = ast.literal_eval(row['dest_vtypes']) if isinstance(row['dest_vtypes'], str) else []
                source_has_parent = any('parent' in str(v).lower() for v in source_vtypes)
                dest_has_parent = any('parent' in str(v).lower() for v in dest_vtypes)
                source_enames = ast.literal_eval(row['source_enames']) if isinstance(row['source_enames'], str) else []
                dest_enames = ast.literal_eval(row['dest_enames']) if isinstance(row['dest_enames'], str) else []
                meaningless_names = {'datum', 'N/A', '},'}
                source_enames = [n for n in source_enames if n not in meaningless_names and len(n) > 1]
                dest_enames = [n for n in dest_enames if n not in meaningless_names and len(n) > 1]
                if source_has_parent and not dest_has_parent and not source_enames and dest_enames:
                    source_enames = dest_enames.copy()
                elif dest_has_parent and not source_has_parent and not dest_enames and source_enames:
                    dest_enames = source_enames.copy()
                if llmfunc and llm_func_dict:
                    if source_id in llm_func_dict and source_id not in id_nodemsg_map:
                        id_nodemsg_map[source_id] = llm_func_dict[source_id]
                        llm_func_stats['enriched'] += 1
                    elif source_id not in llm_func_dict:
                        llm_func_stats['not_found'] += 1
                    if dest_id in llm_func_dict and dest_id not in id_nodemsg_map:
                        id_nodemsg_map[dest_id] = llm_func_dict[dest_id]
                        llm_func_stats['enriched'] += 1
                    elif dest_id not in llm_func_dict:
                        llm_func_stats['not_found'] += 1
                    llm_func_stats['total_uuids'] += 2
                elif llmlabel and llm_label_dict:
                    if source_id in llm_label_dict and source_id not in id_nodemsg_map:
                        id_nodemsg_map[source_id] = llm_label_dict[source_id]
                        llm_func_stats['enriched'] += 1
                    elif source_id not in llm_label_dict:
                        llm_func_stats['not_found'] += 1
                    if dest_id in llm_label_dict and dest_id not in id_nodemsg_map:
                        id_nodemsg_map[dest_id] = llm_label_dict[dest_id]
                        llm_func_stats['enriched'] += 1
                    elif dest_id not in llm_label_dict:
                        llm_func_stats['not_found'] += 1
                    llm_func_stats['total_uuids'] += 2
                else:
                    if source_enames and source_id not in id_nodemsg_map:
                        id_nodemsg_map[source_id] = max(source_enames, key=len)
                    if dest_enames and dest_id not in id_nodemsg_map:
                        id_nodemsg_map[dest_id] = max(dest_enames, key=len)
                should_write_edge = True
                if not is_valid_action(action, action_validation):
                    if nolabel and unknown_actions:
                        action_key = (', '.join(map(str, source_vtypes)), ', '.join(map(str, dest_vtypes)))
                        if action_key in unknown_actions:
                            action = unknown_actions[action_key]
                            if not is_valid_action(action, action_validation):
                                total_dropped_invalid += 1
                                should_write_edge = False
                        else:
                            total_dropped_invalid += 1
                            should_write_edge = False
                    else:
                        total_dropped_invalid += 1
                        should_write_edge = False
                if should_write_edge and fw is not None:
                    fw.write(f"{source_id}\t{id_labels[source_id]}\t{dest_id}\t{id_labels[dest_id]}\t{action}\t{timestamp}\n")
        finally:
            if fw is not None:
                fw.close()
        with open(main_msg_path, "w") as msg_file:
            json.dump(id_nodemsg_map, msg_file, indent=2)
    return id_nodemsg_map


def process_data(file_path):
    id_nodetype_map = {}
    for timestamp_dir in tqdm(glob.glob(file_path), desc="NODE TYPE STORAGE"):
        data = load_log_data(timestamp_dir)
        for line in tqdm(data, desc='Processing Logs'):
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line:
                continue
            if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line:
                continue
            if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line:
                continue
            uuid = extract_uuid(line)
            subject_type = extract_subject_type(line)
            if len(subject_type) < 1:
                if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                    id_nodetype_map[uuid[0]] = 'MemoryObject'
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                    id_nodetype_map[uuid[0]] = 'NetFlowObject'
                    continue
                if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                    id_nodetype_map[uuid[0]] = 'UnnamedPipeObject'
                    continue
            id_nodetype_map[uuid[0]] = subject_type[0]
    return id_nodetype_map


def process_edges(file_path, path_to_save, id_nodetype_map):
    for timestamp_dir in tqdm(glob.glob(file_path), desc="GRAPH CREATION"):
        data = load_log_data(timestamp_dir)
        with open(path_to_save + timestamp_dir.split('/')[-2] + '.txt', 'w') as fw:
            for line in tqdm(data, desc='Processing Logs'):
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    src_id, edge_type, timestamp, dst_id1, dst_id2 = extract_edge_info(line)
                    if src_id is None or src_id not in id_nodetype_map:
                        continue
                    src_type = id_nodetype_map[src_id]
                    if dst_id1 is not None and dst_id1 in id_nodetype_map:
                        fw.write(f"{src_id}\t{src_type}\t{dst_id1}\t{id_nodetype_map[dst_id1]}\t{edge_type}\t{timestamp}\n")
                    if dst_id2 is not None and dst_id2 in id_nodetype_map:
                        fw.write(f"{src_id}\t{src_type}\t{dst_id2}\t{id_nodetype_map[dst_id2]}\t{edge_type}\t{timestamp}\n")


def add_node_properties(nodes, node_id, properties):
    if node_id not in nodes:
        nodes[node_id] = []
    nodes[node_id].extend(properties)


def update_edge_index(edges, edge_index, index):
    for src_id, dst_id in edges:
        edge_index[0].append(index[src_id])
        edge_index[1].append(index[dst_id])


def get_filename(mapping, uuid):
    if uuid not in mapping:
        raise KeyError(f"UUID '{uuid}' not found in mapping.")
    return mapping[uuid]


def get_id2msg_current(df, nodeid2msg):
    id2msg = {}
    for i in range(df.shape[0]):
        if str(df.iloc[i, 0]).lower() in nodeid2msg:
            id2msg[df.iloc[i, 0]] = get_filename(nodeid2msg, str(df.iloc[i, 0]).lower())
    for i in range(df.shape[0]):
        if str(df.iloc[i, 2]).lower() in nodeid2msg and df.iloc[i, 2] not in id2msg:
            id2msg[df.iloc[i, 2]] = get_filename(nodeid2msg, str(df.iloc[i, 2]).lower())
    return id2msg


def prepare_graph(df, rulellm=False, dataset="theia", vtype_combinations=None, id_labels=None, use_semantic_groups=False, vtype_mapping=None):
    nodes, labels, edges = {}, {}, []
    edge_actions = []
    label_encoder = None
    if rulellm:
        if vtype_combinations is None or id_labels is None:
            vtype_combinations, id_labels = getvtypes(dataset.lower(), "../BIGDATA/ExtractedProvGraph/")
        dummies = vtype_combinations
    elif dataset.upper() == "THEIA":
        dummies = {"SUBJECT_PROCESS": 0, "MemoryObject": 1, "FILE_OBJECT_BLOCK": 2, "NetFlowObject": 3, "PRINCIPAL_REMOTE": 4, 'PRINCIPAL_LOCAL': 5}
    else:
        dummies = {"SUBJECT_PROCESS": 0, "MemoryObject": 1, "FILE_OBJECT_BLOCK": 2, "NetFlowObject": 3, "PRINCIPAL_REMOTE": 4, 'PRINCIPAL_LOCAL': 5}
    skipped_count = 0
    for _, row in tqdm(df.iterrows(), desc="PREPARING GRAPH"):
        action = row["action"]
        properties = [row['exec'], action] + ([row['path']] if row.get('path') else [])
        actor_id, object_id = row["actorID"], row["objectID"]
        if rulellm:
            if actor_id not in id_labels or object_id not in id_labels:
                skipped_count += 1
                continue
            if id_labels[actor_id] not in dummies or id_labels[object_id] not in dummies:
                skipped_count += 1
                continue
            actor_label, object_label = dummies[id_labels[actor_id]], dummies[id_labels[object_id]]
        else:
            if row['actor_type'] not in dummies or row['object'] not in dummies:
                skipped_count += 1
                continue
            actor_label, object_label = dummies[row['actor_type']], dummies[row['object']]
        add_node_properties(nodes, actor_id, properties)
        labels[actor_id] = actor_label
        add_node_properties(nodes, object_id, properties)
        labels[object_id] = object_label
        edges.append((actor_id, object_id))
        edge_actions.append(action)
    features, feat_labels, edge_index, index_map = [], [], [[], []], {}
    for node_id, props in tqdm(nodes.items(), desc="PREPARING NODES"):
        features.append(props)
        feat_labels.append(labels[node_id])
        index_map[node_id] = len(features) - 1
    update_edge_index(edges, edge_index, index_map)
    edge_actions_array = list(edge_actions)
    if rulellm and use_semantic_groups and vtype_mapping:
        labels, label_encoder = remap_labels_to_semantic_groups(labels, vtype_combinations, vtype_mapping)
        feat_labels = [labels[node_id] for node_id in index_map.keys()]
    return features, feat_labels, edge_index, list(index_map.keys()), edge_actions_array, label_encoder


def add_csv_attributes(d, file_path, timestamp_par, dataset="theia", nolabel=False, uuid2msg_path=None, llmfunc=False, llm_func_dict=None, llmlabel=False, llm_label_dict=None):
    id_nodemsg_map = {}
    if uuid2msg_path and os.path.exists(uuid2msg_path):
        with open(uuid2msg_path, 'r') as f:
            id_nodemsg_map = json.load(f)
    if id_nodemsg_map:
        d = d.astype(str)
        d['exec'] = d['actorID'].map(id_nodemsg_map).fillna('')
        d['path'] = d['objectID'].map(id_nodemsg_map).fillna('')
        return d
    unknown_actions = load_unknown_actions_mapping(dataset) if nolabel else {}
    csv_dirs = glob.glob(file_path)
    info = []
    for timestamp_dir in tqdm(csv_dirs, desc="CSV ADDING ATTRIBUTES"):
        if timestamp_par not in timestamp_dir.split('/')[-1]:
            continue
        df = load_csv_data(timestamp_dir)
        for _, row in df.iterrows():
            source_id, dest_id = row['source_id'], row['dest_id']
            action, timestamp = row['action'], row['timestamp']
            if action == "NO LABEL" and nolabel and unknown_actions:
                action_key = None
                for key, value in unknown_actions.items():
                    if isinstance(key, tuple) and len(key) == 2 and key == (source_id, dest_id):
                        action = value
                        break
                else:
                    continue
            if action == "NO LABEL":
                continue
            import ast
            source_names = ast.literal_eval(row['source_enames']) if isinstance(row['source_enames'], str) else []
            dest_names = ast.literal_eval(row['dest_enames']) if isinstance(row['dest_enames'], str) else []
            meaningless_names = {'datum', 'N/A', '},'}
            source_name = ' '.join(n for n in source_names if n not in meaningless_names and len(n) > 1)
            dest_name = ' '.join(n for n in dest_names if n not in meaningless_names and len(n) > 1)
            info.append({'actorID': source_id, 'objectID': dest_id, 'action': action, 'timestamp': timestamp, 'exec': source_name, 'path': dest_name})
    rdf = pd.DataFrame.from_records(info).astype(str) if info else pd.DataFrame()
    d = d.astype(str)
    if rdf.empty:
        return d
    return d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()


def load_csv_train_test_data(file_pattern, train_start_date, train_end_date, test_start_date, test_end_date, dataset="theia", id_labels=None, nolabel=False):
    from datetime import datetime
    train_data, test_data = [], []
    txt_files = glob.glob(file_pattern)
    train_start = datetime.strptime(train_start_date, "%Y-%m-%d").date()
    train_end = datetime.strptime(train_end_date, "%Y-%m-%d").date()
    test_start = datetime.strptime(test_start_date, "%Y-%m-%d").date()
    test_end = datetime.strptime(test_end_date, "%Y-%m-%d").date()
    for txt_file in tqdm(txt_files, desc="LOADING CSV TRAIN-TEST"):
        base_name = os.path.basename(txt_file)
        file_date = datetime.strptime(base_name.split(' ')[0], "%Y-%m-%d").date()
        if train_start <= file_date <= train_end:
            with open(txt_file, 'r') as f:
                data = [line.split('\t') for line in f.read().strip().split('\n') if line.strip()]
                train_data.extend(data)
        elif test_start <= file_date <= test_end:
            with open(txt_file, 'r') as f:
                data = [line.split('\t') for line in f.read().strip().split('\n') if line.strip()]
                test_data.extend(data)
    return train_data, test_data


def add_attributes(d, file_path, timestamp_par):
    info = []
    for timestamp_dir in tqdm(glob.glob(file_path), desc="ADDING ATTRIBUTES"):
        if timestamp_par not in timestamp_dir.split('/')[-2]:
            continue
        f_in = load_log_data(timestamp_dir)
        data = [json.loads(x) for x in f_in if "EVENT" in x]
        for x in data:
            try:
                event_data = x.get('datum', {}).get('com.bbn.tc.schema.avro.cdm18.Event')
                if event_data is None:
                    continue
                action = event_data['type']
                actor = event_data['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
                obj = event_data['predicateObject']['com.bbn.tc.schema.avro.cdm18.UUID']
                timestamp = event_data['timestampNanos']
                cmd = event_data.get('properties', {}).get('map', {}).get('cmdLine', '') if event_data.get('properties') else ''
                path = ''
                if event_data.get('predicateObjectPath'):
                    path = event_data['predicateObjectPath'].get('string', '') if isinstance(event_data['predicateObjectPath'], dict) else ''
                path2 = ''
                if event_data.get('predicateObject2Path'):
                    path2 = event_data['predicateObject2Path'].get('string', '') if isinstance(event_data['predicateObject2Path'], dict) else ''
                obj2_data = event_data.get('predicateObject2', {}).get('com.bbn.tc.schema.avro.cdm18.UUID', None) if isinstance(event_data.get('predicateObject2'), dict) else None
                if obj2_data:
                    info.append({'actorID': actor, 'objectID': obj2_data, 'action': action, 'timestamp': timestamp, 'exec': cmd, 'path': path2})
                info.append({'actorID': actor, 'objectID': obj, 'action': action, 'timestamp': timestamp, 'exec': cmd, 'path': path})
            except (KeyError, TypeError, AttributeError):
                continue
    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)
    return d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()


def load_train_test_data(file_pattern, train_start_date, train_end_date, test_start_date, test_end_date):
    from datetime import datetime
    train_data, test_data = [], []
    zip_files = glob.glob(file_pattern)
    train_start = datetime.strptime(train_start_date, "%Y-%m-%d").date()
    train_end = datetime.strptime(train_end_date, "%Y-%m-%d").date()
    test_start = datetime.strptime(test_start_date, "%Y-%m-%d").date()
    test_end = datetime.strptime(test_end_date, "%Y-%m-%d").date()
    for zip_file in tqdm(zip_files, desc="LOADING TRAIN-TEST"):
        base_name = os.path.basename(zip_file)
        file_date = datetime.strptime(base_name.split(' ')[0], "%Y-%m-%d").date()
        if train_start <= file_date <= train_end:
            with open(zip_file, 'r') as f:
                data = [line.split('\t') for line in f.read().strip().split('\n')]
                train_data.extend(data)
        elif test_start <= file_date <= test_end:
            with open(zip_file, 'r') as f:
                data = [line.split('\t') for line in f.read().strip().split('\n')]
                test_data.extend(data)
    return train_data, test_data


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        pass
    def on_epoch_end(self, model):
        self.epoch += 1


class EpochSaver(CallbackAny2Vec):
    def __init__(self, save_dir, dataset):
        self.epoch = 0
        self.save_dir = save_dir
        self.dataset = dataset
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    def on_epoch_end(self, model):
        file_path = os.path.join(self.save_dir, f'word2vec_{self.dataset.lower()}_E3.model')
        model.save(file_path)
        self.epoch += 1


def run_with_args(args):
    dataset = args.dataset.upper()
    dates = get_dataset_dates(
        args.dataset,
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
    )
    if not all(dates.values()):
        missing = [k for k, v in dates.items() if not v]
        raise ValueError(f"Missing dates for dataset '{args.dataset}': {missing}")
    train_start_date = dates["train_start_date"]
    train_end_date = dates["train_end_date"]
    test_start_date = dates["test_start_date"]
    test_end_date = dates["test_end_date"]
    if args.llmfunc and args.llmlabel:
        raise ValueError("--llmfunc and --llmlabel are mutually exclusive")
    if args.rulellm and args.nolabel and args.llmfunc:
        suffix = '_rulellm_nolabel_llmfunc'
    elif args.rulellm and args.nolabel and args.llmlabel:
        suffix = '_rulellm_nolabel_llmlabel'
    elif args.rulellm and args.llmfunc:
        suffix = '_rulellm_llmfunc'
    elif args.rulellm and args.llmlabel:
        suffix = '_rulellm_llmlabel'
    elif args.rulellm and args.nolabel:
        suffix = '_rulellm_nolabel'
    elif args.rulellm:
        suffix = '_rulellm'
    else:
        suffix = ''
    base_dir = f'{args.artifacts_dir}/{dataset}_graphs{suffix}'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f'{base_dir}/{args.embedding}/', exist_ok=True)
    os.makedirs(f'{base_dir}/{args.embedding}/train_graph{suffix}/', exist_ok=True)
    action_validation = {}
    llm_func_dict = {}
    llm_label_dict = {}
    if args.rulellm:
        action_validation = load_action_validation(dataset.lower())
    if args.llmfunc or args.llmlabel:
        step_llm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        timeoh_dir = os.path.join(step_llm_dir, "ename-processing", "behavioral-profiles", "timeoh")
        if args.llmfunc:
            with open(os.path.join(timeoh_dir, f"typed_nodes_functionality_{dataset.lower()}.json"), 'r') as f:
                typed_func = json.load(f)
            with open(os.path.join(timeoh_dir, f"untype2type_nodes_functionality_{dataset.lower()}.json"), 'r') as f:
                untype_func = json.load(f)
            llm_func_dict = {**typed_func, **untype_func}
        if args.llmlabel:
            with open(os.path.join(timeoh_dir, f"typed_nodes_{dataset.lower()}.json"), 'r') as f:
                typed_labels = json.load(f)
            with open(os.path.join(timeoh_dir, f"untype2type_nodes_{dataset.lower()}.json"), 'r') as f:
                untype_labels = json.load(f)
            llm_label_dict = {**typed_labels, **untype_labels}
    if args.rulellm:
        dataset_path = f'../BIGDATA/ExtractedProvGraph/{dataset}/*/'
        vtype_combinations, id_labels = getvtypes(dataset.lower(), "../BIGDATA/ExtractedProvGraph/")
        vtype_mapping = load_vtype_semantic_mapping(dataset.lower())
        uuid2msg_path = f'{base_dir}/{args.embedding}/uuid2msg.json'
        if args.gen_graphs:
            id_nodemsg_map = process_csv_edges_and_msg(
                dataset_path, f'{base_dir}/{args.embedding}/train_graph{suffix}/',
                id_labels, dataset, args.nolabel, train_start_date, train_end_date, test_start_date, test_end_date,
                args.llmfunc, llm_func_dict, args.llmlabel, llm_label_dict, action_validation,
            )
            with open(uuid2msg_path, "w") as f:
                json.dump(id_nodemsg_map, f, indent=4)
        train_data, test = load_csv_train_test_data(
            f'{base_dir}/{args.embedding}/train_graph{suffix}/*.txt',
            train_start_date, train_end_date, test_start_date, test_end_date, dataset, id_labels, args.nolabel,
        )
        df = pd.DataFrame(train_data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
    else:
        dataset_path = f'{args.dataset_path}/{dataset}/*/'
        vtype_mapping = None
        id_nodetype_map = process_data(dataset_path)
        uuid2msg_path = f'{base_dir}/{args.embedding}/uuid2msg{suffix}.json'
        if not os.path.exists(uuid2msg_path):
            id_nodemsg_map = process_msg(dataset_path)
            with open(uuid2msg_path, "w") as f:
                json.dump(id_nodemsg_map, f, indent=4)
        else:
            with open(uuid2msg_path, "r") as f:
                id_nodemsg_map = json.load(f)
        if args.gen_graphs:
            process_edges(dataset_path, f'{base_dir}/{args.embedding}/train_graph{suffix}/', id_nodetype_map)
        train_data, test = load_train_test_data(
            f'{base_dir}/{args.embedding}/train_graph{suffix}/*.txt',
            train_start_date, train_end_date, test_start_date, test_end_date,
        )
        df = pd.DataFrame(train_data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
        id_labels = None
        vtype_combinations = None
    df = df.dropna()
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if args.train_attr:
        if args.rulellm:
            df = add_csv_attributes(
                df, dataset_path, train_end_date, dataset, args.nolabel, uuid2msg_path,
                args.llmfunc, llm_func_dict, args.llmlabel, llm_label_dict,
            )
        else:
            df = add_attributes(df, dataset_path, train_end_date)
        df.reset_index(drop=True, inplace=True)
    root_dir = f'{args.artifacts_dir}/{dataset}_graphs{suffix}/{args.embedding}/train_attr{suffix}/'
    os.makedirs(root_dir, exist_ok=True)
    phrases_exist = os.path.exists(root_dir + 'phrases.pkl') or os.path.exists(root_dir + 'phrases_chunk_0.pkl')
    nodes_exist = os.path.exists(root_dir + 'nodes.pkl')
    if not phrases_exist:
        if args.rulellm:
            phrases, labels, edges, mapp, edge_actions, label_encoder = prepare_graph(
                df, True, dataset, vtype_combinations, id_labels, use_semantic_groups=True, vtype_mapping=vtype_mapping,
            )
        else:
            phrases, labels, edges, mapp, edge_actions, label_encoder = prepare_graph(df, False, dataset)
        if args.llmfunc or args.llmlabel:
            save_phrases_chunked(phrases, root_dir, chunk_size=100000)
        else:
            with open(root_dir + 'phrases.pkl', 'wb') as f:
                pickle.dump(phrases, f)
        with open(root_dir + 'labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
        with open(root_dir + 'edges.pkl', 'wb') as f:
            pickle.dump(edges, f)
        with open(root_dir + 'mapp.pkl', 'wb') as f:
            pickle.dump(mapp, f)
        with open(root_dir + 'edge_actions.pkl', 'wb') as f:
            pickle.dump(edge_actions, f)
        if label_encoder is not None:
            with open(root_dir + 'label_encoder.pkl', 'wb') as f:
                pickle.dump(label_encoder, f)
        text_embedder = TextEmbedder(embedding_type=args.embedding, vector_size=30)
        logger = EpochLogger()
        saver = EpochSaver(root_dir, dataset)
        if args.embedding == "word2vec":
            text_embedder.train_word2vec(phrases, logger, saver)
            if hasattr(text_embedder.model, 'save'):
                text_embedder.model.save(f"{root_dir}word2vec_{dataset.lower()}_E3.model")
            if os.path.exists(os.path.join(root_dir, 'phrases_chunk_0.pkl')):
                chunk_gen, total_size, num_chunks = load_phrases_generator(root_dir)
                phrases = []
                for chunk in tqdm(chunk_gen, total=num_chunks):
                    phrases.extend(chunk)
            else:
                phrases = load_from_zip(root_dir, 'phrases')
            nodes = process_nodes_with_gpu(phrases, text_embedder.model)
        else:
            use_pca = args.embedding in ['mpnet', 'minilm', 'roberta', 'distilbert']
            if args.llmfunc or args.llmlabel:
                nodes, pca_model = process_phrases_to_embeddings(
                    root_dir, text_embedder, use_pca=use_pca, pca_dim=args.pca_dim, batch_size=args.batch_size,
                )
            else:
                phrases = load_from_zip(root_dir, 'phrases')
                text_phrases = [' '.join(phrase) for phrase in phrases]
                nodes, pca_model = text_embedder.get_embeddings(text_phrases, use_pca=use_pca, pca_dim=args.pca_dim)
            if pca_model is not None:
                with open(os.path.join(root_dir, f'pca_model_{args.embedding}.pkl'), 'wb') as f:
                    pickle.dump(pca_model, f)
        with open(root_dir + 'nodes.pkl', 'wb') as f:
            pickle.dump(nodes, f)
    elif not nodes_exist:
        if args.embedding == 'word2vec':
            w2vmodel = Word2Vec.load(f"{root_dir}word2vec_{dataset.lower()}_E3.model")
            if os.path.exists(os.path.join(root_dir, 'phrases_chunk_0.pkl')):
                chunk_gen, _, num_chunks = load_phrases_generator(root_dir)
                phrases = []
                for chunk in tqdm(chunk_gen, total=num_chunks):
                    phrases.extend(chunk)
            else:
                phrases = load_from_zip(root_dir, 'phrases')
            nodes = process_nodes_with_gpu(phrases, w2vmodel)
            with open(root_dir + 'nodes.pkl', 'wb') as f:
                pickle.dump(nodes, f)
        else:
            text_embedder = TextEmbedder(embedding_type=args.embedding, vector_size=30)
            use_pca = args.embedding in ['mpnet', 'minilm', 'roberta', 'distilbert']
            pca_path = os.path.join(root_dir, f'pca_model_{args.embedding}.pkl')
            pca_model = pickle.load(open(pca_path, 'rb')) if (use_pca and os.path.exists(pca_path)) else None
            if args.llmfunc or args.llmlabel or os.path.exists(os.path.join(root_dir, 'phrases_chunk_0.pkl')):
                nodes, _ = process_phrases_to_embeddings(
                    root_dir, text_embedder, use_pca=use_pca, pca_dim=args.pca_dim, pca_model=pca_model, batch_size=args.batch_size,
                )
            else:
                phrases = load_from_zip(root_dir, 'phrases')
                text_phrases = [' '.join(phrase) for phrase in phrases]
                nodes, pca_model = text_embedder.get_embeddings(text_phrases, use_pca=use_pca, pca_dim=args.pca_dim, pca_model=pca_model)
            with open(root_dir + 'nodes.pkl', 'wb') as f:
                pickle.dump(nodes, f)
    if args.train_attr and test is not None and len(test) > 0:
        test_df = pd.DataFrame(test, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
        hourly_windows = split_dataframe_by_hour(test_df)
        use_pca = args.embedding in ['mpnet', 'minilm', 'roberta', 'distilbert']
        for window_key, window_df in tqdm(hourly_windows.items(), desc="Processing hourly windows"):
            window_dir = f'{args.artifacts_dir}/{dataset}_graphs{suffix}/{args.embedding}/test_attr_{window_key}{suffix}/'
            os.makedirs(window_dir, exist_ok=True)
            required_files = ['phrases.pkl', 'labels.pkl', 'edges.pkl', 'mapp.pkl', 'edge_actions.pkl', 'nodes.pkl']
            if all(os.path.exists(window_dir + f) for f in required_files):
                continue
            window_df = window_df.dropna()
            window_df.sort_values(by='timestamp', ascending=True, inplace=True)
            window_df.reset_index(drop=True, inplace=True)
            window_date = window_key.rsplit('_', 1)[0] if '_' in window_key else window_key
            if args.rulellm:
                window_df = add_csv_attributes(
                    window_df, dataset_path, window_date, dataset, args.nolabel, uuid2msg_path,
                    args.llmfunc, llm_func_dict, args.llmlabel, llm_label_dict,
                )
            else:
                window_df = add_attributes(window_df, dataset_path, window_date)
            if args.rulellm:
                window_phrases, window_labels, window_edges, window_mapp, window_edge_actions, window_label_encoder = prepare_graph(
                    window_df, True, dataset, vtype_combinations, id_labels, use_semantic_groups=True, vtype_mapping=vtype_mapping,
                )
            else:
                window_phrases, window_labels, window_edges, window_mapp, window_edge_actions, window_label_encoder = prepare_graph(window_df, False, dataset)
            if (args.llmfunc or args.llmlabel) and len(window_phrases) > 100000:
                chunk_size = 100000
                num_chunks = (len(window_phrases) + chunk_size - 1) // chunk_size
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(window_phrases))
                    with open(f'{window_dir}phrases_chunk_{i}.pkl', 'wb') as f:
                        pickle.dump(window_phrases[start_idx:end_idx], f)
            else:
                with open(window_dir + 'phrases.pkl', 'wb') as f:
                    pickle.dump(window_phrases, f)
            with open(window_dir + 'labels.pkl', 'wb') as f:
                pickle.dump(window_labels, f)
            with open(window_dir + 'edges.pkl', 'wb') as f:
                pickle.dump(window_edges, f)
            with open(window_dir + 'mapp.pkl', 'wb') as f:
                pickle.dump(window_mapp, f)
            with open(window_dir + 'edge_actions.pkl', 'wb') as f:
                pickle.dump(window_edge_actions, f)
            if window_label_encoder is not None:
                with open(window_dir + 'label_encoder.pkl', 'wb') as f:
                    pickle.dump(window_label_encoder, f)
            if args.embedding == 'word2vec':
                w2v_model_path = root_dir + f'word2vec_{dataset.lower()}_E3.model'
                if os.path.exists(w2v_model_path):
                    w2vmodel = Word2Vec.load(w2v_model_path)
                    if len(window_phrases) == 0:
                        window_nodes = []
                    elif (args.llmfunc or args.llmlabel) and len(window_phrases) > 100000:
                        chunk_gen, _, num_chunks = load_phrases_generator(window_dir)
                        window_nodes = []
                        for chunk in tqdm(chunk_gen, total=num_chunks):
                            window_nodes.extend(process_nodes_with_gpu(chunk, w2vmodel))
                    else:
                        window_nodes = process_nodes_with_gpu(window_phrases, w2vmodel)
                    with open(window_dir + 'nodes.pkl', 'wb') as f:
                        pickle.dump(window_nodes, f)
            else:
                text_embedder = TextEmbedder(embedding_type=args.embedding, vector_size=30)
                pca_model_path = root_dir + f'pca_model_{args.embedding}.pkl'
                pca_model = pickle.load(open(pca_model_path, 'rb')) if os.path.exists(pca_model_path) else None
                if len(window_phrases) == 0:
                    window_nodes = []
                elif (args.llmfunc or args.llmlabel) and len(window_phrases) > 100000:
                    window_nodes, _ = process_phrases_to_embeddings(
                        window_dir, text_embedder, use_pca=use_pca, pca_dim=args.pca_dim, pca_model=pca_model, batch_size=args.batch_size,
                    )
                else:
                    text_phrases = [' '.join(phrase) for phrase in window_phrases]
                    window_nodes, _ = text_embedder.get_embeddings(text_phrases, use_pca=use_pca, pca_dim=args.pca_dim, pca_model=pca_model)
                with open(window_dir + 'nodes.pkl', 'wb') as f:
                    pickle.dump(window_nodes, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FLASH graph generation for THEIA: baseline or AutoProv (rulellm+llmlabel). Outputs to BIGDATA/FLASH_artifacts, all zipped."
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Baseline mode",
    )
    parser.add_argument(
        "--autoprov",
        action="store_true",
        help="AutoProv mode",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../BIGDATA/DARPA-E3/",
        help="Base path to raw log dataset (for --baseline)",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory for outputs (default: {DEFAULT_ARTIFACTS_DIR})",
    )
    return parser.parse_args()


def zip_all_artifacts(artifacts_dir: str) -> None:
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        return

    zipped_count = 0
    for root, dirs, files in os.walk(artifacts_path, topdown=True):
        if ".zip" in root:
            continue
        root_path = Path(root)

        for fname in files:
            if fname.endswith(".zip"):
                continue
            path = root_path / fname
            if not path.is_file():
                continue

            if fname.endswith(".pkl") or fname.endswith(".pth"):
                zip_path = path.with_suffix(path.suffix + ".zip")
                try:
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.write(path, arcname=fname)
                    path.unlink()
                    zipped_count += 1
                except Exception:
                    pass

        for dname in list(dirs):
            if dname.startswith("word2vec_") and dname.endswith("_E3.model"):
                dir_path = root_path / dname
                zip_path = dir_path.with_suffix(dir_path.suffix + ".zip")
                try:
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        for f in dir_path.rglob("*"):
                            if f.is_file():
                                arcname = dname + "/" + f.relative_to(dir_path).as_posix()
                                zf.write(f, arcname=arcname)
                    shutil.rmtree(dir_path)
                    dirs.remove(dname)
                    zipped_count += 1
                except Exception:
                    pass

    for root, dirs, files in os.walk(artifacts_path, topdown=True):
        if ".zip" in root:
            continue
        root_path = Path(root)
        for fname in files:
            if fname.endswith(".zip"):
                continue
            if fname.startswith("word2vec_") and fname.endswith("_E3.model"):
                path = root_path / fname
                zip_path = path.with_suffix(path.suffix + ".zip")
                if zip_path.exists():
                    continue
                try:
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.write(path, arcname=fname)
                    path.unlink()
                    zipped_count += 1
                except Exception:
                    pass


def build_graph_gen_args(baseline: bool, artifacts_dir: str, dataset_path: str) -> SimpleNamespace:
    dataset = "theia"
    dates = get_dataset_dates(dataset)
    return SimpleNamespace(
        dataset=dataset,
        dataset_path=dataset_path,
        artifacts_dir=artifacts_dir,
        train_start_date=dates["train_start_date"],
        train_end_date=dates["train_end_date"],
        test_start_date=dates["test_start_date"],
        test_end_date=dates["test_end_date"],
        gen_graphs=True,
        train_attr=True,
        embedding="word2vec",
        infer_w2v=False,
        gen_test_nodes=False,
        resume=False,
        rulellm=not baseline,
        nolabel=False,
        llmfunc=False,
        llmlabel=not baseline,
        pca_dim=128,
        batch_size=32,
    )


def main():
    args = parse_args()

    if args.baseline and args.autoprov:
        print("Error: use exactly one of --baseline or --autoprov")
        sys.exit(1)
    if not args.baseline and not args.autoprov:
        print("Error: must specify either --baseline or --autoprov")
        sys.exit(1)

    artifacts_dir = os.path.abspath(args.artifacts_dir)
    dataset_path = args.dataset_path
    baseline = args.baseline

    os.makedirs(artifacts_dir, exist_ok=True)

    graph_gen_args = build_graph_gen_args(baseline, artifacts_dir, dataset_path)

    run_with_args(graph_gen_args)

    zip_all_artifacts(artifacts_dir)


if __name__ == "__main__":
    main()
