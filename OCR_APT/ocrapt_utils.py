import hashlib
import time
import pytz
from time import mktime
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler, normalize

def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

def ns_time_to_datetime_US(ns):
    tz = pytz.timezone('US/Eastern')
    dt = datetime.fromtimestamp(int(ns) // 1000000000, tz)
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

def ns_time_to_datetime(ns):
    dt = datetime.fromtimestamp(int(ns) // 1000000000)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def datetime_to_ns_time(date):
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000
    return timeStamp

def normalize_edge_features(features_df, edge_type_columns):
    existing_cols = [col for col in edge_type_columns if col in features_df.columns]
    
    if len(existing_cols) == 0:
        return features_df
    
    normalized_data = normalize(features_df[existing_cols], norm='l2', axis=1)
    features_df[existing_cols] = normalized_data
    
    return features_df

def scale_temporal_features(features_df, temporal_columns):
    existing_cols = [col for col in temporal_columns if col in features_df.columns]
    
    if len(existing_cols) == 0:
        return features_df, None
    
    scaler = MinMaxScaler()
    features_df[existing_cols] = scaler.fit_transform(features_df[existing_cols])
    
    return features_df, scaler

def apply_scaler(features_df, temporal_columns, scaler):
    if scaler is None:
        return features_df
    
    existing_cols = [col for col in temporal_columns if col in features_df.columns]
    
    if len(existing_cols) == 0:
        return features_df
    
    features_df[existing_cols] = scaler.transform(features_df[existing_cols])
    
    return features_df

def print_graph_stats(g, name="Graph"):
    print(f"\n{name} Statistics:")
    print(f"  Nodes: {g.number_of_nodes()}")
    print(f"  Edges: {g.number_of_edges()}")
    if g.number_of_nodes() > 0:
        print(f"  Avg degree: {2 * g.number_of_edges() / g.number_of_nodes():.2f}")

def get_node_type_distribution(nodes_dict):
    from collections import Counter
    node_types = []
    for node_info in nodes_dict.values():
        if isinstance(node_info, dict):
            node_type = list(node_info.keys())[0] if node_info else 'unknown'
        else:
            node_type = node_info
        node_types.append(node_type)
    return Counter(node_types)

def get_edge_type_distribution(edges_list):
    from collections import Counter
    edge_types = [edge[2] for edge in edges_list]
    return Counter(edge_types)

def order_edge_types_ocrapt(dataset, edge_types):
    priority_order = [
        'EVENT_EXECUTE', 'EVENT_FORK', 'EVENT_CLONE',
        'EVENT_OPEN', 'EVENT_CLOSE', 'EVENT_READ', 'EVENT_WRITE',
        'EVENT_SENDTO', 'EVENT_RECVFROM', 'EVENT_SENDMSG', 'EVENT_RECVMSG',
        'EVENT_CONNECT', 'EVENT_ACCEPT', 'EVENT_BIND', 'EVENT_LISTEN',
        'EVENT_MMAP', 'EVENT_MPROTECT', 'EVENT_UNLINK', 'EVENT_RENAME',
        'EVENT_LINK', 'EVENT_CHMOD', 'EVENT_TRUNCATE', 'EVENT_DUP',
        'EVENT_FCNTL', 'EVENT_IOCTL',         'EVENT_LOAD_MODULE'
    ]
    
    normalized_types = []
    for et in edge_types:
        if et.startswith('EVENT_'):
            normalized_types.append(et)
        else:
            normalized_types.append('EVENT_' + et.upper())
    
    priority_set = set(priority_order)
    priority_found = [et for et in priority_order if et in normalized_types]
    others_sorted = sorted([et for et in normalized_types if et not in priority_set])
    
    ordered = priority_found + others_sorted
    
    ordered_features = []
    for et in ordered:
        base_name = et.replace('EVENT_', '').lower()
        ordered_features.append(f'out_{base_name}')
        ordered_features.append(f'in_{base_name}')
    
    return ordered_features

def validate_graph_data(nodes, edges, edge_types):
    if not nodes:
        print("Warning: No nodes found in graph")
        return False
    
    if not edges:
        print("Warning: No edges found in graph")
        return False
    
    node_ids = set(nodes.keys())
    invalid_edges = 0
    for src, dst, etype, timestamp in edges:
        if src not in node_ids:
            invalid_edges += 1
        if dst not in node_ids:
            invalid_edges += 1
    
    if invalid_edges > 0:
        print(f"Warning: {invalid_edges} edge endpoints not found in nodes")
    
    edge_types_in_edges = set([e[2] for e in edges])
    if not edge_types_in_edges.issubset(set(edge_types)):
        print(f"Warning: Some edge types in edges not in edge_types list")
        print(f"  Missing: {edge_types_in_edges - set(edge_types)}")
    
    return True

def ensure_dir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def get_output_path(output_dir, dataset, filename):
    import os
    ensure_dir(output_dir)
    dataset_dir = os.path.join(output_dir, dataset)
    ensure_dir(dataset_dir)
    return os.path.join(dataset_dir, filename)

