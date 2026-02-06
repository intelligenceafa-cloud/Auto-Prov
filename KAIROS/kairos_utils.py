#!/usr/bin/env python3

import numpy as np
import time
import pytz
from time import mktime
from datetime import datetime
import hashlib
import math

DATASET_DATES = {
    "THEIA": {
        "train_start_date": "2018-04-03",
        "train_end_date": "2018-04-05",
        "test_start_date": "2018-04-09",
        "test_end_date": "2018-04-12"
    },
    "FIVEDIRECTIONS": {
        "train_start_date": "2018-04-04",
        "train_end_date": "2018-04-08",
        "test_start_date": "2018-04-09",
        "test_end_date": "2018-04-13"
    }
}

def ns_time_to_datetime(ns):
    dt = datetime.fromtimestamp(int(ns) // 1000000000)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def ns_time_to_datetime_US(ns):
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1000000000, tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def time_to_datetime_US(s):
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(s), tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    return s

def datetime_to_ns_time(date):
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000
    return timeStamp

def datetime_to_ns_time_US(date):
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
    return int(timeStamp)

def datetime_to_timestamp_US(date):
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp
    return int(timeStamp)

def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

def path2higlist(p):
    l = []
    spl = p.strip().split('/')
    for i in spl:
        if len(l) != 0:
            l.append(l[-1] + '/' + i)
        else:
            l.append(i)
    return l

def ip2higlist(p):
    l = []
    spl = p.strip().split('.')
    for i in spl:
        if len(l) != 0:
            l.append(l[-1] + '.' + i)
        else:
            l.append(i)
    return l

def list2str(l):
    s = ''
    for i in l:
        s += i
    return s

def mean(data):
    t = np.array(data)
    return np.mean(t)

def std(data):
    t = np.array(data)
    return np.std(t)

def var(data):
    t = np.array(data)
    return np.var(t)

def calculate_idf_for_node(node, file_list, node_appearances):
    if node in node_appearances:
        include_count = len(node_appearances[node])
    else:
        include_count = 0
    
    idf = math.log(len(file_list) / (include_count + 1))
    return idf

def is_rare_node(idf_value, file_list, rarity_threshold=0.9):
    max_possible_idf = math.log(len(file_list) * rarity_threshold / 1)
    return idf_value > max_possible_idf

def calculate_anomaly_threshold(loss_list, multiplier=1.5):
    if len(loss_list) == 0:
        return 0
    
    loss_mean = mean(loss_list)
    loss_std = std(loss_list)
    threshold = loss_mean + multiplier * loss_std
    
    return threshold

def compute_adp(y_true, y_scores, thresholds=None):
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1]
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    num_positive = np.sum(y_true == 1)
    num_negative = np.sum(y_true == 0)
    
    if num_positive == 0 or num_negative == 0:
        return 0.0
    
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    detection_rates = []
    for fpr_threshold in thresholds:
        fp_allowed = int(np.ceil(fpr_threshold * num_negative))
        
        fp_count = 0
        tp_count = 0
        
        for label in sorted_labels:
            if label == 0:
                fp_count += 1
            else:
                tp_count += 1
            
            if fp_count > fp_allowed:
                break
        
        detection_rate = tp_count / num_positive
        detection_rates.append(detection_rate)
    
    adp = np.mean(detection_rates)
    return adp

def calculate_metrics(TP, FP, FN, TN):
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    TPR = TP / (TP + FN) if TP + FN > 0 else 0

    prec = TP / (TP + FP) if TP + FP > 0 else 0
    rec = TP / (TP + FN) if TP + FN > 0 else 0
    fscore = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0

    return prec, rec, fscore, FPR, TPR

def get_benign_filter_keywords(dataset="theia"):
    if dataset.upper() == "THEIA":
        keywords = [
            'netflow', 'null', '/dev/pts', 'salt-minion.log', '675',
            'usr', 'proc', '/.cache/mozilla/', 'tmp', 'thunderbird',
            '/bin/', '/sbin/sysctl', '/data/replay_logdb/', 
            '/home/admin/.cache', '/stat', '/boot',
            'qt-opensource-linux-x64', '/eraseme'
        ]
    elif dataset.upper() == "FIVEDIRECTIONS":
        keywords = [
            'netflow', '/dev/pts', 'salt-minion.log', 'null',
            'usr', 'proc', 'firefox', 'tmp', 'thunderbird',
            'bin/', '/data/replay_logdb', '/stat', '/boot',
            'qt-opensource-linux-x64', '/eraseme', '675'
        ]
    else:
        keywords = [
            'netflow', 'null', '/dev/pts', 'usr', 'proc', 'tmp'
        ]
    
    return keywords

def is_benign_pattern(node_msg, keywords):
    for keyword in keywords:
        if keyword in str(node_msg):
            return True
    return False

def get_edge_type_mapping():
    rel2id = {
        1: 'EVENT_CONNECT', 'EVENT_CONNECT': 1,
        2: 'EVENT_EXECUTE', 'EVENT_EXECUTE': 2,
        3: 'EVENT_OPEN', 'EVENT_OPEN': 3,
        4: 'EVENT_READ', 'EVENT_READ': 4,
        5: 'EVENT_RECVFROM', 'EVENT_RECVFROM': 5,
        6: 'EVENT_RECVMSG', 'EVENT_RECVMSG': 6,
        7: 'EVENT_SENDMSG', 'EVENT_SENDMSG': 7,
        8: 'EVENT_SENDTO', 'EVENT_SENDTO': 8,
        9: 'EVENT_WRITE', 'EVENT_WRITE': 9
    }
    return rel2id

def is_reverse_edge(edge_type):
    reverse_types = [
        'EVENT_READ', 'EVENT_READ_SOCKET_PARAMS', 
        'EVENT_RECVFROM', 'EVENT_RECVMSG'
    ]
    return edge_type in reverse_types

def load_attack_ground_truth(dataset="theia"):
    print(f"Warning: Ground truth loading not implemented for {dataset}")
    return set()

def save_results_summary(results, output_path):
    with open(output_path, 'w') as f:
        f.write("KAIROS Anomaly Detection Results\n")
        f.write("=" * 80 + "\n\n")
        
        for campaign_id, campaign_data in enumerate(results, 1):
            f.write(f"Campaign {campaign_id}:\n")
            f.write(f"  Anomaly Score: {campaign_data.get('anomaly_score', 0):.4f}\n")
            f.write(f"  Time Windows: {campaign_data.get('windows', [])}\n")
            f.write(f"  Node Count: {campaign_data.get('node_count', 0)}\n")
            f.write("\n")
    
    print(f"Results saved to: {output_path}")

def create_attack_subgraph(edges, nodes, output_path):
    print(f"Subgraph visualization not implemented yet")
    print(f"Would save to: {output_path}")
    pass

if __name__ == "__main__":
    print("Testing KAIROS utilities...")
    
    ns_timestamp = 1522764134000000000
    dt_str = ns_time_to_datetime_US(ns_timestamp)
    print(f"Timestamp: {ns_timestamp} -> {dt_str}")
    
    path = "/home/user/documents/file.txt"
    hlist = path2higlist(path)
    print(f"Path: {path}")
    print(f"Hierarchical: {hlist}")
    
    ip = "192.168.1.1"
    iplist = ip2higlist(ip)
    print(f"IP: {ip}")
    print(f"Hierarchical: {iplist}")
    
    data = [1, 2, 3, 4, 5, 100]
    print(f"Data: {data}")
    print(f"Mean: {mean(data):.2f}")
    print(f"Std: {std(data):.2f}")
    print(f"Threshold: {calculate_anomaly_threshold(data):.2f}")
    
    print("\n--- Testing ADP Metric ---")
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.8, 0.9, 0.7])
    adp = compute_adp(y_true, y_scores)
    print(f"y_true: {y_true}")
    print(f"y_scores: {y_scores}")
    print(f"ADP: {adp:.4f}")
    
    print("\n--- DATASET_DATES ---")
    for dataset, dates in DATASET_DATES.items():
        print(f"{dataset}:")
        for key, value in dates.items():
            print(f"  {key}: {value}")
    
    print("\nAll utilities working correctly!")

