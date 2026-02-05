#!/usr/bin/env python3

import hashlib
import time
import pytz
from time import mktime
from datetime import datetime

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

def print_graph_stats(g, name="Graph"):
    print(f"\n{name} Statistics:")
    print(f"  Nodes: {g.number_of_nodes()}")
    print(f"  Edges: {g.number_of_edges()}")
    if g.number_of_nodes() > 0:
        print(f"  Avg degree: {2 * g.number_of_edges() / g.number_of_nodes():.2f}")

def get_node_type_distribution(g):
    from collections import Counter
    node_types = [g.nodes[n]['type'] for n in g.nodes()]
    return Counter(node_types)

def get_edge_type_distribution(g):
    from collections import Counter
    edge_types = [g.edges[e]['type'] for e in g.edges()]
    return Counter(edge_types)

