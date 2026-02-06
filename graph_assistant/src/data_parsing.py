import re
import argparse
import json
import os
import glob
import sys
from datetime import datetime
from graph_splitter import split_graph_by_node_count

class data_parser:

    def __init__(self, data_path, output, attack_name=None, subgraph_size=None, magicsubgraph_idx=None, poisoning_mapping=None, poisoned_enriched_base_dir=None):
        self.data_path = data_path
        self.output = output
        self.attack_name = attack_name
        self.subgraph_size = subgraph_size
        self.magicsubgraph_idx = magicsubgraph_idx
        self._enriched_cache = None
        self.poisoning_mapping = poisoning_mapping or {}  # Dict mapping original names to poisoned names
        self.poisoned_enriched_base_dir = poisoned_enriched_base_dir
        self._enriched_path_logged = False

    def _resolve_enriched_nodes_path(self):
        if self.poisoned_enriched_base_dir:
            poisoned_dir = os.path.join(
                self.poisoned_enriched_base_dir,
                os.path.basename(self.data_path).replace(".json", "_poisoned_enriched/")
            )
            if os.path.exists(poisoned_dir):
                final_path = poisoned_dir if poisoned_dir.endswith("/") else poisoned_dir + "/"
                if not self._enriched_path_logged:
                    self._enriched_path_logged = True
                return final_path

        enriched_nodes_path = self.data_path.replace(".json", "_poisoned_enriched/")
        if not os.path.exists(enriched_nodes_path):
            enriched_nodes_path = self.data_path.replace(".json", "_enriched/")

        final_path = enriched_nodes_path if enriched_nodes_path.endswith("/") else enriched_nodes_path + "/"
        if not self._enriched_path_logged:
            self._enriched_path_logged = True
        return final_path
        
    def _load_enriched_files(self, enriched_nodes_path):
        if self._enriched_cache is not None:
            return self._enriched_cache
            
        self._enriched_cache = {}
        if not os.path.exists(enriched_nodes_path):
            return self._enriched_cache
            
        enriched_files = glob.glob(os.path.join(enriched_nodes_path, "*.json"))
        for enriched_file in enriched_files:
            try:
                with open(enriched_file, "r") as f:
                    enriched_data = json.load(f)
                    source_name = enriched_data.get("source_node_name")
                    target_name = enriched_data.get("target_node_name")
                    if source_name and target_name:
                        key = (source_name, target_name)
                        if key not in self._enriched_cache:
                            self._enriched_cache[key] = []
                        self._enriched_cache[key].append(enriched_data)
            except (json.JSONDecodeError, IOError) as e:
                continue
                
        return self._enriched_cache
    
    def _parse_timestamp(self, timestamp_str):
        if not timestamp_str:
            return None
        try:
            if '.' in timestamp_str:
                return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            else:
                return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return None
    
    def _aggregate_consecutive_edges(self, edges_with_data):
        if not edges_with_data:
            return []
        
        aggregated = []
        current_group = None
        
        for edge_data in edges_with_data:
            edge_key = (
                edge_data.get('src_name', ''),
                edge_data.get('dst_name', ''),
                edge_data.get('edge_type', 'unknown')
            )
            
            if current_group is None:
                current_group = {
                    'key': edge_key,
                    'count': 1,
                    'edge_data': edge_data.copy()
                }
            elif current_group['key'] == edge_key:
                current_group['count'] += 1
            else:
                if current_group['count'] > 1:
                    base_edge_str = current_group['edge_data']['edge_string']
                    current_group['edge_data']['edge_string'] = f"{base_edge_str} (x{current_group['count']})"
                aggregated.append(current_group['edge_data'])
                
                current_group = {
                    'key': edge_key,
                    'count': 1,
                    'edge_data': edge_data.copy()
                }
        
        if current_group:
            if current_group['count'] > 1:
                base_edge_str = current_group['edge_data']['edge_string']
                current_group['edge_data']['edge_string'] = f"{base_edge_str} (x{current_group['count']})"
            aggregated.append(current_group['edge_data'])
        
        return aggregated
    
    def _find_enriched_file(self, source, target, index, enriched_nodes_path):
        source_start = source.split("-")[0]
        target_start = target.split("-")[0]
        edge_path = f"edge_{source_start}_to_{target_start}_{index}.json"
        full_path = os.path.join(enriched_nodes_path, edge_path)
        
        if os.path.exists(full_path):
            try:
                with open(full_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        enriched_cache = self._load_enriched_files(enriched_nodes_path)
        key = (source, target)
        if key in enriched_cache:
            if enriched_cache[key]:
                return enriched_cache[key][0]
        
        return None
        
    def parse(self):
        malicious_edges = []
        with open(self.data_path, "r") as f:
            data = f.read()
            subgraph_dict = json.loads(data)

        enriched_nodes_path = self._resolve_enriched_nodes_path()

        split_data = self.data_path.replace(".json", "").split("/")
        
        path_parts = [part.replace(" ", "_") for part in split_data if part]
        if len(path_parts) >= 4:
            unique_output_path = "_".join(path_parts[-4:])
        elif len(path_parts) >= 3:
            unique_output_path = "_".join(path_parts[-3:])
        else:
            unique_output_path = path_parts[-1] if path_parts else "unknown"

        missing_enriched_count = 0
        edges_with_data = []  # Store edges with metadata for temporal sorting
        
        for edge_dict in subgraph_dict["links"]:
            source = edge_dict["source"]
            target = edge_dict["target"]
            index = edge_dict["key"]
            timestamp = edge_dict.get("timestamp")

            entity_information_dict = self._find_enriched_file(source, target, index, enriched_nodes_path)
            
            if entity_information_dict is None:
                missing_enriched_count += 1
                src_name = source
                dst_name = target
                edge_type = edge_dict.get("type", "unknown")
                src_information = "unknown"
                dst_information = "unknown"
            else:
                is_atlas_format = "source_node_name" in entity_information_dict or "target_node_name" in entity_information_dict
                
                source_enames = entity_information_dict.get("source_enames", [])
                target_enames = entity_information_dict.get("target_enames", [])
                
                if is_atlas_format:
                    if not source_enames or (len(source_enames) == 1 and source_enames[0] == "unknown"):
                        src_name = entity_information_dict.get("source_node_name", source)
                    else:
                        src_name = " ".join(source_enames)
                    
                    if not target_enames or (len(target_enames) == 1 and target_enames[0] == "unknown"):
                        dst_name = entity_information_dict.get("target_node_name", target)
                    else:
                        dst_name = " ".join(target_enames)
                else:
                    src_name = " ".join(source_enames) if source_enames else source
                    dst_name = " ".join(target_enames) if target_enames else target
                
                if self.poisoning_mapping:
                    if src_name in self.poisoning_mapping:
                        src_name = self.poisoning_mapping[src_name]
                    if dst_name in self.poisoning_mapping:
                        dst_name = self.poisoning_mapping[dst_name]
                
                edge_type = entity_information_dict.get("edge_action") or entity_information_dict.get("edge_type", edge_dict.get("type", "unknown"))
                
                src_information = entity_information_dict.get("source_type", "unknown")
                dst_information = entity_information_dict.get("target_type", "unknown")
            
            edge_string = f"{src_name} --{edge_type}--> {dst_name}"
            
            edges_with_data.append({
                'edge_string': edge_string,
                'timestamp': timestamp,
                'datetime': self._parse_timestamp(timestamp) if timestamp else None,
                'src_name': src_name,
                'dst_name': dst_name,
                'edge_type': edge_type,
                'src_information': src_information,
                'dst_information': dst_information
            })
        
        edges_with_timestamps = [e for e in edges_with_data if e['datetime'] is not None]
        edges_without_timestamps = [e for e in edges_with_data if e['datetime'] is None]
        
        if edges_with_timestamps:
            edges_with_timestamps.sort(key=lambda x: x['datetime'])
            sorted_edges_data = edges_with_timestamps + edges_without_timestamps
        else:
            sorted_edges_data = edges_with_data
        
        aggregated_edges_data = self._aggregate_consecutive_edges(sorted_edges_data)
        
        malicious_edges = [e['edge_string'] for e in aggregated_edges_data]
        
        malicious_nodes = []
        malicious_nodes_func = []
        for e in aggregated_edges_data:
            malicious_nodes.append(e['src_name'])
            malicious_nodes_func.append(e['src_information'])
            malicious_nodes.append(e['dst_name'])
            malicious_nodes_func.append(e['dst_information'])

        node_func_map = {}
        for i, node in enumerate(malicious_nodes):
            if node not in node_func_map:
                node_func_map[node] = malicious_nodes_func[i]
        
        malicious_nodes = list(node_func_map.keys())
        malicious_nodes_func = [node_func_map[node] for node in malicious_nodes]


        with open(f"{self.output}/parsed_data/malicious_edges_{unique_output_path}.txt", "w") as f:
            for item in malicious_edges:
                f.write(item + "\n")
        with open(f"{self.output}/parsed_data/malicious_nodes_{unique_output_path}.txt", "w") as f:
            for item in malicious_nodes:
                f.write(item + "\n")
        return malicious_edges, malicious_nodes, malicious_nodes_func, unique_output_path
    
    def _parse_subgraph_edges(self, subgraph_edges, enriched_nodes_path):
        malicious_edges = []
        
        missing_enriched_count = 0
        edges_with_data = []
        
        for edge_dict in subgraph_edges:
            source = edge_dict.get("source")
            target = edge_dict.get("target")
            index = edge_dict.get("key", 0)
            timestamp = edge_dict.get("timestamp")
            
            entity_information_dict = self._find_enriched_file(source, target, index, enriched_nodes_path)
            
            if entity_information_dict is None:
                missing_enriched_count += 1
                src_name = source
                dst_name = target
                edge_type = edge_dict.get("type", "unknown")
                src_information = "unknown"
                dst_information = "unknown"
            else:
                is_atlas_format = "source_node_name" in entity_information_dict or "target_node_name" in entity_information_dict
                
                source_enames = entity_information_dict.get("source_enames", [])
                target_enames = entity_information_dict.get("target_enames", [])
                
                if is_atlas_format:
                    if not source_enames or (len(source_enames) == 1 and source_enames[0] == "unknown"):
                        src_name = entity_information_dict.get("source_node_name", source)
                    else:
                        src_name = " ".join(source_enames)
                    
                    if not target_enames or (len(target_enames) == 1 and target_enames[0] == "unknown"):
                        dst_name = entity_information_dict.get("target_node_name", target)
                    else:
                        dst_name = " ".join(target_enames)
                else:
                    src_name = " ".join(source_enames) if source_enames else source
                    dst_name = " ".join(target_enames) if target_enames else target
                
                if self.poisoning_mapping:
                    if src_name in self.poisoning_mapping:
                        src_name = self.poisoning_mapping[src_name]
                    if dst_name in self.poisoning_mapping:
                        dst_name = self.poisoning_mapping[dst_name]
                
                edge_type = entity_information_dict.get("edge_action") or entity_information_dict.get("edge_type", edge_dict.get("type", "unknown"))
                src_information = entity_information_dict.get("source_type", "unknown")
                dst_information = entity_information_dict.get("target_type", "unknown")
            
            edge_string = f"{src_name} --{edge_type}--> {dst_name}"
            
            edges_with_data.append({
                'edge_string': edge_string,
                'timestamp': timestamp,
                'datetime': self._parse_timestamp(timestamp) if timestamp else None,
                'src_name': src_name,
                'dst_name': dst_name,
                'edge_type': edge_type,
                'src_information': src_information,
                'dst_information': dst_information
            })
        
        edges_with_timestamps = [e for e in edges_with_data if e['datetime'] is not None]
        edges_without_timestamps = [e for e in edges_with_data if e['datetime'] is None]
        
        if edges_with_timestamps:
            edges_with_timestamps.sort(key=lambda x: x['datetime'])
            sorted_edges_data = edges_with_timestamps + edges_without_timestamps
        else:
            sorted_edges_data = edges_with_data
        
        aggregated_edges_data = self._aggregate_consecutive_edges(sorted_edges_data)
        
        malicious_edges = [e['edge_string'] for e in aggregated_edges_data]
        
        malicious_nodes = []
        malicious_nodes_func = []
        for e in aggregated_edges_data:
            malicious_nodes.append(e['src_name'])
            malicious_nodes_func.append(e['src_information'])
            malicious_nodes.append(e['dst_name'])
            malicious_nodes_func.append(e['dst_information'])
        
        node_func_map = {}
        for i, node in enumerate(malicious_nodes):
            if node not in node_func_map:
                node_func_map[node] = malicious_nodes_func[i]
        
        malicious_nodes = list(node_func_map.keys())
        malicious_nodes_func = [node_func_map[node] for node in malicious_nodes]
        
        return malicious_edges, malicious_nodes, malicious_nodes_func
    
    def split_and_parse(self, attack_name):
        from tqdm import tqdm
        
        with open(self.data_path, "r") as f:
            data = f.read()
            subgraph_dict = json.loads(data)
        
        enriched_nodes_path = self._resolve_enriched_nodes_path()
        
        if self.subgraph_size == -1:
            subgraph_edges = subgraph_dict["links"]
            malicious_edges, malicious_nodes, malicious_nodes_func = self._parse_subgraph_edges(
                subgraph_edges, enriched_nodes_path
            )
            return [(self.magicsubgraph_idx if self.magicsubgraph_idx is not None else 0, malicious_edges, malicious_nodes, malicious_nodes_func)]
        
        edges_list = subgraph_dict["links"]
        from graph_splitter import build_adjacency_list
        adjacency, all_nodes = build_adjacency_list(edges_list)
        
        if len(all_nodes) <= self.subgraph_size:
            malicious_edges, malicious_nodes, malicious_nodes_func = self._parse_subgraph_edges(
                edges_list, enriched_nodes_path
            )
            return [(self.magicsubgraph_idx if self.magicsubgraph_idx is not None else 0, malicious_edges, malicious_nodes, malicious_nodes_func)]
        
        subgraphs = split_graph_by_node_count(edges_list, self.subgraph_size)
        
        results = []
        for split_idx, subgraph_edges in enumerate(tqdm(subgraphs, desc="Parsing subgraphs", leave=False)):
            malicious_edges, malicious_nodes, malicious_nodes_func = self._parse_subgraph_edges(
                subgraph_edges, enriched_nodes_path
            )
            
            results.append((split_idx, malicious_edges, malicious_nodes, malicious_nodes_func))
        
        return results



class kairos_data_parser:
    def __init__(self, data_path, output):
        self.data_path = data_path
        self.output = output
        
    def parse(self):
        try:
            with open(self.data_path, "r") as f:
                file_data = f.readlines()
        except FileNotFoundError:
            return
        
        file_data.pop(0)
        file_data.pop(0)
        file_data.pop(0)
        file_data.pop(-1)
        
        input_file = self.data_path.split("/")[-1]
        malicious_nodes = []
        benign_nodes = []
        malicious_edges = []
        with open(f"{self.output}/malicious_theia_edges_{input_file}", "w") as f:
            for i in range(0, len(file_data), 3):
                data_buffer = file_data[i:i+3]
                if len(data_buffer) < 3:
                    continue
                if self.find_malicious_edges(data_buffer):
                    edge = self.parse_edge(data_buffer)
                    f.write(edge)
                    malicious_edges.append(edge)
                    self.node_classification(data_buffer[0:2], malicious_nodes, benign_nodes)
                    
        with open(f"{self.output}/malicious_theia_node_classification_{input_file}", "w") as f:
            unique_malicious_nodes = list(set(malicious_nodes))
            unique_benign_nodes = list(set(benign_nodes))

            f.write(f"Malicious nodes: {str(unique_malicious_nodes)}\n")
            f.write(f"Benign nodes: {str(unique_benign_nodes)}\n")
        
        return malicious_edges, unique_malicious_nodes, unique_benign_nodes



    def node_classification(self,data_buffer, malicious_nodes, benign_nodes):
        src = data_buffer[0].split(" ", 1)[1].strip()
        dst = data_buffer[1].split(" ", 1)[1].strip()

        color1 = self.get_node_color(src)
        color2 = self.get_node_color(dst)

        src_label = self.extract_entity_name(src)
        dst_label = self.extract_entity_name(dst)

        if color1 == "red":
            malicious_nodes.append(src_label)
        else:
            benign_nodes.append(src_label)
        
        if color2 == "red":
            malicious_nodes.append(dst_label)
        else:
            benign_nodes.append(dst_label)
            
    def parse_edge(self, data_buffer):
        first_node = data_buffer[0].split(" ", 1)[1].strip()
        second_node = data_buffer[1].split(" ", 1)[1].strip()
        edge_data = data_buffer[2].split(" ", 3)[3].strip()


        first_node_label = self.extract_entity_name(first_node)
        second_node_label = self.extract_entity_name(second_node)
        
        edge_label = re.search(r'label=([^\s;,\)]+)', edge_data).group(1)
        
        final_edge = f"{first_node_label} --{edge_label}--> {second_node_label}\n"
        return final_edge
            
            
            
    def find_malicious_edges(self, data):
        for edge in data:
            match = re.search(r'color=([^\s;,\)]+)', edge)
            if "red" in match.group(1):
                return True
        return False

    def find_malicious_edges_sparse(self, data):
        edge_info = data[2]
        match = re.search(r'color=([^\s;,\)]+)', edge_info)
        if "red" in match.group(1):
            return True
        return False

    def get_node_color(self, node):
        match = re.search(r'color=([^\s;,\)]+)', node)
        return match.group(1) if match else None

    def extract_entity_name(self, node):
        match = re.search(r'label=\"([^\"]+)\"', node)
        match2 = re.search(r"'[^']+':\s*'([^']*)'", match.group(1))
        return match2.group(1) if match2 else match.group(1)
