#!/usr/bin/env python3

import pickle
import json
import re
import os
from typing import List, Dict, Any, Tuple, Optional
import argparse
from datetime import datetime
import random
from pathlib import Path

class LLMRegexGenerator:
    def __init__(self, model_type: str = "ollama", model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", dataset: str = "ATLAS", embedding: str = None, log_type: str = "audit", **kwargs):
        self.model_type = model_type
        self.model_name = model_name
        self.dataset = dataset
        self.embedding = embedding
        self.log_type = log_type
        self.cee = kwargs.get('cee', None)
        self.has_builtin_ids = kwargs.get('has_builtin_ids', 'yes')
        self.edges_data = []
        self.enames_data = []
        self.vtypes_data = []
        self.show_gpu_usage = kwargs.get('show_gpu_usage', False)
        self.auto_restart_ollama = kwargs.get('auto_restart_ollama', True)
        self.incremental_save_interval = kwargs.get('incremental_save_interval', 10)
        self.save_dir = kwargs.get('save_dir', './temp')
        
        if self.model_type != "ollama":
            raise ValueError("Only Ollama models are supported in this version")
        
        self._initialize_model(**kwargs)
    
    def _initialize_model(self, **kwargs):
        if self.model_type == "ollama":
            self._initialize_ollama(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _initialize_ollama(self, **kwargs):
        try:
            import requests
            import os
            
            self.ollama_url = kwargs.get('ollama_url')
            if not self.ollama_url:
                raise ValueError("ollama_url is required")
            
            gpu_cards = kwargs.get('gpu_cards')
            if not gpu_cards:
                gpu_cards = self._get_all_available_gpus()
            
            self._setup_gpu_environment(gpu_cards)
            
            self._check_gpu_usage()
            
            self._verify_ollama_gpu_usage(gpu_cards)
            
            try:
                response = requests.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    pass
                else:
                    raise ConnectionError(f"Ollama server responded with status {response.status_code}")
            except requests.exceptions.ConnectionError:
                raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_url}. Make sure Ollama is running.")
                
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
    
    def _setup_gpu_environment(self, gpu_cards):
        import os
        import subprocess
        
        if not gpu_cards:
            return
        
        self._validate_gpu_cards(gpu_cards)
        
        ollama_running = self._check_ollama_running()
        
        if ollama_running:
            if self.auto_restart_ollama:
                self._restart_ollama_with_gpu_restriction(gpu_cards)
            else:
                pass
        else:
            if self.auto_restart_ollama:
                self._start_ollama_with_gpu_restriction(gpu_cards)
            else:
                pass
    
    def _check_ollama_running(self):
        try:
            import requests
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _restart_ollama_with_gpu_restriction(self, gpu_cards):
        import subprocess
        import time
        
        try:
            result = subprocess.run(['ollama', 'stop'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                pass
            
            time.sleep(2)
            
            self._start_ollama_with_gpu_restriction(gpu_cards)
            
        except subprocess.TimeoutExpired:
            raise Exception("Failed to restart Ollama")
        except Exception as e:
            raise
    
    def _start_ollama_with_gpu_restriction(self, gpu_cards):
        import subprocess
        import os
        import time
        
        try:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu_cards
            
            process = subprocess.Popen(['ollama', 'serve'], 
                                     env=env,
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
            
            max_wait = 30
            for i in range(max_wait):
                time.sleep(1)
                if self._check_ollama_running():
                    return
                pass
            
            process.terminate()
            raise Exception("Ollama failed to start within 30 seconds")
            
        except Exception as e:
            raise
    
    def _get_all_available_gpus(self):
        try:
            import subprocess
            
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                available_gpus = len(result.stdout.strip().split('\n'))
                if available_gpus > 0:
                    gpu_list = ','.join(str(i) for i in range(available_gpus))
                    return gpu_list
            return None
        except Exception:
            return None
    
    def _validate_gpu_cards(self, gpu_cards):
        try:
            import subprocess
            
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                available_gpus = len(result.stdout.strip().split('\n'))
                
                requested_gpus = []
                for card in gpu_cards.split(','):
                    try:
                        gpu_id = int(card.strip())
                        requested_gpus.append(gpu_id)
                    except ValueError:
                        continue
                
                max_requested = max(requested_gpus) if requested_gpus else 0
                if max_requested >= available_gpus:
                    raise ValueError(f"GPU {max_requested} not available. Only {available_gpus} GPUs found (0-{available_gpus-1}).")
                
            else:
                pass
                
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            pass
    
    def _check_gpu_usage(self):
        try:
            import subprocess
            
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                pass
            else:
                pass
                
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            pass
    
    def _verify_ollama_gpu_usage(self, gpu_cards):
        try:
            import subprocess
            
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if result.returncode == 0:
                ollama_processes = [line for line in result.stdout.split('\n') if 'ollama' in line.lower()]
                
                if ollama_processes:
                    for process in ollama_processes:
                        if 'serve' in process:
                            if f'CUDA_VISIBLE_DEVICES={gpu_cards}' in process:
                                pass
                            elif 'CUDA_VISIBLE_DEVICES' in process:
                                import re
                                match = re.search(r'CUDA_VISIBLE_DEVICES=([^\s]+)', process)
                                if match:
                                    actual_gpus = match.group(1)
                                    pass
                                else:
                                    pass
                            else:
                                pass
                            break
                    else:
                        pass
                else:
                    pass
                    
        except Exception as e:
            pass
    
    def generate_text(self, prompt: str, max_length: int = 4000, temperature: float = 0.0) -> str:
        if self.model_type == "ollama":
            return self._generate_ollama(prompt, max_length, temperature)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _generate_ollama(self, prompt: str, max_length: int, temperature: float) -> str:
        import requests
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_length
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=data)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {e}")
    
    def cleanup_model(self):
        if self.model_type == "ollama":
            try:
                import requests
                import subprocess
                
                try:
                    response = requests.post(f"{self.ollama_url}/api/stop", json={"name": self.model_name})
                    if response.status_code == 200:
                        pass
                    else:
                        result = subprocess.run(['ollama', 'stop', self.model_name], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            pass
                        else:
                            pass
                except Exception as e:
                    pass
                
            except Exception as e:
                pass
        
        pass
         
    def load_training_data(self, file_type: str = "edges", log_type: str = "audit", embedding: str = None, candidates_dir: str = "./candidates-atlas", cee: str = None):
        
        if cee:
            candidates_path = os.path.join(candidates_dir, embedding, cee, log_type, "candidate-output.json")
        else:
            candidates_path = os.path.join(candidates_dir, embedding, log_type, "candidate-output.json")
        if not os.path.exists(candidates_path):
            raise FileNotFoundError(f"Candidate file not found: {candidates_path}")
        
        with open(candidates_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        def parse_input(input_str):
            try:
                parsed = json.loads(input_str)
                return parsed
            except (json.JSONDecodeError, TypeError):
                return input_str
        
        if file_type == "edges":
            self.edges_data = []
            for entry in json_data:
                input_log = parse_input(entry['input'])
                edges = entry.get('edges', '')
                
                if isinstance(edges, list):
                    self.edges_data.append({
                        'input': input_log,
                        'target': edges,
                        'is_dict_format': True
                    })
                else:
                    self.edges_data.append({
                        'input': input_log,
                        'target': edges,
                        'is_dict_format': False
                    })
            return self.edges_data
        elif file_type == "enames":
            self.enames_data = []
            total_duplicates_removed = 0
            
            for entry in json_data:
                input_log = parse_input(entry['input'])
                target_enames = entry['enames']
                
                filtered_lines = []
                for line in target_enames.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    if " = " in line:
                        parts = line.split(" = ", 1)
                        if len(parts) == 2:
                            entity_id = parts[0].strip()
                            if " List" not in entity_id:
                                filtered_lines.append(line)
                
                if filtered_lines:
                    deduplicated_lines = self.remove_duplicate_id_assignments(filtered_lines)
                    total_duplicates_removed += len(filtered_lines) - len(deduplicated_lines)
                    
                    if deduplicated_lines:
                        filtered_example = {
                            'input': input_log,
                            'target': '\n'.join(deduplicated_lines)
                        }
                        self.enames_data.append(filtered_example)
            
            return self.enames_data
        elif file_type == "vtypes":
            self.vtypes_data = []
            total_duplicates_removed = 0
            
            for entry in json_data:
                input_log = parse_input(entry['input'])
                target_vtypes = entry['vtypes']
                
                if isinstance(target_vtypes, str) and target_vtypes != "NOTHING TO BE EXTRACTED":
                    lines = [line.strip() for line in target_vtypes.split('\n') if line.strip()]
                    deduplicated_lines = self.remove_duplicate_id_assignments(lines)
                    total_duplicates_removed += len(lines) - len(deduplicated_lines)
                    
                    if deduplicated_lines:
                        deduplicated_example = {
                            'input': input_log,
                            'target': '\n'.join(deduplicated_lines)
                        }
                        self.vtypes_data.append(deduplicated_example)
                else:
                    self.vtypes_data.append({
                        'input': input_log,
                        'target': target_vtypes
                    })
            
            return self.vtypes_data
        else:
            raise ValueError(f"Unknown file type: {file_type}. Must be 'edges', 'enames', or 'vtypes'")
    
    def remove_duplicate_id_assignments(self, lines: List[str]) -> List[str]:
        if not lines:
            return lines
        
        unique_assignments = {}
        
        for line in lines:
            if " = " in line and line != "No Regex":
                parts = line.split(" = ", 1)
                if len(parts) == 2:
                    entity_id, entity_value = parts
                    entity_id = entity_id.strip()
                    entity_value = entity_value.strip()
                    
                    if entity_id not in unique_assignments:
                        unique_assignments[entity_id] = entity_value
        
        deduplicated_lines = [f"{id_val} = {value}" for id_val, value in unique_assignments.items()]
        return deduplicated_lines

    def classify_entity_type(self, entity: str) -> str:
        if not entity or not isinstance(entity, str):
            return "unknown"
        
        entity = entity.strip()
        if not entity:
            return "unknown"
        
        has_english = bool(re.search(r'[a-zA-Z]', entity))
        
        has_digits = bool(re.search(r'\d', entity))
        
        has_dots = '.' in entity
        
        has_colons = ':' in entity
        
        if has_english:
            return "website_filename"
        elif has_digits and has_dots and not has_english:
            if has_colons:
                return "ip_with_port"
            else:
                return "ip_no_port"
        elif has_digits and not has_dots and not has_english:
            return "integer_only"
        else:
            return "unknown"
    
    def parse_edge_format(self, edge_line: str) -> dict:
        edge_info = {
            'source': None,
            'destination': None,
            'action': None,
            'timestamp': None,
            'has_label': True,
            'has_timestamp': True
        }
        
        import re
        
        uuid_pattern = r'\(([^,]+),\s*([^)]+)\)'
        uuid_match = re.search(uuid_pattern, edge_line)
        
        if uuid_match:
            uuid_a = uuid_match.group(1).strip()
            uuid_b = uuid_match.group(2).strip()
            
            direction_pattern = r'\{D=([<>-]+)\}'
            direction_match = re.search(direction_pattern, edge_line)
            
            if direction_match:
                direction = direction_match.group(1)
                if direction == '<-':
                    edge_info['source'] = uuid_b
                    edge_info['destination'] = uuid_a
                elif direction == '->':
                    edge_info['source'] = uuid_a
                    edge_info['destination'] = uuid_b
                else:
                    edge_info['source'] = uuid_a
                    edge_info['destination'] = uuid_b
            else:
                edge_info['source'] = uuid_a
                edge_info['destination'] = uuid_b
        
        action_pattern = r'A:\s*\[([^\]]+)\]'
        action_match = re.search(action_pattern, edge_line)
        
        if action_match:
            action = action_match.group(1).strip()
            if action == "NO LABEL":
                edge_info['action'] = "NO LABEL"
                edge_info['has_label'] = False
            else:
                edge_info['action'] = action
        
        timestamp_pattern = r'timestamp=([^)]+)'
        timestamp_match = re.search(timestamp_pattern, edge_line)
        
        if timestamp_match:
            timestamp = timestamp_match.group(1).strip()
            if timestamp == "..." or timestamp == "NO TIMESTAMP":
                edge_info['timestamp'] = "NO TIMESTAMP"
                edge_info['has_timestamp'] = False
            else:
                edge_info['timestamp'] = timestamp
        else:
            edge_info['timestamp'] = "NO TIMESTAMP"
            edge_info['has_timestamp'] = False
        
        return edge_info
    
    def analyze_patterns(self, data: List[Dict], num_samples: int = 10) -> Dict[str, Any]:
        patterns = {
            'input_patterns': [],
            'target_patterns': [],
            'common_fields': set(),
            'target_formats': set()
        }
        
        sample_count = 0
        for sample in data:
            if sample_count >= num_samples:
                break
                
            input_text = sample['input']
            target_text = sample['target']
            
            try:
                json_start = input_text.find('{')
                json_end = input_text.rfind('}') + 1
                json_str = input_text[json_start:json_end]
                parsed_json = json.loads(json_str)
                
                if 'datum' in parsed_json:
                    for key in parsed_json['datum'].keys():
                        patterns['common_fields'].add(key)
                        
            except Exception as e:
                continue
                
            patterns['input_patterns'].append(input_text[:200] + "...")
            patterns['target_patterns'].append(target_text)
            patterns['target_formats'].add(self._extract_target_format(target_text))
            
            sample_count += 1
                
        return patterns
    
    def _extract_target_format(self, target: str) -> str:
        uuid_pattern = r'[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}'
        format_pattern = re.sub(uuid_pattern, '<UUID>', target)
        
        timestamp_pattern = r'\d{19}'
        format_pattern = re.sub(timestamp_pattern, '<TIMESTAMP>', format_pattern)
        
        number_pattern = r'\d+'
        format_pattern = re.sub(number_pattern, '<NUMBER>', format_pattern)
        
        return format_pattern

    def generate_regex_for_edges(self) -> str:
        if self.cee:
            folder_name = f"{self.cee.lower()}_{self.model_name.lower()}"
            folder_name_sanitized = sanitize_cee_name(folder_name)
            rules_dir = os.path.join(self.save_dir, 'ATLAS', self.embedding, folder_name_sanitized, self.log_type, 'edges')
        else:
            rules_dir = os.path.join(self.save_dir, 'ATLAS', self.embedding, self.log_type, 'edges')
        os.makedirs(rules_dir, exist_ok=True)
        
        all_regex_patterns = []
        
        total_logs_processed = 0
        total_edges_processed = 0
        total_edges_skipped = 0
        
        examples = self.edges_data
        
        for log_idx, example in enumerate(examples):
            input_log = example['input']
            target_edges = example['target']
            is_dict_format = example.get('is_dict_format', False)
        
            if is_dict_format:
                edge_dicts = target_edges if isinstance(target_edges, list) else []
                
                for edge_idx, edge_dict in enumerate(edge_dicts):
                    source = edge_dict.get('source', '')
                    dest = edge_dict.get('dest', '')
                    action = edge_dict.get('Action', '')
                    timestamp = edge_dict.get('timestamp', '')
                    
                    source_type = self.classify_entity_type(source)
                    dest_type = self.classify_entity_type(dest)
                    
                    if source_type == "integer_only" or dest_type == "integer_only":
                        total_edges_skipped += 1
                        continue
                    
                    if source_type == "website_filename":
                        source_regex_response = self.generate_ename_name_regex_for_entity(input_log, source)
                        source_regex = self.parse_single_regex_from_response(source_regex_response)
                    elif source_type == "ip_with_port":
                        source_info = {'source': source}
                        source_regex_response = self.generate_source_id_regex(input_log, source_info)
                        source_regex = self.parse_id_regex_from_response(source_regex_response)
                    elif source_type == "ip_no_port":
                        source_regex_response = self.generate_ip_regex_no_port(input_log, source)
                        source_regex = self.parse_single_regex_from_response(source_regex_response)
                    else:
                        total_edges_skipped += 1
                        continue
                    
                    if dest_type == "website_filename":
                        dest_regex_response = self.generate_ename_name_regex_for_entity(input_log, dest)
                        dest_regex = self.parse_single_regex_from_response(dest_regex_response)
                    elif dest_type == "ip_with_port":
                        dest_info = {'destination': dest}
                        dest_regex_response = self.generate_dest_id_regex(input_log, dest_info)
                        dest_regex = self.parse_id_regex_from_response(dest_regex_response)
                    elif dest_type == "ip_no_port":
                        dest_regex_response = self.generate_ip_regex_no_port(input_log, dest)
                        dest_regex = self.parse_single_regex_from_response(dest_regex_response)
                    else:
                        total_edges_skipped += 1
                        continue
                    
                    action_info = {'action': action}
                    action_regex = self.generate_action_regex(input_log, action_info)
                    if action_regex == "NO LABEL":
                        action_regex = "NO LABEL"
                    else:
                        action_regex = self.parse_single_regex_from_response(action_regex)
                    
                    timestamp_info = {'timestamp': timestamp}
                    timestamp_regex = self.generate_timestamp_regex(input_log, timestamp_info)
                    if timestamp_regex == "NO TIMESTAMP":
                        timestamp_regex = "NO TIMESTAMP"
                    else:
                        timestamp_regex = self.parse_single_regex_from_response(timestamp_regex)
                    
                    regex_dict = {
                        'source': source_regex,
                        'dest': dest_regex,
                        'action': action_regex,
                        'timestamp': timestamp_regex
                    }
                    
                    
                    all_regex_patterns.append(regex_dict)
                    total_edges_processed += 1
            else:
                edge_lines = [line.strip() for line in target_edges.split('\n') if line.strip()]
            
                for edge_idx, edge_line in enumerate(edge_lines):
                    edge_info = self.parse_edge_format(edge_line)
                    
                    llm_response = self.generate_regex_for_single_edge(input_log, edge_line, edge_info)

                    if self.show_gpu_usage:
                        self._check_gpu_usage()

                    parsed_regexes = self.parse_llm_regex_response(llm_response)
                    
                    regex_dict = {
                        'Source ID': parsed_regexes['Source ID'],
                        'Destination ID': parsed_regexes['Destination ID'],
                        'A': parsed_regexes['A'],
                        'Timestamp': parsed_regexes['Timestamp']
                    }
                    
                    all_regex_patterns.append(regex_dict)
                    total_edges_processed += 1
            
            total_logs_processed += 1
            
            if total_logs_processed % self.incremental_save_interval == 0:
                self._save_incremental_patterns(all_regex_patterns, rules_dir, 'edges', total_logs_processed)
        
        if total_logs_processed % self.incremental_save_interval != 0:
            self._save_incremental_patterns(all_regex_patterns, rules_dir, 'edges', total_logs_processed)
        
        unique_patterns = []
        for pattern in all_regex_patterns:
            if pattern not in unique_patterns:
                unique_patterns.append(pattern)
        
        master_patterns_file = os.path.join(rules_dir, 'master_patterns_edges.pkl')
        with open(master_patterns_file, 'wb') as f:
            pickle.dump(unique_patterns, f)
        
        
        if self.has_builtin_ids == "no":
            summary = f"""
Processed {total_logs_processed} logs with {total_edges_processed} edges.
Skipped {total_edges_skipped} edges (integer_only or unknown type).

Total unique patterns: {len(unique_patterns)}
Master patterns saved to: {master_patterns_file}

Master structure:
master_patterns_edges.pkl = [
    {{"source": regex, "dest": regex, "action": regex, "timestamp": regex}},
    {{"source": regex, "dest": regex, "action": regex, "timestamp": regex}},
    ...
]
"""
        else:
            summary = f"""
Processed {total_logs_processed} logs with {total_edges_processed} edges.
Skipped {total_edges_skipped} edges (integer_only or unknown type).

Total unique patterns: {len(unique_patterns)}
Master patterns saved to: {master_patterns_file}

Master structure:
master_patterns_edges.pkl = [
    {{"Source ID": regex, "Destination ID": regex, "A": regex, "Timestamp": regex}},
    {{"Source ID": regex, "Destination ID": regex, "A": regex, "Timestamp": regex}},
    ...
]
"""
        
        return summary

    def generate_regex_for_single_edge(self, input_log: str, edge_line: str, edge_info: dict) -> str:
        source_id_regex = self.generate_source_id_regex(input_log, edge_info)
        
        dest_id_regex = self.generate_dest_id_regex(input_log, edge_info)
        
        action_regex = self.generate_action_regex(input_log, edge_info)
        
        timestamp_regex = self.generate_timestamp_regex(input_log, edge_info)
        
        final_response = f"Source ID: {source_id_regex}\nDestination ID: {dest_id_regex}\nA: {action_regex}\nTimestamp: {timestamp_regex}"
        
        return final_response

    def generate_source_id_regex(self, input_log: str, edge_info: dict) -> str:
        prompt = f"""
You are a regex expert. I need to generate a regex pattern that can extract a specific ID or VALUE from a CDM18 JSON log.

INPUT JSON LOG:
{input_log}

TARGET ID/VALUE TO EXTRACT: {edge_info['source']}

TASK: Create a generic regex pattern that extracts this specific specific ID value from the JSON log.

IMPORTANT: Your response must contain ONLY the regex pattern, with NO additional text, NO explanations, NO Incomplete Regex, NO backticks, and NO other formatting.

Rules:
- The regex should extract the exact ID value "{edge_info['source']}" from the log
- If the target ID contains multiple parts (like IP:port), then extract both parts as REGEX 1: <regex_pattern_IP-Address> and REGEX 2: <regex_pattern_Port>
- Extract the actual field names, not generic IP/port or other patterns
- Use capturing groups () to extract the desired values
- For combined values, use lookahead or other techniques to ensure proper extraction
- Do not return an incomplete regex pattern.
- Return ONLY the regex patterns, nothing else

STRICT OUTPUT FORMAT:
    Id/Value:
        - REGEX 1: <complete_regex_pattern_1>
        - REGEX 2: <complete_regex_pattern_2> or NOT APPLICABLE

Do NOT provide any explanations, examples, or additional text. Return ONLY the regex pattern.
"""
        
        response = self.generate_text(prompt, max_length=500, temperature=0.0)
        return response.strip()
    
    def generate_dest_id_regex(self, input_log: str, edge_info: dict) -> str:
        prompt = f"""
You are a regex expert. I need to generate a regex pattern that can extract a specific ID or VALUE from a CDM18 JSON log.

INPUT LOG:
{input_log}

TARGET ID/VALUE TO EXTRACT: {edge_info['destination']}

TASK: Create a generic regex pattern that extracts this specific ID value from the log.

IMPORTANT: Your response must contain ONLY the regex pattern, with NO additional text, NO explanations, NO Incomplete Regex, NO backticks, and NO other formatting.

Rules:
- The regex should extract the exact ID value "{edge_info['destination']}" from the log
- If the target ID contains multiple parts (like IP:port), then extract both parts as REGEX 1: <regex_pattern_IP-Address> and REGEX 2: <regex_pattern_Port>
- Extract the actual field names, not generic IP/port or other patterns
- Use capturing groups () to extract the desired values
- For combined values, use lookahead or other techniques to ensure proper extraction
- Do not return an incomplete regex pattern.
- Return ONLY the regex pattern, nothing else

STRICT OUTPUT FORMAT:
    Id/Value:
        - REGEX 1: <complete_regex_pattern_1>
        - REGEX 2: <complete_regex_pattern_2> or NOT APPLICABLE

Do NOT provide any explanations, examples, or additional text. Return ONLY the correct regex pattern.
"""
        response = self.generate_text(prompt, max_length=500, temperature=0.0)
        return response.strip()
    
    def generate_action_regex(self, input_log: str, edge_info: dict) -> str:
        
        if edge_info['action'] == "NO LABEL":
            return "NO LABEL"
        
        prompt = f"""
You are a regex expert. I need to generate a GENERIC, REUSABLE regex pattern that can extract action types from logs with similar structure.

INPUT LOG:
{input_log}

TARGET ACTION TO EXTRACT: {edge_info['action']}

TASK: Create a GENERIC regex pattern that extracts action types from the SAME POSITION/STRUCTURE in the log, NOT the exact action "{edge_info['action']}".

CRITICAL REQUIREMENTS:
- The regex must be GENERIC and REUSABLE - it should match ANY action type in the same position/structure
- DO NOT include the specific action "{edge_info['action']}" in the regex pattern
- Identify the FIELD NAME or POSITION where the action appears in the log structure
- Use generic patterns like ([A-Za-z ]+) or ([A-Za-z0-9_]+) combined with field/position context (if any)
- The regex should work for ANY log with the same structure, not just this specific log
- Extract the actual field names or position patterns, not the specific action value

IMPORTANT: Your response must contain ONLY the regex pattern, with NO additional text, NO explanations, NO backticks, and NO other formatting.

Rules:
- The regex pattern must NOT contain the specific action "{edge_info['action']}"
- Use capturing groups () to extract the desired value
- Focus on the FIELD/POSITION pattern where actions appear, not the exact action text
- Return ONLY the regex pattern, nothing else

Do NOT provide any explanations, examples, or additional text. Return ONLY the regex pattern.
"""
        
        response = self.generate_text(prompt, max_length=500, temperature=0.0)
        return response.strip()
    
    def generate_timestamp_regex(self, input_log: str, edge_info: dict) -> str:
        
        if edge_info['timestamp'] == "NO TIMESTAMP":
            return "NO TIMESTAMP"
        
        prompt = f"""
You are a regex expert. I need to generate a GENERIC, REUSABLE regex pattern that can extract timestamps from logs with similar structure.

INPUT LOG:
{input_log}

TARGET TIMESTAMP TO EXTRACT: {edge_info['timestamp']}

TASK: Create a GENERIC regex pattern that extracts timestamps from the SAME POSITION/STRUCTURE in the log, NOT the exact timestamp "{edge_info['timestamp']}".

CRITICAL REQUIREMENTS:
- The regex must be GENERIC and REUSABLE - it should match ANY timestamp in the same position/structure
- DO NOT include the specific timestamp "{edge_info['timestamp']}" in the regex pattern
- Identify the FIELD NAME or POSITION where the timestamp appears in the log structure
- Use generic timestamp patterns like (\\d{{4}}-\\d{{2}}-\\d{{2}} \\d{{2}}:\\d{{2}}:\\d{{2}}\\.\\d{{6}}) combined with field/position context (if any)
- The regex should work for ANY log with the same structure, not just this specific log
- Extract the actual field names or position patterns, not the specific timestamp value

IMPORTANT: Your response must contain ONLY the regex pattern, with NO additional text, NO explanations, NO backticks, and NO other formatting.

Rules:
- The regex pattern must NOT contain the specific timestamp "{edge_info['timestamp']}"
- Use capturing groups () to extract the desired value
- Focus on the FIELD/POSITION pattern where timestamps appear, not the exact timestamp value
- Return ONLY the regex pattern, nothing else

Do NOT provide any explanations, examples, or additional text. Return ONLY the regex pattern.
"""
        response = self.generate_text(prompt, max_length=500, temperature=0.0)
        return response.strip()
    
    def generate_ip_regex_no_port(self, input_log: str, target_ip: str) -> str:
        prompt = f"""
You are a regex expert. I need to generate a GENERIC, REUSABLE regex pattern that can extract IP addresses from logs with similar structure.

INPUT LOG:
{input_log}

TARGET IP ADDRESS TO EXTRACT: {target_ip}

TASK: Create a GENERIC regex pattern that extracts IP addresses from the SAME POSITION/STRUCTURE in the log, NOT the exact IP address "{target_ip}".

CRITICAL REQUIREMENTS:
- The regex must be GENERIC and REUSABLE - it should match ANY IP address in the same position/structure
- DO NOT include the specific IP address "{target_ip}" in the regex pattern
- Identify the FIELD NAME or POSITION where the IP address appears in the log structure
- Use a generic IP pattern like (\\d{{1,3}}\\.\\d{{1,3}}\\.\\d{{1,3}}\\.\\d{{1,3}}) combined with the field/position context (if any)
- The regex should work for ANY log with the same structure, not just this specific log
- Extract the actual field names or position patterns, not the specific IP value

IMPORTANT: Your response must contain ONLY the regex pattern, with NO additional text, NO explanations, NO Incomplete Regex, NO backticks, and NO other formatting.

STRICT OUTPUT FORMAT:
    IP Address:
        - REGEX: <generic_regex_pattern_that_matches_any_ip_in_this_position>

Do NOT provide any explanations, examples, or additional text. Return ONLY the regex pattern.
"""
        
        response = self.generate_text(prompt, max_length=500, temperature=0.0)
        return response.strip()
    
    def generate_ename_name_regex_for_entity(self, input_log: str, entity_name: str) -> str:
        ename_info = {'name': entity_name}
        return self.generate_ename_name_regex(input_log, ename_info)
    
    def parse_id_regex_from_response(self, llm_response: str):
        if not llm_response:
            return None
        
        lines = llm_response.strip().split('\n')
        regexes = []
        
        for line in lines:
            line = line.strip()
            if 'REGEX 1:' in line or '- REGEX 1:' in line:
                if '- REGEX 1:' in line:
                    regex = line.split('- REGEX 1:', 1)[1].strip()
                else:
                    regex = line.split('REGEX 1:', 1)[1].strip()
                regex = self.clean_regex_pattern(regex)
                if regex and regex.lower() not in ['not applicable', 'none', 'null']:
                    regexes.append(regex)
            elif 'REGEX 2:' in line or '- REGEX 2:' in line:
                if '- REGEX 2:' in line:
                    regex = line.split('- REGEX 2:', 1)[1].strip()
                else:
                    regex = line.split('REGEX 2:', 1)[1].strip()
                regex = self.clean_regex_pattern(regex)
                if regex and regex.lower() not in ['not applicable', 'none', 'null']:
                    regexes.append(regex)
        
        if len(regexes) == 0:
            return None
        elif len(regexes) == 1:
            return regexes[0]
        else:
            return tuple(regexes)
    
    def parse_single_regex_from_response(self, llm_response: str) -> str:
        if not llm_response:
            return None
        
        lines = llm_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if 'REGEX:' in line or '- REGEX:' in line:
                if '- REGEX:' in line:
                    regex = line.split('- REGEX:', 1)[1].strip()
                elif 'REGEX:' in line:
                    regex = line.split('REGEX:', 1)[1].strip()
                else:
                    continue
                
                regex = self.clean_regex_pattern(regex)
                if regex and regex.lower() not in ['not applicable', 'none', 'null', 'no regex']:
                    return regex
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            if '(' in line and ')' in line:
                cleaned = self.clean_regex_pattern(line)
                if cleaned and len(cleaned) > 3:
                    return cleaned
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                cleaned = self.clean_regex_pattern(line)
                if cleaned:
                    return cleaned
        
        return None
    
    def generate_regex_for_enames(self) -> str:
        if self.cee:
            folder_name = f"{self.cee.lower()}_{self.model_name.lower()}"
            folder_name_sanitized = sanitize_cee_name(folder_name)
            rules_dir = os.path.join(self.save_dir, 'ATLAS', self.embedding, folder_name_sanitized, self.log_type, 'enames')
        else:
            rules_dir = os.path.join(self.save_dir, 'ATLAS', self.embedding, self.log_type, 'enames')
        os.makedirs(rules_dir, exist_ok=True)
        
        all_regex_patterns = []
        
        total_logs_processed = 0
        total_enames_processed = 0
        total_names_replaced_with_none = 0
        
        examples = self.enames_data
        
        for log_idx, example in enumerate(examples):
            input_log = example['input']
            target_enames = example['target']
            
            ename_lines = [line.strip() for line in target_enames.split('\n') if line.strip()]
            
            for ename_idx, ename_line in enumerate(ename_lines):
                ename_info = self.parse_ename_format(ename_line)
                
                if not ename_info:
                    continue
                
                if ename_info['name'] != 'NONE':
                    if ename_info['name'] not in input_log:
                        ename_line = f"{ename_info['id']} = NONE"
                        ename_info['name'] = 'NONE'
                        total_names_replaced_with_none += 1
                
                llm_response = self.generate_regex_for_single_ename(input_log, ename_line, ename_info)

                parsed_regexes = self.parse_llm_ename_response(llm_response)
                
                regex_dict = {
                    'ID': parsed_regexes['ID'],
                    'Name': parsed_regexes['Name']
                }
                
                if regex_dict['ID'] is not None or regex_dict['Name'] != 'No Regex':
                    if self.validate_ename_pattern(regex_dict):
                        all_regex_patterns.append(regex_dict)
                
                total_enames_processed += 1
            
            total_logs_processed += 1
            
            if total_logs_processed % self.incremental_save_interval == 0:
                self._save_incremental_patterns(all_regex_patterns, rules_dir, 'enames', total_logs_processed)
        
        if total_logs_processed % self.incremental_save_interval != 0:
            self._save_incremental_patterns(all_regex_patterns, rules_dir, 'enames', total_logs_processed)
        
        unique_patterns = []
        for pattern in all_regex_patterns:
            if pattern not in unique_patterns:
                unique_patterns.append(pattern)
        
        master_patterns_file = os.path.join(rules_dir, 'master_patterns_enames.pkl')
        with open(master_patterns_file, 'wb') as f:
            pickle.dump(unique_patterns, f)
        
        
        summary = f"""
Processed {total_logs_processed} logs with {total_enames_processed} entity name pairs.
Names replaced with NONE due to validation: {total_names_replaced_with_none}

Total unique patterns: {len(unique_patterns)}
Master patterns saved to: {master_patterns_file}

Master structure:
master_patterns_enames.pkl = [
    {{"ID": (regex1, regex2), "Name": regex}},
    {{"ID": (regex1,), "Name": regex}},
    ...
]
"""
        
        return summary

    def generate_regex_for_vtypes(self) -> str:
        if self.cee:
            folder_name = f"{self.cee.lower()}_{self.model_name.lower()}"
            folder_name_sanitized = sanitize_cee_name(folder_name)
            rules_dir = os.path.join(self.save_dir, 'ATLAS', self.embedding, folder_name_sanitized, self.log_type, 'vtypes')
        else:
            rules_dir = os.path.join(self.save_dir, 'ATLAS', self.embedding, self.log_type, 'vtypes')
        os.makedirs(rules_dir, exist_ok=True)
        
        all_regex_patterns = []
        
        total_logs_processed = 0
        total_vtypes_processed = 0
        
        examples = self.vtypes_data
        
        for log_idx, example in enumerate(examples):
            input_log = example['input']
            target_vtypes = example['target']
            
            vtype_lines = [line.strip() for line in target_vtypes.split('\n') if line.strip()]
            
            for vtype_idx, vtype_line in enumerate(vtype_lines):
                vtype_info = self.parse_vtype_format(vtype_line)
                
                llm_response = self.generate_regex_for_single_vtype(input_log, vtype_line, vtype_info)

                parsed_regexes = self.parse_llm_vtype_response(llm_response)
                
                regex_dict = {
                    'ID': parsed_regexes['ID'],
                    'Type': parsed_regexes['Type']
                }
                
                regex_dict_gen = {
                    'ID': parsed_regexes['ID'],
                    'Type': parsed_regexes['Generalized_Type']
                }
                
                if regex_dict['Type'] != 'No Regex':
                    all_regex_patterns.append(regex_dict)
                
                if regex_dict_gen['Type'] != 'No Regex':
                    all_regex_patterns.append(regex_dict_gen)
                    
                total_vtypes_processed += 1
            
            total_logs_processed += 1
            
            if total_logs_processed % self.incremental_save_interval == 0:
                self._save_incremental_patterns(all_regex_patterns, rules_dir, 'vtypes', total_logs_processed)
        
        if total_logs_processed % self.incremental_save_interval != 0:
            self._save_incremental_patterns(all_regex_patterns, rules_dir, 'vtypes', total_logs_processed)
        
        unique_patterns = []
        for pattern in all_regex_patterns:
            if pattern not in unique_patterns:
                unique_patterns.append(pattern)
        
        master_patterns_file = os.path.join(rules_dir, 'master_patterns_vtypes.pkl')
        with open(master_patterns_file, 'wb') as f:
            pickle.dump(unique_patterns, f)
        
        
        summary = f"""
Processed {total_logs_processed} logs with {total_vtypes_processed} ID-type pairs.

Total unique patterns: {len(unique_patterns)}
Master patterns saved to: {master_patterns_file}

Master structure:
master_patterns_vtypes.pkl = [
    {{"ID": (regex1, regex2), "Type": regex}},
    {{"ID": (regex1,), "Type": regex}},
    ...
]
"""
        
        return summary

    def parse_vtype_format(self, vtype_line: str) -> dict:
        vtype_info = {
            'id': None,
            'type': None
        }
        
        if ' = ' in vtype_line:
            parts = vtype_line.split(' = ', 1)
            vtype_info['id'] = parts[0].strip()
            vtype_info['type'] = parts[1].strip()
        
        return vtype_info
    
    def parse_ename_format(self, ename_line: str) -> dict:
        if " = " not in ename_line:
            return {}
        
        parts = ename_line.split(" = ", 1)
        if len(parts) != 2:
            return {}
        
        entity_id, entity_name = parts
        
        if " List" in entity_id:
            return {}
        
        return {
            'id': entity_id.strip(),
            'name': entity_name.strip()
        }

    def generate_regex_for_single_ename(self, input_log: str, ename_line: str, ename_info: dict) -> str:
        id_regex = self.generate_ename_id_regex(input_log, ename_info)
        
        name_regex = self.generate_ename_name_regex(input_log, ename_info)
        
        final_response = f"ID: {id_regex}\nName: {name_regex}"
        return final_response

    def generate_ename_id_regex(self, input_log: str, ename_info: dict) -> str:
        prompt = f"""
You are a regex expert. I need to generate a regex pattern that can extract a specific ID or VALUE from a CDM18 JSON log.

INPUT JSON LOG:
{input_log}

TARGET ID/VALUE TO EXTRACT: {ename_info['id']}

TASK: Create a generic regex pattern that extracts this specific ID value from the JSON log.

IMPORTANT: Your response must contain ONLY the regex pattern, with NO additional text, NO explanations, NO Incomplete Regex, NO backticks, and NO other formatting.

Rules:
- The regex should extract the exact ID value "{ename_info['id']}" from the log
- If the target ID contains multiple parts (like IP:port), then extract both parts as REGEX 1: <regex_pattern_IP> and REGEX 2: <regex_pattern_Port>
- Extract the actual field names, not generic IP/port or other patterns
- Use capturing groups () to extract the desired values
- For combined values, use lookahead or other techniques to ensure proper extraction
- Never include the specific ID or the IP Address/Port in the regex pattern
- Do not return an incomplete regex pattern.
- Return ONLY the regex patterns, nothing else

STRICT OUTPUT FORMAT:
    Id/Value:
        - REGEX 1: <complete_regex_pattern_1>
        - REGEX 2: <complete_regex_pattern_2> or NOT APPLICABLE

Do NOT provide any explanations, examples, or additional text. Return ONLY the regex pattern.
"""
        
        response = self.generate_text(prompt, max_length=500, temperature=0.0)
        return response.strip()
    
    def generate_ename_name_regex(self, input_log: str, ename_info: dict) -> str:
        if ename_info['name'] == "NONE":
            return "REGEX: No Regex"
        
        prompt = f"""
You are a regex expert. I need to generate a GENERIC, REUSABLE regex pattern that can extract domain names, website names, or file paths from logs with similar structure.

INPUT LOG:
{input_log}

TARGET ENTITY NAME TO EXTRACT: {ename_info['name']}

TASK: Create a GENERIC regex pattern that extracts domain names/websites/filenames from the SAME POSITION/STRUCTURE in the log, NOT the exact entity name "{ename_info['name']}".

CRITICAL REQUIREMENTS:
- The regex must be GENERIC and REUSABLE - it should match ANY domain name/website/filename in the same position/structure
- DO NOT include the specific entity name "{ename_info['name']}" in the regex pattern
- Identify the FIELD NAME or POSITION where the domain/website/filename appears in the log structure
- Use generic patterns like ([a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}) for domains, or ([a-zA-Z0-9._/-]+) for filenames, combined with field/position context
- The regex should work for ANY log with the same structure, not just this specific log
- Extract the actual field names or position patterns, not the specific entity value

THINKING STEP BY STEP:
1. Identify WHERE in the log structure the entity name appears (field name, position, context)
2. Create a regex that matches the STRUCTURE/PATTERN, not the exact value
3. Use generic patterns for domains/filenames combined with field position
4. Ensure the regex is reusable across different logs with similar structure

IMPORTANT: Your response must contain ONLY the regex pattern, with NO additional text, NO explanations, NO backticks, and NO other formatting.

Rules:
- The regex pattern must NOT contain the specific entity name "{ename_info['name']}"
- Use capturing groups () to extract the value
- Focus on the FIELD/POSITION pattern (if any), not the exact value
- If NO suitable pattern exists, return exactly: No Regex
- Return ONLY the regex pattern, nothing else

STRICT OUTPUT FORMAT:
    Name:
        - REGEX: <generic_regex_pattern_that_matches_any_domain_or_filename_in_this_position> or No Regex

Do NOT provide any explanations, examples, or additional text. Return ONLY the regex pattern.
"""
        
        response = self.generate_text(prompt, max_length=500, temperature=0.0)
        return response.strip()

    def generate_regex_for_single_vtype(self, input_log: str, vtype_line: str, vtype_info: dict) -> str:
        
        id_regex = self.generate_id_regex(input_log, vtype_info)
        
        type_regex = self.generate_type_regex(input_log, vtype_info, vtype_info['id'])
        
        generalized_type_regex = "No Regex"
        if type_regex != "No Regex" and type_regex.strip():
            specific_pattern = type_regex.replace("Type:", "").replace("- REGEX:", "").replace("REGEX:", "").strip()
            if specific_pattern and specific_pattern != "No Regex" and specific_pattern.strip():
                generalized_response = self.generate_generalized_type_regex(input_log, specific_pattern)
                if "Type:" in generalized_response and "- REGEX:" in generalized_response:
                    generalized_type_regex = generalized_response.replace("Type:", "").replace("- REGEX:", "").strip()
                elif "REGEX:" in generalized_response:
                    generalized_type_regex = generalized_response.replace("Type:", "").replace("REGEX:", "").strip()
                else:
                    generalized_type_regex = generalized_response.replace("Type:", "").strip()
        
        final_response = f"ID: {id_regex}\nType: {type_regex}\nGeneralized_Type: {generalized_type_regex}"
        return final_response
    
    def generate_id_regex(self, input_log: str, vtype_info: dict) -> str:
        prompt = f"""
You are a regex expert. I need to generate a regex pattern that can extract a specific ID or VALUE from a CDM18 JSON log.

INPUT JSON LOG:
{input_log}

TARGET ID/VALUE TO EXTRACT: {vtype_info['id']}

TASK: Create a generic regex pattern that extracts this specific ID value from the JSON log.

IMPORTANT: Your response must contain ONLY the regex pattern, with NO additional text, NO explanations, NO Incomplete Regex, NO backticks, and NO other formatting.

Rules:
- The regex should extract the exact ID value "{vtype_info['id']}" from the log
- If the target ID contains multiple parts (like IP:port), then extract both parts as REGEX 1: <regex_pattern_IP> and REGEX 2: <regex_pattern_Port>
- Extract the actual field names, not generic IP/port or other patterns
- Use capturing groups () to extract the desired values
- For combined values, use lookahead or other techniques to ensure proper extraction
- Never include the specific ID or the IP Address/Port in the regex pattern
- Do not return an incomplete regex pattern.
- Return ONLY the regex patterns, nothing else

STRICT OUTPUT FORMAT:
    Id/Value:
        - REGEX 1: <complete_regex_pattern_1>
        - REGEX 2: <complete_regex_pattern_2> or NOT APPLICABLE

Do NOT provide any explanations, examples, or additional text. Return ONLY the regex pattern.
"""
        
        response = self.generate_text(prompt, max_length=500, temperature=0.0)
        return response.strip()
    
    def generate_type_regex(self, input_log: str, vtype_info: dict, target_id: str = None) -> str:
        step1_response = self.generate_type_check_regex(input_log, vtype_info)

        if "NOT FOUND" not in step1_response.strip():
            return step1_response.strip()

        if target_id:
            step2_response = self.generate_field_name_regex(input_log, vtype_info, target_id)
            return step2_response.strip()
        else:
            return "No Regex"

    def generate_type_check_regex(self, input_log: str, vtype_info: dict) -> str:
        word_found = self.find_exact_word_in_log(input_log, vtype_info['type'])
        
        if not word_found:
            return "NOT FOUND"
        
        regex_pattern = self.generate_word_regex_pattern(input_log, vtype_info['type'])
        
        return regex_pattern
    
    def find_exact_word_in_log(self, input_log: str, target_word: str) -> bool:
        
        prompt = f"""
You are a word finder. Check if a specific word exists in this JSON log.

LOG: {input_log}

TARGET WORD TO FIND: {target_word}

TASK: 
1. Search for the exact word "{target_word}" in the log
2. If found: return exactly "FOUND"
3. If not found: return exactly "NOT FOUND"

IMPORTANT: Your response must contain ONLY "FOUND" or "NOT FOUND", with NO additional text, NO explanations, NO backticks, and NO other formatting.

VERY STRICT RULES:
- Look for the word "{target_word}" (case sensitive)
- Pay attention to the EXACT formatting of the word in the LOG (spacing, punctuation, uppercase, lowercase, etc.)
- Return the word exactly as it appears in the JSON structure
- Return only "FOUND" or "NOT FOUND", nothing else

Return ONLY "FOUND" or "NOT FOUND".
"""
        
        response = self.generate_text(prompt, max_length=50, temperature=0.0)
        return response.strip() == "FOUND"
    
    def generate_word_regex_pattern(self, input_log: str, target_word: str) -> str:
        target_word_formatted = self.analyze_word_formatting(input_log, target_word)
        
        if not target_word_formatted:
            return "No Regex"
        
        regex_pattern = self.generate_formatted_word_regex(target_word_formatted)
        
        return regex_pattern
    
    def analyze_word_formatting(self, input_log: str, target_word: str) -> str:
        occurrences = self.find_word_occurrences(input_log, target_word)
        
        if not occurrences:
            return None
        
        extracted_word = self.extract_subword_portion(occurrences, target_word)
        
        return extracted_word
    
    def find_word_occurrences(self, input_log: str, target_word: str) -> str:
        
        prompt = f"""
You are a word occurrence finder. Find all occurrences of a target word in this JSON log.

LOG: {input_log}

TARGET WORD: {target_word}

TASK: Find all words in the log that contain the target word "{target_word}" (case insensitive).

SEARCH RULES:
- Look for "{target_word}" as part of larger words (case insensitive)
- Include words where "{target_word}" appears at the beginning, middle, or end
- Return ALL words that contain the target word, separated by commas
- If no occurrences found, return "NOT FOUND"

IMPORTANT: Your response must contain ONLY the list of words or "NOT FOUND", with NO additional text, NO explanations, NO backticks.

Return ONLY the comma-separated list of words or "NOT FOUND".
"""
        
        response = self.generate_text(prompt, max_length=200, temperature=0.0)
        result = response.strip()
        return result if result != "NOT FOUND" else None
    
    def extract_subword_portion(self, occurrences: str, target_word: str) -> str:
        
        prompt = f"""
You are a sub-word extractor. Extract the target word portion from a list of containing words.

CONTAINING WORDS: {occurrences}

TARGET WORD: {target_word}

TASK: From the first word in the list, extract ONLY the portion that matches the target word "{target_word}".

EXTRACTION RULES:
- Take the first word from the containing words list
- Find the portion that matches "{target_word}" (case insensitive matching)
- Extract ONLY that matching portion with its exact formatting (case, spacing, punctuation)
- Do NOT include any surrounding characters, numbers, or other text
- Preserve the exact case as it appears in the containing word

IMPORTANT: Your response must contain ONLY the extracted sub-word portion, with NO additional text, NO explanations, NO backticks.

CRITICAL: Extract only the matching portion, not the entire containing word.

Return ONLY the extracted sub-word portion.
"""
        
        response = self.generate_text(prompt, max_length=100, temperature=0.0)
        return response.strip()
    
    def generate_formatted_word_regex(self, target_word_formatted: str) -> str:
        
        prompt = f"""
You are a regex expert. Generate a regex pattern to extract a specific word.

TARGET WORD TO EXTRACT: {target_word_formatted}

TASK: Create a simple regex pattern that extracts ONLY the word "{target_word_formatted}" from any log, using capturing groups ()

IMPORTANT: Your response must contain ONLY the regex pattern, with NO additional text, NO explanations, NO backticks, and NO other formatting.

VERY STRICT RULES:
- Use capturing groups () to extract just the word "{target_word_formatted}", nothing else
- Do NOT include word boundaries or other patterns that might extract additional text
- The regex pattern should be simple and focused only on extracting the target word "{target_word_formatted}"
- Never include the action type, IDs, values, or other parts of the log in the regex pattern
- Return only the regex pattern, nothing else

STRICT OUTPUT FORMAT:
Type:
    - REGEX: <simple_regex_pattern_to_extract_only_the_word>

Return ONLY the regex pattern.
"""
        
        response = self.generate_text(prompt, max_length=200, temperature=0.0)
        return response.strip()

    def generate_field_name_regex(self, input_log: str, vtype_info: dict, target_id: str) -> str:
        field_name = self.find_field_name_containing_id(input_log, target_id, vtype_info['type'])
        
        if not field_name:
            return "No Regex"
        
        regex_pattern = self.generate_field_name_regex_pattern(field_name)
        
        return regex_pattern
    
    def find_field_name_containing_id(self, input_log: str, target_id: str, target_type: str) -> str:
        
        prompt = f"""
You are a field name finder. Find the field name in this JSON log.

LOG: {input_log}

TARGET ID TO FIND: {target_id}

TARGET TYPE WORD: {target_type}

TASK: 
1. First, try to find the field name that contains the ID "{target_id}"
2. If found: return the EXACT field name as it appears in the log
3. If not found: find the single field name that closely matches the target type "{target_type}"
4. If neither found: return exactly "NOT FOUND"

IMPORTANT: Your response must contain ONLY the field name or "NOT FOUND", with NO additional text, NO explanations, NO backticks, and NO other formatting.

STRICT RULES:
- Look for field names in the JSON structure (keys, property names)
- Pay attention to the EXACT formatting of the field name in the LOG (spacing, punctuation, uppercase, lowercase, etc.)
- Return the field name exactly as it appears in the JSON structure
- Return only the field name or "NOT FOUND", nothing else

Return ONLY the field name or "NOT FOUND".
"""
        
        response = self.generate_text(prompt, max_length=100, temperature=0.0)
        result = response.strip()
        return result if result != "NOT FOUND" else None
    
    def generate_field_name_regex_pattern(self, field_name: str) -> str:
        
        prompt = f"""
You are a regex expert. Generate a simple regex pattern to extract a field name.

FIELD NAME TO EXTRACT: {field_name}

TASK: Create a simple regex pattern that extracts ONLY the field name "{field_name}" from any log, using capturing groups ()

IMPORTANT: Your response must contain ONLY the regex pattern, with NO additional text, NO explanations, NO backticks, and NO other formatting.

VERY STRICT RULES:
- Use capturing groups () to extract just the field name "{field_name}", nothing else
- Pay attention to the EXACT formatting of the field name in the LOG (spacing, punctuation, uppercase, lowercase, etc.)
- Do NOT include word boundaries or other patterns that might extract additional text
- The regex pattern should be simple and focused only on extracting the field name "{field_name}"
- The regex should be generic, so, NEVER include the action type or IDs in the regex pattern
- Never include IDs, values, or other parts of the log in the regex pattern
- Return only the regex pattern, nothing else

STRICT OUTPUT FORMAT:
Type:
    - REGEX: <simple_regex_pattern_to_extract_only_the_field_name>

Return ONLY the regex pattern.
"""
        
        response = self.generate_text(prompt, max_length=200, temperature=0.0)
        return response.strip()
    
    def generate_generalized_type_regex(self, input_log: str, specific_type_regex: str) -> str:
        log_location = self.find_type_location_in_log(input_log, specific_type_regex)
        log_location = log_location.strip(':{}')
        
        if not log_location or log_location == "NOT FOUND":
            return "No Regex"
        
        generalized_pattern = self.create_generalized_pattern_for_location(log_location, specific_type_regex)
        
        verified_pattern = self.verify_and_fix_generalized_regex(input_log, log_location, generalized_pattern, specific_type_regex)
        
        return verified_pattern
    
    def verify_and_fix_generalized_regex(self, input_log: str, log_location: str, generated_regex: str, specific_type_regex: str) -> str:
        
        specific_word = specific_type_regex.replace('(', '').replace(')', '').strip()
        regex_pattern = generated_regex.replace("Type:", "").replace("- REGEX:", "").replace("REGEX:", "").strip()
        
        is_correct = self.check_regex_correctness(input_log, log_location, regex_pattern, specific_word)
        
        if "INCORRECT" in is_correct.upper():
            reason = "No specific reason provided"
            if "- Reason:" in is_correct:
                reason = is_correct.split("- Reason:", 1)[1].strip()
            
            fixed_pattern = self.fix_incorrect_regex(log_location, regex_pattern, specific_word, reason)
            return fixed_pattern
        else:
            return generated_regex
    
    def fix_incorrect_regex(self, log_location: str, regex_pattern: str, specific_word: str, reason: str = "") -> str:
        
        prompt = f"""
You are a regex fixer. Fix an incorrect regex pattern to properly match a log fragment.

LOG FRAGMENT: {log_location}

INCORRECT REGEX: {regex_pattern}

WORD TO GENERALIZE: {specific_word}

REASON FOR INCORRECTNESS: {reason}

TASK: Create a corrected regex pattern that properly matches the log fragment.

FIX RULES:
- Match the exact structure of the log fragment
- Replace only the word "{specific_word}" with [A-Za-z]+ not with patterns that capture the entire log structure (.+)
- Use proper regex escaping for special characters
- Include capturing groups () around the generalized part
- Don't give unnecessary braces or other patterns in the regex pattern

IMPORTANT: Your response must contain ONLY the corrected regex pattern, with NO additional text, NO explanations, NO backticks.

STRICT OUTPUT FORMAT:
Type:
    - REGEX: <corrected_regex_pattern>

Return ONLY the corrected regex pattern.
"""
        
        response = self.generate_text(prompt, max_length=200, temperature=0.0)
        return response.strip()
    
    def check_regex_correctness(self, input_log: str, log_location: str, regex_pattern: str, specific_word: str) -> str:
        
        prompt = f"""
You are a regex checker. Check if a regex pattern correctly matches a log fragment.

FULL LOG:
{input_log}

LOG FRAGMENT: {log_location}

REGEX PATTERN: {regex_pattern}

WORD BEING GENERALIZED: {specific_word}

TASK: Analyse the regex pattern step by step and check if the regex pattern would correctly extract ONLY the specific word from the FULL LOG or not.

IMPORTANT: Your response must contain ONLY "CORRECT" or "INCORRECT", along with the reason for the correctness or incorrectness, with NO additional text.

Output format:
    - Outcome: CORRECT or INCORRECT
    - Reason: <reason_for_correctness_or_incorrectness>

Return ONLY the output format, with NO additional text.
"""
        
        response = self.generate_text(prompt, max_length=50, temperature=0.0)
        return response.strip()
    
    def find_type_location_in_log(self, input_log: str, specific_type_regex: str) -> str:
        specific_word = specific_type_regex.replace('(', '').replace(')', '').strip()
        
        prompt = f"""
You are a pattern finder. Find where a specific word appears in a log.

LOG:
{input_log}

SPECIFIC WORD TO FIND: {specific_word}

TASK: Find the exact location in the log where this word appears and return ONLY that specific field.

SEARCH RULES:
- Look for the word "{specific_word}" anywhere in the log structure
- Return the smallest log fragment that contains this word, without putting unnecessary patterns like {{}} or patterns like {{...}} or other patterns in the output
- Include the field name and the value containing the word
- Do NOT return the entire log structure, or unnecessary extentions of the fragment.
- Return only the specific part where the word appears

IMPORTANT: Your response must contain ONLY the log fragment, with NO additional text, NO explanations, NO backticks.

Return ONLY the log fragment containing the word, or "NOT FOUND" if not present.
"""
        
        response = self.generate_text(prompt, max_length=200, temperature=0.0)
        return response.strip()
    
    def create_generalized_pattern_for_location(self, log_location: str, specific_type_regex: str) -> str:
        specific_word = specific_type_regex.replace('(', '').replace(')', '').strip()
        
        prompt = f"""
You are a regex pattern creator. Create a generic regex pattern to extract from a specific log fragment.

LOG FRAGMENT: {log_location}

SPECIFIC WORD TO GENERALIZE: {specific_word}

TASK: Create a generic regex pattern that matches this log fragment but generalizes the specific word to match any similar word.

STRICT RULES:
- DO NOT create patterns that return entire log structure like .+
- Keep ALL structural elements exactly as they are (quotes, dots, colons, brackets, etc.)
- Replace ONLY the word "{specific_word}" ONLY with [A-Za-z]+ to match any alphabetic word, not with patterns that capture the entire log structure (.+)
- Escape special regex characters in the structural elements (dots become \\., etc.)
- Use capturing groups () around the generalized part


IMPORTANT: Your response must contain ONLY the regex pattern, with NO additional text, NO explanations, NO backticks.

STRICT OUTPUT FORMAT:
Type:
    - REGEX: <regex_pattern_for_this_specific_location>

Return ONLY the regex pattern.
"""
        
        response = self.generate_text(prompt, max_length=200, temperature=0.0)
        return response.strip()

    def parse_llm_regex_response(self, llm_response: str) -> dict:
        result = {
            'Source ID': None,
            'Destination ID': None,
            'A': None,
            'Timestamp': None
        }
        
        lines = llm_response.strip().split('\n')
        source_id_regexes = []
        dest_id_regexes = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('Source ID:'):
                j = i + 1
                while j < len(lines) and (lines[j].strip().startswith('REGEX 1:') or lines[j].strip().startswith('REGEX 2:') or lines[j].strip().startswith('- REGEX 1:') or lines[j].strip().startswith('- REGEX 2:')):
                    regex_line = lines[j].strip()
                    if regex_line.startswith('REGEX 1:') or regex_line.startswith('- REGEX 1:'):
                        regex1 = regex_line.replace('REGEX 1:', '').replace('- REGEX 1:', '').strip()
                        regex1 = self.clean_regex_pattern(regex1)
                        if regex1 and regex1.lower() != 'not applicable':
                            source_id_regexes.append(regex1)
                    elif regex_line.startswith('REGEX 2:') or regex_line.startswith('- REGEX 2:'):
                        regex2 = regex_line.replace('REGEX 2:', '').replace('- REGEX 2:', '').strip()
                        regex2 = self.clean_regex_pattern(regex2)
                        if regex2 and regex2.lower() != 'not applicable':
                            source_id_regexes.append(regex2)
                    j += 1
                i = j - 1
                
            elif line.startswith('Destination ID:'):
                j = i + 1
                while j < len(lines) and (lines[j].strip().startswith('REGEX 1:') or lines[j].strip().startswith('REGEX 2:') or lines[j].strip().startswith('- REGEX 1:') or lines[j].strip().startswith('- REGEX 2:')):
                    regex_line = lines[j].strip()
                    if regex_line.startswith('REGEX 1:') or regex_line.startswith('- REGEX 1:'):
                        regex1 = regex_line.replace('REGEX 1:', '').replace('- REGEX 1:', '').strip()
                        regex1 = self.clean_regex_pattern(regex1)
                        if regex1 and regex1.lower() != 'not applicable':
                            dest_id_regexes.append(regex1)
                    elif regex_line.startswith('REGEX 2:') or regex_line.startswith('- REGEX 2:'):
                        regex2 = regex_line.replace('REGEX 2:', '').replace('- REGEX 2:', '').strip()
                        regex2 = self.clean_regex_pattern(regex2)
                        if regex2 and regex2.lower() != 'not applicable':
                            dest_id_regexes.append(regex2)
                    j += 1
                i = j - 1
                
            elif line.startswith('A:'):
                result['A'] = line.replace('A:', '').strip()
            elif line.startswith('Timestamp:'):
                result['Timestamp'] = line.replace('Timestamp:', '').strip()
            
            i += 1
        
        if source_id_regexes:
            result['Source ID'] = tuple(source_id_regexes)
        else:
            result['Source ID'] = None
        
        if dest_id_regexes:
            result['Destination ID'] = tuple(dest_id_regexes)
        else:
            result['Destination ID'] = None
        
        return result
    
    def clean_regex_pattern(self, regex_pattern: str) -> str:
        if not regex_pattern:
            return regex_pattern
        
        cleaned = regex_pattern.strip()
        
        import re
        cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
        
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1]
        
        cleaned = cleaned.strip('`')
        
        cleaned = cleaned.replace('｛', '{').replace('｝', '}')
        
        cleaned = re.sub(r'\\+', r'\\', cleaned)
        
        standard_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !@#$%^&*()-_=+[]{}\\|;:'\",.<>/?`~")
        non_standard = [c for c in cleaned if c not in standard_chars]
        if non_standard:
            for c in non_standard:
                if c in '｛｝':
                    cleaned = cleaned.replace('｛', '{').replace('｝', '}')
                else:
                    cleaned = cleaned.replace(c, '')
        
        return cleaned.strip()
    
    def parse_llm_ename_response(self, llm_response: str) -> dict:
        result = {
            'ID': None,
            'Name': 'No Regex'
        }
        
        if not llm_response or "No Regex" in llm_response:
            return result
        
        try:
            id_regex_patterns = []
            
            id_section_match = re.search(r'ID:.*?(?=Name:|$)', llm_response, re.DOTALL)
            if id_section_match:
                id_section = id_section_match.group(0)
                
                for i in range(1, 10):
                    pattern_match = re.search(rf'REGEX\s+{i}:\s*([^\n]+)', id_section)
                    if pattern_match:
                        pattern = pattern_match.group(1).strip()
                        pattern = self.clean_regex_pattern(pattern)
                        if pattern and pattern.lower() not in ['not applicable', 'none', 'null']:
                            id_regex_patterns.append(pattern)
                    else:
                        break
                
                if not id_regex_patterns:
                    single_regex_match = re.search(r'REGEX:\s*([^\n]+)', id_section)
                    if single_regex_match:
                        pattern = single_regex_match.group(1).strip()
                        pattern = self.clean_regex_pattern(pattern)
                        if pattern and pattern.lower() not in ['not applicable', 'none', 'null']:
                            id_regex_patterns.append(pattern)
            
            if id_regex_patterns:
                if len(id_regex_patterns) == 1:
                    result['ID'] = id_regex_patterns[0]
                else:
                    result['ID'] = tuple(id_regex_patterns)
            
            name_section_match = re.search(r'Name:.*?$', llm_response, re.DOTALL)
            if name_section_match:
                name_section = name_section_match.group(0)
                
                name_regex_match = re.search(r'REGEX:\s*([^\n]+)', name_section)
                if name_regex_match:
                    name_regex = name_regex_match.group(1).strip()
                    name_regex = self.clean_regex_pattern(name_regex)
                    if name_regex and name_regex.lower() not in ['not applicable', 'none', 'null', 'no regex']:
                        result['Name'] = name_regex
            
            if isinstance(result['ID'], str) and ('｛' in result['ID'] or '｝' in result['ID']):
                result['ID'] = result['ID'].replace('｛', '{').replace('｝', '}')
            
            if '｛' in result['Name'] or '｝' in result['Name']:
                result['Name'] = result['Name'].replace('｛', '{').replace('｝', '}')

            if isinstance(result['ID'], str) and " List" in result['ID']:
                result['ID'] = None
                result['Name'] = 'No Regex'
            
        except Exception as e:
            pass
        
        return result
    
    def is_valid_regex_pattern(self, pattern: str) -> bool:
        if not pattern or pattern.lower() in ['not applicable', 'none', 'null', 'no regex']:
            return False
        
        has_capturing_groups = '(' in pattern and ')' in pattern
        has_regex_chars = any(char in pattern for char in ['[', ']', '{', '}', '\\', '+', '*', '?', '^', '$', '.'])
        
        if not (has_capturing_groups or has_regex_chars):
            return False
        
        if len(pattern) < 3:
            return False
        
        pattern_stripped = pattern.strip()
        if pattern_stripped.startswith('Name:') or pattern_stripped.startswith('ID:'):
            return False
        if pattern_stripped.startswith('REGEX:'):
            return False
        
        if pattern.count('REGEX:') > 1:
            return False
        
        return True
    
    def parse_llm_vtype_response(self, llm_response: str) -> dict:
        result = {
            'ID': None,
            'Type': 'No Regex',  # Default fallback
            'Generalized_Type': 'No Regex'  # Default fallback for generalized pattern
        }
        
        lines = llm_response.strip().split('\n')
        id_regexes = []
        type_regex = None
        generalized_type_regex = None
        
        in_id_section = False
        for line in lines:
            line = line.strip()
            
            if line.startswith('ID:'):
                in_id_section = True
                continue
            elif line.startswith('Type:'):
                in_id_section = False
                type_part = line.replace('Type:', '').strip()
                if type_part.startswith('- REGEX:'):
                    type_regex = type_part.replace('- REGEX:', '').strip()
                elif type_part.startswith('REGEX:'):
                    type_regex = type_part.replace('REGEX:', '').strip()
                else:
                    type_regex = type_part
                continue
            elif line.startswith('Generalized_Type:'):
                in_id_section = False
                gen_type_part = line.replace('Generalized_Type:', '').strip()
                if gen_type_part.startswith('- REGEX:'):
                    generalized_type_regex = gen_type_part.replace('- REGEX:', '').strip()
                elif gen_type_part.startswith('REGEX:'):
                    generalized_type_regex = gen_type_part.replace('REGEX:', '').strip()
                else:
                    generalized_type_regex = gen_type_part
                continue
            
            if in_id_section:
                if line.startswith('- REGEX 1:'):
                    regex1 = line.replace('- REGEX 1:', '').strip()
                    regex1 = self.clean_regex_pattern(regex1)
                    if regex1 and regex1.lower() != 'not applicable':
                        id_regexes.append(regex1)
                elif line.startswith('- REGEX 2:'):
                    regex2 = line.replace('- REGEX 2:', '').strip()
                    regex2 = self.clean_regex_pattern(regex2)
                    if regex2 and regex2.lower() != 'not applicable':
                        id_regexes.append(regex2)
                elif line.startswith('REGEX 1:'):
                    regex1 = line.replace('REGEX 1:', '').strip()
                    if regex1 and regex1.lower() != 'not applicable':
                        id_regexes.append(regex1)
                elif line.startswith('REGEX 2:'):
                    regex2 = line.replace('REGEX 2:', '').strip()
                    if regex2 and regex2.lower() != 'not applicable':
                        id_regexes.append(regex2)
                elif line.startswith('Id/Value:'):
                    continue
                elif line and not line.startswith('-') and not line.startswith('REGEX'):
                    if line and line.lower() not in ['not applicable', 'none', 'null']:
                        id_regexes.append(line)
        
        if id_regexes:
            result['ID'] = tuple(id_regexes)
        else:
            result['ID'] = None
        
        if type_regex and type_regex.lower() not in ['none', 'null', 'no regex', '']:
            result['Type'] = type_regex
        else:
            result['Type'] = 'No Regex'
        
        if generalized_type_regex and generalized_type_regex.lower() not in ['none', 'null', 'no regex', '']:
            result['Generalized_Type'] = generalized_type_regex
        else:
            result['Generalized_Type'] = 'No Regex'
        
        return result

    def run_full_generation(self, file_type: str = "edges"):
        if file_type == "vtypes" and self.has_builtin_ids == "no":
            return "Skipped: vtypes not applicable for this log type"
        
        if file_type == "edges":
            result = self.generate_regex_for_edges()
        elif file_type == "enames":
            result = self.generate_regex_for_enames()
        elif file_type == "vtypes":
            result = self.generate_regex_for_vtypes()
        else:
            raise ValueError(f"Unknown file type: {file_type}")
        
        return result

    def validate_ename_pattern(self, pattern_dict: dict) -> bool:
        if not isinstance(pattern_dict, dict):
            return False
        
        if 'ID' not in pattern_dict or 'Name' not in pattern_dict:
            return False
        
        id_field = pattern_dict['ID']
        if id_field is not None:
            if isinstance(id_field, tuple):
                if len(id_field) == 0:
                    return False
                for part in id_field:
                    if not self.is_valid_regex_pattern(part):
                        return False
            elif isinstance(id_field, str):
                if not self.is_valid_regex_pattern(id_field):
                    return False
            else:
                return False
        
        name_field = pattern_dict['Name']
        if name_field != 'No Regex':
            if not isinstance(name_field, str) or not self.is_valid_regex_pattern(name_field):
                return False
        
        if id_field is None and name_field == 'No Regex':
            return False
        
        return True

    def _save_incremental_patterns(self, all_patterns: List[dict], rules_dir: str, pattern_type: str, logs_processed: int):
        try:
            unique_patterns = []
            for pattern in all_patterns:
                if pattern not in unique_patterns:
                    unique_patterns.append(pattern)
            
            incremental_filename = f'master_patterns_{pattern_type}_incremental_{logs_processed}logs.pkl'
            incremental_filepath = os.path.join(rules_dir, incremental_filename)
            
            with open(incremental_filepath, 'wb') as f:
                pickle.dump(unique_patterns, f)
            
            summary_filename = f'master_patterns_{pattern_type}_incremental_{logs_processed}logs_summary.txt'
            summary_filepath = os.path.join(rules_dir, summary_filename)
            
            with open(summary_filepath, 'w') as f:
                f.write(f"Incremental Patterns Summary - {pattern_type.upper()}\n")
                f.write(f"Logs Processed: {logs_processed}\n")
                f.write(f"Total Patterns: {len(unique_patterns)}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                for i, pattern in enumerate(unique_patterns, 1):
                    f.write(f"Pattern {i}:\n")
                    f.write(f"  {pattern}\n\n")
            
            
        except Exception as e:
            pass


def sanitize_cee_name(cee: str) -> str:
    sanitized = cee.replace(':', '_')
    sanitized = sanitized.replace('/', '_')
    return sanitized


def load_log_characteristics(log_characteristics_path: str = None, cee: str = None) -> dict:
    if log_characteristics_path is None:
        script_dir = Path(__file__).parent
        if cee:
            log_characteristics_path = script_dir / f"../../candidate_edge_extractor/ATLAS/ablation/{cee}/log_idcharacteristics.json"
        else:
            log_characteristics_path = script_dir / "../../candidate_edge_extractor/ATLAS/log_idcharacteristics.json"
        log_characteristics_path = log_characteristics_path.resolve()
    
    if not os.path.exists(log_characteristics_path):
        return {}
    
    with open(log_characteristics_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate regex")
    parser.add_argument("--cee", type=str, required=True,
                        help="Candidate Edge Extractor name (e.g., gpt-3.5-turbo, llama3:70b)")
    parser.add_argument("--model-name", default="llama3:70b", 
                        help="Name of the Ollama model to use")
    parser.add_argument("--file-type", default="edges", choices=["edges", "enames", "vtypes"],
                        help="Type of data to process")
    parser.add_argument("--ollama-url", required=True,
                        help="Ollama server URL (required)")
    parser.add_argument("--gpu-cards", default=None, 
                        help="Comma-separated list of GPU card numbers to use (default: all available GPUs)")
    parser.add_argument("--show-gpu-usage", action="store_true", 
                        help="Show detailed GPU usage information during processing")
    parser.add_argument("--auto-restart-ollama", action="store_true", default=True,
                        help="Automatically restart Ollama with GPU restrictions (default: True)")
    parser.add_argument("--no-auto-restart-ollama", action="store_true",
                        help="Don't automatically restart Ollama (manual restart required)")
    parser.add_argument("--incremental-save-interval", type=int, default=10,
                        help="Save incremental patterns every N logs (default: 10)")

    parser.add_argument("--embedding", type=str, required=True, help="Name of the embedding (e.g., roberta)")
    parser.add_argument("--log-type", type=str, default="audit", help="Log type (e.g., audit, dns, firefox)")
    parser.add_argument("--candidates-dir", type=str, default="../../../clusterlogs_atlas/candidates-atlas_ablation", 
                        help="Directory containing candidate data (default: ../../../clusterlogs_atlas/candidates-atlas_ablation)")
    parser.add_argument("--save-dir", type=str, default="./rules", 
                        help="Directory to save generated patterns")
    parser.add_argument("--log-characteristics", type=str, default=None,
                        help="Path to log_idcharacteristics.json (default: auto-detect from ablation/{cee}/)")
    
    args = parser.parse_args()
    
    log_characteristics = load_log_characteristics(args.log_characteristics, cee=args.cee)
    has_builtin_ids = log_characteristics.get(args.log_type, "unknown")
    
    if has_builtin_ids == "no" and args.file_type == "vtypes":
        return
    
    auto_restart = args.auto_restart_ollama and not args.no_auto_restart_ollama
    
    kwargs = {
        "ollama_url": args.ollama_url,
        "gpu_cards": args.gpu_cards,
        "show_gpu_usage": args.show_gpu_usage,
        "auto_restart_ollama": auto_restart,
        "incremental_save_interval": args.incremental_save_interval,
        "save_dir": args.save_dir,
        "has_builtin_ids": has_builtin_ids,
        "cee": args.cee
    }
    
    generator = None
    try:
        generator = LLMRegexGenerator(
            model_type="ollama",
            model_name=args.model_name,
            dataset="ATLAS",
            embedding=args.embedding,
            log_type=args.log_type,
            **kwargs
        )
        
        generator.load_training_data(args.file_type, args.log_type, args.embedding, args.candidates_dir, cee=args.cee)
        
        result = generator.run_full_generation(args.file_type)
        
    except Exception as e:
        pass
    
    finally:
        if generator is not None:
            generator.cleanup_model()

if __name__ == "__main__":
    main()
