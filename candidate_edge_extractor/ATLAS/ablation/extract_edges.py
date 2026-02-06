#!/usr/bin/env python3
import os
import json
import yaml
import pickle
import argparse
import re
import random
from tqdm import tqdm
from datetime import datetime

import llm_utils
import llm_utils2

from consistency_utils import (
    get_consistent_vanetype,
    extract_consistent_graph,
    filter_process_dict,
    extract_final_output,
    extract_final_desc,
    extract_final_pairs,
    extract_last_entity_types_block
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract provenance graph edges from ATLAS candidates")
    parser.add_argument("--llm_name", type=str, default="gpt-3.5-turbo", help="LLM model to use (default: gpt-3.5-turbo)")
    parser.add_argument("--model_type", type=str, default="openai", choices=["openai", "ollama"],
                       help="Model backend type: 'openai' or 'ollama' (default: openai)")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config.yaml file containing OpenAI API key (required)")
    parser.add_argument("--ollama_url", type=str, default=None,
                       help="Ollama server URL (required when model_type is ollama)")
    parser.add_argument("--embedding", type=str, default="roberta", help="Embedding type (default: roberta)")
    parser.add_argument("--log_types", nargs='+', default=["audit", "dns", "firefox"], 
                       help="Log types to process (default: audit dns firefox)")
    parser.add_argument("--llm_iterations", type=int, default=7, 
                       help="Number of LLM iterations for self-consistency (default: 7)")
    parser.add_argument("--candidates_dir", type=str, 
                       default="../../../clusterlogs_atlas/candidates-atlas",
                       help="Path to candidates-atlas directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for results")
    parser.add_argument("--continual_learning", action="store_true",
                       help="Enable continual learning (use previous outputs as examples)")
    parser.add_argument("--start_examples_after", type=int, default=5,
                       help="Start using examples after N candidates (default: 5)")
    parser.add_argument("--num_examples", type=int, default=10,
                       help="Number of examples to use (default: 10)")
    parser.add_argument("--start_from", type=int, default=0,
                       help="Start from candidate index (for resuming)")
    parser.add_argument("--log_characteristics", type=str, default=None,
                       help="Path to log ID characteristics JSON file (default: ablation/{llm_name}/log_idcharacteristics.json)")
    return parser.parse_args()


def load_candidates(candidates_dir, embedding, log_type):
    candidate_log_path = os.path.join(candidates_dir, embedding, log_type, "candidate.pkl")
    candidate_ids_path = os.path.join(candidates_dir, embedding, log_type, "candidate_ids.pkl")
    
    if not os.path.exists(candidate_log_path) or not os.path.exists(candidate_ids_path):
        return None, None
    
    with open(candidate_log_path, 'rb') as f:
        candidate_logs = pickle.load(f)
    
    with open(candidate_ids_path, 'rb') as f:
        candidate_ids = pickle.load(f)
    
    return candidate_logs, candidate_ids


def extract_eid_from_etypes(etypes):
    ids = ''
    output = ''
    for line in etypes.split('\n'):
        if '=' in line:
            ids += line.strip().split('=')[0].replace('"','').replace("'",'').replace("{",'').replace("}",'').replace(")",'').replace("(",'')+', '
            output += line.strip()+'\n'
    
    return ids[:-2], output


def filter_tuples(text):
    def is_valid_id(id_str):
        return any(char.isdigit() for char in id_str)
    
    filtered_lines = []
    
    for line in text.strip().splitlines():
        match = re.search(r'\(([^,]+),\s*([^)]+)\)', line)
        if match:
            id1, id2 = match.groups()
            if is_valid_id(id1) and is_valid_id(id2) and (id1 != id2):
                filtered_lines.append(line)
                
    return '\n'.join(filtered_lines)


def filter_action_equals(graph_str):
    if not graph_str or not graph_str.strip():
        return graph_str
    
    filtered_lines = []
    
    for line in graph_str.split('\n'):
        if not line.strip():
            filtered_lines.append(line)
            continue
        def replace_action(match):
            action_content = match.group(1)
            if '=' in action_content:
                first_part = action_content.split('=')[0].strip()
                return f"A: [{first_part}]"
            else:
                return match.group(0)
        filtered_line = re.sub(r'A:\s*\[([^\]]+)\]', replace_action, line)
        filtered_lines.append(filtered_line)
    
    return '\n'.join(filtered_lines)


def filter_edge_quality(graph_str):
    if not graph_str or not graph_str.strip():
        return graph_str
    parsed_lines = []
    edges_data = []
    for line in graph_str.split('\n'):
        line_stripped = line.strip()
        edge_match = re.search(r'\(([^,]+),\s*([^)]+)\)\s+A:\s*\[([^\]]+)\]\s+\{D=([-><]+)\}\s*(\(timestamp=([^)]+)\))?', line)
        if edge_match:
            src, dst, action, direction, _, timestamp = edge_match.groups()
            has_valid_timestamp = timestamp is not None and timestamp.strip() not in ('...', '.', '')
            edge_info = {
                'line': line,
                'src': src,
                'dst': dst,
                'action': action.strip(),
                'direction': direction,
                'timestamp': timestamp if timestamp else None,
                'has_timestamp': has_valid_timestamp
            }
            parsed_lines.append(('edge', len(edges_data)))
            edges_data.append(edge_info)
        else:
            parsed_lines.append(('non_edge', line))
    if not edges_data:
        return graph_str
    valid_edge_indices = [i for i, e in enumerate(edges_data) if len(e['action']) > 1]
    if not valid_edge_indices:
        result_lines = [line for tag, line in parsed_lines if tag == 'non_edge']
        return '\n'.join(result_lines)
    valid_edges = [edges_data[i] for i in valid_edge_indices]
    has_timestamp_count = sum(1 for e in valid_edges if e['has_timestamp'])
    no_timestamp_count = len(valid_edges) - has_timestamp_count
    if has_timestamp_count > 0 and no_timestamp_count > 0:
        valid_edge_indices = [i for i in valid_edge_indices if edges_data[i]['has_timestamp']]
    result_lines = []
    valid_edge_set = set(valid_edge_indices)
    
    for tag, content in parsed_lines:
        if tag == 'non_edge':
            result_lines.append(content)
        elif tag == 'edge' and content in valid_edge_set:
            result_lines.append(edges_data[content]['line'])
    
    return '\n'.join(result_lines)


def process_single_candidate(candidate_log, candidate_id, llm_name, api_key, os_type,
                             iterations, llm_module, model_type="openai", ollama_url="http://localhost:11434",
                             use_examples=False, examples_dict=None, filter_actions=False):
    actual_model_name = llm_name
    logs = f'"""\n{candidate_log}\n"""'
    
    results = {
        'candidate_id': candidate_id,
        'success': False,
        'no_graph': False
    }
    responses_desc = {}
    responses_vanetype = {}
    responses_eidtup = {}
    responses_graph = {}
    no_edge_count = 0
    for i in range(iterations):
        if no_edge_count == 4:
            break
        try:
            if use_examples and examples_dict and examples_dict.get('entity_names'):
                ans = llm_module.llm_desc_cont(actual_model_name, api_key, logs, 
                                          examples_dict['entity_ids'], 
                                          examples_dict['entity_names'], 
                                          os_type, model_type, ollama_url)
            else:
                ans = llm_module.llm_desc(actual_model_name, api_key, logs, os_type, model_type, ollama_url)
            
            desc = extract_final_desc(ans.choices[0].message.content)
            responses_desc[i] = desc
        except Exception as e:
            responses_desc[i] = ""
            desc = ""
        try:
            ans = llm_module.llm_entitytypExt(actual_model_name, api_key, logs, desc, os_type, model_type, ollama_url)
            etypes = extract_last_entity_types_block(ans.choices[0].message.content)
            if etypes:
                eid_input, van_etype = extract_eid_from_etypes(etypes)
                responses_vanetype[i] = van_etype
            else:
                responses_vanetype[i] = ""
                eid_input = ""
        except Exception as e:
            responses_vanetype[i] = ""
            eid_input = ""
        try:
            if use_examples and examples_dict and examples_dict.get('edge_pairs'):
                graph_ans = llm_module.llm_GlobalEntity_contEg(actual_model_name, api_key, desc, 
                                                           eid_input, logs, 
                                                           examples_dict['edge_pairs'], 
                                                           os_type, model_type, ollama_url)
            else:
                graph_ans = llm_module.llm_GlobalEntity(actual_model_name, api_key, desc, 
                                                    eid_input, logs, os_type, model_type, ollama_url)
            
            related_entities = extract_final_pairs(graph_ans.choices[0].message.content)
            related_entities = related_entities.replace('"', '').replace("{","")
            
            if len(related_entities) < 1:
                no_edge_count += 1
                responses_eidtup[i] = ""
                continue
            related_entities = filter_tuples(related_entities)
            responses_eidtup[i] = related_entities
            
        except Exception as e:
            responses_eidtup[i] = ""
            no_edge_count += 1
            continue
        try:
            if use_examples and examples_dict and examples_dict.get('edges'):
                graphact_ans = llm_module.llm_GlobalAct_cont(actual_model_name, api_key, desc, 
                                                        related_entities, logs, 
                                                        examples_dict['edges'], 
                                                        os_type, model_type, ollama_url)
            else:
                graphact_ans = llm_module.llm_GlobalAct(actual_model_name, api_key, logs, desc, 
                                                   related_entities, os_type, model_type, ollama_url)
            
            graph_content = graphact_ans.choices[0].message.content
            if filter_actions:
                graph_content = filter_action_equals(graph_content)
            graph_content = filter_edge_quality(graph_content)
            
            responses_graph[i] = graph_content
            
        except Exception as e:
            responses_graph[i] = ""
    if no_edge_count == 4:
        results['no_graph'] = True
        results['description'] = ""
        results['entity_types'] = {}
        results['entity_pairs'] = ""
        results['final_graph'] = ""
        results['entity_names'] = {}
        results['description_iterations'] = responses_desc
        results['entity_types_iterations'] = responses_vanetype
        results['entity_pairs_iterations'] = responses_eidtup
        results['graph_iterations'] = responses_graph
        return results
    final_graph = extract_consistent_graph(responses_graph, 0)
    if filter_actions:
        final_graph = filter_action_equals(final_graph)
    final_graph = filter_edge_quality(final_graph)
    if responses_desc:
        valid_descs = {k: v for k, v in responses_desc.items() if v and v.strip()}
        if valid_descs:
            selected_iteration = random.choice(list(valid_descs.keys()))
            consistent_desc = valid_descs[selected_iteration]
        else:
            consistent_desc = ""
    else:
        consistent_desc = ""
    consistent_vanetypes = get_consistent_vanetype(list(responses_vanetype.values()))
    results['description'] = consistent_desc
    results['description_iterations'] = responses_desc
    results['entity_types'] = consistent_vanetypes
    results['entity_types_iterations'] = responses_vanetype
    results['entity_pairs'] = ""  # No single consensus for pairs, they vary per iteration
    results['entity_pairs_iterations'] = responses_eidtup
    results['final_graph'] = final_graph
    results['graph_iterations'] = responses_graph
    if len(final_graph) > 1:
        responses_ename = {}
        pattern = r"\(([\w\.\-]+(?:\:\d+)?),\s*([\w\.\-]+(?:\:\d+)?)\)"
        current_pairs_list = re.findall(pattern, final_graph)
        current_pairs_example = ''
        
        for pair in current_pairs_list:
            current_pairs_example += f"({pair[0]}, {pair[1]})\n"
        
        for i in range(iterations):
            try:
                if use_examples and examples_dict and examples_dict.get('entity_names'):
                    entity_name_ans = llm_module.llm_entityNames(actual_model_name, api_key, logs, consistent_desc, 
                                                            current_pairs_example, 
                                                            examples_dict['entity_names'], 
                                                            os_type, model_type, ollama_url)
                else:
                    entity_name_ans = llm_module.llm_entityNames_init(actual_model_name, api_key, logs, consistent_desc, 
                                                                  current_pairs_example, os_type, model_type, ollama_url)
                
                responses_ename[i] = extract_final_output(entity_name_ans.choices[0].message.content)
                
            except Exception as e:
                responses_ename[i] = ""
        entity_id_library = filter_process_dict(responses_ename)
        results['entity_names'] = entity_id_library
        results['entity_names_iterations'] = responses_ename
    else:
        results['entity_names'] = {}
        results['entity_names_iterations'] = {}
    
    results['success'] = True
    return results


def update_examples_pool(results_history, num_examples=10):
    if not results_history:
        return None
    all_entity_names = []
    for result in results_history:
        if result.get('entity_names'):
            for entity_id, names in result['entity_names'].items():
                if names and names[0].upper() != "NONE":
                    all_entity_names.append(f"{names[0]}")
    unique_entity_names = list(set(all_entity_names))
    if len(unique_entity_names) > num_examples:
        import random
        sampled_names = random.sample(unique_entity_names, num_examples)
    else:
        sampled_names = unique_entity_names
    
    entity_names_str = ',\n'.join(sampled_names)
    all_edges = []
    for result in results_history:
        if result.get('final_graph'):
            for line in result['final_graph'].split('\n'):
                if 'A:' in line and len(line.strip()) > 0:
                    all_edges.append(line.strip())
    unique_edges = list(set(all_edges))
    if len(unique_edges) > num_examples:
        import random
        sampled_edges = random.sample(unique_edges, num_examples)
    else:
        sampled_edges = unique_edges
    
    edges_str = '\n'.join(sampled_edges)
    edge_pairs = []
    pattern = r"\(([\w\.\-]+(?:\:\d+)?),\s*([\w\.\-]+(?:\:\d+)?)\)"
    for edge in sampled_edges:
        matches = re.findall(pattern, edge)
        for match in matches:
            edge_pairs.append(f"({match[0]}, {match[1]}) A: [...]\n")
    
    edge_pairs_str = ''.join(list(set(edge_pairs))[:num_examples])
    entity_ids = []
    for pair in edge_pairs[:num_examples]:
        matches = re.findall(r'\(([\w\.\-:]+)', pair)
        entity_ids.extend(matches)
    
    entity_ids_str = ', '.join(list(set(entity_ids))[:20])
    return {
        'entity_names': entity_names_str if entity_names_str else None,
        'edges': edges_str if edges_str else None,
        'edge_pairs': edge_pairs_str if edge_pairs_str else None,
        'entity_ids': entity_ids_str if entity_ids_str else None
    }


def save_candidate_results(output_dir, log_type, candidate_idx, candidate_log, results):
    candidate_dir = os.path.join(output_dir, log_type, f"candidate_{candidate_idx}")
    os.makedirs(candidate_dir, exist_ok=True)
    with open(os.path.join(candidate_dir, "log.txt"), 'w') as f:
        f.write(candidate_log)
    with open(os.path.join(candidate_dir, "candidate_id.txt"), 'w') as f:
        f.write(results['candidate_id'])
    with open(os.path.join(candidate_dir, "1_description.txt"), 'w') as f:
        f.write(results.get('description', ''))
    with open(os.path.join(candidate_dir, "1_description_7runs.json"), 'w') as f:
        json.dump(results.get('description_iterations', {}), f, indent=2)
    with open(os.path.join(candidate_dir, "2_entity_types.json"), 'w') as f:
        json.dump(results.get('entity_types', {}), f, indent=2)
    
    with open(os.path.join(candidate_dir, "2_entity_types_7runs.json"), 'w') as f:
        json.dump(results.get('entity_types_iterations', {}), f, indent=2)
    with open(os.path.join(candidate_dir, "3_entity_pairs.txt"), 'w') as f:
        f.write(results.get('entity_pairs', ''))
    with open(os.path.join(candidate_dir, "3_entity_pairs_7runs.json"), 'w') as f:
        json.dump(results.get('entity_pairs_iterations', {}), f, indent=2)
    with open(os.path.join(candidate_dir, "4_final_graph.txt"), 'w') as f:
        f.write(results.get('final_graph', ''))
    
    with open(os.path.join(candidate_dir, "4_graph_7runs.json"), 'w') as f:
        json.dump(results.get('graph_iterations', {}), f, indent=2)
    
    with open(os.path.join(candidate_dir, "5_entity_names.json"), 'w') as f:
        json.dump(results.get('entity_names', {}), f, indent=2)
    
    with open(os.path.join(candidate_dir, "5_entity_names_7runs.json"), 'w') as f:
        json.dump(results.get('entity_names_iterations', {}), f, indent=2)
    metadata = {
        'candidate_id': results['candidate_id'],
        'success': results['success'],
        'no_graph': results['no_graph'],
        'processed_at': datetime.now().isoformat()
    }
    with open(os.path.join(candidate_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    api_key = None
    if args.model_type == "openai":
        with open(args.config) as f:
            config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        if 'token' not in config_yaml:
            raise ValueError("config.yaml must contain 'token' field with OpenAI API key")
        api_key = config_yaml['token']
    elif args.model_type == "ollama":
        if not args.ollama_url:
            raise ValueError("--ollama_url is required when --model_type is ollama")
    if args.log_characteristics:
        log_characteristics_path = args.log_characteristics
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        characteristics_model_name = args.llm_name
        log_characteristics_path = os.path.join(script_dir, characteristics_model_name, 'log_idcharacteristics.json')
    if not os.path.exists(log_characteristics_path):
        return
    
    with open(log_characteristics_path) as f:
        log_characteristics_raw = json.load(f)
    if "parsed_results" in log_characteristics_raw:
        log_characteristics = log_characteristics_raw["parsed_results"]
    else:
        log_characteristics = log_characteristics_raw
    for log_type in args.log_types:
        candidate_logs, candidate_ids = load_candidates(args.candidates_dir, args.embedding, log_type)
        if candidate_logs is None:
            continue
        has_builtin_ids = log_characteristics.get(log_type, "unknown")
        if has_builtin_ids == "unclear":
            has_builtin_ids = "no"
        if has_builtin_ids == "yes":
            llm_module = llm_utils
        else:
            llm_module = llm_utils2
        output_dir = os.path.join(args.output_dir, args.embedding, args.llm_name)
        os.makedirs(output_dir, exist_ok=True)
        results_history = []
        candidate_indices = list(range(args.start_from, len(candidate_logs)))
        random.shuffle(candidate_indices)
        processed_count = 0
        total_to_process = len(candidate_indices)
        for idx in candidate_indices:
            processed_count += 1
            candidate_log = candidate_logs[idx]
            candidate_id = candidate_ids[idx]
            candidate_dir = os.path.join(args.output_dir, args.embedding, args.llm_name, log_type, f"candidate_{idx}")
            metadata_path = os.path.join(candidate_dir, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception:
                    pass
                continue
            use_examples = args.continual_learning and len(results_history) >= args.start_examples_after
            examples_dict = None
            
            if use_examples and results_history:
                examples_dict = update_examples_pool(results_history, args.num_examples)
            if log_type == "audit":
                os_type = "WINDOWS" 
            if log_type == "dns":
                os_type = "NETWORK"
            if log_type == "firefox":
                os_type = "FIREFOX"
            try:
                filter_actions = (has_builtin_ids != "yes")
                
                results = process_single_candidate(
                    candidate_log, 
                    candidate_id, 
                    args.llm_name, 
                    api_key, 
                    os_type,
                    args.llm_iterations,
                    llm_module,  # Pass the appropriate module
                    model_type=args.model_type,
                    ollama_url=args.ollama_url,
                    use_examples=use_examples,
                    examples_dict=examples_dict,
                    filter_actions=filter_actions
                )
                save_candidate_results(output_dir, log_type, idx, candidate_log, results)
                if results['success'] and not results['no_graph']:
                    results_history.append(results)
            except Exception as e:
                pass


if __name__ == "__main__":
    main()

