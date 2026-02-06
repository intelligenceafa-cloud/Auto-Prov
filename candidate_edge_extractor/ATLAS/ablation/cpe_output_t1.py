#!/usr/bin/env python3
import os
import json
import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="CPE: Convert ATLAS outputs to THEIA format (resolved edges) - Ablation")
    parser.add_argument("--llm_name", type=str, required=True,
                       help="LLM model name (e.g., gpt-3.5-turbo, llama3:70b)")
    parser.add_argument("--outputs_dir", type=str, default="./outputs",
                       help="Path to outputs directory (default: ./outputs)")
    parser.add_argument("--candidates_dir", type=str,
                       default="../../../clusterlogs_atlas/candidates-atlas_ablation",
                       help="Path to candidates-atlas_ablation directory (default: ../../../clusterlogs_atlas/candidates-atlas_ablation)")
    parser.add_argument("--embedding", type=str, default="roberta",
                       help="Embedding type (default: roberta)")
    parser.add_argument("--log_types", nargs='+', default=["dns", "firefox"],
                       help="Log types to process (default: dns firefox)")
    return parser.parse_args()


def is_full_integer(s):
    if not s or not isinstance(s, str):
        return False
    s = s.strip()
    return s.isdigit() or (s.startswith('-') and s[1:].isdigit())


def clean_enames_dict(enames_dict):
    if not enames_dict:
        return {}
    cleaned = {}
    for entity_id, name_list in enames_dict.items():
        if "**Step" in entity_id or "- Step" in entity_id or entity_id.startswith("-"):
            continue
        if entity_id.startswith('"-') or '\\"' in entity_id:
            continue
        if name_list:
            filtered_name_list = [n for n in name_list if "**Step" not in n]
            if filtered_name_list:
                cleaned[entity_id] = filtered_name_list
    return cleaned


def resolve_entity_id(entity_id: str, enames_dict: dict) -> str:
    if entity_id.startswith("id-"):
        if entity_id in enames_dict:
            name_list = enames_dict[entity_id]
            if name_list:
                for name in name_list:
                    if name.upper() != "NONE" and name != entity_id:
                        return name
                return name_list[0] if name_list else "NONE"
        return "NONE"
    else:
        return entity_id


def parse_edge_line(edge_line: str):
    if not edge_line or not edge_line.strip():
        return None
    edge_pattern = r'\(([^)]+),\s*([^)]+)\)\s+A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(timestamp=([^)]+)\)'
    match = re.match(edge_pattern, edge_line.strip())
    if not match:
        return None
    id1, id2, action, direction, timestamp = match.groups()
    action = action.strip()
    timestamp = timestamp.strip()
    if timestamp.endswith(' UTC'):
        timestamp = timestamp[:-4].strip()
    return {
        'source_id': id1.strip(),
        'dest_id': id2.strip(),
        'action': action,
        'direction': direction.strip(),
        'timestamp': timestamp
    }


def convert_edges_to_dict_list(edges_string: str, enames_dict: dict) -> list:
    if not edges_string:
        return []
    cleaned_enames = clean_enames_dict(enames_dict)
    edge_dicts = []
    lines = edges_string.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parsed = parse_edge_line(line)
        if not parsed:
            continue
        if parsed['action'] == "NO LABEL":
            continue
        source_id = parsed['source_id']
        dest_id = parsed['dest_id']
        source_name = resolve_entity_id(source_id, cleaned_enames)
        dest_name = resolve_entity_id(dest_id, cleaned_enames)
        if parsed['direction'] == '<-':
            source_name, dest_name = dest_name, source_name
        if source_name.upper() == "NONE" or dest_name.upper() == "NONE":
            continue
        if is_full_integer(source_name) or is_full_integer(dest_name):
            continue
        edge_dict = {
            "source": source_name,
            "dest": dest_name,
            "Action": parsed['action'],
            "timestamp": parsed['timestamp']
        }
        edge_dicts.append(edge_dict)
    return edge_dicts


def split_multi_action_edges(edges_string):
    if not edges_string:
        return ""
    lines = edges_string.split('\n')
    result_lines = []
    edge_pattern = r'\(([^)]+),\s*([^)]+)\)\s+A:\s*\[([^\]]+)\]\s*(\{[^}]+\})\s*(\(timestamp=[^)]*\))'
    for line in lines:
        line = line.strip()
        if not line:
            result_lines.append("")
            continue
        match = re.match(edge_pattern, line)
        if match:
            id1, id2, actions_str, direction, timestamp = match.groups()
            if ',' in actions_str:
                actions = []
                current_action = ""
                paren_depth = 0
                for char in actions_str:
                    if char == '(':
                        paren_depth += 1
                        current_action += char
                    elif char == ')':
                        paren_depth -= 1
                        current_action += char
                    elif char == ',' and paren_depth == 0:
                        actions.append(current_action.strip())
                        current_action = ""
                    else:
                        current_action += char
                if current_action.strip():
                    actions.append(current_action.strip())
                for action in actions:
                    action = action.strip()
                    if action:
                        new_edge = f"({id1}, {id2}) A: [{action}] {direction} {timestamp}"
                        result_lines.append(new_edge)
            else:
                result_lines.append(line)
        else:
            result_lines.append(line)
    return '\n'.join(result_lines)


def process_cpe(cpe_dir):
    cpe_path = Path(cpe_dir)
    metadata_path = cpe_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if metadata.get('no_graph', False):
                return None, 0
    log_path = cpe_path / "log.txt"
    if not log_path.exists():
        return None, 0
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read().strip()
    input_json = json.dumps(log_content, ensure_ascii=False)
    enames_path = cpe_path / "5_entity_names.json"
    enames_dict = {}
    if enames_path.exists():
        with open(enames_path, 'r') as f:
            enames_dict = json.load(f)
    edges_path = cpe_path / "4_final_graph.txt"
    edges_string = ""
    if edges_path.exists():
        with open(edges_path, 'r', encoding='utf-8') as f:
            edges_string = f.read().strip()
    if not edges_string:
        return None, 0
    edges_string = split_multi_action_edges(edges_string)
    no_label_count = edges_string.count('A: [NO LABEL]')
    edges_list = convert_edges_to_dict_list(edges_string, enames_dict)
    if not edges_list:
        return None, 0
    return {
        "input": input_json,
        "edges": edges_list
    }, no_label_count


def process_log_type(outputs_dir, candidates_dir, embedding, log_type, llm_name):
    source_dir = Path(outputs_dir) / embedding / llm_name / log_type
    if not source_dir.exists():
        return
    cpe_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir() and d.name.startswith("candidate_")])
    if not cpe_dirs:
        return
    results = []
    skipped = 0
    total_no_label_removed = 0
    for cpe_dir in cpe_dirs:
        result = process_cpe(cpe_dir)
        cpe_data, no_label_count = result
        if cpe_data is None:
            skipped += 1
            continue
        total_no_label_removed += no_label_count
        results.append(cpe_data)
    if not results:
        return
    output_dir = Path(candidates_dir) / embedding / llm_name / log_type
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "candidate-output.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def main():
    args = parse_args()
    for log_type in args.log_types:
        process_log_type(
            args.outputs_dir,
            args.candidates_dir,
            args.embedding,
            log_type,
            args.llm_name
        )


if __name__ == "__main__":
    main()

