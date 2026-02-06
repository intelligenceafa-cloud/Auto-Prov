#!/usr/bin/env python3
import os
import json
import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="CPE: Convert ATLAS outputs to THEIA format (vtypes, enames, edges) - Ablation")
    parser.add_argument("--llm_name", type=str, required=True,
                       help="LLM model name (e.g., gpt-3.5-turbo, llama3:70b)")
    parser.add_argument("--outputs_dir", type=str, default="./outputs",
                       help="Path to outputs directory (default: ./outputs)")
    parser.add_argument("--candidates_dir", type=str,
                       default="../../../clusterlogs_atlas/candidates-atlas_ablation",
                       help="Path to candidates-atlas_ablation directory (default: ../../../clusterlogs_atlas/candidates-atlas_ablation)")
    parser.add_argument("--embedding", type=str, default="roberta",
                       help="Embedding type (default: roberta)")
    parser.add_argument("--log_type", type=str, default="audit",
                       help="Log type to process (default: audit)")
    return parser.parse_args()


def convert_entity_types_to_string(vtypes_dict):
    if not vtypes_dict:
        return ""
    lines = []
    for entity_id, entity_type in vtypes_dict.items():
        lines.append(f"{entity_id} = {entity_type}")
    return "\n".join(lines)


def convert_entity_names_to_string(enames_dict):
    if not enames_dict:
        return ""
    lines = []
    for entity_id, name_list in enames_dict.items():
        if "**Step" in entity_id:
            continue
        filtered_name_list = [n for n in name_list if "**Step" not in n] if name_list else []
        if filtered_name_list and all(n == entity_id for n in filtered_name_list):
            continue
        if not filtered_name_list:
            name = "NONE"
        elif len(filtered_name_list) == 1 and filtered_name_list[0].upper() == "NONE":
            name = "NONE"
        else:
            name = None
            for n in filtered_name_list:
                if n.upper() != "NONE" and n != entity_id:
                    name = n
                    break
            if name is None:
                for n in filtered_name_list:
                    if n != entity_id:
                        name = n
                        break
            if name is None:
                name = "NONE"
        if "**Step" in name:
            continue
        if name == entity_id:
            continue
        lines.append(f"{entity_id} = {name}")
    return "\n".join(lines)


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


def filter_no_label_edges(edges_string):
    if not edges_string:
        return "", 0
    lines = [line.strip() for line in edges_string.split('\n') if line.strip()]
    filtered_lines = [line for line in lines if 'A: [NO LABEL]' not in line]
    removed_count = len(lines) - len(filtered_lines)
    return '\n'.join(filtered_lines), removed_count


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
    vtypes_path = cpe_path / "2_entity_types.json"
    vtypes_dict = {}
    if vtypes_path.exists():
        with open(vtypes_path, 'r') as f:
            vtypes_dict = json.load(f)
    vtypes_string = convert_entity_types_to_string(vtypes_dict)
    enames_path = cpe_path / "5_entity_names.json"
    enames_dict = {}
    if enames_path.exists():
        with open(enames_path, 'r') as f:
            enames_dict = json.load(f)
    enames_string = convert_entity_names_to_string(enames_dict)
    edges_path = cpe_path / "4_final_graph.txt"
    edges_string = ""
    if edges_path.exists():
        with open(edges_path, 'r', encoding='utf-8') as f:
            edges_string = f.read().strip()
    if not edges_string:
        return None, 0
    edges_string = split_multi_action_edges(edges_string)
    edges_string, no_label_count = filter_no_label_edges(edges_string)
    if not edges_string or not edges_string.strip():
        return None, 0
    return {
        "input": input_json,
        "vtypes": vtypes_string,
        "enames": enames_string,
        "edges": edges_string
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
    process_log_type(
        args.outputs_dir,
        args.candidates_dir,
        args.embedding,
        args.log_type,
        args.llm_name
    )


if __name__ == "__main__":
    main()

