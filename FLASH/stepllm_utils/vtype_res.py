#!/usr/bin/env python3

import os
import sys
import pandas as pd
import ast
import json
from pathlib import Path
import csv
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import argparse


def load_action_validation(dataset="theia"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    step_llm_dir = os.path.dirname(os.path.dirname(script_dir))
    validation_file = os.path.join(step_llm_dir, "ename-processing", f"edge_type_validation_{dataset.lower()}.json")
    
    if os.path.exists(validation_file):
        with open(validation_file, 'r') as f:
            return json.load(f)
    else:
        return {}


def is_valid_action(action, validation_dict):
    if action == "NO LABEL":
        return False
    if not validation_dict:
        return True
    status = validation_dict.get(action, "VALID")
    return status == "VALID"


def create_patterns_dict(patterns):
    patterns_dict = {}
    for pattern in patterns:
        patterns_dict[str(pattern)] = list(pattern)
    
    return patterns_dict


def create_dummies_dict(patterns):
    flattened_strings = []
    for pattern in patterns:
        flattened = []
        for sublist in pattern:
            flattened.extend(sublist)
        unique_elements = set([elem.lower() for elem in flattened])
        if unique_elements:
            joined_string = '+'.join(sorted(unique_elements))
            flattened_strings.append(joined_string)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(flattened_strings)
    dummies_dict = {}
    for string, label in zip(flattened_strings, labels):
        dummies_dict[string] = int(label)
    
    return dummies_dict


def create_id_labels_dict(final_patterns, dummies_dict):
    id_labels_dict = {}
    for id_key, patterns in final_patterns.items():
        flattened = []
        for vtype_list in patterns:
            flattened.extend(vtype_list)
        unique_elements = set([elem.lower() for elem in flattened])
        if unique_elements:
            joined_string = '+'.join(sorted(unique_elements))
            if joined_string in dummies_dict:
                id_labels_dict[id_key] = joined_string
            else:
                id_labels_dict[id_key] = None
    return id_labels_dict


def analyze_no_label_vtype_combinations_internal(dataset_path, id_labels_dict, dataset_name, inter_info_dir, existing_file=None):
    timestamp_dirs = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not timestamp_dirs:
        return None
    
    no_label_combinations = set()
    total_rows_analyzed = 0
    
    for timestamp_dir in sorted(timestamp_dirs):
        timestamp_path = os.path.join(dataset_path, timestamp_dir)
        
        csv_files = [f for f in os.listdir(timestamp_path) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            csv_path = os.path.join(timestamp_path, csv_file)
            
            try:
                with open(csv_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    header = next(csv_reader)
                    total_file_rows = sum(1 for row in csv_reader)
                
                with open(csv_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    header = next(csv_reader)
                    
                    pbar = tqdm(total=total_file_rows, 
                               desc=f"Analyzing {os.path.basename(csv_path)}", 
                               unit="rows",
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
                    
                    for row in csv_reader:
                        if len(row) >= 6:
                            action = row[3] if len(row) > 3 else ""
                            source_id = row[1] if len(row) > 1 else ""
                            dest_id = row[2] if len(row) > 2 else ""
                            
                            if action == "NO LABEL":
                                source_label = id_labels_dict.get(source_id)
                                dest_label = id_labels_dict.get(dest_id)
                                
                                if source_label and dest_label:
                                    combination = (source_label, dest_label)
                                    no_label_combinations.add(combination)
                            
                            total_rows_analyzed += 1
                            
                            if total_rows_analyzed % 1000 == 0:
                                pbar.set_postfix({'No Label Combinations': len(no_label_combinations)})
                            pbar.update(1)
                    
                    pbar.close()
                    
            except Exception as e:
                continue
    output_file = inter_info_dir / f"unknown_actions_{dataset_name.lower()}.txt"
    existing_combinations = set()
    if existing_file and existing_file.exists():
        with open(existing_file, 'r') as f:
            content = f.read()
            import re
            pattern = r'(\d+)\. source_vtypes: (.+)\n    dest_vtypes: (.+)\n    action: (.+)'
            matches = re.findall(pattern, content)
            for match in matches:
                source_vtypes = match[1]
                dest_vtypes = match[2]
                existing_combinations.add((source_vtypes, dest_vtypes))
    combinations_list = sorted(list(no_label_combinations))
    new_combinations = [combo for combo in combinations_list if combo not in existing_combinations]
    
    if not new_combinations:
        return combinations_list

    file_mode = 'a' if existing_file and existing_file.exists() else 'w'
    with open(output_file, file_mode) as f:
        for i, (source_label, dest_label) in enumerate(new_combinations, 1):
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f"source_vtypes: {source_label}")
            print(f"dest_vtypes: {dest_label}")
            user_action = input("Enter the action label for this combination: ")
            f.write(f"{len(existing_combinations) + i:3d}. source_vtypes: {source_label}\n")
            f.write(f"    dest_vtypes: {dest_label}\n")
            f.write(f"    action: {user_action}\n\n")
    
    return combinations_list


def analyze_no_label_vtype_combinations(dataset_name, base_path=None):
    dummies_dict, id_labels_dict = getvtypes(dataset_name, base_path)
    if base_path is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        extracted_prov_graph_path = project_root / "BIGDATA" / "ExtractedProvGraph"
    else:
        extracted_prov_graph_path = Path(base_path)
    
    if not extracted_prov_graph_path.exists():
        raise FileNotFoundError(f"ExtractedProvGraph directory not found at {extracted_prov_graph_path}")
    dataset_path = extracted_prov_graph_path / dataset_name.upper()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found at {dataset_path}")
    timestamp_dirs = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not timestamp_dirs:
        return None
    
    no_label_combinations = set()
    total_rows_analyzed = 0
    
    for timestamp_dir in sorted(timestamp_dirs):
        timestamp_path = os.path.join(dataset_path, timestamp_dir)
        
        csv_files = [f for f in os.listdir(timestamp_path) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            csv_path = os.path.join(timestamp_path, csv_file)
            
            try:
                with open(csv_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    header = next(csv_reader)
                    total_file_rows = sum(1 for row in csv_reader)
                
                with open(csv_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    header = next(csv_reader)
                    
                    pbar = tqdm(total=total_file_rows, 
                               desc=f"Analyzing {os.path.basename(csv_path)}", 
                               unit="rows",
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
                    
                    for row in csv_reader:
                        if len(row) >= 6:
                            action = row[3] if len(row) > 3 else ""
                            source_id = row[1] if len(row) > 1 else ""
                            dest_id = row[2] if len(row) > 2 else ""
                            
                            if action == "NO LABEL":
                                source_label = id_labels_dict.get(source_id)
                                dest_label = id_labels_dict.get(dest_id)
                                
                                if source_label and dest_label:
                                    combination = (source_label, dest_label)
                                    no_label_combinations.add(combination)
                            
                            total_rows_analyzed += 1
                            
                            if total_rows_analyzed % 1000 == 0:
                                pbar.set_postfix({'No Label Combinations': len(no_label_combinations)})
                            pbar.update(1)
                    
                    pbar.close()
                    
            except Exception as e:
                continue
    inter_info_dir = Path(__file__).parent.parent / "inter_info"
    inter_info_dir.mkdir(exist_ok=True)
    output_file = inter_info_dir / f"unknown_actions_{dataset_name.lower()}.txt"
    combinations_list = sorted(list(no_label_combinations))
    with open(output_file, 'w') as f:
        for i, (source_label, dest_label) in enumerate(combinations_list, 1):
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f"source_vtypes: {source_label}")
            print(f"dest_vtypes: {dest_label}")
            user_action = input("Enter the action label for this combination: ")
            f.write(f"{i:3d}. source_vtypes: {source_label}\n")
            f.write(f"    dest_vtypes: {dest_label}\n")
            f.write(f"    action: {user_action}\n\n")
    
    return combinations_list


def extract_vtype_list_from_string(vtype_str):
    if pd.isna(vtype_str) or vtype_str == '[]' or vtype_str == '':
        return tuple()
    try:
        if isinstance(vtype_str, str):
            result = ast.literal_eval(vtype_str)
            if isinstance(result, list):
                return tuple(sorted(result))
    except:
        pass
    if isinstance(vtype_str, str):
        import re
        matches = re.findall(r"'([^']+)'", vtype_str)
        if matches:
            return tuple(sorted(matches))
        words = re.findall(r'\b[A-Z_][A-Za-z0-9_]*\b', vtype_str)
        if words:
            return tuple(sorted(words))
    return tuple()


def analyze_csv_simple(csv_path, dataset_name="theia", action_validation=None):
    if action_validation is None:
        action_validation = load_action_validation(dataset_name)
    interactive_logs = set()
    non_interactive_logs = set()
    id_interactive_patterns = {}
    id_non_interactive_patterns = {}
    total_rows = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            total_file_rows = sum(1 for row in csv_reader)
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            log_actions = {}
            for row in csv_reader:
                if len(row) >= 4:
                    log_idx = int(row[0]) if row[0].isdigit() else 0
                    action = row[3] if len(row) > 3 else ""
                    
                    if log_idx not in log_actions:
                        log_actions[log_idx] = set()
                    log_actions[log_idx].add(action)
        for log_idx, actions in log_actions.items():
            valid_actions = [a for a in actions if is_valid_action(a, action_validation)]
            if len(valid_actions) == 0:
                non_interactive_logs.add(log_idx)
            else:
                interactive_logs.add(log_idx)
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            pbar = tqdm(total=total_file_rows, 
                       desc=f"Processing {os.path.basename(csv_path)}", 
                       unit="rows",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
            
            for row_idx, row in enumerate(csv_reader):
                if len(row) >= 9:
                    log_idx = int(row[0]) if row[0].isdigit() else 0
                    source_id = row[1] if len(row) > 1 else ""
                    dest_id = row[2] if len(row) > 2 else ""
                    source_vtype_str = row[4] if len(row) > 4 else ""
                    dest_vtype_str = row[5] if len(row) > 5 else ""
                    source_vtype_list = extract_vtype_list_from_string(source_vtype_str)
                    dest_vtype_list = extract_vtype_list_from_string(dest_vtype_str)
                    if log_idx in interactive_logs:
                        source_patterns = id_interactive_patterns
                        dest_patterns = id_interactive_patterns
                    else:
                        source_patterns = id_non_interactive_patterns
                        dest_patterns = id_non_interactive_patterns
                    if source_id and source_vtype_list:
                        if source_id not in source_patterns:
                            source_patterns[source_id] = set()
                        source_patterns[source_id].add(source_vtype_list)
                    if dest_id and dest_vtype_list:
                        if dest_id not in dest_patterns:
                            dest_patterns[dest_id] = set()
                        dest_patterns[dest_id].add(dest_vtype_list)
                    total_rows += 1
                    if total_rows % 1000 == 0:
                        pbar.set_postfix({
                            'Interactive': len(interactive_logs),
                            'Non-Interactive': len(non_interactive_logs),
                            'Rows': total_rows
                        })
                    pbar.update(1)
            
            pbar.close()
    
    except Exception as e:
        return None
    
    return {
        'id_interactive_patterns': id_interactive_patterns,
        'id_non_interactive_patterns': id_non_interactive_patterns,
        'interactive_logs': interactive_logs,
        'non_interactive_logs': non_interactive_logs,
        'total_rows': total_rows
    }


def analyze_dataset_simple(dataset_path):
    dataset_name = os.path.basename(dataset_path).lower()
    action_validation = load_action_validation(dataset_name)

    all_id_interactive_patterns = {}
    all_id_non_interactive_patterns = {}
    total_rows = 0
    csv_files_processed = 0
    total_interactive_logs = 0
    total_non_interactive_logs = 0
    timestamp_dirs = [d for d in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, d))]
    if not timestamp_dirs:
        return None
    for timestamp_dir in sorted(timestamp_dirs):
        timestamp_path = os.path.join(dataset_path, timestamp_dir)
        csv_files = [f for f in os.listdir(timestamp_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            csv_path = os.path.join(timestamp_path, csv_file)
            result = analyze_csv_simple(csv_path, dataset_name, action_validation)
            if result:
                for id_key, patterns in result['id_interactive_patterns'].items():
                    if id_key not in all_id_interactive_patterns:
                        all_id_interactive_patterns[id_key] = set()
                    all_id_interactive_patterns[id_key].update(patterns)
                for id_key, patterns in result['id_non_interactive_patterns'].items():
                    if id_key not in all_id_non_interactive_patterns:
                        all_id_non_interactive_patterns[id_key] = set()
                    all_id_non_interactive_patterns[id_key].update(patterns)
                
                
                total_rows += result['total_rows']
                total_interactive_logs += len(result['interactive_logs'])
                total_non_interactive_logs += len(result['non_interactive_logs'])
                csv_files_processed += 1
    final_patterns = {}
    all_ids = set(all_id_interactive_patterns.keys()) | set(all_id_non_interactive_patterns.keys())
    for id_key in all_ids:
        in_interactive = id_key in all_id_interactive_patterns
        in_non_interactive = id_key in all_id_non_interactive_patterns
        if in_interactive and in_non_interactive:
            final_patterns[id_key] = all_id_non_interactive_patterns[id_key]
        elif in_interactive:
            final_patterns[id_key] = all_id_interactive_patterns[id_key]
        else:
            final_patterns[id_key] = all_id_non_interactive_patterns[id_key]
    unique_patterns = set()
    for id_key, patterns in final_patterns.items():
        pattern = tuple(sorted(patterns))
        if pattern:
            unique_patterns.add(pattern)
    
    return {
        'dataset': os.path.basename(dataset_path),
        'timestamp_dirs': len(timestamp_dirs),
        'csv_files_processed': csv_files_processed,
        'total_rows': total_rows,
        'total_interactive_logs': total_interactive_logs,
        'total_non_interactive_logs': total_non_interactive_logs,
        'unique_ids': len(final_patterns),
        'unique_patterns': sorted(list(unique_patterns)),
        'pattern_count': len(unique_patterns),
        'final_patterns': final_patterns
    }


def getvtypes(dataset_name, base_path=None):
    inter_info_dir = Path(__file__).parent.parent / "inter_info"
    inter_info_dir.mkdir(exist_ok=True)
    
    dummies_file = inter_info_dir / f"dummies_{dataset_name.lower()}.json"
    id_labels_file = inter_info_dir / f"id_labels_{dataset_name.lower()}.json"
    unknown_actions_file = inter_info_dir / f"unknown_actions_{dataset_name.lower()}.txt"

    if dummies_file.exists() and id_labels_file.exists():
        with open(dummies_file, 'r') as f:
            dummies_dict = json.load(f)
        with open(id_labels_file, 'r') as f:
            id_labels_dict = json.load(f)
        if base_path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            extracted_prov_graph_path = project_root / "BIGDATA" / "ExtractedProvGraph"
        else:
            extracted_prov_graph_path = Path(base_path)
        dataset_path = extracted_prov_graph_path / dataset_name.upper()
        if unknown_actions_file.exists():
            pass
        else:
            analyze_no_label_vtype_combinations_internal(dataset_path, id_labels_dict, dataset_name, inter_info_dir)
        
        return dummies_dict, id_labels_dict
    if base_path is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        extracted_prov_graph_path = project_root / "BIGDATA" / "ExtractedProvGraph"
    else:
        extracted_prov_graph_path = Path(base_path)
    if not extracted_prov_graph_path.exists():
        raise FileNotFoundError(f"ExtractedProvGraph directory not found at {extracted_prov_graph_path}")
    dataset_path = extracted_prov_graph_path / dataset_name.upper()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found at {dataset_path}")
    result = analyze_dataset_simple(dataset_path)
    if not result:
        raise ValueError(f"Failed to analyze dataset {dataset_name}")
    dummies_dict = create_dummies_dict(result['unique_patterns'])
    dummies_file = inter_info_dir / f"dummies_{dataset_name.lower()}.json"
    with open(dummies_file, 'w') as f:
        json.dump(dummies_dict, f, indent=2)
    id_labels_dict = create_id_labels_dict(result['final_patterns'], dummies_dict)
    id_labels_file = inter_info_dir / f"id_labels_{dataset_name.lower()}.json"
    with open(id_labels_file, 'w') as f:
        json.dump(id_labels_dict, f, indent=2)
    analyze_no_label_vtype_combinations_internal(dataset_path, id_labels_dict, dataset_name, inter_info_dir)
    
    return dummies_dict, id_labels_dict


def main():
    parser = argparse.ArgumentParser(description='Analyze vertex types from CSV files')
    parser.add_argument('dataset', nargs='?', help='Dataset name (e.g., THEIA, FIVEDIRECTIONS)')
    parser.add_argument('--path', '-p', default='../BIGDATA/ExtractedProvGraph/', help='Path to ExtractedProvGraph directory')
    
    args = parser.parse_args()
    if args.path:
        extracted_prov_graph_path = Path(args.path)
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        extracted_prov_graph_path = project_root / "BIGDATA" / "ExtractedProvGraph"
    if not extracted_prov_graph_path.exists():
        print(f"Error: ExtractedProvGraph directory not found at {extracted_prov_graph_path}")
        sys.exit(1)
    if args.dataset:
        dataset_name = args.dataset.upper()
        dataset_path = extracted_prov_graph_path / dataset_name
        
        if not dataset_path.exists():
            print(f"Error: Dataset '{dataset_name}' not found at {dataset_path}")
            sys.exit(1)
        
        datasets_to_analyze = [dataset_path]
    else:
        datasets_to_analyze = [d for d in extracted_prov_graph_path.iterdir() 
                              if d.is_dir()]
        
        if not datasets_to_analyze:
            print(f"Error: No datasets found in {extracted_prov_graph_path}")
            sys.exit(1)
    script_dir = Path(__file__).parent
    inter_info_dir = script_dir.parent / "inter_info"
    inter_info_dir.mkdir(exist_ok=True)
    for dataset_path in datasets_to_analyze:
        dataset_name = dataset_path.name
        getvtypes(dataset_name, args.path)


if __name__ == "__main__":
    main()