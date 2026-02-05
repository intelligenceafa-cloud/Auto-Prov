#!/usr/bin/env python3

import os
import pickle
import re
import csv
import argparse
import signal
import threading
import json
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager

try:
    import re2
    re_fast = re2
    HAS_REGEX = True
except ImportError:
    try:
        import regex as re_fast
        HAS_REGEX = True
    except ImportError:
        re_fast = re
        HAS_REGEX = False


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Regex pattern matching timed out")


@contextmanager
def timeout_context(seconds):
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(int(seconds) if seconds > 0 else 1)
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        yield


def clean_pattern(pattern):
    if isinstance(pattern, tuple):
        cleaned_tuple = []
        for p in pattern:
            if isinstance(p, str):
                cleaned_p = clean_string_pattern(p)
                cleaned_tuple.append(cleaned_p)
            else:
                cleaned_tuple.append(p)
        return tuple(cleaned_tuple)
    elif isinstance(pattern, str):
        return clean_string_pattern(pattern)
    else:
        return pattern


def clean_string_pattern(pattern_str):
    if not isinstance(pattern_str, str):
        return pattern_str
    
    cleaned = pattern_str.replace('\n', '').replace('\r', '')
    
    cleaned = re.sub(r'\s*-\s*NOT\s+APPLICABLE.*$', '', cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\s*\'+\s*$', '', cleaned)
    
    cleaned = cleaned.strip()
    
    return cleaned


def compile_patterns(master_patterns: List[dict]) -> List[dict]:
    compiled_patterns = []
    
    for pattern_dict in master_patterns:
        compiled_dict = {}
        for field, pattern in pattern_dict.items():
            if pattern and pattern not in ['No Regex', 'NO LABEL', 'NO TIMESTAMP']:
                try:
                    cleaned_pattern = clean_pattern(pattern)
                    
                    if isinstance(cleaned_pattern, tuple):
                        compiled_tuple = []
                        for p in cleaned_pattern:
                            if p and p not in ['No Regex', 'NO LABEL', 'NO TIMESTAMP']:
                                compiled_tuple.append(re_fast.compile(p))
                            else:
                                compiled_tuple.append(p)
                        compiled_dict[field] = tuple(compiled_tuple)
                    else:
                        compiled_dict[field] = re_fast.compile(cleaned_pattern)
                except Exception:
                    compiled_dict[field] = pattern
            else:
                compiled_dict[field] = pattern
        compiled_patterns.append(compiled_dict)
    
    return compiled_patterns


def apply_single_regex(compiled_pattern, text: str, timeout_seconds: float = 0.5, return_groups_separately: bool = False) -> Optional[Union[str, List[str]]]:
    if compiled_pattern in ['NO LABEL', 'NO TIMESTAMP', 'No Regex']:
        return compiled_pattern
    
    try:
        with timeout_context(int(timeout_seconds)):
            match = compiled_pattern.search(text)
            if match:
                if match.groups():
                    if len(match.groups()) > 1:
                        if return_groups_separately:
                            second_group = match.group(2)
                            if not second_group.isdigit():
                                return list(match.groups())
                        result = ':'.join(match.groups())
                        return result
                    else:
                        result = match.group(1)
                        return result
                else:
                    result = match.group(0)
                    return result
        return None
    except TimeoutError:
        return None
    except Exception:
        return None


def apply_tuple_regex(pattern_tuple: Tuple, text: str, timeout_seconds: float = 0.5) -> Optional[str]:
    if not pattern_tuple:
        return None
    
    if 'NOT APPLICABLE' in pattern_tuple:
        for pattern in pattern_tuple:
            if pattern != 'NOT APPLICABLE':
                result = apply_single_regex(pattern, text, timeout_seconds)
                return result
        return None
    
    results: List[str] = []
    for pattern in pattern_tuple:
        result = apply_single_regex(pattern, text, timeout_seconds)
        if result is None:
            return None
        results.append(result)
    
    return ':'.join(results)


def detect_pattern_format(pattern_dict: dict) -> str:
    if 'Source ID' in pattern_dict or 'Destination ID' in pattern_dict or 'A' in pattern_dict:
        return 'audit'
    elif 'source' in pattern_dict or 'dest' in pattern_dict or 'action' in pattern_dict:
        return 'no_builtin_ids'
    else:
        return 'audit'


def test_single_pattern(pattern_dict: dict, log_text: str, timeout_seconds: float = 0.5) -> Tuple[bool, Union[dict, List[dict]]]:
    extracted_data = {}
    
    pattern_format = detect_pattern_format(pattern_dict)
    
    try:
        if pattern_format == 'audit':
            source_groups = None
            dest_groups = None
            
            for field in ['Source ID', 'Destination ID']:
                if field in pattern_dict:
                    pattern = pattern_dict[field]
                    if pattern is None:
                        extracted_data[field] = None
                        continue
                    
                    if isinstance(pattern, tuple):
                        extracted_value = apply_tuple_regex(pattern, log_text, timeout_seconds)
                        extracted_data[field] = extracted_value
                    else:
                        extracted_value = apply_single_regex(pattern, log_text, timeout_seconds, return_groups_separately=True)
                        
                        if extracted_value is None and pattern not in ['NO LABEL', 'NO TIMESTAMP']:
                            return False, {}
                        
                        if isinstance(extracted_value, list):
                            if field == 'Source ID':
                                source_groups = extracted_value
                            elif field == 'Destination ID':
                                dest_groups = extracted_value
                            extracted_data[field] = extracted_value[0]
                        else:
                            extracted_data[field] = extracted_value
            
            for field, pattern in pattern_dict.items():
                if field in ['Source ID', 'Destination ID']:
                    continue
                
                if pattern is None:
                    extracted_data[field] = None
                    continue
                
                if isinstance(pattern, tuple):
                    extracted_value = apply_tuple_regex(pattern, log_text, timeout_seconds)
                else:
                    extracted_value = apply_single_regex(pattern, log_text, timeout_seconds)
                    
                if extracted_value is None and pattern not in ['NO LABEL', 'NO TIMESTAMP']:
                    return False, {}
                
                extracted_data[field] = extracted_value
            
            if source_groups or dest_groups:
                result_list = []
                source_list = source_groups if source_groups else [extracted_data['Source ID']]
                dest_list = dest_groups if dest_groups else [extracted_data['Destination ID']]
                
                for src in source_list:
                    for dst in dest_list:
                        edge_data = extracted_data.copy()
                        edge_data['Source ID'] = src
                        edge_data['Destination ID'] = dst
                        result_list.append(edge_data)
                
                return True, result_list
        
        else:
            source_groups = None
            dest_groups = None
            
            for field in ['source', 'dest']:
                if field in pattern_dict:
                    pattern = pattern_dict[field]
                    if pattern is None:
                        extracted_data[field] = None
                        continue
                    
                    if isinstance(pattern, tuple):
                        extracted_value = apply_tuple_regex(pattern, log_text, timeout_seconds)
                        extracted_data[field] = extracted_value
                    else:
                        extracted_value = apply_single_regex(pattern, log_text, timeout_seconds, return_groups_separately=True)
                        
                        if extracted_value is None and pattern not in ['NO LABEL', 'NO TIMESTAMP']:
                            return False, {}
                        
                        if isinstance(extracted_value, list):
                            if field == 'source':
                                source_groups = extracted_value
                            elif field == 'dest':
                                dest_groups = extracted_value
                            extracted_data[field] = extracted_value[0]
                        else:
                            extracted_data[field] = extracted_value
            
            for field, pattern in pattern_dict.items():
                if field in ['source', 'dest']:
                    continue
                
                if pattern is None:
                    extracted_data[field] = None
                    continue
                
                if isinstance(pattern, tuple):
                    extracted_value = apply_tuple_regex(pattern, log_text, timeout_seconds)
                else:
                    extracted_value = apply_single_regex(pattern, log_text, timeout_seconds)
                    
                if extracted_value is None and pattern not in ['NO LABEL', 'NO TIMESTAMP']:
                    return False, {}
                
                extracted_data[field] = extracted_value
            
            if source_groups or dest_groups:
                result_list = []
                source_list = source_groups if source_groups else [extracted_data['source']]
                dest_list = dest_groups if dest_groups else [extracted_data['dest']]
                
                for src in source_list:
                    for dst in dest_list:
                        edge_data = extracted_data.copy()
                        edge_data['source'] = src
                        edge_data['dest'] = dst
                        result_list.append(edge_data)
                
                return True, result_list
        
        return True, extracted_data
    except TimeoutError:
        return False, {}
    except Exception:
        return False, {}


def categorize_patterns(compiled_patterns: List[dict]) -> Tuple[List[dict], List[dict]]:
    if not compiled_patterns:
        return [], []
    
    pattern_format = detect_pattern_format(compiled_patterns[0])
    
    specific_patterns = []
    generic_patterns = []
    
    for pattern_dict in compiled_patterns:
        if pattern_format == 'audit':
            action_pattern = pattern_dict.get('A', '')
            timestamp_pattern = pattern_dict.get('Timestamp', '')
        else:
            action_pattern = pattern_dict.get('action', '')
            timestamp_pattern = pattern_dict.get('timestamp', '')
        
        if (action_pattern and action_pattern not in ['NO LABEL', 'NO TIMESTAMP']) or \
           (timestamp_pattern and timestamp_pattern not in ['NO LABEL', 'NO TIMESTAMP']):
            specific_patterns.append(pattern_dict)
        else:
            generic_patterns.append(pattern_dict)
    
    return specific_patterns, generic_patterns


def format_edge_output(extracted_data: dict) -> str:
    if 'Source ID' in extracted_data or 'Destination ID' in extracted_data:
        source_id = extracted_data.get('Source ID', '')
        dest_id = extracted_data.get('Destination ID', '')
        action = extracted_data.get('A', 'NO LABEL')
        timestamp = extracted_data.get('Timestamp', 'NO TIMESTAMP')
    else:
        source_id = extracted_data.get('source', '')
        dest_id = extracted_data.get('dest', '')
        action = extracted_data.get('action', 'NO LABEL')
        timestamp = extracted_data.get('timestamp', 'NO TIMESTAMP')
    
    source_id = str(source_id).strip('"\'') if source_id else ''
    dest_id = str(dest_id).strip('"\'') if dest_id else ''
    
    if timestamp == 'NO TIMESTAMP' or timestamp is None:
        timestamp_str = 'timestamp=...'
    else:
        timestamp_str = f'timestamp={timestamp}'
    
    if action == 'NO LABEL' or action is None:
        action_str = 'NO LABEL'
    else:
        action_str = str(action).strip('"\'')
    
    result = f"({source_id}, {dest_id}) A: [{action_str}] {{D=->}} ({timestamp_str})"
    return result


def test_single_ename_pattern(pattern_dict: dict, log_text: str, timeout_seconds: float = 0.5) -> Tuple[bool, dict]:
    extracted_data = {}
    
    try:
        if 'ID' in pattern_dict:
            pattern = pattern_dict['ID']
            if pattern is None:
                return False, {}
            
            if isinstance(pattern, tuple):
                extracted_value = apply_tuple_regex(pattern, log_text, timeout_seconds)
            else:
                extracted_value = apply_single_regex(pattern, log_text, timeout_seconds)
            
            if extracted_value is None:
                return False, {}
            
            extracted_data['ID'] = extracted_value
        
        if 'Name' in pattern_dict:
            pattern = pattern_dict['Name']
            if pattern is None:
                extracted_data['Name'] = None
            elif pattern == 'No Regex':
                extracted_data['Name'] = 'No Regex'
            else:
                if isinstance(pattern, tuple):
                    extracted_value = apply_tuple_regex(pattern, log_text, timeout_seconds)
                else:
                    extracted_value = apply_single_regex(pattern, log_text, timeout_seconds)
                
                if extracted_value is None and pattern != 'No Regex':
                    extracted_data['Name'] = None
                else:
                    extracted_data['Name'] = extracted_value
        
        return True, extracted_data
    except TimeoutError:
        return False, {}
    except Exception:
        return False, {}


def format_ename_output(extracted_data: dict) -> str:
    entity_id = extracted_data.get('ID', '')
    entity_name = extracted_data.get('Name', 'No Regex')
    
    entity_id = str(entity_id).strip('"\'') if entity_id else ''
    entity_name = str(entity_name).strip('"\'') if entity_name else 'No Regex'
    
    if entity_id and entity_name != 'No Regex':
        return f"{entity_id} = {entity_name}"
    else:
        return "No Regex"


def filter_single_character_enames(enames: List[str]) -> List[str]:
    filtered_enames = []
    
    for ename in enames:
        if not ename or " = " not in ename:
            continue
        
        parts = ename.split(" = ", 1)
        if len(parts) != 2:
            continue
        
        entity_id = parts[0].strip()
        entity_name = parts[1].strip()
        
        if len(entity_name) == 1:
            continue
        
        filtered_enames.append(ename)
    
    return filtered_enames


def filter_edges_with_timestamp_as_action(edges: List[str]) -> List[str]:
    filtered_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            filtered_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        action = action.strip()
        timestamp = timestamp.strip()
        
        if timestamp.startswith('timestamp='):
            timestamp_value = timestamp[len('timestamp='):].strip()
        else:
            timestamp_value = timestamp
        
        action_normalized = action.strip()
        timestamp_normalized = timestamp_value.strip()
        
        if timestamp_normalized.endswith(':PM') or timestamp_normalized.endswith(':AM'):
            timestamp_normalized = timestamp_normalized[:-3].strip()
        
        if action_normalized == timestamp_normalized or action_normalized in timestamp_normalized:
            continue
        
        if action_normalized in timestamp_value:
            continue
        
        filtered_edges.append(edge)
    
    return filtered_edges


def filter_edges_with_ename_matches(edges: List[str], enames: List[str]) -> List[str]:
    ename_names = set()
    for ename in enames:
        if " = " in ename:
            parts = ename.split(" = ", 1)
            if len(parts) == 2:
                name = parts[1].strip()
                if name:
                    ename_names.add(name)
    
    if not ename_names:
        return edges
    
    filtered_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            filtered_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        action = action.strip()
        
        if source in ename_names or dest in ename_names or action in ename_names:
            continue
        
        filtered_edges.append(edge)
    
    return filtered_edges


def remove_trailing_quotes_from_enames(enames: List[str]) -> List[str]:
    cleaned_enames = []
    
    for ename in enames:
        if not ename or " = " not in ename:
            cleaned_enames.append(ename)
            continue
        
        parts = ename.split(" = ", 1)
        if len(parts) != 2:
            cleaned_enames.append(ename)
            continue
        
        entity_id = parts[0].strip()
        entity_name = parts[1].strip()
        
        while entity_name and entity_name.endswith('"'):
            entity_name = entity_name[:-1].strip()
        
        if entity_id and entity_name:
            cleaned_ename = f"{entity_id} = {entity_name}"
            cleaned_enames.append(cleaned_ename)
        else:
            continue
    
    return cleaned_enames


def preprocess_enames(enames: List[str]) -> List[str]:
    cleaned_enames = []
    
    for ename in enames:
        if not ename or " = " not in ename:
            continue
        
        parts = ename.split(" = ", 1)
        if len(parts) != 2:
            continue
        
        entity_id = parts[0].strip()
        entity_name = parts[1].strip()
        
        if '\n' in entity_name:
            lines = entity_name.split('\n')
            first_line = None
            for line in lines:
                line = line.strip()
                if line:
                    first_line = line
                    break
            if first_line:
                entity_name = first_line
            else:
                continue
        
        if '\t' in entity_name or re.search(r' {3,}', entity_name):
            split_parts = re.split(r'\t+| {3,}', entity_name)
            split_parts = [p.strip() for p in split_parts if p.strip()]
            
            if len(split_parts) >= 2:
                final_name = None
                for part in split_parts:
                    if part.endswith(':') and not re.search(r'[A-Za-z]:\\', part):
                        continue
                    
                    path_duplicate_pattern = re.search(r'^([A-Za-z]:\\[^:]+):([A-Za-z]:\\[^:]+)$', part)
                    if path_duplicate_pattern:
                        first_path = path_duplicate_pattern.group(1).strip()
                        second_path = path_duplicate_pattern.group(2).strip()
                        
                        if first_path == second_path:
                            final_name = second_path
                            break
                        elif second_path.startswith(first_path):
                            final_name = second_path
                            break
                        else:
                            final_name = second_path
                            break
                    else:
                        colon_positions = []
                        for i, char in enumerate(part):
                            if char == ':' and i > 0:
                                if not (i == 1 and part[0].isalpha()):
                                    colon_positions.append(i)
                        
                        if colon_positions:
                            last_colon_idx = colon_positions[-1]
                            before_last_colon = part[:last_colon_idx].strip()
                            after_last_colon = part[last_colon_idx + 1:].strip()
                            
                            if before_last_colon == after_last_colon:
                                final_name = after_last_colon
                                break
                            elif after_last_colon.startswith(before_last_colon):
                                final_name = after_last_colon
                                break
                            else:
                                if len(before_last_colon) < 30 and ' ' in before_last_colon:
                                    final_name = after_last_colon
                                    break
                                else:
                                    final_name = after_last_colon
                                    break
                        else:
                            final_name = part
                            break
                
                if final_name:
                    entity_name = final_name
                else:
                    for part in reversed(split_parts):
                        if not (part.endswith(':') and not any(c.isalnum() or c in '\\/' for c in part[:-1])):
                            entity_name = part
                            break
                    else:
                        continue
            elif len(split_parts) == 1:
                part = split_parts[0]
                path_duplicate_pattern = re.search(r'^([A-Za-z]:\\[^:]+):([A-Za-z]:\\[^:]+)$', part)
                if path_duplicate_pattern:
                    first_path = path_duplicate_pattern.group(1).strip()
                    second_path = path_duplicate_pattern.group(2).strip()
                    
                    if first_path == second_path:
                        entity_name = second_path
                    elif second_path.startswith(first_path):
                        entity_name = second_path
                    else:
                        entity_name = second_path
                else:
                    colon_positions = []
                    for i, char in enumerate(part):
                        if char == ':' and i > 0:
                            if not (i == 1 and part[0].isalpha()):
                                colon_positions.append(i)
                    
                    if colon_positions:
                        last_colon_idx = colon_positions[-1]
                        before_last_colon = part[:last_colon_idx].strip()
                        after_last_colon = part[last_colon_idx + 1:].strip()
                        
                        if before_last_colon == after_last_colon:
                            entity_name = after_last_colon
                        elif after_last_colon.startswith(before_last_colon):
                            entity_name = after_last_colon
                        elif len(colon_positions) > 1:
                            first_colon_idx = colon_positions[0]
                            before_first_colon = part[:first_colon_idx].strip()
                            if len(before_first_colon) < 30 and ' ' in before_first_colon:
                                entity_name = after_last_colon
                            else:
                                entity_name = after_last_colon
                        else:
                            entity_name = after_last_colon
                    else:
                        entity_name = part
            else:
                continue
        
        if entity_id and entity_name:
            cleaned_ename = f"{entity_id} = {entity_name}"
            cleaned_enames.append(cleaned_ename)
    
    return cleaned_enames


def split_edges_by_colon_in_action(edges: List[str]) -> List[str]:
    split_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            split_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        direction = direction.strip()
        timestamp = timestamp.strip()
        action = action.strip()
        
        if ':' in action:
            action_parts = action.split(':')
            for action_part in action_parts:
                action_part = action_part.strip()
                if action_part:
                    split_edge = f"({source}, {dest}) A: [{action_part}] {{D={direction}}} ({timestamp})"
                    split_edges.append(split_edge)
        else:
            split_edges.append(edge)
    
    return split_edges


def split_multi_action_edges(edges: List[str]) -> List[str]:
    split_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            split_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        direction = direction.strip()
        timestamp = timestamp.strip()
        
        action_lines = []
        
        if '\n' in action:
            lines = action.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    action_lines.append(line)
        elif '\t' in action or re.search(r' {3,}', action):
            parts = re.split(r'\t+| {3,}', action)
            for part in parts:
                part = part.strip()
                if part:
                    action_lines.append(part)
        else:
            split_edges.append(edge)
            continue
        
        for action_line in action_lines:
            colon_with_gap_pattern = r':(\t+| {3,})'
            match = re.search(colon_with_gap_pattern, action_line)
            
            if match:
                parts = re.split(colon_with_gap_pattern, action_line, maxsplit=1)
                
                if len(parts) >= 3:
                    before_colon = parts[0].strip()
                    if before_colon:
                        if not before_colon.endswith(':'):
                            before_colon += ':'
                        split_edges.append(f"({source}, {dest}) A: [{before_colon}] {{D={direction}}} ({timestamp})")
                    else:
                        split_edges.append(f"({source}, {dest}) A: [:] {{D={direction}}} ({timestamp})")
                    
                    after_gap = parts[2].strip()
                    if after_gap:
                        split_edges.append(f"({source}, {dest}) A: [{after_gap}] {{D={direction}}} ({timestamp})")
                else:
                    split_edges.append(f"({source}, {dest}) A: [{action_line}] {{D={direction}}} ({timestamp})")
            else:
                split_edges.append(f"({source}, {dest}) A: [{action_line}] {{D={direction}}} ({timestamp})")
    
    return split_edges


def filter_edges_with_invalid_timestamps(edges: List[str]) -> List[str]:
    filtered_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    invalid_timestamp_patterns = ['PM', 'AM', 'pm', 'am']
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            filtered_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        timestamp = timestamp.strip()
        
        if timestamp.startswith('timestamp='):
            timestamp_value = timestamp[len('timestamp='):].strip()
        else:
            timestamp_value = timestamp
        
        if timestamp_value.upper() in ['PM', 'AM']:
            continue
        
        if len(timestamp_value) < 5 and timestamp_value != '...':
            continue
        
        if timestamp_value == '...':
            filtered_edges.append(edge)
            continue
        
        if not re.search(r'\d', timestamp_value):
            continue
        
        filtered_edges.append(edge)
    
    return filtered_edges


def filter_invalid_action_edges(edges: List[str]) -> List[str]:
    filtered_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]*)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            filtered_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        action = action.strip()
        
        if not action or action == '':
            continue
        
        if len(action) < 3:
            continue
        
        if action.isdigit():
            continue
        
        filtered_edges.append(edge)
    
    return filtered_edges


def clean_dns_prefixes_from_edges(edges: List[str]) -> List[str]:
    cleaned_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            cleaned_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        action = action.strip()
        direction = direction.strip()
        timestamp = timestamp.strip()
        
        if source.startswith('A:'):
            source = source[2:].strip()
        elif source.startswith('AAAA:'):
            source = source[5:].strip()
        
        if dest.startswith('A:'):
            dest = dest[2:].strip()
        elif dest.startswith('AAAA:'):
            dest = dest[5:].strip()
        
        cleaned_edge = f"({source}, {dest}) A: [{action}] {{D={direction}}} ({timestamp})"
        cleaned_edges.append(cleaned_edge)
    
    return cleaned_edges


def remove_integers_from_actions(edges: List[str]) -> List[str]:
    cleaned_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            cleaned_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        direction = direction.strip()
        timestamp = timestamp.strip()
        action = action.strip()
        
        action_parts = action.split()
        cleaned_parts = []
        for part in action_parts:
            if not part.isdigit():
                cleaned_parts.append(part)
        
        cleaned_action = ' '.join(cleaned_parts).strip()
        
        if not cleaned_action:
            continue
        
        cleaned_edge = f"({source}, {dest}) A: [{cleaned_action}] {{D={direction}}} ({timestamp})"
        cleaned_edges.append(cleaned_edge)
    
    return cleaned_edges


def filter_edges_without_english_in_action(edges: List[str]) -> List[str]:
    filtered_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]*)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            filtered_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        action = action.strip()
        
        if not re.search(r'[a-zA-Z]', action):
            continue
        
        filtered_edges.append(edge)
    
    return filtered_edges


def clean_trailing_colons_from_source_dest(edges: List[str]) -> List[str]:
    cleaned_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            cleaned_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        action = action.strip()
        direction = direction.strip()
        timestamp = timestamp.strip()
        
        source = source.rstrip(':')
        dest = dest.rstrip(':')
        
        cleaned_edge = f"({source}, {dest}) A: [{action}] {{D={direction}}} ({timestamp})"
        cleaned_edges.append(cleaned_edge)
    
    return cleaned_edges


def filter_self_loop_edges(edges: List[str]) -> List[str]:
    filtered_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            filtered_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip().rstrip(':')
        dest = dest.strip().rstrip(':')
        
        if source == dest:
            continue
        
        filtered_edges.append(edge)
    
    return filtered_edges


def filter_edges_with_integer_source_dest(edges: List[str]) -> List[str]:
    filtered_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            filtered_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip().rstrip(':')
        dest = dest.strip().rstrip(':')
        
        if source.isdigit():
            continue
        
        if dest.isdigit():
            continue
        
        filtered_edges.append(edge)
    
    return filtered_edges


def filter_edges_with_empty_source_dest(edges: List[str]) -> List[str]:
    filtered_edges = []
    
    edge_pattern = re.compile(r'\(([^,]*),\s*([^)]*)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            filtered_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        
        if not source or not dest:
            continue
        
        filtered_edges.append(edge)
    
    return filtered_edges


def filter_edges_with_similar_source_dest(edges: List[str]) -> List[str]:
    filtered_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    def differ_by_one_char(s1: str, s2: str) -> bool:
        if s1 == s2:
            return False
        
        len1, len2 = len(s1), len(s2)
        
        if abs(len1 - len2) > 1:
            return False
        
        if len1 == len2:
            differences = sum(1 for a, b in zip(s1, s2) if a != b)
            return differences == 1
        
        if len1 == len2 + 1:
            for i in range(len1):
                if s1[:i] + s1[i+1:] == s2:
                    return True
            return False
        else:
            for i in range(len2):
                if s2[:i] + s2[i+1:] == s1:
                    return True
            return False
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            filtered_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        
        if differ_by_one_char(source, dest):
            continue
        
        filtered_edges.append(edge)
    
    return filtered_edges


def filter_edges_where_action_matches_source_dest(edges: List[str]) -> List[str]:
    if not edges:
        return edges
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    all_sources_dests = set()
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if match:
            source = match.group(1).strip()
            dest = match.group(2).strip()
            all_sources_dests.add(source)
            all_sources_dests.add(dest)
    
    filtered_edges = []
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            filtered_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        action = action.strip()
        
        if action in all_sources_dests:
            continue
        
        filtered_edges.append(edge)
    
    return filtered_edges


def filter_edges_with_single_char_source_dest(edges: List[str]) -> List[str]:
    filtered_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            filtered_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        
        if len(source) == 1 or len(dest) == 1:
            continue
        
        filtered_edges.append(edge)
    
    return filtered_edges


def clean_extra_brackets_from_actions(edges: List[str]) -> List[str]:
    cleaned_edges = []
    
    action_pattern = re.compile(r'A:\s*\[(.*?)\]\s*\{D=', re.DOTALL)
    
    for edge in edges:
        action_match = action_pattern.search(edge)
        if not action_match:
            cleaned_edges.append(edge)
            continue
        
        action_content = action_match.group(1)
        
        cleaned_action = action_content.strip()
        while cleaned_action.startswith('[') and cleaned_action.endswith(']'):
            inner = cleaned_action[1:-1].strip()
            if inner:
                cleaned_action = inner
            else:
                break
        
        start_pos = action_match.start()
        end_pos = action_match.end()
        
        before_action = edge[:start_pos]
        after_action = edge[end_pos:]
        cleaned_edge = f"{before_action}A: [{cleaned_action}] {{D={after_action}"
        cleaned_edges.append(cleaned_edge)
    
    return cleaned_edges


def clean_uri_prefix_from_edges(edges: List[str]) -> List[str]:
    cleaned_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            cleaned_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        action = action.strip()
        direction = direction.strip()
        timestamp = timestamp.strip()
        
        if source.startswith('uri='):
            source = source[4:].strip()
        
        if dest.startswith('uri='):
            dest = dest[4:].strip()
        
        cleaned_edge = f"({source}, {dest}) A: [{action}] {{D={direction}}} ({timestamp})"
        cleaned_edges.append(cleaned_edge)
    
    return cleaned_edges


def clean_http_actions(edges: List[str]) -> List[str]:
    cleaned_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            cleaned_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        direction = direction.strip()
        timestamp = timestamp.strip()
        action = action.strip()
        
        if 'http' in action.lower():
            action = 'http'
        
        cleaned_edge = f"({source}, {dest}) A: [{action}] {{D={direction}}} ({timestamp})"
        cleaned_edges.append(cleaned_edge)
    
    return cleaned_edges


def normalize_string_for_grouping(s: str) -> str:
    s = s.strip()
    
    has_english = bool(re.search(r'[a-zA-Z]', s))
    
    if not has_english:
        return s
    
    while s and s[0] in [':', '/', '\\']:
        s = s[1:].strip()
    
    while s and s[-1] in [':', '/', '\\']:
        s = s[:-1].strip()
    
    s = s.replace('://', '')
    
    s = s.strip()
    
    return s


def merge_actions_for_same_source_dest(edges: List[str], use_smart_normalization: bool = False) -> List[str]:
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    edge_groups = {}
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            if "UNPARSEABLE" not in edge_groups:
                edge_groups["UNPARSEABLE"] = []
            edge_groups["UNPARSEABLE"].append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source_original = source.strip().rstrip(':')
        dest_original = dest.strip().rstrip(':')
        
        if use_smart_normalization:
            source_has_english = bool(re.search(r'[a-zA-Z]', source_original))
            dest_has_english = bool(re.search(r'[a-zA-Z]', dest_original))
            
            if source_has_english and dest_has_english:
                source_key = normalize_string_for_grouping(source_original)
                dest_key = normalize_string_for_grouping(dest_original)
            else:
                source_key = source_original
                dest_key = dest_original
        else:
            source_key = source_original
            dest_key = dest_original
        
        key = (source_key, dest_key)
        
        if key not in edge_groups:
            edge_groups[key] = []
        edge_groups[key].append({
            'source': source_original,
            'dest': dest_original,
            'action': action.strip(),
            'direction': direction.strip(),
            'timestamp': timestamp.strip()
        })
    
    merged_edges = []
    
    for key, group_edges in edge_groups.items():
        if key == "UNPARSEABLE":
            merged_edges.extend(group_edges)
            continue
        
        if len(group_edges) == 1:
            edge_data = group_edges[0]
            merged_edge = f"({edge_data['source']}, {edge_data['dest']}) A: [{edge_data['action']}] {{D={edge_data['direction']}}} ({edge_data['timestamp']})"
            merged_edges.append(merged_edge)
        else:
            all_words = []
            
            for edge_data in group_edges:
                action = edge_data['action']
                words = action.split()
                all_words.extend(words)
            
            unique_words = sorted(set(all_words), key=str.lower)
            
            merged_action = ' '.join(word.lower() for word in unique_words)
            
            first_edge = group_edges[0]
            merged_edge = f"({first_edge['source']}, {first_edge['dest']}) A: [{merged_action}] {{D={first_edge['direction']}}} ({first_edge['timestamp']})"
            merged_edges.append(merged_edge)
    
    return merged_edges


def clean_timestamps_in_edges(edges: List[str]) -> List[str]:
    cleaned_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            cleaned_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        action = action.strip()
        direction = direction.strip()
        timestamp = timestamp.strip()
        
        if timestamp.startswith('timestamp='):
            timestamp_value = timestamp[len('timestamp='):].strip()
        else:
            timestamp_value = timestamp
        
        timestamp_parts = timestamp_value.split()
        
        if len(timestamp_parts) == 3:
            cleaned_parts = timestamp_parts[1:]
            cleaned_timestamp = ' '.join(cleaned_parts)
            if cleaned_timestamp and cleaned_timestamp[0] == ':':
                cleaned_timestamp = cleaned_timestamp[1:].strip()
        elif len(timestamp_parts) == 2:
            first_part = timestamp_parts[0]
            if ':' in first_part:
                colon_idx = first_part.rfind(':')
                if colon_idx >= 0:
                    first_part = first_part[colon_idx + 1:]
                cleaned_timestamp = ' '.join([first_part, timestamp_parts[1]]).strip()
            else:
                cleaned_timestamp = ' '.join(timestamp_parts)
        else:
            cleaned_timestamp = timestamp_value
        
        cleaned_edge = f"({source}, {dest}) A: [{action}] {{D={direction}}} (timestamp={cleaned_timestamp})"
        cleaned_edges.append(cleaned_edge)
    
    return cleaned_edges


def remove_trailing_colons_from_actions(edges: List[str]) -> List[str]:
    cleaned_edges = []
    
    edge_pattern = re.compile(r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]*)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)')
    
    for edge in edges:
        match = edge_pattern.match(edge.strip())
        if not match:
            cleaned_edges.append(edge)
            continue
        
        source, dest, action, direction, timestamp = match.groups()
        source = source.strip()
        dest = dest.strip()
        direction = direction.strip()
        timestamp = timestamp.strip()
        action = action.strip()
        
        while action and action[0] in [':', '"', '.']:
            action = action[1:].strip()
        
        while action and action[-1] in [':', '"', '.']:
            action = action[:-1].strip()
        
        cleaned_edge = f"({source}, {dest}) A: [{action}] {{D={direction}}} ({timestamp})"
        cleaned_edges.append(cleaned_edge)
    
    return cleaned_edges


def apply_ename_patterns_to_log(log_text: str, compiled_patterns: List[dict], verbose: bool = False, max_patterns: int = None, timeout_seconds: float = 0.5) -> List[str]:
    processed_log_text = log_text.replace('\\"', '"')
    
    all_extracted_results = []
    
    patterns_to_use = compiled_patterns[:max_patterns] if max_patterns else compiled_patterns
    total_patterns = len(patterns_to_use)
    patterns_tried = 0
    
    for pattern_dict in patterns_to_use:
        patterns_tried += 1
        
        id_pattern = pattern_dict.get('ID')
        name_pattern = pattern_dict.get('Name')
        
        if not id_pattern:
            continue
        
        try:
            if isinstance(id_pattern, tuple):
                id_match = apply_tuple_regex(id_pattern, processed_log_text, timeout_seconds)
            else:
                id_match = apply_single_regex(id_pattern, processed_log_text, timeout_seconds)
            
            if id_match:
                if name_pattern and name_pattern != 'No Regex':
                    try:
                        if isinstance(name_pattern, tuple):
                            name_match = apply_tuple_regex(name_pattern, processed_log_text, timeout_seconds)
                        else:
                            name_match = apply_single_regex(name_pattern, processed_log_text, timeout_seconds)
                        
                        if name_match:
                            extraction = f"{id_match} = {name_match}"
                            all_extracted_results.append(extraction)
                    except (TimeoutError, Exception):
                        pass
        except (TimeoutError, Exception):
            continue
    
    
    seen = set()
    unique_enames = []
    for ename in all_extracted_results:
        if ename not in seen and ename != "No Regex":
            seen.add(ename)
            unique_enames.append(ename)
    
    return unique_enames


EDGE_PARSE_PATTERN = re.compile(
    r'\(([^,]+),\s*([^)]+)\)\s*A:\s*\[([^\]]+)\]\s*\{D=([^}]+)\}\s*\(([^)]+)\)'
)


def group_edges_by_source_dest(edges: List[str]) -> List[Tuple[Tuple[str, str], List[str]]]:
    edge_groups = {}
    
    for edge in edges:
        match = EDGE_PARSE_PATTERN.match(edge.strip())
        if not match:
            source = "UNKNOWN"
            dest = "UNKNOWN"
        else:
            source, dest, action, direction, timestamp = match.groups()
            source = source.strip().rstrip(':')
            dest = dest.strip().rstrip(':')
        
        key = (source, dest)
        
        if key not in edge_groups:
            edge_groups[key] = []
        edge_groups[key].append(edge)
    
    sorted_groups = sorted(edge_groups.items(), key=lambda x: (x[0][0], x[0][1]))
    
    return sorted_groups


def apply_patterns_to_log(log_text: str, compiled_patterns: List[dict], verbose: bool = False, max_patterns: int = None, timeout_seconds: float = 0.5) -> List[str]:
    extracted_edges = []

    specific_patterns, generic_patterns = categorize_patterns(compiled_patterns)
    
    if max_patterns:
        specific_patterns = specific_patterns[:max_patterns]
        generic_patterns = []
    
    total_patterns = len(specific_patterns) + len(generic_patterns)
    patterns_tried = 0
    
    for idx, pattern_dict in enumerate(specific_patterns):
        patterns_tried += 1
        try:
            success, extracted_data = test_single_pattern(pattern_dict, log_text, timeout_seconds)
            if success:
                if isinstance(extracted_data, list):
                    for edge_data in extracted_data:
                        edge_str = format_edge_output(edge_data)
                        extracted_edges.append(edge_str)
                else:
                    edge_str = format_edge_output(extracted_data)
                    extracted_edges.append(edge_str)
        except (TimeoutError, Exception):
            continue
    
    for idx, pattern_dict in enumerate(generic_patterns):
        patterns_tried += 1
        try:
            success, extracted_data = test_single_pattern(pattern_dict, log_text, timeout_seconds)
            if success:
                if isinstance(extracted_data, list):
                    for edge_data in extracted_data:
                        edge_str = format_edge_output(edge_data)
                        extracted_edges.append(edge_str)
                else:
                    edge_str = format_edge_output(extracted_data)
                    extracted_edges.append(edge_str)
        except (TimeoutError, Exception):
            continue
    
    seen = set()
    unique_edges = []
    for edge in extracted_edges:
        if edge not in seen:
            seen.add(edge)
            unique_edges.append(edge)
    
    processed_edges = split_multi_action_edges(unique_edges)
    
    cleaned_edges = remove_trailing_colons_from_actions(processed_edges)
    
    filtered_edges = filter_invalid_action_edges(cleaned_edges)
    
    return filtered_edges


def get_completed_logs_from_edges_file(edges_file_path: str) -> set:
    completed_logs = set()
    
    if not os.path.exists(edges_file_path):
        return completed_logs
    
    try:
        log_pattern = re.compile(r'^LOG (\d+)$')
        
        with open(edges_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                match = log_pattern.match(line)
                if match:
                    try:
                        log_num = int(match.group(1))
                        completed_logs.add(log_num)
                    except ValueError:
                        continue
                    
    except Exception as e:
        return set()
    
    return completed_logs


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


def parse_args():
    parser = argparse.ArgumentParser(description="Apply master patterns to ATLAS dataset logs (Ablation version)")
    parser.add_argument("--cee", type=str, required=True,
                       help="Candidate Edge Extractor name (e.g., gpt-3.5-turbo, llama3:70b)")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name used for rule generation (e.g., llama3:70b)")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name")
    parser.add_argument("--log-type", type=str, required=True,
                       help="Log type (default: audit)")
    parser.add_argument("--rules-dir", type=str, default="./rules", 
                       help="Directory containing rules (default: ./rules)")
    parser.add_argument("--atlas-dir", type=str, required=True,
                       help="Directory containing ATLAS datasets")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to store results")
    parser.add_argument("--max-logs", type=int, default=None,
                       help="Maximum number of logs to process per timestamp (default: all)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed progress for each log")
    parser.add_argument("--max-patterns", type=int, default=None,
                       help="Maximum number of patterns to try per log (for testing, default: all)")
    parser.add_argument("--timeout", type=float, default=0.5,
                       help="Timeout in seconds for each pattern match (default: 0.5)")
    parser.add_argument("--log-characteristics", type=str, default=None,
                       help="Path to log_idcharacteristics.json (default: auto-detect)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    folder_name = f"{args.cee.lower()}_{args.model_name.lower()}"
    folder_name_sanitized = sanitize_cee_name(folder_name)
    
    log_characteristics = load_log_characteristics(args.log_characteristics, cee=args.cee)
    has_builtin_ids = log_characteristics.get(args.log_type, "unknown")
    
    dataset_upper = args.dataset.upper()
    embedding = "roberta"
    
    edge_patterns_path = os.path.join(args.rules_dir, "ATLAS", embedding, folder_name_sanitized, args.log_type, 
                                      "edges", "master_patterns_edges.pkl")
    
    if not os.path.exists(edge_patterns_path):
        return
    
    with open(edge_patterns_path, 'rb') as f:
        raw_edge_patterns = pickle.load(f)
    
    compiled_edge_patterns = compile_patterns(raw_edge_patterns)
    
    ename_patterns_path = os.path.join(args.rules_dir, "ATLAS", embedding, folder_name_sanitized, args.log_type, 
                                       "enames", "master_patterns_enames.pkl")
    
    compiled_ename_patterns = []
    if os.path.exists(ename_patterns_path):
        try:
            with open(ename_patterns_path, 'rb') as f:
                raw_ename_patterns = pickle.load(f)
            
            if raw_ename_patterns:
                compiled_ename_patterns = compile_patterns(raw_ename_patterns)
        except Exception:
            pass
    
    dataset_path = os.path.join(args.atlas_dir, dataset_upper, args.log_type)
    
    if not os.path.exists(dataset_path):
        return
    
    timestamp_folders = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            pkl_file = os.path.join(item_path, f"{args.log_type}.pkl")
            if os.path.exists(pkl_file):
                timestamp_folders.append(item)
    
    if not timestamp_folders:
        return
    
    timestamp_folders.sort()
    
    dataset_output_root = os.path.join(args.output_dir, folder_name_sanitized, dataset_upper)
    fully_completed_timestamps = []
    partially_completed_timestamps = []
    
    for timestamp_folder in timestamp_folders:
        output_dir = os.path.join(dataset_output_root, args.log_type, timestamp_folder)
        edge_output_path = os.path.join(output_dir, "edges.txt")
        
        if os.path.exists(edge_output_path):
            completed_logs = get_completed_logs_from_edges_file(edge_output_path)
            if len(completed_logs) > 0:
                timestamp_path = os.path.join(dataset_path, timestamp_folder)
                pkl_file = os.path.join(timestamp_path, f"{args.log_type}.pkl")
                try:
                    with open(pkl_file, 'rb') as f:
                        logs = pickle.load(f)
                    if isinstance(logs, list):
                        total_logs = len(logs[:args.max_logs]) if args.max_logs else len(logs)
                        if len(completed_logs) >= total_logs:
                            fully_completed_timestamps.append(timestamp_folder)
                        else:
                            partially_completed_timestamps.append((timestamp_folder, len(completed_logs), total_logs))
                except:
                    partially_completed_timestamps.append((timestamp_folder, len(completed_logs), None))
    
    
    total_processed = 0
    total_timestamps = len(timestamp_folders)
    
    for timestamp_idx, timestamp_folder in enumerate(timestamp_folders, 1):
        if timestamp_folder in fully_completed_timestamps:
            continue
        
        timestamp_path = os.path.join(dataset_path, timestamp_folder)
        pkl_file = os.path.join(timestamp_path, f"{args.log_type}.pkl")
        
        try:
            with open(pkl_file, 'rb') as f:
                logs = pickle.load(f)
        except Exception as e:
            continue
        
        if not isinstance(logs, list):
            continue
        
        logs_to_process = logs[:args.max_logs] if args.max_logs else logs
        
        dataset_output_root = os.path.join(args.output_dir, folder_name_sanitized, dataset_upper)
        output_dir = os.path.join(dataset_output_root, args.log_type, timestamp_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        edge_output_path = os.path.join(output_dir, "edges.txt")
        csv_path = os.path.join(output_dir, "provgraph.csv")
        completed_logs = get_completed_logs_from_edges_file(edge_output_path)
        
        edge_file_exists = os.path.exists(edge_output_path) and len(completed_logs) > 0
        csv_file_exists = os.path.exists(csv_path) and len(completed_logs) > 0
        
        edge_file_mode = 'a' if edge_file_exists else 'w'
        csv_file_mode = 'a' if csv_file_exists else 'w'
        
        edge_file = open(edge_output_path, edge_file_mode, encoding='utf-8')
        csv_file = open(csv_path, csv_file_mode, encoding='utf-8', newline='')
        csv_writer = csv.writer(csv_file)
        
        if not csv_file_exists:
            if has_builtin_ids == "no":
                csv_writer.writerow(["log_idx", "source_name", "dest_name", "action", "timestamp"])
            else:
                csv_writer.writerow(["log_idx", "source_id", "source_name", "dest_id", "dest_name", "action", "timestamp"])
        
        ename_file = None
        if has_builtin_ids != "no":
            ename_output_path = os.path.join(output_dir, "enames.txt")
            ename_file_exists = os.path.exists(ename_output_path) and len(completed_logs) > 0
            ename_file_mode = 'a' if ename_file_exists else 'w'
            ename_file = open(ename_output_path, ename_file_mode, encoding='utf-8')
        
        csv_rows_last_printed = 0
        csv_row_count = 0
        
        logs_to_actually_process = []
        for log_idx, log in enumerate(logs_to_process, 1):
            if log_idx not in completed_logs:
                logs_to_actually_process.append((log_idx, log))
        
        if not logs_to_actually_process:
            edge_file.close()
            csv_file.close()
            if ename_file:
                ename_file.close()
            continue
        
        try:
            for log_idx, input_log in tqdm(logs_to_actually_process, desc=f"[{timestamp_idx}/{total_timestamps}] {timestamp_folder[:40]}", leave=True):
                if not isinstance(input_log, str):
                    input_log = str(input_log)
                
                extracted_edges = apply_patterns_to_log(input_log, compiled_edge_patterns, verbose=False, max_patterns=args.max_patterns, timeout_seconds=args.timeout)
                
                extracted_edges = filter_edges_with_invalid_timestamps(extracted_edges)
                
                extracted_edges = split_edges_by_colon_in_action(extracted_edges)
                
                extracted_edges = filter_edges_without_english_in_action(extracted_edges)
                
                extracted_edges = remove_integers_from_actions(extracted_edges)
                
                extracted_edges = clean_dns_prefixes_from_edges(extracted_edges)
                
                extracted_edges = clean_trailing_colons_from_source_dest(extracted_edges)
                
                extracted_edges = clean_timestamps_in_edges(extracted_edges)
                
                extracted_edges = clean_extra_brackets_from_actions(extracted_edges)
                
                extracted_edges = clean_uri_prefix_from_edges(extracted_edges)
                
                if has_builtin_ids == "no":
                    extracted_edges = clean_http_actions(extracted_edges)
                
                seen_edges = set()
                unique_edges_for_log = []
                for edge in extracted_edges:
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        unique_edges_for_log.append(edge)
                
                unique_edges_for_log = filter_edges_with_empty_source_dest(unique_edges_for_log)
                
                unique_edges_for_log = filter_edges_with_integer_source_dest(unique_edges_for_log)
                
                if args.log_type == "dns":
                    unique_edges_for_log = merge_actions_for_same_source_dest(unique_edges_for_log, use_smart_normalization=False)
                elif args.log_type == "firefox":
                    unique_edges_for_log = merge_actions_for_same_source_dest(unique_edges_for_log, use_smart_normalization=True)
                
                unique_edges_for_log = filter_self_loop_edges(unique_edges_for_log)
                
                extracted_enames = []
                if has_builtin_ids != "no" and compiled_ename_patterns:
                    try:
                        extracted_enames = apply_ename_patterns_to_log(input_log, compiled_ename_patterns, verbose=False, max_patterns=args.max_patterns, timeout_seconds=args.timeout)
                        extracted_enames = preprocess_enames(extracted_enames)
                        extracted_enames = filter_single_character_enames(extracted_enames)
                        extracted_enames = remove_trailing_quotes_from_enames(extracted_enames)
                        seen_enames = set()
                        unique_enames = []
                        for ename in extracted_enames:
                            if ename not in seen_enames:
                                seen_enames.add(ename)
                                unique_enames.append(ename)
                        extracted_enames = unique_enames
                    except Exception:
                        extracted_enames = []
                
                ename_map = {}
                if has_builtin_ids != "no":
                    for ename in extracted_enames:
                        if " = " in ename:
                            parts = ename.split(" = ", 1)
                            if len(parts) == 2:
                                ename_map[parts[0].strip()] = parts[1].strip()
                    
                    unique_edges_for_log = filter_edges_with_ename_matches(unique_edges_for_log, extracted_enames)

                unique_edges_for_log = filter_edges_with_timestamp_as_action(unique_edges_for_log)
                
                unique_edges_for_log = filter_invalid_action_edges(unique_edges_for_log)
                
                unique_edges_for_log = filter_edges_with_similar_source_dest(unique_edges_for_log)
                
                unique_edges_for_log = filter_edges_with_single_char_source_dest(unique_edges_for_log)
                
                unique_edges_for_log = filter_edges_where_action_matches_source_dest(unique_edges_for_log)

                for edge in unique_edges_for_log:
                    match = EDGE_PARSE_PATTERN.match(edge)
                    if not match:
                        continue
                    src, dest, action, direction, timestamp = match.groups()
                    src = src.strip()
                    dest = dest.strip()
                    action = action.strip()
                    timestamp = timestamp.strip()
                    if timestamp.startswith('timestamp='):
                        timestamp = timestamp[len('timestamp='):].strip()
                    if timestamp.endswith(':PM'):
                        timestamp = timestamp[:-3].strip()
                    elif timestamp.endswith(':AM'):
                        timestamp = timestamp[:-3].strip()
                    
                    if has_builtin_ids == "no":
                        if '<-' in direction:
                            src, dest = dest, src
                        csv_writer.writerow([log_idx, src, dest, action, timestamp])
                    else:
                        src_name = ename_map.get(src, "")
                        dest_name = ename_map.get(dest, "")
                        if '<-' in direction:
                            src, dest = dest, src
                            src_name, dest_name = dest_name, src_name
                        csv_writer.writerow([log_idx, src, src_name, dest, dest_name, action, timestamp])
                    csv_row_count += 1

                    if args.verbose and csv_row_count - csv_rows_last_printed >= 100:
                        csv_rows_last_printed = csv_row_count

                edge_file.write(f"{'='*70}\n")
                edge_file.write(f"LOG {log_idx}\n")
                edge_file.write(f"{'='*70}\n")
                edge_file.write(f"\nINPUT LOG:\n{input_log}\n")
                edge_file.write(f"\nEXTRACTED EDGES ({len(unique_edges_for_log)}):\n")
                if unique_edges_for_log:
                    grouped_edges = group_edges_by_source_dest(unique_edges_for_log)
                    
                    for (source, dest), edges_list in grouped_edges:
                        for edge in edges_list:
                            edge_file.write(edge + "\n")
                else:
                    edge_file.write("(No edges extracted)\n")
                edge_file.write("\n")
                edge_file.flush()
                
                if ename_file is not None:
                    ename_file.write(f"{'='*70}\n")
                    ename_file.write(f"LOG {log_idx}\n")
                    ename_file.write(f"{'='*70}\n")
                    ename_file.write(f"\nINPUT LOG:\n{input_log}\n")
                    ename_file.write(f"\nEXTRACTED ENAMES ({len(extracted_enames)}):\n")
                    if extracted_enames:
                        for ename in extracted_enames:
                            ename_file.write(ename + "\n")
                    else:
                        ename_file.write("(No enames extracted)\n")
                    ename_file.write("\n")
                    ename_file.flush()
                
                csv_file.flush()
        finally:
            edge_file.close()
            if ename_file is not None:
                ename_file.close()
            csv_file.close()
        
        total_processed += len(logs_to_process)
    


if __name__ == "__main__":
    main()

