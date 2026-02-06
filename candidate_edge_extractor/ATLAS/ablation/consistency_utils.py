from collections import defaultdict, Counter
import re


def filter_process_dict(process_dict):
    combined_entries = ''.join(process_dict.values())
    name_frequencies = {}
    for entry in combined_entries.strip().split('\n'):
        parts = entry.strip().split('=', 1)
        if len(parts) == 2:
            pid = parts[0].strip().strip('"')
            process_name = parts[1].strip().strip('"')
            if pid not in name_frequencies:
                name_frequencies[pid] = {}
            name_frequencies[pid][process_name] = name_frequencies[pid].get(process_name, 0) + 1
    filtered_dict = {}
    for id, process_name_freq in name_frequencies.items():
        most_frequent_name = max(process_name_freq, key=process_name_freq.get)
        filtered_dict[id] = [most_frequent_name]
    return filtered_dict


def get_consistent_vanetype(iterations):
    id_value_counts = defaultdict(lambda: Counter())
    only_none_ids = set()
    for iteration in iterations:
        pairs = [line.strip() for line in iteration.strip().split('\n') if '=' in line]
        for pair in pairs:
            key, value = map(str.strip, pair.split('=', 1))
            value = value.lower()
            clean_key = key.replace('"', '').replace("'", '').replace("{", '').replace("}", '').replace(")", '').replace("()", '')
            clean_value = value.replace('"', '').replace("'", '').replace("{", '')
            if clean_value == "NONE":
                if clean_key not in id_value_counts:
                    only_none_ids.add(clean_key)
            else:
                only_none_ids.discard(clean_key)
                id_value_counts[clean_key][clean_value] += 1
    consistent_output = {}
    for key in set(id_value_counts.keys()).union(only_none_ids):
        if key in only_none_ids:
            consistent_output[key] = "NONE"
        else:
            most_common = id_value_counts[key].most_common()
            max_count = most_common[0][1]
            top_values = [val.lower() for val, count in most_common if count == max_count]
            consistent_output[key] = top_values[0]
    return consistent_output


def handle_direction_tie(action_frequencies, window_no):
    pass


def extract_consistent_graph(responses_graph, window_no):
    edge_frequencies = {}
    for iteration, graph_str in responses_graph.items():
        lines = graph_str.strip().split('\n\n')
        header = lines[0]
        edges = lines[1:]
        for edge_line in edges:
            parts = edge_line.split(' A:')
            endpoints = parts[0].strip().strip('()')
            if len(endpoints.split(',', 1)) > 1:
                src, dst = [x.strip() for x in endpoints.split(',', 1)]
                action = parts[1].split('{')[0].strip()
                if '(timestamp=' in edge_line:
                    timestamp = edge_line.split('(timestamp=')[1].strip(')')
                direction = '{D=->}' in edge_line
                edge_key = (sorted([src, dst])[0], sorted([src, dst])[1], timestamp)
                if edge_key not in edge_frequencies:
                    edge_frequencies[edge_key] = {}
                action_direction = (direction if src == sorted([src, dst])[0] else not direction)
                if action not in edge_frequencies[edge_key]:
                    edge_frequencies[edge_key][action] = {True: 0, False: 0}
                edge_frequencies[edge_key][action][action_direction] += 1
    consistent_edges = []
    for (src, dst, timestamp), actions_dict in edge_frequencies.items():
        best_action = max(
            actions_dict.keys(),
            key=lambda a: sum(actions_dict[a].values())
        )
        action_frequencies = actions_dict[best_action]
        forward_count = action_frequencies[True]
        reverse_count = action_frequencies[False]
        if forward_count > reverse_count:
            preferred_direction = True
        elif reverse_count > forward_count:
            preferred_direction = False
        else:
            preferred_direction = handle_direction_tie(action_frequencies, window_no)
        edge_str = f"({src}, {dst}) A: {best_action} {{D={'->' if preferred_direction else '<-'}}} (timestamp={timestamp})"
        consistent_edges.append(edge_str)
    final_graph = '\n\n'.join(consistent_edges)
    return final_graph


def extract_final_output(response):
    f_output = ''
    status = 0
    for line in response.split('\n'):
        if '### Final Output' in line:
            status = 1
        if status == 1:
            if '=' in line:
                f_output += line + '\n'
    return f_output


def extract_final_desc(response):
    f_output = ''
    status = 0
    for line in response.split('\n'):
        if '### Final Output' in line:
            status = 1
        if status == 1:
            if len(line) > 1 and '### Final Output' not in line:
                f_output += line + '\n'
    return f_output


def extract_final_pairs(response):
    f_output = ''
    status = 0
    for line in response.split('\n'):
        if '### final output' in line.lower():
            status = 1
        if status == 1:
            if 'a: [...]' in line.lower():
                f_output += line + '\n'
    return f_output


def extract_last_entity_types_block(text):
    matches = list(re.finditer(r"\[ENTITY TYPES\](?:\n[^\[]+)+", text, re.MULTILINE))
    return matches[-1].group(0) if matches else None
