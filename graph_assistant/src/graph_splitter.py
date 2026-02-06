from collections import defaultdict, deque


def build_adjacency_list(edges):
    adjacency = defaultdict(set)
    all_nodes = set()
    
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source and target:
            adjacency[source].add(target)
            adjacency[target].add(source)
            all_nodes.add(source)
            all_nodes.add(target)
    
    return adjacency, all_nodes


def split_graph_by_node_count(edges, max_nodes, return_metadata=False):
    if not edges or max_nodes is None:
        if return_metadata:
            return [edges], {
                'node_to_subgraph': {},
                'subgraph_nodes': {0: set()},
                'frontier_nodes': {0: {'incoming': [], 'outgoing': []}},
                'connections': {}
            }
        return [edges]
    
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    
    adjacency, all_nodes = build_adjacency_list(edges)
    
    if len(all_nodes) <= max_nodes:
        if return_metadata:
            return [edges], {
                'node_to_subgraph': {node: 0 for node in all_nodes},
                'subgraph_nodes': {0: all_nodes},
                'frontier_nodes': {0: {'incoming': [], 'outgoing': []}},
                'connections': {}
            }
        return [edges]
    
    subgraphs = []
    visited = set()
    remaining_nodes = all_nodes.copy()
    
    node_to_subgraph = {}
    subgraph_nodes = {}
    
    frontier_nodes = set()
    
    frontier_metadata = {}
    connections = {}
    
    subgraph_idx = 0
    
    while remaining_nodes:
        if frontier_nodes:
            start_node = next(iter(frontier_nodes))
            frontier_nodes.remove(start_node)
        else:
            start_node = next(iter(remaining_nodes))
        
        remaining_nodes.remove(start_node)
        visited.add(start_node)
        
        current_subgraph_nodes = {start_node}
        node_to_subgraph[start_node] = subgraph_idx
        queue = deque([start_node])
        
        while queue and len(current_subgraph_nodes) < max_nodes:
            current = queue.popleft()
            
            neighbors = adjacency.get(current, set())
            for neighbor in neighbors:
                if neighbor in remaining_nodes:
                    if len(current_subgraph_nodes) < max_nodes:
                        current_subgraph_nodes.add(neighbor)
                        visited.add(neighbor)
                        remaining_nodes.remove(neighbor)
                        node_to_subgraph[neighbor] = subgraph_idx
                        frontier_nodes.discard(neighbor)
                        queue.append(neighbor)
        
        subgraph_nodes[subgraph_idx] = current_subgraph_nodes.copy()
        
        incoming_frontiers = set()
        outgoing_frontiers = set()
        
        for node in current_subgraph_nodes:
            neighbors = adjacency.get(node, set())
            for neighbor in neighbors:
                neighbor_subgraph = node_to_subgraph.get(neighbor)
                if neighbor_subgraph is not None and neighbor_subgraph < subgraph_idx:
                    incoming_frontiers.add(node)
                    conn_key = (neighbor_subgraph, subgraph_idx)
                    if conn_key not in connections:
                        connections[conn_key] = set()
                    connections[conn_key].add(node)
                    connections[conn_key].add(neighbor)
        
        for node in current_subgraph_nodes:
            neighbors = adjacency.get(node, set())
            for neighbor in neighbors:
                if neighbor in remaining_nodes and neighbor not in visited:
                    frontier_nodes.add(neighbor)
                    outgoing_frontiers.add(node)
        
        frontier_metadata[subgraph_idx] = {
            'incoming': incoming_frontiers,
            'outgoing': outgoing_frontiers
        }
        
        subgraph_edges = []
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            
            source_in_current = source in current_subgraph_nodes
            target_in_current = target in current_subgraph_nodes
            
            if source_in_current and target_in_current:
                subgraph_edges.append(edge)
            elif source_in_current and source in node_to_subgraph:
                target_subgraph = node_to_subgraph.get(target)
                if target_subgraph is not None and target_subgraph < subgraph_idx:
                    subgraph_edges.append(edge)
            elif target_in_current and target in node_to_subgraph:
                source_subgraph = node_to_subgraph.get(source)
                if source_subgraph is not None and source_subgraph < subgraph_idx:
                    subgraph_edges.append(edge)
        
        if subgraph_edges:
            subgraphs.append(subgraph_edges)
        
        subgraph_idx += 1
    
    if return_metadata:
        metadata = {
            'node_to_subgraph': node_to_subgraph,
            'subgraph_nodes': {idx: list(nodes) for idx, nodes in subgraph_nodes.items()},
            'frontier_nodes': {
                idx: {
                    'incoming': list(frontier['incoming']),
                    'outgoing': list(frontier['outgoing'])
                }
                for idx, frontier in frontier_metadata.items()
            },
            'connections': {
                f"{src_idx}->{dst_idx}": list(nodes)
                for (src_idx, dst_idx), nodes in connections.items()
            }
        }
        return subgraphs if subgraphs else [edges], metadata
    
    return subgraphs if subgraphs else [edges]


def count_unique_nodes_in_edges(edges):
    nodes = set()
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source:
            nodes.add(source)
        if target:
            nodes.add(target)
    return len(nodes)

