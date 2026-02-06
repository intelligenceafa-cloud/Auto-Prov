#!/usr/bin/env python3

import os
import json
import pickle
import argparse
from typing import List, Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Create candidate seeds from ATLAS candidate output")
    parser.add_argument("--embedding", type=str, default="roberta", help="Name of the embedding (e.g., mptnet, all-miniLM, roberta)")
    parser.add_argument("--log_type", type=str, default="audit", help="Log type to process (default: audit)")
    parser.add_argument("--candidates_dir", type=str, default="./candidates-atlas", 
                       help="Directory containing candidate data")
    return parser.parse_args()


def process_log_type(candidates_dir: str, embedding: str, log_type: str):
    log_type_dir = os.path.join(candidates_dir, embedding, log_type)
    candidate_output_path = os.path.join(log_type_dir, "candidate-output.json")
    
    if not os.path.exists(candidate_output_path):
        return False
    
    with open(candidate_output_path, 'r') as f:
        candidate_data = json.load(f)
    
    candidate_edges = []
    candidate_enames = []
    candidate_vtypes = []
    
    total_edges_before = 0
    total_edges_after = 0
    no_label_edges_removed = 0
    
    for entry in candidate_data:
        input_log = entry.get("input", "")
        edges = entry.get("edges", "")
        enames = entry.get("enames", "")
        vtypes = entry.get("vtypes", "")
        
        if edges:
            edge_lines = [line.strip() for line in edges.split('\n') if line.strip()]
            total_edges_before += len(edge_lines)
            filtered_edges = [line for line in edge_lines if 'A: [NO LABEL]' not in line]
            no_label_edges_removed += len(edge_lines) - len(filtered_edges)
            total_edges_after += len(filtered_edges)
            edges = '\n'.join(filtered_edges)
        
        candidate_edges.append({"input": input_log, "target": edges})
        candidate_enames.append({"input": input_log, "target": enames})
        candidate_vtypes.append({"input": input_log, "target": vtypes})
    
    edges_path = os.path.join(log_type_dir, "candidate_edges.pkl")
    with open(edges_path, 'wb') as f:
        pickle.dump(candidate_edges, f)
    
    enames_path = os.path.join(log_type_dir, "candidate_enames.pkl")
    with open(enames_path, 'wb') as f:
        pickle.dump(candidate_enames, f)
    
    vtypes_path = os.path.join(log_type_dir, "candidate_vtypes.pkl")
    with open(vtypes_path, 'wb') as f:
        pickle.dump(candidate_vtypes, f)
    
    return True


def main():
    args = parse_args()
    
    processed_count = 0
    
    success = process_log_type(args.candidates_dir, args.embedding, args.log_type)
    if success:
        processed_count += 1


if __name__ == "__main__":
    main()
