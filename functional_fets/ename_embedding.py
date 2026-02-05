#!/usr/bin/env python3

import argparse
import json
import pickle
import os
import sys
import numpy as np
from tqdm import tqdm


class TextEmbedder:
    
    def __init__(self, embedding_type="roberta", vector_size=16):
        from sentence_transformers import SentenceTransformer
        
        self.embedding_type = embedding_type
        self.vector_size = vector_size

        if embedding_type == "mpnet":
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        elif embedding_type == "minilm":
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        elif embedding_type == "roberta":
            self.model = SentenceTransformer('roberta-base')
        elif embedding_type == "distilbert":
            self.model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

    def get_embedding(self, text):
        embedding = self.model.encode(text)
        return embedding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute ename embeddings for classification"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["theia", "fivedirections", "atlas"],
        help="Dataset name"
    )
    
    parser.add_argument(
        "--llmfets-model",
        type=str,
        default=None,
        help="LLM model name used for feature extraction (required for ATLAS, e.g., llama3:70b, gpt-4o)"
    )
    
    parser.add_argument(
        "--embedding",
        type=str,
        required=True,
        choices=["mpnet", "minilm", "roberta", "distilbert", "all"],
        help="Embedding type (or 'all' to compute all types)"
    )
    
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Load enames from causal profiles directory"
    )
    
    parser.add_argument(
        "--timeoh",
        action="store_true",
        help="Load enames from timeoh profiles directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset.lower() == "atlas" and not args.llmfets_model:
        parser.error("--llmfets-model is required for ATLAS dataset")
    
    return args


def compute_embeddings_for_type(dataset, embedding_type, base_dir, model_normalized=None):
    typed_file = f"{base_dir}/typed_nodes_enames_{dataset}.json"
    untyped_file = f"{base_dir}/untyped_nodes_enames_{dataset}.json"
    
    try:
        with open(typed_file, 'r') as f:
            typed_enames = json.load(f)
    except FileNotFoundError:
        typed_enames = {}
    try:
        with open(untyped_file, 'r') as f:
            untyped_enames = json.load(f)
    except FileNotFoundError:
        untyped_enames = {}
    all_enames = set()
    for ename_list in tqdm(typed_enames.values(), desc=f"Collecting typed enames", ncols=100, leave=False):
        all_enames.update(ename_list)
    for ename_list in tqdm(untyped_enames.values(), desc=f"Collecting untyped enames", ncols=100, leave=False):
        all_enames.update(ename_list)
    
    all_enames = sorted(list(all_enames))
    
    if not all_enames:
        return None
    text_embedder = TextEmbedder(embedding_type=embedding_type, vector_size=128)
    
    ename_embeddings = {}
    failed_count = 0
    
    pbar = tqdm(all_enames, desc=f"Computing {embedding_type} embeddings", ncols=100)
    for ename in pbar:
        try:
            embedding = text_embedder.get_embedding(ename)
            ename_embeddings[ename] = embedding
        except Exception as e:
            failed_count += 1
            continue
        
        pbar.set_postfix_str(f"Success: {len(ename_embeddings):,}/{len(all_enames):,}, Failed: {failed_count}")
    
    pbar.close()
    
    if model_normalized:
        cache_dir = f"./ename-embeddings/{model_normalized}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/enameemb_{dataset}_{embedding_type}.pkl"
    else:
        os.makedirs("./ename-embeddings", exist_ok=True)
        cache_file = f"./ename-embeddings/enameemb_{dataset}_{embedding_type}.pkl"
    
    with open(cache_file, 'wb') as f:
        pickle.dump(ename_embeddings, f)
    return cache_file


def main():
    args = parse_args()
    
    is_atlas = args.dataset.lower() == "atlas"
    if is_atlas and args.llmfets_model:
        model_normalized = args.llmfets_model.lower().replace(':', '_')
    else:
        model_normalized = None
    
    if is_atlas and model_normalized:
        if args.causal:
            base_dir = f"./behavioral-profiles/{model_normalized}/causal"
        elif args.timeoh:
            base_dir = f"./behavioral-profiles/{model_normalized}/timeoh"
        else:
            base_dir = f"./behavioral-profiles/{model_normalized}"
    else:
        if args.causal:
            base_dir = "./behavioral-profiles/causal"
        elif args.timeoh:
            base_dir = "./behavioral-profiles/timeoh"
        else:
            base_dir = "./behavioral-profiles"
    
    if args.embedding == "all":
        embedding_types = ["mpnet", "minilm", "roberta", "distilbert"]
        
        main_pbar = tqdm(embedding_types, desc="Processing all embeddings", ncols=100)
        for emb_type in main_pbar:
            main_pbar.set_description(f"Processing {emb_type}")
            cache_file = compute_embeddings_for_type(args.dataset, emb_type, base_dir, model_normalized)
        
        main_pbar.close()
    else:
        compute_embeddings_for_type(args.dataset, args.embedding, base_dir, model_normalized)


if __name__ == "__main__":
    main()
