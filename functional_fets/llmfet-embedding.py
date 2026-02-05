import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA


class TextEmbedder:
    def __init__(self, embedding_type="mpnet", vector_size=16):
        self.embedding_type = embedding_type
        self.vector_size = vector_size
        
        from sentence_transformers import SentenceTransformer

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
        return self.model.encode(text)

    def get_embeddings(self, texts, use_pca=False, pca_dim=128, pca_model=None, desc="Generating embeddings"):
        embeddings = np.vstack([self.get_embedding(text) for text in tqdm(texts, desc=desc)])
        
        if use_pca:
            if pca_model is None:
                pca = PCA(n_components=pca_dim)
                embeddings = pca.fit_transform(embeddings)
                return embeddings, pca
            else:
                embeddings = pca_model.transform(embeddings)
                return embeddings, pca_model
        
        return embeddings, None


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings for LLM-extracted file type features (ATLAS version)")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["theia", "fivedirections", "atlas"],
                        help="Dataset name")
    parser.add_argument("--llmfets-model", type=str, default=None,
                        help="LLM model name used for feature extraction (required for ATLAS, e.g., llama3:70b, gpt-4o)")
    parser.add_argument("--embedding", type=str, required=True, 
                        choices=["all", "mpnet", "minilm", "roberta", "distilbert"],
                        help="Embedding type to use ('all' will run all embedding types)")
    parser.add_argument("--saving_path", type=str, default="../BIGDATA/llmfets-embedding/",
                        help="Directory to save embeddings (default: ../BIGDATA/llmfets-embedding/)")
    parser.add_argument("--timeoh", action="store_true",
                        help="If set, read from behavioral-profiles/{model}/timeoh/, otherwise from behavioral-profiles/{model}/")
    parser.add_argument("--causal", action="store_true",
                        help="If set, read from behavioral-profiles/{model}/causal/")
    
    args = parser.parse_args()
    
    if args.dataset.lower() == "atlas" and not args.llmfets_model:
        parser.error("--llmfets-model is required for ATLAS dataset")
    
    return args


def load_behavioral_profile_data(dataset_name, llmfets_model=None, timeoh=False, causal=False):
    is_atlas = dataset_name.lower() == "atlas"
    if is_atlas and llmfets_model:
        model_normalized = llmfets_model.lower().replace(':', '_')
    else:
        model_normalized = None
    
    current_dir = Path(__file__).parent
    if is_atlas and model_normalized:
        if causal:
            base_dir = current_dir / "behavioral-profiles" / model_normalized / "causal"
        elif timeoh:
            base_dir = current_dir / "behavioral-profiles" / model_normalized / "timeoh"
        else:
            base_dir = current_dir / "behavioral-profiles" / model_normalized
    else:
        if causal:
            base_dir = current_dir / "behavioral-profiles" / "causal"
        elif timeoh:
            base_dir = current_dir / "behavioral-profiles" / "timeoh"
        else:
            base_dir = current_dir / "behavioral-profiles"
    
    files = {
        'untype2type_nodes': base_dir / f"untype2type_nodes_{dataset_name}.json",
        'typed_nodes': base_dir / f"typed_nodes_{dataset_name}.json",
        'untype2type_functionality': base_dir / f"untype2type_nodes_functionality_{dataset_name}.json",
        'typed_functionality': base_dir / f"typed_nodes_functionality_{dataset_name}.json"
    }
    
    missing_files = []
    for name, filepath in files.items():
        if not filepath.exists():
            missing_files.append(str(filepath))
    
    if missing_files:
        return None, None
    
    file_list = [
        ('untype2type_nodes', files['untype2type_nodes']),
        ('typed_nodes', files['typed_nodes']),
        ('untype2type_func', files['untype2type_functionality']),
        ('typed_func', files['typed_functionality'])
    ]
    
    loaded_data = {}
    try:
        for name, filepath in tqdm(file_list, desc="Loading JSON files"):
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data[name] = json.load(f)
    except Exception as e:
        return None, None
    
    merged_types = {**loaded_data['untype2type_nodes'], **loaded_data['typed_nodes']}
    merged_functionalities = {**loaded_data['untype2type_func'], **loaded_data['typed_func']}
    
    filtered_types = {}
    filtered_func = {}
    
    for node_id, type_text in tqdm(merged_types.items(), desc="Filtering NO LABEL types"):
        if type_text != "NO LABEL":
            filtered_types[node_id] = type_text
    
    for node_id, func_text in merged_functionalities.items():
        if node_id in filtered_types:
            filtered_func[node_id] = func_text
    
    return filtered_types, filtered_func


def batch_embed_dict(data_dict, text_embedder, desc="Embedding"):
    if not data_dict:
        return {}
    
    ids = list(data_dict.keys())
    texts = [data_dict[id_] for id_ in ids]
    
    embeddings, _ = text_embedder.get_embeddings(texts, use_pca=False, desc=desc)
    
    embeddings_dict = {}
    for i, id_ in enumerate(ids):
        embeddings_dict[id_] = embeddings[i]
    
    return embeddings_dict


def save_embeddings(embeddings_dict, output_file, saving_path):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(saving_path, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)


def process_single_embedding(dataset, embedding_type, saving_path, llmfets_model=None, timeoh=False, causal=False):
    type_dict, func_dict = load_behavioral_profile_data(dataset, llmfets_model, timeoh, causal)
    
    if type_dict is None or func_dict is None:
        return False
    
    text_embedder = TextEmbedder(embedding_type=embedding_type, vector_size=30)
    
    type_embeddings = batch_embed_dict(type_dict, text_embedder, desc=f"Embedding types ({len(type_dict)} nodes)")
    func_embeddings = batch_embed_dict(func_dict, text_embedder, desc=f"Embedding functionality ({len(func_dict)} nodes)")
    
    is_atlas = dataset.lower() == "atlas"
    if is_atlas and llmfets_model:
        model_normalized = llmfets_model.lower().replace(':', '_')
        output_dir = os.path.join(saving_path, dataset.lower(), model_normalized)
        os.makedirs(output_dir, exist_ok=True)
        type_output = os.path.join(output_dir, f"{embedding_type.lower()}_type.pkl")
        func_output = os.path.join(output_dir, f"{embedding_type.lower()}_functionality.pkl")
        output_dir_for_save = output_dir
    else:
        type_output = os.path.join(saving_path, f"{embedding_type.lower()}_{dataset.lower()}_type.pkl")
        func_output = os.path.join(saving_path, f"{embedding_type.lower()}_{dataset.lower()}_functionality.pkl")
        output_dir_for_save = saving_path
    
    save_embeddings(type_embeddings, type_output, output_dir_for_save)
    save_embeddings(func_embeddings, func_output, output_dir_for_save)
    
    return True


def main():
    args = parse_args()
    if args.embedding == 'all':
        embedding_types = ['roberta', 'mpnet', 'minilm', 'distilbert']
        
        success_count = 0
        for embedding_type in embedding_types:
            success = process_single_embedding(
                args.dataset, embedding_type, args.saving_path, 
                args.llmfets_model, args.timeoh, args.causal
            )
            if success:
                success_count += 1
    else:
        process_single_embedding(
            args.dataset, args.embedding, args.saving_path,
            args.llmfets_model, args.timeoh, args.causal
        )


if __name__ == "__main__":
    main()

