import os
import math
import torch
import numpy as np
import torch.nn.functional as F
from gensim.models import Word2Vec
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv, GATConv
import torch.nn as nn
from tqdm import tqdm
import zipfile
import pickle
from torch.utils.data import DataLoader, Dataset

class PositionalEncoder:

    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]

def infer(document, w2vmodel):
    encoder = PositionalEncoder(30)
    word_embeddings = [w2vmodel.wv[word] for word in document if word in  w2vmodel.wv]
    
    if not word_embeddings:
        return np.zeros(20)

    output_embedding = torch.tensor(word_embeddings, dtype=torch.float)
    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)

class PositionalEncoder_GPU:

    def __init__(self, d_model, max_len=100000, device="cuda:0"):
        self.device = device
        self.pe = self._generate_positional_encodings(d_model, max_len).to(self.device)

    @staticmethod
    def _generate_positional_encodings(d_model, max_len):
        position = torch.arange(max_len, device="cuda:0").unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device="cuda:0") * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, device="cuda:0")
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def embed(self, x):
        if x.device != self.pe.device:
            raise ValueError("Input tensor and positional encodings must be on the same device.")
        return x + self.pe[:x.size(0), :]


def infer_gpu_batch(documents, encoder, wv_tensor, word2index, batch_size=128):
    encoder = PositionalEncoder_GPU(30, device="cuda:0")
    
    results = []
    for batch in tqdm(DataLoader(documents, batch_size=batch_size, collate_fn=list), desc=f"Processing Nodes with Cuda"):
        batch_word_embeddings = []
        for document in batch:
            word_indices = [
                word2index[word] for word in document if word in word2index
            ]
            if not word_indices:
                batch_word_embeddings.append(torch.zeros(30, device="cuda:0"))
                continue

            word_indices = torch.tensor(word_indices, device="cuda:0")
            word_embeddings = wv_tensor[word_indices]

            if len(document) < 100000:
                word_embeddings = encoder.embed(word_embeddings)

            mean_embedding = word_embeddings.mean(dim=0)
            batch_word_embeddings.append(mean_embedding)

        batch_embeddings = torch.stack(batch_word_embeddings)
        results.append(batch_embeddings)

    return torch.cat(results).cpu().numpy()

def process_nodes_with_gpu(phrases, w2vmodel):
    encoder = PositionalEncoder_GPU(30, device="cuda:0")

    wv_tensor = torch.tensor(w2vmodel.wv.vectors, device="cuda:0", dtype=torch.float)
    word2index = {word: idx for idx, word in enumerate(w2vmodel.wv.index_to_key)}

    nodes = infer_gpu_batch(phrases, encoder, wv_tensor, word2index)

    return nodes


class GCN(torch.nn.Module):
    def __init__(self, in_channel, out_channel, hidden_units=32, num_layers=2, dropout_rate=0.5):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout_rate)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channel, hidden_units, normalize=True))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_units, hidden_units, normalize=True))
        self.convs.append(SAGEConv(hidden_units, out_channel, normalize=True))

    def forward(self, x, edge_index):
        x = self.get_embeddings(x, edge_index)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = x.relu()
            x = self.dropout(x)
        return x

def load_from_zip(base_dir, name):
    pkl_path = f"{base_dir}/{name}.pkl"
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_phrases_chunked(phrases, root_dir, chunk_size=100000):
    num_chunks = (len(phrases) + chunk_size - 1) // chunk_size

    for i in tqdm(range(num_chunks), desc="Saving phrase chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(phrases))
        chunk = phrases[start_idx:end_idx]
        
        chunk_file = os.path.join(root_dir, f'phrases_chunk_{i}.pkl')
        with open(chunk_file, 'wb') as f:
            pickle.dump(chunk, f)


def load_phrases_generator(root_dir):
    chunk_0_file = os.path.join(root_dir, 'phrases_chunk_0.pkl')
    
    if os.path.exists(chunk_0_file):
        chunk_idx = 0
        chunk_files = []
        while True:
            chunk_file = os.path.join(root_dir, f'phrases_chunk_{chunk_idx}.pkl')
            if os.path.exists(chunk_file):
                chunk_files.append(chunk_file)
                chunk_idx += 1
            else:
                break
        
        num_chunks = len(chunk_files)
        with open(chunk_files[0], 'rb') as f:
            first_chunk = pickle.load(f)
        first_chunk_size = len(first_chunk)
        del first_chunk
        estimated_total = first_chunk_size * (num_chunks - 1)
        if num_chunks > 1:
            with open(chunk_files[-1], 'rb') as f:
                last_chunk = pickle.load(f)
            estimated_total += len(last_chunk)
            del last_chunk
        else:
            estimated_total = first_chunk_size
        import gc
        gc.collect()

        def chunk_gen():
            for chunk_file in chunk_files:
                with open(chunk_file, 'rb') as f:
                    chunk = pickle.load(f)
                yield chunk
        
        return chunk_gen(), estimated_total, num_chunks
    else:
        phrases_file = os.path.join(root_dir, 'phrases.pkl')
        
        if os.path.exists(phrases_file):
            with open(phrases_file, 'rb') as f:
                phrases = pickle.load(f)

            def single_chunk_gen():
                yield phrases
            
            return single_chunk_gen(), len(phrases), 1
        else:
            raise FileNotFoundError(f"No phrases file found in {root_dir}")


def process_phrases_to_embeddings(root_dir, text_embedder, use_pca=False, pca_dim=128, pca_model=None, batch_size=64):
    import gc
    import shutil
    temp_dir = os.path.join(root_dir, '.embeddings_temp')
    os.makedirs(temp_dir, exist_ok=True)
    chunk_gen, total_size, num_chunks = load_phrases_generator(root_dir)

    for chunk_idx, phrases_chunk in enumerate(tqdm(chunk_gen, total=num_chunks, desc="Generating embeddings", position=0)):
        chunk_emb_path = os.path.join(temp_dir, f'embeddings_chunk_{chunk_idx}.npy')
        if os.path.exists(chunk_emb_path):
            continue
        chunk_embeddings = []
        num_batches = (len(phrases_chunk) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(phrases_chunk), batch_size), 
                     desc=f"  Chunk {chunk_idx}/{num_chunks-1}", 
                     position=1, 
                     leave=False,
                     total=num_batches):
            batch = phrases_chunk[i:i+batch_size]
            text_batch = [' '.join(phrase) if isinstance(phrase, (list, tuple)) else str(phrase) for phrase in batch]
            embeddings_batch = text_embedder.model.encode(text_batch, show_progress_bar=False)
            chunk_embeddings.append(embeddings_batch)
            
            del batch, text_batch, embeddings_batch
        chunk_embeddings_combined = np.vstack(chunk_embeddings)
        np.save(chunk_emb_path, chunk_embeddings_combined)
        del phrases_chunk, chunk_embeddings, chunk_embeddings_combined
        gc.collect()

    if use_pca and pca_model is None:
        from sklearn.decomposition import IncrementalPCA
        
        pca = IncrementalPCA(n_components=pca_dim, batch_size=10000)
        
        for chunk_idx in tqdm(range(num_chunks), desc="Fitting PCA", position=0):
            chunk_emb_path = os.path.join(temp_dir, f'embeddings_chunk_{chunk_idx}.npy')
            embeddings_chunk = np.load(chunk_emb_path)
            pca.partial_fit(embeddings_chunk)
            
            del embeddings_chunk
            
            if chunk_idx % 5 == 0:
                gc.collect()
        
        pca_model = pca

    elif use_pca and pca_model is not None:
        pass
    else:
        pca_model = None

    final_embeddings = []
    
    for chunk_idx in tqdm(range(num_chunks), desc="Transforming chunks", position=0):
        chunk_emb_path = os.path.join(temp_dir, f'embeddings_chunk_{chunk_idx}.npy')
        embeddings_chunk = np.load(chunk_emb_path)
        if use_pca and pca_model is not None:
            transformed_chunk = pca_model.transform(embeddings_chunk)
        else:
            transformed_chunk = embeddings_chunk
        
        final_embeddings.append(transformed_chunk)
        
        del embeddings_chunk, transformed_chunk
        
        if chunk_idx % 5 == 0:
            gc.collect()
    nodes = np.vstack(final_embeddings)
    del final_embeddings
    gc.collect()
    shutil.rmtree(temp_dir)

    return nodes, pca_model