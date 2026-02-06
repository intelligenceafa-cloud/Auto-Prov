#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator, LastNeighborLoader
from torch_geometric.data import TemporalData
from tqdm import tqdm
import pickle
import numpy as np
import glob
import zipfile
import tempfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_6_CONFIG = {
    "memory_dim": 128,
    "embedding_dim": 128,
    "lr": 0.00001,
    "batch_size": 512,
    "weight_decay": 0.001
}

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=8,
                                    dropout=0.0, edge_dim=edge_dim)
        self.conv2 = TransformerConv(in_channels*8, out_channels, heads=1, concat=False,
                             dropout=0.0, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        last_update = last_update.to(device)
        x = x.to(device)
        t = t.to(device)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, num_edge_types=4):
        super(LinkPredictor, self).__init__()
        self.lin_src = nn.Linear(in_channels, in_channels*2)
        self.lin_dst = nn.Linear(in_channels, in_channels*2)
        
        self.lin_seq = nn.Sequential(
            nn.Linear(in_channels*4, in_channels*8),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(in_channels*8, in_channels*2),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(in_channels*2, in_channels//2),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(in_channels//2, num_edge_types)
        )
        
    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)      
        h = self.lin_seq(h)
        return h

def load_torch_from_zip_or_file(file_path, map_location='cpu'):
    if file_path.endswith('.zip'):
        zip_path = file_path
        file_name = os.path.basename(file_path).replace('.zip', '')
    elif os.path.exists(file_path + '.zip'):
        zip_path = file_path + '.zip'
        file_name = os.path.basename(file_path)
    else:
        if os.path.exists(file_path):
            return torch.load(file_path, map_location=map_location, weights_only=False)
        else:
            return None
    
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmp_dir:
                extracted_path = os.path.join(tmp_dir, file_name)
                zip_ref.extract(file_name, tmp_dir)
                data = torch.load(extracted_path, map_location=map_location, weights_only=False)
                return data
    else:
        return None

def load_pickle_from_zip_or_file(file_path):
    if os.path.exists(file_path + '.zip'):
        with zipfile.ZipFile(file_path + '.zip', 'r') as zip_ref:
            file_name = os.path.basename(file_path)
            with zip_ref.open(file_name) as f:
                return pickle.load(f)
    elif os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        return None

def zip_file(file_path):
    if not os.path.exists(file_path):
        return
    
    zip_path = file_path + '.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, os.path.basename(file_path))
    
    os.remove(file_path)

def train(train_data, memory, gnn, link_pred, optimizer, criterion, neighbor_loader, 
          batch_size=1024, num_edge_types=4):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()
    neighbor_loader.reset_state()

    total_loss = 0
    num_events = len(train_data.src)
    total_batches = (num_events + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, num_events, batch_size), desc="Training", total=total_batches, leave=False):
        optimizer.zero_grad()
        
        end_idx = min(batch_idx + batch_size, num_events)
        
        src = train_data.src[batch_idx:end_idx]
        pos_dst = train_data.dst[batch_idx:end_idx]
        t = train_data.t[batch_idx:end_idx]
        msg = train_data.msg[batch_idx:end_idx]
        
        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        
        src = src.to(device, non_blocking=True)
        pos_dst = pos_dst.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        msg = msg.to(device, non_blocking=True)
        n_id = n_id.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        
        assoc = torch.empty(train_data.num_nodes, dtype=torch.long, device=device)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, 
                train_data.t[e_id.cpu()].to(device), 
                train_data.msg[e_id.cpu()].to(device))
        
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])       
        y_pred = torch.cat([pos_out], dim=0)
        
        node_feat_dim = 16
        edge_type_vecs = msg[:, node_feat_dim:node_feat_dim+num_edge_types]
        y_true = torch.argmax(edge_type_vecs, dim=1).to(torch.long)
        
        loss = criterion(y_pred, y_true)

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src.cpu(), pos_dst.cpu())

        loss.backward()
        optimizer.step()
        memory.detach()
        
        total_loss += float(loss) * len(src)
    
    return total_loss / train_data.num_events

def train_model_6(base_dir, max_epochs=40):
    print(f"\n{'='*70}")
    print(f"Training Model 6 (AutoProv)")
    print(f"  Memory dim: {MODEL_6_CONFIG['memory_dim']}, Embedding dim: {MODEL_6_CONFIG['embedding_dim']}, "
          f"LR: {MODEL_6_CONFIG['lr']}, Batch size: {MODEL_6_CONFIG['batch_size']}, Weight decay: {MODEL_6_CONFIG['weight_decay']}")
    print(f"{'='*70}\n")
    
    nodeid2msg_path = f'{base_dir}/processed_data/node_mappings/nodeid2msg.pkl'
    nodeid2msg = load_pickle_from_zip_or_file(nodeid2msg_path)
    if nodeid2msg is None:
        raise FileNotFoundError(f"nodeid2msg.pkl not found at {nodeid2msg_path} (or {nodeid2msg_path}.zip)")
    
    max_node_num = len([k for k in nodeid2msg.keys() if type(k) == int])
    print(f"  Max node number: {max_node_num}")
    
    num_edge_types = 4
    edge_label_encoder_path = f'{base_dir}/processed_data/node_features/edge_label_encoder.pkl'
    edge_label_encoder = load_pickle_from_zip_or_file(edge_label_encoder_path)
    if edge_label_encoder is not None:
        num_edge_types = len(edge_label_encoder.classes_)
        print(f"  Loaded edge label encoder: {num_edge_types} edge types")
    else:
        rel2id_path = f'{base_dir}/processed_data/node_features/rel2id.pkl'
        rel2id = load_pickle_from_zip_or_file(rel2id_path)
        if rel2id is not None:
            num_edge_types = len([k for k in rel2id.keys() if isinstance(k, int)])
            print(f"  Loaded from rel2id: {num_edge_types} edge types")
        else:
            print(f"  Warning: Edge type encoder not found, using default: {num_edge_types} edge types")
    
    print(f"  Number of edge types: {num_edge_types}")
    
    print("\n[Step 2/4] Loading training graphs (benign edges only)...")
    train_graphs = []
    
    def filter_benign_edges(graph_data, edge_labels):
        benign_mask = [label == False for label in edge_labels]
        if not any(benign_mask):
            return None
        
        filtered_graph = TemporalData()
        filtered_graph.src = graph_data.src[benign_mask]
        filtered_graph.dst = graph_data.dst[benign_mask]
        filtered_graph.t = graph_data.t[benign_mask]
        filtered_graph.msg = graph_data.msg[benign_mask]
        filtered_graph.num_nodes = graph_data.num_nodes
        
        return filtered_graph
    
    train_graphs_dir = f'{base_dir}/train/temporal_graphs'
    train_labels_dir = f'{base_dir}/train/edge_labels'
    
    if not os.path.exists(train_graphs_dir):
        raise FileNotFoundError(f"Training graphs directory not found: {train_graphs_dir}")
    
    graph_files = sorted(glob.glob(os.path.join(train_graphs_dir, '*.pt'))) + sorted(glob.glob(os.path.join(train_graphs_dir, '*.pt.zip')))
    for graph_path in graph_files:
        graph_data = load_torch_from_zip_or_file(graph_path, map_location='cpu')
        if graph_data is None:
            continue
        
        graph_name = os.path.basename(graph_path).replace('.pt.zip', '').replace('.pt', '')
        label_name = graph_name.replace('graph_', 'labels_')
        label_path = os.path.join(train_labels_dir, label_name + '.pkl')
        
        edge_labels = load_pickle_from_zip_or_file(label_path)
        if edge_labels is not None:
            filtered_graph = filter_benign_edges(graph_data, edge_labels)
            if filtered_graph is not None:
                train_graphs.append((graph_name, filtered_graph))
                num_benign = sum(1 for label in edge_labels if label == False)
                num_malicious = sum(1 for label in edge_labels if label != False)
                print(f"  Loaded {graph_name}: {len(graph_data.src)} total edges ({num_benign} benign, {num_malicious} malicious) -> {len(filtered_graph.src)} edges for training")
            else:
                print(f"  Warning: {graph_name} has no benign edges, skipping")
        else:
            print(f"  Warning: Labels not found for {graph_name}, using all edges")
            train_graphs.append((graph_name, graph_data))
    
    if len(train_graphs) == 0:
        raise RuntimeError("No training graphs found!")
    
    msg_dim = train_graphs[0][1].msg.size(-1)
    print(f"\n  Message dimension: {msg_dim}")
    
    node_feat_dim = 16
    calculated_num_edge_types = msg_dim - 2 * node_feat_dim
    
    if calculated_num_edge_types != num_edge_types:
        print(f"  Warning: Calculated num_edge_types ({calculated_num_edge_types}) != loaded value ({num_edge_types})")
        print(f"  Using calculated value: {calculated_num_edge_types}")
        num_edge_types = calculated_num_edge_types
    
    print("\n[Step 3/4] Initializing TGN model...")
    
    time_dim = 100
    memory = TGNMemory(
        max_node_num, msg_dim, MODEL_6_CONFIG['memory_dim'], time_dim,
        message_module=IdentityMessage(msg_dim, MODEL_6_CONFIG['memory_dim'], time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)
    
    gnn = GraphAttentionEmbedding(
        in_channels=MODEL_6_CONFIG['memory_dim'],
        out_channels=MODEL_6_CONFIG['embedding_dim'],
        msg_dim=msg_dim,
        time_enc=memory.time_enc,
    ).to(device)
    
    link_pred = LinkPredictor(in_channels=MODEL_6_CONFIG['embedding_dim'], num_edge_types=num_edge_types).to(device)
    
    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=MODEL_6_CONFIG['lr'], eps=1e-08, weight_decay=MODEL_6_CONFIG['weight_decay']
    )
    
    criterion = CrossEntropyLoss()
    neighbor_loader = LastNeighborLoader(max_node_num, size=20, device='cpu')
    
    model_dir = f'{base_dir}/models/model_6/'
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n[Step 4/4] Training for {max_epochs} epochs...")
    
    for epoch in range(1, max_epochs + 1):
        epoch_loss = 0
        
        for window_name, graph_data in train_graphs:
            loss = train(
                graph_data, memory, gnn, link_pred, optimizer, criterion,
                neighbor_loader, MODEL_6_CONFIG['batch_size'], num_edge_types
            )
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(train_graphs)
        print(f'Epoch: {epoch:02d}, Loss: {avg_loss:.4f}')
        
        checkpoint = {
            'epoch': epoch,
            'memory_state_dict': memory.state_dict(),
            'gnn_state_dict': gnn.state_dict(),
            'link_pred_state_dict': link_pred.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': {
                'max_node_num': max_node_num,
                'msg_dim': msg_dim,
                'memory_dim': MODEL_6_CONFIG['memory_dim'],
                'time_dim': time_dim,
                'embedding_dim': MODEL_6_CONFIG['embedding_dim'],
                'num_edge_types': num_edge_types,
            }
        }
        checkpoint_path = f'{model_dir}tgn_model_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        zip_file(checkpoint_path)
        print(f'  Checkpoint saved: tgn_model_epoch_{epoch}.pt.zip')
    
    print(f"\n{'='*70}")
    print(f"Model 6 training completed!")
    print(f"Saved to: {model_dir}")
    print(f"{'='*70}\n")
    
    del memory, gnn, link_pred, neighbor_loader, optimizer
    torch.cuda.empty_cache()

def train_baseline(base_dir, epochs=50):
    print(f"\n{'='*70}")
    print(f"Training Baseline Model")
    print(f"{'='*70}\n")
    
    nodeid2msg_path = f'{base_dir}/processed_data/node_mappings/nodeid2msg.pkl'
    nodeid2msg = load_pickle_from_zip_or_file(nodeid2msg_path)
    if nodeid2msg is None:
        raise FileNotFoundError(f"nodeid2msg.pkl not found at {nodeid2msg_path} (or {nodeid2msg_path}.zip)")
    
    max_node_num = len([k for k in nodeid2msg.keys() if type(k) == int])
    print(f"  Max node number: {max_node_num}")
    
    num_edge_types = 4
    edge_label_encoder_path = f'{base_dir}/processed_data/node_features/edge_label_encoder.pkl'
    edge_label_encoder = load_pickle_from_zip_or_file(edge_label_encoder_path)
    if edge_label_encoder is not None:
        num_edge_types = len(edge_label_encoder.classes_)
        print(f"  Loaded edge label encoder: {num_edge_types} edge types")
    else:
        rel2id_path = f'{base_dir}/processed_data/node_features/rel2id.pkl'
        rel2id = load_pickle_from_zip_or_file(rel2id_path)
        if rel2id is not None:
            num_edge_types = len([k for k in rel2id.keys() if isinstance(k, int)])
            print(f"  Loaded from rel2id: {num_edge_types} edge types")
        else:
            print(f"  Warning: Edge type encoder not found, using default: {num_edge_types} edge types")
    
    print(f"  Number of edge types: {num_edge_types}")
    
    print("\n[Step 2/4] Loading training graphs (benign edges only)...")
    train_graphs = []
    
    def filter_benign_edges(graph_data, edge_labels):
        benign_mask = [label == False for label in edge_labels]
        if not any(benign_mask):
            return None
        
        filtered_graph = TemporalData()
        filtered_graph.src = graph_data.src[benign_mask]
        filtered_graph.dst = graph_data.dst[benign_mask]
        filtered_graph.t = graph_data.t[benign_mask]
        filtered_graph.msg = graph_data.msg[benign_mask]
        filtered_graph.num_nodes = graph_data.num_nodes
        
        return filtered_graph
    
    train_graphs_dir = f'{base_dir}/train/temporal_graphs'
    train_labels_dir = f'{base_dir}/train/edge_labels'
    
    if not os.path.exists(train_graphs_dir):
        raise FileNotFoundError(f"Training graphs directory not found: {train_graphs_dir}")
    
    graph_files = sorted(glob.glob(os.path.join(train_graphs_dir, '*.pt'))) + sorted(glob.glob(os.path.join(train_graphs_dir, '*.pt.zip')))
    for graph_path in graph_files:
        graph_data = load_torch_from_zip_or_file(graph_path, map_location='cpu')
        if graph_data is None:
            continue
        
        graph_name = os.path.basename(graph_path).replace('.pt.zip', '').replace('.pt', '')
        label_name = graph_name.replace('graph_', 'labels_')
        label_path = os.path.join(train_labels_dir, label_name + '.pkl')
        
        edge_labels = load_pickle_from_zip_or_file(label_path)
        if edge_labels is not None:
            filtered_graph = filter_benign_edges(graph_data, edge_labels)
            if filtered_graph is not None:
                train_graphs.append((graph_name, filtered_graph))
                num_benign = sum(1 for label in edge_labels if label == False)
                num_malicious = sum(1 for label in edge_labels if label != False)
                print(f"  Loaded {graph_name}: {len(graph_data.src)} total edges ({num_benign} benign, {num_malicious} malicious) -> {len(filtered_graph.src)} edges for training")
            else:
                print(f"  Warning: {graph_name} has no benign edges, skipping")
        else:
            print(f"  Warning: Labels not found for {graph_name}, using all edges")
            train_graphs.append((graph_name, graph_data))
    
    if len(train_graphs) == 0:
        raise RuntimeError("No training graphs found!")
    
    msg_dim = train_graphs[0][1].msg.size(-1)
    print(f"\n  Message dimension: {msg_dim}")
    
    node_feat_dim = 16
    calculated_num_edge_types = msg_dim - 2 * node_feat_dim
    
    if calculated_num_edge_types != num_edge_types:
        print(f"  Warning: Calculated num_edge_types ({calculated_num_edge_types}) != loaded value ({num_edge_types})")
        print(f"  Using calculated value: {calculated_num_edge_types}")
        num_edge_types = calculated_num_edge_types
    
    print("\n[Step 3/4] Initializing TGN model...")
    
    memory_dim = 100
    embedding_dim = 100
    time_dim = 100
    lr = 0.00005
    batch_size = 1024
    weight_decay = 0.01
    
    memory = TGNMemory(
        max_node_num, msg_dim, memory_dim, time_dim,
        message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)
    
    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=msg_dim,
        time_enc=memory.time_enc,
    ).to(device)
    
    link_pred = LinkPredictor(in_channels=embedding_dim, num_edge_types=num_edge_types).to(device)
    
    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=lr, eps=1e-08, weight_decay=weight_decay
    )
    
    criterion = CrossEntropyLoss()
    neighbor_loader = LastNeighborLoader(max_node_num, size=20, device='cpu')
    
    model_dir = f'{base_dir}/models/'
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n[Step 4/4] Training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        
        for window_name, graph_data in train_graphs:
            loss = train(
                graph_data, memory, gnn, link_pred, optimizer, criterion,
                neighbor_loader, batch_size, num_edge_types
            )
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(train_graphs)
        print(f'Epoch: {epoch:02d}, Loss: {avg_loss:.4f}')
        
        if epoch == 50 or epoch == epochs:
            checkpoint = {
                'epoch': epoch,
                'memory_state_dict': memory.state_dict(),
                'gnn_state_dict': gnn.state_dict(),
                'link_pred_state_dict': link_pred.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'max_node_num': max_node_num,
                    'msg_dim': msg_dim,
                    'memory_dim': memory_dim,
                    'time_dim': time_dim,
                    'embedding_dim': embedding_dim,
                    'num_edge_types': num_edge_types,
                }
            }
            checkpoint_path = f'{model_dir}tgn_model_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            zip_file(checkpoint_path)
            print(f'  Checkpoint saved: tgn_model_epoch_{epoch}.pt.zip')
    
    final_model = {
        'memory': memory,
        'gnn': gnn,
        'link_pred': link_pred,
        'neighbor_loader': neighbor_loader,
        'config': {
            'max_node_num': max_node_num,
            'msg_dim': msg_dim,
            'memory_dim': memory_dim,
            'time_dim': time_dim,
            'embedding_dim': embedding_dim,
            'num_edge_types': num_edge_types,
        }
    }
    final_path = f'{model_dir}tgn_model_final.pt'
    torch.save(final_model, final_path)
    zip_file(final_path)
    print(f'  Final model saved: tgn_model_final.pt.zip')
    
    print(f"\n{'='*70}")
    print("Baseline training completed!")
    print(f"{'='*70}\n")
    
    del memory, gnn, link_pred, neighbor_loader, optimizer
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description="KAIROS Graph Learning for ATLAS")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--baseline", action='store_true',
                        help="Train baseline model (original ATLAS graphs)")
    mode_group.add_argument("--autoprov", action='store_true',
                        help="Train AutoProv model (RuleLLM graphs with LLM embeddings)")
    
    parser.add_argument("--artifacts_dir", type=str, default=None,
                        help="Directory with preprocessed ATLAS artifacts (auto-set based on mode)")
    
    parser.add_argument("--cee", type=str, default="gpt-4o",
                        help="Candidate Edge Extractor name (e.g., 'gpt-4o', 'llama3_70b')")
    parser.add_argument("--rule_generator", type=str, default="llama3_70b",
                        help="Rule Generator name (e.g., 'llama3_70b', 'qwen2_72b')")
    parser.add_argument("--embedding", type=str, default="mpnet",
                        choices=['roberta', 'mpnet', 'minilm', 'distilbert'],
                        help="Embedding model name (only used with --autoprov)")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (for baseline, default: 50)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    artifacts_root = os.path.join(autoprov_dir, 'BIGDATA', 'KAIROS_artifacts', 'ATLAS_artifacts')
    
    if args.baseline:
        if args.artifacts_dir is None:
            artifacts_dir = os.path.join(artifacts_root, 'original_atlas_graph')
        else:
            artifacts_dir = args.artifacts_dir
        
        if not os.path.exists(artifacts_dir):
            print(f"ERROR: Artifacts directory not found: {artifacts_dir}")
            print(f"       Please run graph_gen_atlas.py --baseline first")
            return
        
        train_baseline(artifacts_dir, epochs=args.epochs)
    
    elif args.autoprov:
        folder_name = f"{args.cee.lower()}_{args.rule_generator.lower()}"
        
        if args.artifacts_dir is None:
            artifacts_dir = os.path.join(
                artifacts_root,
                f"rulellm_llmlabel_{args.embedding.lower()}",
                folder_name
            )
        else:
            artifacts_dir = args.artifacts_dir
        
        if not os.path.exists(artifacts_dir):
            print(f"ERROR: Artifacts directory not found: {artifacts_dir}")
            print(f"       Please run graph_gen_atlas.py --autoprov first")
            return
        
        train_model_6(artifacts_dir, max_epochs=40)

if __name__ == "__main__":
    main()

