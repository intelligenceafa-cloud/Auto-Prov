#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import torch
import torch.nn as nn
import pickle
import json
import networkx as nx
import numpy as np
from tqdm import tqdm
import dgl
import torch.nn.functional as F
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm" or name == "BatchNorm":
        return nn.BatchNorm1d
    else:
        return None


class GATConv(nn.Module):
    def __init__(self, in_dim, e_dim, out_dim, n_heads, feat_drop=0.0, attn_drop=0.0,
                 negative_slope=0.2, residual=False, activation=None, norm=None, concat_out=True):
        super(GATConv, self).__init__()
        self.n_heads = n_heads
        self.out_feat = out_dim
        self.concat_out = concat_out
        
        self.fc = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.edge_fc = nn.Linear(e_dim, out_dim * n_heads, bias=False)
        self.attn_h = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self.bias = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        
        if residual:
            if in_dim != n_heads * out_dim:
                self.res_fc = nn.Linear(in_dim, n_heads * out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        
        self.reset_parameters()
        self.activation = activation
        self.norm = norm
        if norm is not None:
            self.norm = norm(n_heads * out_dim)
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
    
    def forward(self, graph, feat, get_attention=False):
        edge_feature = graph.edata['attr']
        h = self.feat_drop(feat)
        feat_transformed = self.fc(h).view(-1, self.n_heads, self.out_feat)
        
        eh = (feat_transformed * self.attn_h).sum(-1).unsqueeze(-1)
        et = (feat_transformed * self.attn_t).sum(-1).unsqueeze(-1)
        
        graph.ndata.update({'hs': feat_transformed, 'eh': eh, 'et': et})
        
        feat_edge = self.edge_fc(edge_feature).view(-1, self.n_heads, self.out_feat)
        ee = (feat_edge * self.attn_e).sum(-1).unsqueeze(-1)
        
        graph.edata.update({'ee': ee})
        graph.apply_edges(dgl.function.u_add_e('eh', 'ee', 'ee'))
        graph.apply_edges(dgl.function.e_add_v('ee', 'et', 'e'))
        
        e = self.leaky_relu(graph.edata.pop('e'))
        graph.edata['a'] = self.attn_drop(dgl.ops.edge_softmax(graph, e))
        
        graph.update_all(dgl.function.u_mul_e('hs', 'a', 'm'),
                        dgl.function.sum('m', 'hs'))
        
        rst = graph.ndata['hs'].view(-1, self.n_heads, self.out_feat)
        rst = rst + self.bias.view(1, self.n_heads, self.out_feat)
        
        if self.res_fc is not None:
            resval = self.res_fc(h).view(-1, self.n_heads, self.out_feat)
            rst = rst + resval
        
        if self.concat_out:
            rst = rst.flatten(1)
        else:
            rst = torch.mean(rst, dim=1)
        
        if self.norm is not None:
            rst = self.norm(rst)
        
        if self.activation:
            rst = self.activation(rst)
        
        if get_attention:
            return rst, graph.edata['a']
        else:
            return rst


class GAT(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, out_dim, n_layers, n_heads, n_heads_out,
                 activation, feat_drop, attn_drop, negative_slope, residual, norm,
                 concat_out=False, encoding=False):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.gats = nn.ModuleList()
        self.concat_out = concat_out
        
        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None
        
        if self.n_layers == 1:
            self.gats.append(GATConv(
                n_dim, e_dim, out_dim, n_heads_out, feat_drop, attn_drop, negative_slope,
                last_residual, norm=last_norm, concat_out=self.concat_out
            ))
        else:
            self.gats.append(GATConv(
                n_dim, e_dim, hidden_dim, n_heads, feat_drop, attn_drop, negative_slope,
                residual, create_activation(activation),
                norm=norm, concat_out=self.concat_out
            ))
            for _ in range(1, self.n_layers - 1):
                self.gats.append(GATConv(
                    hidden_dim * self.n_heads, e_dim, hidden_dim, n_heads,
                    feat_drop, attn_drop, negative_slope,
                    residual, create_activation(activation),
                    norm=norm, concat_out=self.concat_out
                ))
            self.gats.append(GATConv(
                hidden_dim * self.n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, last_activation, norm=last_norm, concat_out=self.concat_out
            ))
        self.head = nn.Identity()
    
    def forward(self, g, input_feature, return_hidden=False):
        h = input_feature
        hidden_list = []
        for layer in range(self.n_layers):
            h = self.gats[layer](g, h)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)


class GMAEModel(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, mask_rate=0.5, loss_fn="sce", alpha_l=2):
        super(GMAEModel, self).__init__()
        self._mask_rate = mask_rate
        self._output_hidden_size = hidden_dim
        self.recon_loss = nn.BCELoss(reduction='mean')
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.edge_recon_fc.apply(init_weights)
        
        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads
        enc_nhead = n_heads
        
        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim
        
        self.encoder = GAT(
            n_dim=n_dim,
            e_dim=e_dim,
            hidden_dim=enc_num_hidden,
            out_dim=enc_num_hidden,
            n_layers=n_layers,
            n_heads=enc_nhead,
            n_heads_out=enc_nhead,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=True,
        )
        
        self.decoder = GAT(
            n_dim=dec_in_dim,
            e_dim=e_dim,
            hidden_dim=dec_num_hidden,
            out_dim=n_dim,
            n_layers=1,
            n_heads=n_heads,
            n_heads_out=1,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=False,
        )
        
        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim * n_layers, dec_in_dim, bias=False)
        
        from functools import partial
        if loss_fn == "sce":
            self.criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
    
    @property
    def output_hidden_dim(self):
        return self._output_hidden_size
    
    def encoding_mask_noise(self, g, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)
        
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        
        masked_attr = g.ndata["attr"].clone()
        masked_attr[mask_nodes] = self.enc_mask_token
        
        original_attr = g.ndata["attr"]
        g.ndata["attr"] = masked_attr
        
        return g, (mask_nodes, keep_nodes), original_attr
    
    def forward(self, g):
        loss = self.compute_loss(g)
        return loss
    
    def compute_loss(self, g):
        masked_g, (mask_nodes, keep_nodes), original_attr = self.encoding_mask_noise(g, self._mask_rate)
        masked_x = masked_g.ndata['attr'].to(masked_g.device)
        
        enc_rep, all_hidden = self.encoder(masked_g, masked_x, return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)
        
        recon = self.decoder(masked_g, rep)
        x_init = original_attr[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init)
        
        g.ndata["attr"] = original_attr
        
        threshold = min(10000, g.num_nodes())
        
        negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
        import random
        positive_edge_pairs = random.sample(range(g.number_of_edges()), min(threshold, g.number_of_edges()))
        positive_edge_pairs = (g.edges()[0][positive_edge_pairs], g.edges()[1][positive_edge_pairs])
        sample_src = enc_rep[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
        sample_dst = enc_rep[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)
        y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)
        y = torch.cat([torch.ones(len(positive_edge_pairs[0])), torch.zeros(len(negative_edge_pairs[0]))]).to(g.device)
        loss += self.recon_loss(y_pred, y)
        return loss
    
    def embed(self, g):
        x = g.ndata['attr'].to(g.device)
        rep = self.encoder(g, x)
        return rep


def load_entity_level_dataset(base_dir, window_key, use_llm_features=False, global_node_embeddings=None):
    graph_path = f'{base_dir}/processed_data/graphs/graph_{window_key}.pkl'
    
    with open(graph_path, 'rb') as f:
        data = pickle.load(f)
    
    g_nx = nx.node_link_graph(data['graph'], edges='links')
    
    if use_llm_features:
        if 'node_embeddings' in data:
            node_features = data['node_embeddings']
        elif global_node_embeddings is not None and 'node_indices' in data:
            node_indices = data['node_indices']
            node_features = global_node_embeddings[node_indices]
        else:
            raise ValueError(f"LLM features requested but not found in {graph_path}")
    else:
        if 'node_labels' in data:
            node_features = data['node_labels']
        else:
            node_features = None
    
    return g_nx, node_features


def transform_graph(g, node_feature_dim, edge_feature_dim, node_features=None, use_llm_features=False):
    for node in g.nodes():
        if 'type' not in g.nodes[node]:
            g.nodes[node]['type'] = 0
    
    for src, dst in g.edges():
        if 'type' not in g.edges[src, dst]:
            edge_data = g.edges[src, dst]
            if 'type' not in edge_data:
                g.edges[src, dst]['type'] = 0
    
    try:
        g_dgl = dgl.from_networkx(g, node_attrs=['type'], edge_attrs=['type'])
    except Exception as e:
        g_dgl = dgl.from_networkx(g)
        if g_dgl.number_of_nodes() > 0:
            g_dgl.ndata['type'] = torch.zeros(g_dgl.number_of_nodes(), dtype=torch.long)
        if g_dgl.number_of_edges() > 0:
            g_dgl.edata['type'] = torch.zeros(g_dgl.number_of_edges(), dtype=torch.long)
    
    if use_llm_features and node_features is not None:
        g_dgl.ndata["attr"] = torch.tensor(node_features, dtype=torch.float32)
    else:
        if 'type' in g_dgl.ndata:
            g_dgl.ndata["attr"] = F.one_hot(g_dgl.ndata["type"].view(-1).long(), num_classes=node_feature_dim).float()
        else:
            g_dgl.ndata["attr"] = torch.zeros((g_dgl.number_of_nodes(), node_feature_dim), dtype=torch.float32)
    
    if 'type' in g_dgl.edata:
        g_dgl.edata["attr"] = F.one_hot(g_dgl.edata["type"].view(-1).long(), num_classes=edge_feature_dim).float()
    else:
        g_dgl.edata["attr"] = torch.zeros((g_dgl.number_of_edges(), edge_feature_dim), dtype=torch.float32)
    
    return g_dgl


def train_model_simple(model, base_dir, model_dir, train_windows, metadata, optimizer, max_epoch, device,
                       use_llm_features=False, global_node_embeddings=None):
    model.train()
    
    n_dim = metadata['node_feature_dim']
    e_dim = metadata['edge_feature_dim']
    n_train = len(train_windows)
    
    epoch_iter = tqdm(range(max_epoch), desc=f"Training {os.path.basename(model_dir)}")
    for epoch in epoch_iter:
        epoch_loss = 0.0
        for i, window_key in enumerate(train_windows):
            try:
                g_nx, node_features = load_entity_level_dataset(
                    base_dir, window_key, use_llm_features, global_node_embeddings
                )
                
                if g_nx.number_of_nodes() == 0 or g_nx.number_of_edges() == 0:
                    continue
                
                g = transform_graph(g_nx, n_dim, e_dim, node_features, use_llm_features).to(device)
                
                model.train()
                loss = model(g)
                loss /= n_train
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                del g, g_nx, loss
            except Exception as e:
                print(f"Warning: Failed to process window {window_key}: {e}")
                continue
        
        epoch_iter.set_description(f"{os.path.basename(model_dir)} | Epoch {epoch+1} | loss: {epoch_loss:.4f}")
        
        checkpoint_path = f'{model_dir}/checkpoint_epoch_{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return model


def train_entity_level(model, base_dir, train_windows, metadata, optimizer, max_epoch, device,
                       use_llm_features=False, global_node_embeddings=None, save_every_epoch=False):
    model.train()
    
    n_dim = metadata['node_feature_dim']
    e_dim = metadata['edge_feature_dim']
    n_train = len(train_windows)
    
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        epoch_loss = 0.0
        for i, window_key in enumerate(train_windows):
            try:
                g_nx, node_features = load_entity_level_dataset(
                    base_dir, window_key, use_llm_features, global_node_embeddings
                )
                
                if g_nx.number_of_nodes() == 0 or g_nx.number_of_edges() == 0:
                    continue
                
                g = transform_graph(g_nx, n_dim, e_dim, node_features, use_llm_features).to(device)
                
                model.train()
                loss = model(g)
                loss /= n_train
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                del g, g_nx, loss
            except Exception as e:
                print(f"\nWarning: Failed to process window {window_key}: {e}")
                continue
        
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
        
        if save_every_epoch or (epoch + 1) % 10 == 0:
            checkpoint_path = f'{base_dir}/models/checkpoint_epoch_{epoch+1}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            if save_every_epoch or (epoch + 1) % 10 == 0:
                print(f"\n  Saved checkpoint: {checkpoint_path}")
        
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return model


def get_available_artifact_folders(base_artifacts_dir):
    if not os.path.exists(base_artifacts_dir):
        return []
    
    available_folders = []
    items = os.listdir(base_artifacts_dir)
    
    for item in items:
        item_path = os.path.join(base_artifacts_dir, item)
        
        if item.startswith('.'):
            continue
        
        if os.path.isdir(item_path):
            processed_data_path = os.path.join(item_path, 'processed_data')
            if os.path.exists(processed_data_path):
                available_folders.append(item)
    
    return sorted(available_folders)


def parse_args():
    parser = argparse.ArgumentParser(description="MAGIC Graph Learning for ATLAS")
    
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
    parser.add_argument("--llmfets-model", type=str, default="llama3:70b",
                        help="LLM model name used for feature extraction (default: llama3:70b, e.g., gpt-4o)")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay")
    
    args = parser.parse_args()
    
    return args


def train_model_10(base_dir, train_windows, metadata, device,
                   use_llm_features=False, global_node_embeddings=None, max_epochs=80):
    model_10_config = {
        'model_id': 10,
        'name': 'low_mask',
        'num_hidden': 64,
        'num_layers': 3,
        'n_heads': 4,
        'mask_rate': 0.3,
        'lr': 0.001,
        'feat_drop': 0.1,
        'activation': 'prelu',
        'weight_decay': 5e-4,
        'alpha_l': 3,
        'negative_slope': 0.2,
        'norm': 'BatchNorm',
        'residual': True,
        'attn_drop': 0.0
    }
    
    n_dim = metadata['node_feature_dim']
    e_dim = metadata['edge_feature_dim']
    
    model_dir = f"{base_dir}/models/model_10"
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training Model 10 (AutoProv)")
    print(f"  Hidden dim: {model_10_config['num_hidden']}, Layers: {model_10_config['num_layers']}, "
          f"Heads: {model_10_config['n_heads']}, Mask rate: {model_10_config['mask_rate']}, "
          f"LR: {model_10_config['lr']}, Dropout: {model_10_config['feat_drop']}")
    print(f"{'='*70}\n")
    
    hyperparams_path = f"{model_dir}/hyperparams.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(model_10_config, f, indent=2)
    
    model = GMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=model_10_config['num_hidden'],
        n_layers=model_10_config['num_layers'],
        n_heads=model_10_config['n_heads'],
        activation=model_10_config['activation'],
        feat_drop=model_10_config['feat_drop'],
        negative_slope=model_10_config['negative_slope'],
        residual=model_10_config['residual'],
        mask_rate=model_10_config['mask_rate'],
        norm=model_10_config['norm'],
        loss_fn='sce',
        alpha_l=model_10_config['alpha_l']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model_10_config['lr'], weight_decay=model_10_config['weight_decay'])
    
    model = train_model_simple(
        model, base_dir, model_dir, train_windows, metadata, optimizer, max_epochs, device,
        use_llm_features=use_llm_features, global_node_embeddings=global_node_embeddings
    )
    
    final_path = f'{model_dir}/checkpoint_final.pt'
    torch.save(model.state_dict(), final_path)
    
    model_config = {
        'n_dim': n_dim,
        'e_dim': e_dim,
        'hidden_dim': model_10_config['num_hidden'],
        'n_layers': model_10_config['num_layers'],
        'n_heads': model_10_config['n_heads'],
        'activation': model_10_config['activation'],
        'feat_drop': model_10_config['feat_drop'],
        'negative_slope': model_10_config['negative_slope'],
        'residual': model_10_config['residual'],
        'mask_rate': model_10_config['mask_rate'],
        'norm': model_10_config['norm'],
        'loss_fn': 'sce',
        'alpha_l': model_10_config['alpha_l'],
        'lr': model_10_config['lr'],
        'weight_decay': model_10_config['weight_decay'],
        'use_llm_features': use_llm_features
    }
    with open(f'{model_dir}/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n{'='*70}")
    print(f"Model 10 training completed!")
    print(f"Saved to: {model_dir}")
    print(f"{'='*70}\n")


def train_single_folder(args, base_dir, folder_name="", mode="baseline"):
    print(f"\n{'='*70}")
    print(f"MAGIC Graph Learning - ATLAS")
    if mode == "baseline":
        print(f"Mode: Baseline")
    elif mode == "autoprov":
        print(f"Mode: AutoProv")
        if folder_name:
            print(f"Folder: {folder_name}")
        print(f"LLM Embeddings: {args.embedding}")
    print(f"{'='*70}\n")
    
    os.makedirs(f'{base_dir}/models', exist_ok=True)
    os.makedirs(f'{base_dir}/results', exist_ok=True)
    
    with open(f'{base_dir}/processed_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    use_llm_features = metadata.get('use_llm_features', False)
    
    print(f"Dataset: ATLAS")
    print(f"Mode: {metadata.get('mode', 'unknown')}")
    print(f"LLM Features: {use_llm_features}")
    print(f"Node feature dim: {metadata['node_feature_dim']}")
    print(f"Edge feature dim: {metadata['edge_feature_dim']}")
    print(f"Total nodes: {metadata['total_nodes']}")
    
    with open(f'{base_dir}/processed_data/train_windows.json', 'r') as f:
        train_windows = json.load(f)
    
    print(f"Training windows: {len(train_windows)}")
    
    global_node_embeddings = None
    if use_llm_features:
        emb_path = f'{base_dir}/processed_data/node_features/node_embeddings.npy'
        if os.path.exists(emb_path):
            global_node_embeddings = np.load(emb_path)
            print(f"Loaded global node embeddings: {global_node_embeddings.shape}")
    
    if mode == "autoprov":
        train_model_10(
            base_dir, train_windows, metadata, device,
            use_llm_features=use_llm_features,
            global_node_embeddings=global_node_embeddings,
            max_epochs=80
        )
        return
    
    num_hidden = 64
    num_layers = 3
    negative_slope = 0.2
    mask_rate = 0.5
    alpha_l = 3
    n_dim = metadata['node_feature_dim']
    e_dim = metadata['edge_feature_dim']
    
    print(f"\nBuilding MAGIC model...")
    print(f"  Hidden dim: {num_hidden}")
    print(f"  Layers: {num_layers}")
    print(f"  Mask rate: {mask_rate}")
    print(f"  Node feature dim: {n_dim}")
    print(f"  Edge feature dim: {e_dim}")
    
    model = GMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=negative_slope,
        residual=True,
        mask_rate=mask_rate,
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=alpha_l
    )
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"\nStarting training...")
    model = train_entity_level(
        model, base_dir, train_windows, metadata, optimizer, args.epochs, device,
        use_llm_features=use_llm_features, global_node_embeddings=global_node_embeddings,
        save_every_epoch=False
    )
    
    final_path = f'{base_dir}/models/checkpoint_final.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model: {final_path}")
    
    model_config = {
        'n_dim': n_dim,
        'e_dim': e_dim,
        'hidden_dim': num_hidden,
        'n_layers': num_layers,
        'n_heads': 4,
        'activation': 'prelu',
        'feat_drop': 0.1,
        'negative_slope': negative_slope,
        'residual': True,
        'mask_rate': mask_rate,
        'norm': 'BatchNorm',
        'loss_fn': 'sce',
        'alpha_l': alpha_l,
        'use_llm_features': use_llm_features
    }
    with open(f'{base_dir}/models/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Training completed successfully!")
    print(f"{'='*70}\n")


def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    artifacts_root = os.path.join(autoprov_dir, 'BIGDATA', 'MAGIC_artifacts', 'ATLAS_artifacts')
    
    if args.baseline:
        if args.artifacts_dir is None:
            artifacts_dir = os.path.join(artifacts_root, 'original_atlas_graph')
        else:
            artifacts_dir = args.artifacts_dir
        
        if not os.path.exists(artifacts_dir):
            print(f"ERROR: Artifacts directory not found: {artifacts_dir}")
            print(f"       Please run graph_gen_atlas.py --baseline first")
            return
        
        train_single_folder(args, artifacts_dir, mode="baseline")
    
    elif args.autoprov:
        llmfets_model_normalized = args.llmfets_model.lower().replace(':', '_')
        folder_name = f"{args.cee.lower()}_{args.rule_generator.lower()}"
        
        if args.artifacts_dir is None:
            artifacts_dir = os.path.join(
                artifacts_root,
                f"rulellm_llmlabel_{args.embedding.lower()}",
                folder_name,
                llmfets_model_normalized
            )
        else:
            artifacts_dir = args.artifacts_dir
        
        if not os.path.exists(artifacts_dir):
            print(f"ERROR: Artifacts directory not found: {artifacts_dir}")
            print(f"       Please run graph_gen_atlas.py --autoprov first")
            return
        
        train_single_folder(args, artifacts_dir, folder_name=folder_name, mode="autoprov")


if __name__ == "__main__":
    main()
