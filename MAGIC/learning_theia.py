#!/usr/bin/env python3

import os
import sys
import argparse

def parse_args_early():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids (e.g. 0,1,2); default empty = use all available")
    args, _ = parser.parse_known_args()
    return args.gpus

gpus = parse_args_early()
if gpus and gpus.strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import pickle
import glob
import json
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import dgl
import torch.nn.functional as F
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import gc

try:
    import faiss
    FAISS_AVAILABLE = True
    FAISS_GPU = faiss.get_num_gpus() > 0
    if FAISS_GPU:
        print("✓ FAISS available with GPU support - using ultra-fast KNN")
    else:
        print("✓ FAISS available (CPU only) - using fast KNN")
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠ FAISS not available - falling back to sklearn KNN")
    print("  💡 For faster KNN, install: pip install faiss-gpu")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATASET_DATES = {
    "THEIA": {
        "train_start_date": "2018-04-03",
        "train_end_date": "2018-04-05",
        "test_start_date": "2018-04-09",
        "test_end_date": "2018-04-12"
    }
}


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


def load_entity_level_dataset(base_dir, time_window):
    graph_path = f'{base_dir}/processed_data/graphs/graph_{time_window}.pkl'
    
    with open(graph_path, 'rb') as f:
        data = pickle.load(f)
    

    g_nx = nx.node_link_graph(data['graph'], edges='links')
    return g_nx, data['node_labels']

def transform_graph(g, node_feature_dim, edge_feature_dim, node_embeddings=None):
    g_dgl = dgl.from_networkx(g, node_attrs=['type'], edge_attrs=['type'])
    

    if node_embeddings is not None:

        node_indices = g_dgl.ndata["type"].view(-1).long().cpu().numpy()
        node_features = torch.tensor(node_embeddings[node_indices], dtype=torch.float32)
        g_dgl.ndata["attr"] = node_features
    else:

        g_dgl.ndata["attr"] = F.one_hot(g_dgl.ndata["type"].view(-1).long(), num_classes=node_feature_dim).float()
    

    g_dgl.edata["attr"] = F.one_hot(g_dgl.edata["type"].view(-1).long(), num_classes=edge_feature_dim).float()
    
    return g_dgl



def train_model_simple(model, base_dir, model_dir, train_windows, metadata, optimizer, max_epoch, device, node_embeddings=None):
    model.train()
    
    n_dim = metadata['node_feature_dim']
    e_dim = metadata['edge_feature_dim']
    n_train = len(train_windows)
    
    epoch_iter = tqdm(range(max_epoch), desc=f"Training {os.path.basename(model_dir)}")
    for epoch in epoch_iter:
        epoch_loss = 0.0
        for i, window in enumerate(train_windows):
            try:
                g_nx, _ = load_entity_level_dataset(base_dir, window)
                g = transform_graph(g_nx, n_dim, e_dim, node_embeddings).to(device)
                
                model.train()
                loss = model(g)
                loss /= n_train
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                del g, g_nx, loss
            except Exception as e:
                print(f"Warning: Failed to process window {window}: {e}")
                continue
        
        epoch_iter.set_description(f"{os.path.basename(model_dir)} | Epoch {epoch+1} | loss: {epoch_loss:.4f}")
        
        checkpoint_path = f'{model_dir}/checkpoint_epoch_{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return model

def train_model_15(base_dir, train_windows, metadata, device, node_embeddings=None, max_epochs=100):
    model_15_config = {
        'model_id': 15,
        'name': 'high_dropout',
        'num_hidden': 64,
        'num_layers': 3,
        'n_heads': 4,
        'mask_rate': 0.5,
        'lr': 0.001,
        'feat_drop': 0.3,
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
    
    model_dir = f"{base_dir}/model_15"
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training Model 15 (AutoProv)")
    print(f"  Hidden dim: {model_15_config['num_hidden']}, Layers: {model_15_config['num_layers']}, "
          f"Heads: {model_15_config['n_heads']}, Mask rate: {model_15_config['mask_rate']}, "
          f"LR: {model_15_config['lr']}, Dropout: {model_15_config['feat_drop']}")
    print(f"{'='*70}\n")
    
    hyperparams_path = f"{model_dir}/hyperparams.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(model_15_config, f, indent=2)
    
    model = GMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=model_15_config['num_hidden'],
        n_layers=model_15_config['num_layers'],
        n_heads=model_15_config['n_heads'],
        activation=model_15_config['activation'],
        feat_drop=model_15_config['feat_drop'],
        negative_slope=model_15_config['negative_slope'],
        residual=model_15_config['residual'],
        mask_rate=model_15_config['mask_rate'],
        norm=model_15_config['norm'],
        loss_fn='sce',
        alpha_l=model_15_config['alpha_l']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model_15_config['lr'], weight_decay=model_15_config['weight_decay'])
    
    model = train_model_simple(
        model, base_dir, model_dir, train_windows, metadata, optimizer, max_epochs, device, node_embeddings
    )
    
    final_path = f'{model_dir}/checkpoint_final.pt'
    torch.save(model.state_dict(), final_path)
    
    model_config = {
        'n_dim': n_dim,
        'e_dim': e_dim,
        'hidden_dim': model_15_config['num_hidden'],
        'n_layers': model_15_config['num_layers'],
        'n_heads': model_15_config['n_heads'],
        'activation': model_15_config['activation'],
        'feat_drop': model_15_config['feat_drop'],
        'negative_slope': model_15_config['negative_slope'],
        'residual': model_15_config['residual'],
        'mask_rate': model_15_config['mask_rate'],
        'norm': model_15_config['norm'],
        'loss_fn': 'sce',
        'alpha_l': model_15_config['alpha_l'],
        'lr': model_15_config['lr'],
        'weight_decay': model_15_config['weight_decay']
    }
    with open(f'{model_dir}/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n{'='*70}")
    print(f"Model 15 training completed!")
    print(f"Saved to: {model_dir}")
    print(f"{'='*70}\n")

def train_entity_level(model, base_dir, train_windows, metadata, optimizer, max_epoch, device, 
                       node_embeddings=None, test_data=None, attack_scenarios=None, dataset_name=None):
    model.train()
    
    n_dim = metadata['node_feature_dim']
    e_dim = metadata['edge_feature_dim']
    n_train = len(train_windows)
    

    eval_results = []
    
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        epoch_loss = 0.0
        for i, window in enumerate(train_windows):
            g_nx, _ = load_entity_level_dataset(base_dir, window)
            g = transform_graph(g_nx, n_dim, e_dim, node_embeddings).to(device)
            
            model.train()
            loss = model(g)
            loss /= n_train
            optimizer.zero_grad()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            del g
        
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
        

        checkpoint_path = f'{base_dir}/models/checkpoint_epoch_{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        if (epoch + 1) % 10 == 0 or epoch == max_epoch - 1:
            print(f"  Saved checkpoint: {checkpoint_path}")
        

        if test_data is not None and attack_scenarios is not None:
            model.eval()
            

            x_train = []
            with torch.no_grad():
                for window in train_windows[:min(len(train_windows), 50)]:
                    g_nx, _ = load_entity_level_dataset(base_dir, window)
                    g = transform_graph(g_nx, n_dim, e_dim, node_embeddings).to(device)
                    embeddings = model.embed(g).detach().cpu().numpy()
                    x_train.append(embeddings)
                    del g
            x_train = np.concatenate(x_train, axis=0)
            

            x_test = []
            test_windows = test_data['test_windows']
            for window in test_windows:
                g_nx, _ = load_entity_level_dataset(base_dir, window)
                g = transform_graph(g_nx, n_dim, e_dim, node_embeddings).to(device)
                embeddings = model.embed(g).detach().cpu().numpy()
                x_test.append(embeddings)
                del g
            x_test = np.concatenate(x_test, axis=0)
            

            for attack_name, _, attack_date in attack_scenarios:

                y_test = test_data['attack_labels'][attack_name]
                
                if np.sum(y_test) > 0:

                    save_path = f'{base_dir}/results/distance_save_{dataset_name.lower()}_{attack_name}_epoch{epoch+1}.pkl'
                    auc_roc, auc_pr, scores, metrics = evaluate_entity_level_using_knn(
                        dataset_name, x_train, x_test, y_test, save_path
                    )
                    

                    eval_results.append({
                        'epoch': epoch + 1,
                        'attack_name': attack_name,
                        'AUC_ROC': float(auc_roc),
                        'AUC_PR': float(auc_pr)
                    })
            

            if eval_results:
                results_dir = f'{base_dir}/results'
                os.makedirs(results_dir, exist_ok=True)
                results_df = pd.DataFrame(eval_results)
                csv_path = f"{results_dir}/train_infer_{dataset_name.lower()}.csv"
                results_df.to_csv(csv_path, index=False)
            

            del x_train, x_test
            model.train()
    
    return model


def get_attack_scenarios(dataset):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    step_llm_dir = os.path.dirname(script_dir)
    pids_gt_dir = os.path.join(step_llm_dir, "PIDS_GT")
    
    attacks = {
        "THEIA": [
            ("Firefox_Backdoor_Drakon", os.path.join(pids_gt_dir, "THEIA", "node_Firefox_Backdoor_Drakon_In_Memory.csv"), "2018-04-10"),
            ("Browser_Extension_Drakon", os.path.join(pids_gt_dir, "THEIA", "node_Browser_Extension_Drakon_Dropper.csv"), "2018-04-12")
        ]
    }
    return attacks.get(dataset.upper(), [])

def load_malicious_ids_from_csv(csv_path):
    malicious_ids = set()
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                uuid = line.split(',')[0]
                malicious_ids.add(uuid)
    return malicious_ids

def evaluate_entity_level_using_knn(dataset, x_train, x_test, y_test, save_path=None, k=None):
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train_norm = (x_train - x_train_mean) / (x_train_std + 1e-6)
    x_test_norm = (x_test - x_train_mean) / (x_train_std + 1e-6)
    

    x_train_norm = x_train_norm.astype(np.float32)
    x_test_norm = x_test_norm.astype(np.float32)
    

    if k is not None:
        n_neighbors = k
    elif dataset.lower() == 'cadets':
        n_neighbors = 200
    else:
        n_neighbors = 10
    

    if save_path and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            mean_distance, distances = pickle.load(f)
        del x_train_norm, x_test_norm
    else:



        if FAISS_AVAILABLE:
            try:
                embedding_dim = x_train_norm.shape[1]
                

                index = faiss.IndexFlatL2(embedding_dim)
                

                if FAISS_GPU:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                

                index.add(x_train_norm)
                

                import random
                idx = list(range(x_train_norm.shape[0]))
                random.shuffle(idx)
                sample_size = min(50000, x_train_norm.shape[0])
                distances_train, _ = index.search(x_train_norm[idx][:sample_size], n_neighbors)
                

                distances_train = np.sqrt(distances_train)
                mean_distance = distances_train.mean()
                del distances_train
                

                distances, _ = index.search(x_test_norm, n_neighbors)
                distances = np.sqrt(distances)
                distances = distances.mean(axis=1)
                
                del index, x_train_norm, x_test_norm
                
            except Exception as e:
                print(f"\n⚠ FAISS failed: {str(e)[:100]}")
                print("  Falling back to sklearn...")
                FAISS_SUCCESS = False
            else:
                FAISS_SUCCESS = True
        else:
            FAISS_SUCCESS = False
        



        if not FAISS_SUCCESS:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
            nbrs.fit(x_train_norm)
            
            idx = list(range(x_train_norm.shape[0]))
            import random
            random.shuffle(idx)
            distances_train, _ = nbrs.kneighbors(x_train_norm[idx][:min(50000, x_train_norm.shape[0])], n_neighbors=n_neighbors)
            del x_train_norm
            mean_distance = distances_train.mean()
            del distances_train
            distances, _ = nbrs.kneighbors(x_test_norm, n_neighbors=n_neighbors)
            del x_test_norm
            distances = distances.mean(axis=1)
            del nbrs
        

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_dict = [mean_distance, distances]
            with open(save_path, 'wb') as f:
                pickle.dump(save_dict, f)
    

    score = distances / mean_distance
    del distances
    

    if y_test is not None and len(y_test) > 0:
        auc_roc = roc_auc_score(y_test, score)
        auc_pr = average_precision_score(y_test, score)
        prec, rec, threshold = precision_recall_curve(y_test, score)
        f1 = 2 * prec * rec / (rec + prec + 1e-9)
        best_idx = np.argmax(f1)
        
        return auc_roc, auc_pr, score, (prec, rec, threshold, f1)
    else:
        return None, None, score, None


def compute_attack_detection_precision(scores, attack_label_dict):
    if not attack_label_dict:
        return None

    num_nodes = len(scores)
    if num_nodes == 0:
        return None


    attack_indices = {}
    index_to_attacks = {}
    y_global = np.zeros(num_nodes, dtype=np.int8)

    for attack_name, labels in attack_label_dict.items():
        if labels is None or len(labels) != num_nodes:
            return None
        positive_idx = np.flatnonzero(labels > 0)
        if positive_idx.size > 0:
            attack_indices[attack_name] = positive_idx
            y_global[positive_idx] = 1
            for idx in positive_idx:
                index_to_attacks.setdefault(idx, set()).add(attack_name)

    if not attack_indices:
        return 0.0

    total_attacks = len(attack_indices)

    sorted_idx = np.argsort(-scores)
    sorted_labels = y_global[sorted_idx]

    tp = 0.0
    fp = 0.0
    seen_attacks = set()

    precisions = [0.0]
    detections = [0.0]

    for rank, idx in enumerate(sorted_idx):
        if sorted_labels[rank] == 1:
            tp += 1.0
        else:
            fp += 1.0

        if idx in index_to_attacks:
            seen_attacks.update(index_to_attacks[idx])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        detection_fraction = len(seen_attacks) / total_attacks

        precisions.append(precision)
        detections.append(detection_fraction)

    precisions = np.array(precisions, dtype=np.float64)
    detections = np.array(detections, dtype=np.float64)


    precision_to_detection = {}
    for precision, detection in zip(precisions, detections):
        key = round(float(precision), 6)
        if key in precision_to_detection:
            precision_to_detection[key] = max(precision_to_detection[key], detection)
        else:
            precision_to_detection[key] = detection

    unique_precisions = np.array(sorted(precision_to_detection.keys()), dtype=np.float64)
    unique_detections = np.array(
        [precision_to_detection[key] for key in unique_precisions], dtype=np.float64
    )

    if unique_precisions[0] > 0.0:
        unique_precisions = np.insert(unique_precisions, 0, 0.0)
        unique_detections = np.insert(unique_detections, 0, 0.0)
    if unique_precisions[-1] < 1.0:
        unique_precisions = np.append(unique_precisions, 1.0)
        unique_detections = np.append(unique_detections, unique_detections[-1])

    adp = np.trapz(unique_detections, unique_precisions)
    adp = float(max(0.0, min(1.0, adp)))
    return adp



def parse_args():
    parser = argparse.ArgumentParser(description="MAGIC Graph Learning for THEIA")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--baseline", action='store_true',
                        help="Train baseline model (original THEIA graphs)")
    mode_group.add_argument("--autoprov", action='store_true',
                        help="Train AutoProv model (RuleLLM graphs with LLM embeddings)")
    
    parser.add_argument("--dataset", type=str, default="theia",
                        help="Dataset name (theia)")
    parser.add_argument("--artifacts_dir", type=str, default=None,
                        help="Directory with preprocessed artifacts (auto-set based on mode)")
    parser.add_argument("--embedding", type=str, default="mpnet",
                        choices=["mpnet", "minilm", "roberta", "distilbert"],
                        help="Embedding model name (only used with --autoprov)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    dataset = args.dataset.upper()
    
    if dataset not in DATASET_DATES:
        print(f"Error: Dataset '{dataset}' not found in DATASET_DATES. Available: {list(DATASET_DATES.keys())}")
        return
    
    dataset_dates = DATASET_DATES[dataset]
    train_start_date = dataset_dates.get("train_start_date")
    train_end_date = dataset_dates.get("train_end_date")
    
    if train_start_date is None or train_end_date is None:
        print(f"Error: train_start_date or train_end_date not defined for dataset '{dataset}' in DATASET_DATES")
        return
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autoprov_dir = os.path.dirname(script_dir)
    artifacts_root = os.path.join(autoprov_dir, 'BIGDATA', 'MAGIC_artifacts')
    
    if args.baseline:
        if args.artifacts_dir is None:
            base_dir = os.path.join(artifacts_root, dataset)
        else:
            base_dir = args.artifacts_dir
        
        if not os.path.exists(base_dir):
            print(f"ERROR: Artifacts directory not found: {base_dir}")
            print(f"       Please run graph_gen_theia.py --baseline first")
            return
        
        print(f"\n{'='*70}")
        print(f"MAGIC Graph Learning - {dataset}")
        print(f"Mode: Baseline")
        print(f"{'='*70}\n")
        
        os.makedirs(f'{base_dir}/models', exist_ok=True)
        os.makedirs(f'{base_dir}/results', exist_ok=True)
        
        with open(f'{base_dir}/processed_data/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"Dataset: {dataset}")
        print(f"Node types: {metadata['node_feature_dim']}")
        print(f"Edge types: {metadata['edge_feature_dim']}")
        print(f"Training period: {train_start_date} to {train_end_date}")
        
        node_embeddings = None
        
        from datetime import datetime, timedelta
        start_date = datetime.strptime(train_start_date, "%Y-%m-%d")
        end_date = datetime.strptime(train_end_date, "%Y-%m-%d")
        
        train_windows = []
        current_date = start_date
        while current_date <= end_date:
            for hour in range(24):
                window_key = f"{current_date.strftime('%Y-%m-%d')}_{hour:02d}"
                graph_path = f'{base_dir}/processed_data/graphs/graph_{window_key}.pkl'
                if os.path.exists(graph_path):
                    train_windows.append(window_key)
            current_date += timedelta(days=1)
        
        print(f"Training windows: {len(train_windows)}")
        
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
        model = train_entity_level(model, base_dir, train_windows, metadata, optimizer, args.epochs, device,
                                  node_embeddings, test_data=None, attack_scenarios=None, dataset_name=dataset)
        
        final_path = f'{base_dir}/models/checkpoint_final.pt'
        torch.save(model.state_dict(), final_path)
        print(f"\nSaved final model: {final_path}")
        
        print(f"\n{'='*70}")
        print("Training completed successfully!")
        print(f"{'='*70}\n")
    
    elif args.autoprov:
        if args.artifacts_dir is None:
            base_dir = os.path.join(artifacts_root, f'{dataset}_rulellm_llmlabel_{args.embedding.lower()}')
        else:
            base_dir = args.artifacts_dir
        
        if not os.path.exists(base_dir):
            print(f"ERROR: Artifacts directory not found: {base_dir}")
            print(f"       Please run graph_gen_theia.py --autoprov first")
            return
        
        print(f"\n{'='*70}")
        print(f"MAGIC Graph Learning - {dataset}")
        print(f"Mode: AutoProv")
        print(f"LLM Embeddings: {args.embedding}")
        print(f"{'='*70}\n")
        
        os.makedirs(f'{base_dir}/results', exist_ok=True)
        
        with open(f'{base_dir}/processed_data/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"Dataset: {dataset}")
        print(f"Node types: {metadata['node_feature_dim']}")
        print(f"Edge types: {metadata['edge_feature_dim']}")
        print(f"Training period: {train_start_date} to {train_end_date}")
        
        embedding_path = f'{base_dir}/processed_data/node_embeddings.npy'
        if os.path.exists(embedding_path):
            node_embeddings = np.load(embedding_path)
            print(f"Loaded node embeddings: {node_embeddings.shape}")
        else:
            print(f"Error: Node embeddings not found at {embedding_path}")
            print(f"Please run graph_gen_theia.py --autoprov first")
            return
        
        from datetime import datetime, timedelta
        start_date = datetime.strptime(train_start_date, "%Y-%m-%d")
        end_date = datetime.strptime(train_end_date, "%Y-%m-%d")
        
        train_windows = []
        current_date = start_date
        while current_date <= end_date:
            for hour in range(24):
                window_key = f"{current_date.strftime('%Y-%m-%d')}_{hour:02d}"
                graph_path = f'{base_dir}/processed_data/graphs/graph_{window_key}.pkl'
                if os.path.exists(graph_path):
                    train_windows.append(window_key)
            current_date += timedelta(days=1)
        
        print(f"Training windows: {len(train_windows)}")
        
        train_model_15(
            base_dir, train_windows, metadata, device,
            node_embeddings=node_embeddings,
            max_epochs=100
        )

if __name__ == "__main__":
    main()

