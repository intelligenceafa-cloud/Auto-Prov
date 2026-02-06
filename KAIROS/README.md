# KAIROS


1. [Displaying Results](#displaying-results)
2. [End-to-End Pipeline](#end-to-end-pipeline)
3. [Artifacts and Paths](#artifacts-and-paths)
4. [Complete Workflow Examples](#complete-workflow-examples)

---

## Displaying Results

To evaluate and display ATLAS results (baseline and AutoProv):

```bash
cd AutoProv/KAIROS
python eval_atlas.py
```

**Default behavior:**
- Evaluates both baseline and AutoProv models
- Baseline: `{artifacts_root}/original_atlas_graph` (model: `tgn_model_epoch_50.pt` or `tgn_model_final.pt`)
- AutoProv: `{artifacts_root}/rulellm_llmlabel_{embedding}/{cee}_{rule_generator}` (model: `models/model_6/tgn_model_epoch_40.pt`)
- Default `artifacts_root`: `AutoProv/BIGDATA/KAIROS_artifacts/ATLAS_artifacts`
- Default AutoProv config: `embedding=mpnet`, `cee=gpt-4o`, `rule_generator=llama3_70b`

**Customize evaluation:**

```bash
# Specify embedding and model names
python eval_atlas.py --embedding mpnet --cee gpt-4o --rule_generator llama3_70b

# Use custom artifacts root
python eval_atlas.py --artifacts_root /path/to/ATLAS_artifacts

# Override baseline or autoprov directory
python eval_atlas.py --baseline_dir /path/to/original_atlas_graph --autoprov_dir /path/to/rulellm_artifacts

```

---

## End-to-End Pipeline

### Step 1: Graph Generation

**Baseline mode** (original ATLAS graphs; FeatureHasher node features, no LLM):

```bash
cd AutoProv/KAIROS
python graph_gen_atlas.py --baseline
```

Default input: `AutoProv/rule_generator/ATLAS/original_atlas_graph`. Override with:

```bash
python graph_gen_atlas.py --baseline --original_atlas_graph_dir /path/to/original_atlas_graph
```

Output: `AutoProv/KAIROS/ATLAS_artifacts/original_atlas_graph/` (train/test temporal graphs, edge labels, node mappings, etc.; files are zipped).

---

**AutoProv mode** (RuleLLM graphs with LLM embeddings):

Requires `--cee` and `--rule_generator`. Input graphs are read from:

`AutoProv/rule_generator/ATLAS/ablation/autoprov_atlas_graph/{cee}_{rule_generator}/`

Pre-computed embeddings are read from `AutoProv/BIGDATA/llmfets-pca-embedding` and `AutoProv/BIGDATA/llmfets-embedding` (override with `--pca_embedding_path` and `--embedding_path`).

```bash
cd AutoProv/KAIROS
python graph_gen_atlas.py --autoprov \
    --cee gpt-4o \
    --rule_generator llama3_70b \
    --embedding mpnet \
    --llmfets-model llama3:70b
```

Optional:
- `--pca_dim 128` (default), `--no_pca` to disable PCA
- `--pca_embedding_path`, `--embedding_path` for custom embedding roots
- `--llmfets-model` (default: `llama3:70b`)

Output: `AutoProv/KAIROS/ATLAS_artifacts/rulellm_llmlabel_{embedding}/{cee}_{rule_generator}/{llmfets_model}/` (e.g. `rulellm_llmlabel_mpnet/gpt-4o_llama3_70b/llama3_70b/`).

---

### Step 2: Model Training

**Baseline mode:**

Expects artifacts under `AutoProv/BIGDATA/KAIROS_artifacts/ATLAS_artifacts/original_atlas_graph` by default (see [Artifacts and Paths](#artifacts-and-paths)).

```bash
cd AutoProv/KAIROS
python learning_atlas.py --baseline --epochs 50
```

Checkpoints: `models/tgn_model_epoch_50.pt.zip`, `tgn_model_final.pt.zip`.

**AutoProv mode:**

Expects artifacts under `AutoProv/BIGDATA/KAIROS_artifacts/ATLAS_artifacts/rulellm_llmlabel_{embedding}/{cee}_{rule_generator}/`.

```bash
cd AutoProv/KAIROS
python learning_atlas.py --autoprov \
    --cee gpt-4o \
    --rule_generator llama3_70b \
    --embedding mpnet
```

Trains “model 6” for 40 epochs. Checkpoints: `models/model_6/tgn_model_epoch_{1..40}.pt.zip`.

Override artifacts location:

```bash
python learning_atlas.py --baseline --artifacts_dir /path/to/original_atlas_graph
python learning_atlas.py --autoprov --artifacts_dir /path/to/rulellm_llmlabel_mpnet/gpt-4o_llama3_70b
```

---

### Step 3: Evaluation

```bash
cd AutoProv/KAIROS
python eval_atlas.py
```

With custom parameters:

```bash
python eval_atlas.py \
    --embedding mpnet \
    --cee gpt-4o \
    --rule_generator llama3_70b
```

Evaluation uses log-level ground truth under `STEP-LLM/rule_generator/ATLAS/ablation/log_level_ground_truth` and edge-to-log mappings in the artifacts. Results are printed as a table (attack type, AUC-ROC, AUC-PR, ADP).

---

## Artifacts and Paths

| Script            | Default artifacts root / output |
|-------------------|---------------------------------|
| **graph_gen_atlas.py** | **Output:** `AutoProv/KAIROS/ATLAS_artifacts/` (no `BIGDATA`). Baseline: `original_atlas_graph/`; AutoProv: `rulellm_llmlabel_{embedding}/{cee}_{rule_generator}/{llmfets_model}/`. |
| **learning_atlas.py**  | **Reads:** `AutoProv/BIGDATA/KAIROS_artifacts/ATLAS_artifacts/` (same layout as above under this root). |
| **eval_atlas.py**      | **Reads:** `AutoProv/BIGDATA/KAIROS_artifacts/ATLAS_artifacts/` (same as learning). |

To run the full pipeline without moving files:

1. After graph generation, point learning and eval at the graph-gen output, e.g.:

   ```bash
   python learning_atlas.py --baseline --artifacts_dir AutoProv/KAIROS/ATLAS_artifacts/original_atlas_graph
   python learning_atlas.py --autoprov --artifacts_dir AutoProv/KAIROS/ATLAS_artifacts/rulellm_llmlabel_mpnet/gpt-4o_llama3_70b/llama3_70b
   python eval_atlas.py --artifacts_root AutoProv/KAIROS/ATLAS_artifacts
   ```

2. Or copy/symlink `AutoProv/KAIROS/ATLAS_artifacts` to `AutoProv/BIGDATA/KAIROS_artifacts/ATLAS_artifacts` and use the default flags for learning and eval.

---

## Complete Workflow Examples

### Baseline – Full pipeline

```bash
cd AutoProv/KAIROS

# 1. Generate baseline graphs
python graph_gen_atlas.py --baseline --original_atlas_graph_dir ../rule_generator/ATLAS/original_atlas_graph

# 2. Train (use same artifacts root; if you use BIGDATA, copy artifacts there first or set --artifacts_dir)
python learning_atlas.py --baseline --epochs 50 --artifacts_dir AutoProv/KAIROS/ATLAS_artifacts/original_atlas_graph

# 3. Evaluate
python eval_atlas.py --artifacts_root AutoProv/KAIROS/ATLAS_artifacts
```

### AutoProv – Full pipeline

```bash
cd AutoProv/KAIROS

# 1. Generate AutoProv graphs
python graph_gen_atlas.py --autoprov --cee gpt-4o --rule_generator llama3_70b --embedding mpnet --llmfets-model llama3:70b

# 2. Train
python learning_atlas.py --autoprov --cee gpt-4o --rule_generator llama3_70b --embedding mpnet \
    --artifacts_dir AutoProv/KAIROS/ATLAS_artifacts/rulellm_llmlabel_mpnet/gpt-4o_llama3_70b/llama3_70b

# 3. Evaluate
python eval_atlas.py --artifacts_root AutoProv/KAIROS/ATLAS_artifacts --embedding mpnet --cee gpt-4o --rule_generator llama3_70b
```

---

## File Roles

| File | Role |
|------|------|
| **graph_gen_atlas.py** | Builds ATLAS temporal graphs from original or RuleLLM graphs; baseline uses FeatureHasher node features; AutoProv uses pre-computed LLM embeddings (with optional PCA). Writes train/test graphs, edge labels, node mappings, and metadata (all zipped). |
| **kairos_utils.py** | Shared helpers: dataset dates (THEIA, FIVEDIRECTIONS), time conversions, path/IP hierarchical lists, ADP and metrics, benign filters, edge-type mapping. Used by other KAIROS/STEP-LLM code; not required by the minimal graph_gen → learning → eval flow above. |
| **learning_atlas.py** | Trains TGN (memory + GNN + link predictor): baseline 50 epochs (saves epoch 50 and final), AutoProv “model 6” 40 epochs. Reads zipped graphs/labels from the artifacts dir; writes zipped checkpoints. |
| **eval_atlas.py** | Loads baseline or AutoProv TGN checkpoint, runs inference on test temporal graphs, aggregates edge losses to log level via edge-to-log mapping, loads log-level ground truth, computes AUC-ROC, AUC-PR, and ADP and prints the results table. |
