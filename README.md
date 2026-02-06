# AutoProv: KAIROS / MAGIC / OCR_APT run guide (concise)

This README gives the minimum steps to run `eval_{dataset}.py` and the end-to-end graph generation → training → inference for:
- `AutoProv/KAIROS`
- `AutoProv/MAGIC`
- `AutoProv/OCR_APT`

All paths below assume datasets and features live under `AutoProv/BIGDATA`. Override with CLI flags if your paths differ.

## 1) Dataset and feature placement (required)

### Raw logs (baseline mode: THEIA / FIVEDIRECTIONS)
Used by KAIROS, MAGIC, OCR_APT baseline graph generation.

```
AutoProv/BIGDATA/DARPA-E3/
  THEIA/
    <timestamp_dir>/
      logs.pkl            # or logs.pkl.zip
  FIVEDIRECTIONS/
    <timestamp_dir>/
      logs.pkl            # or logs.pkl.zip
```

Each `<timestamp_dir>` is a single time window (directory name includes date/time). The code loads `logs.pkl` or `logs.pkl.zip` from each directory.

### Extracted provenance graphs (RuleLLM / AutoProv mode)
Used by KAIROS/MAGIC/OCR_APT AutoProv graph generation.

```
AutoProv/BIGDATA/ExtractedProvGraph/
  THEIA/
    <timestamp_dir>/
      *.csv   # one CSV per timestamp dir
  FIVEDIRECTIONS/
    <timestamp_dir>/
      *.csv
```

Each CSV must contain columns like: `source_id`, `dest_id`, `action`, `timestamp`,
`source_enames`, `dest_enames`.

### LLM embeddings (for --llmlabel / --llmfunc)

PCA embeddings (preferred, required by OCR_APT AutoProv):
```
AutoProv/BIGDATA/llmfets-pca-embedding/
  theia|fivedirections/
    <embedding>/
      type_pca{pca_dim}_all.pkl
      type_pca{pca_dim}_model.pkl
      functionality_pca{pca_dim}_all.pkl
      functionality_pca{pca_dim}_model.pkl
```

Raw embeddings (fallback):
```
AutoProv/BIGDATA/llmfets-embedding/
  theia|fivedirections/
    <embedding>/
      type_all.pkl
      functionality_all.pkl
```

### ATLAS labels (used by MAGIC/KAIROS ATLAS eval)
```
AutoProv/BIGDATA/ATLAS/labels/
  S1/  S2/  S3/  S4/
```

### ATLAS embeddings (for candidate generation)
Required for clustering pipeline that generates candidate logs.
```
AutoProv/BIGDATA/ATLAS_embeddings/
  S1|S2|S3|S4/
    audit|dns|firefox/
      <timestamp_dir>/
        embeddings.pkl  # List of numpy arrays (one per log entry)
```

### ATLAS graphs (baseline + RuleLLM)
Used by KAIROS/MAGIC ATLAS graph generation.
```
AutoProv/rule_generator/ATLAS/original_atlas_graph/              # baseline
AutoProv/rule_generator/ATLAS/ablation/autoprov_atlas_graph/     # autoprov
  <cee>_<rule_generator>/
```

## 2) KAIROS

### Eval scripts
```
python AutoProv/KAIROS/eval_theia.py --artifacts_dir AutoProv/BIGDATA/KAIROS_artifacts
python AutoProv/KAIROS/eval_atlas.py --artifacts_root AutoProv/BIGDATA/KAIROS_artifacts/ATLAS_artifacts
```

### End-to-end: THEIA / FIVEDIRECTIONS
Rule generation → graph extraction → graph gen → train → inference
```
# Step 1: Generate regex patterns (for RuleLLM mode)
cd AutoProv/rule_generator/THEIA
python rule-gen-llm.py \
  --dataset THEIA \
  --embedding mpnet \
  --file-type edges \
  --ollama-url <ollama_url> \
  --model-name llama3:70b \
  --candidates-dir ../../../clusterlogs_theia/candidates-simple \
  --save-dir ./rules
# Repeat for enames and vtypes (--file-type enames, --file-type vtypes)

# Step 2: Apply patterns to extract graphs
python apply-rules.py \
  --dataset THEIA \
  --embedding mpnet \
  --base-path ../../../BIGDATA/DARPA-E3 \
  --output-path ../../../BIGDATA/ExtractedProvGraph

# Step 3: Graph generation (raw logs - baseline)
cd ../../KAIROS
python graph_gen.py \
  --dataset theia \
  --dataset_path AutoProv/BIGDATA/DARPA-E3 \
  --artifacts_dir AutoProv/BIGDATA/KAIROS_artifacts

# Step 4: Graph generation (RuleLLM + LLM features)
python graph_gen.py \
  --dataset theia --rulellm --llmlabel --embedding mpnet \
  --extracted_graph_path AutoProv/BIGDATA/ExtractedProvGraph \
  --embedding_path AutoProv/BIGDATA/llmfets-embedding \
  --pca_embedding_path AutoProv/BIGDATA/llmfets-pca-embedding \
  --artifacts_dir AutoProv/BIGDATA/KAIROS_artifacts

# Training
python AutoProv/KAIROS/graph_learning.py \
  --dataset theia \
  --artifacts_dir AutoProv/BIGDATA/KAIROS_artifacts

# Inference
python AutoProv/KAIROS/infer_our.py \
  --dataset theia \
  --artifacts_dir AutoProv/BIGDATA/KAIROS_artifacts
```

### End-to-end: ATLAS
```
# Prerequisite: Generate candidate logs (if not already done)
# See AutoProv/clusterlogs_atlas/README.md for full details
# This generates candidate.pkl and candidate_ids.pkl files needed by extract_edges.py
cd AutoProv/clusterlogs_atlas
# Step 1: Seed selection
python micro-cluster-atlas.py --embedding roberta --k 1000 \
  --embeddings_dir ../BIGDATA/ATLAS_embeddings/ \
  --output_dir ./sample-micro-cluster-atlas
# Step 2: Continual clustering
python cluster-micro-cluster-atlas.py --embedding roberta --method dbstream \
  --base_dir ./sample-micro-cluster-atlas \
  --embeddings_dir ../BIGDATA/ATLAS_embeddings/ \
  --output_dir ./continual-clusters-atlas
# Step 3: Sample candidates
python sample-atlas.py --embedding roberta --n_samples 5 \
  --continual_clusters_dir ./continual-clusters-atlas \
  --original_data_dir ../BIGDATA/ATLAS/ \
  --candidates_dir ./candidates-atlas
# Output: candidates-atlas/{embedding}/{log_type}/candidate.pkl, candidate_ids.pkl

# Step 0: Generate candidate edges (required before rule generation)
# Note: Create config.yaml with OpenAI API key: token: "your-api-key"
cd ../candidate_edge_extractor/ATLAS/ablation

# Step 0a: Check log characteristics
python check_log_characteristics.py \
  --llm_name gpt-4o \
  --model_type openai \
  --config ./config.yaml
# Or for Ollama: add --ollama_url <ollama_url> and --model_type ollama

# Step 0b: Extract edges using LLM
# Requires: candidate.pkl and candidate_ids.pkl from clustering pipeline (see Prerequisite above)
python extract_edges.py \
  --llm_name gpt-4o \
  --model_type openai \
  --config ./config.yaml \
  --embedding roberta \
  --log_types audit dns firefox \
  --llm_iterations 7 \
  --candidates_dir ../../../clusterlogs_atlas/candidates-atlas
# Or for Ollama: add --ollama_url <ollama_url> and --model_type ollama
# This creates outputs in ./outputs/{embedding}/{llm_name}/{log_type}/

# Step 0c: Convert outputs to candidate format
# For DNS/Firefox (resolved edges):
python cpe_output_t1.py \
  --llm_name gpt-4o \
  --embedding roberta \
  --log_types dns firefox \
  --candidates_dir ../../../clusterlogs_atlas/candidates-atlas_ablation

# For Audit (vtypes, enames, edges):
python cpe_output_t2.py \
  --llm_name gpt-4o \
  --embedding roberta \
  --log_type audit \
  --candidates_dir ../../../clusterlogs_atlas/candidates-atlas_ablation
# Output: candidates-atlas_ablation/{embedding}/{llm_name}/{log_type}/candidate-output.json

# Step 1: Generate regex patterns (uses candidate-output.json from Step 0)
cd ../../rule_generator/ATLAS/ablation
python rule-gen-llm-atlas.py \
  --cee gpt-4o \
  --model-name llama3:70b \
  --file-type edges \
  --ollama-url <ollama_url> \
  --embedding roberta \
  --log-type audit \
  --candidates-dir ../../../clusterlogs_atlas/candidates-atlas_ablation \
  --save-dir ./rules
# Repeat for enames, vtypes and other log-types (dns, firefox)

# Step 2: Apply patterns to extract graphs
python apply_rules.py \
  --cee gpt-4o \
  --model-name llama3:70b \
  --dataset S1 \
  --log-type audit \
  --rules-dir ./rules \
  --atlas-dir ../../../BIGDATA/ATLAS \
  --output-dir ./Extracted_Graph
# Repeat for S2, S3, S4 and other log-types

# Step 3: Fix CSV columns (DNS/Firefox only)
python fix_csv_columns.py \
  --extracted-graph-dir ./Extracted_Graph/ATLAS \
  --datasets S1 S2 S3 S4

# Step 4: Merge and label graphs
python labelling.py \
  --extracted-graph-dir ./Extracted_Graph/ATLAS \
  --atlas-dir ../../../BIGDATA/ATLAS \
  --labels-dir ../../../BIGDATA/ATLAS/labels \
  --output-dir ./Extracted_Graph \
  --datasets S1 S2 S3 S4

# Step 5: Create AutoProv format graphs
python create_autoprov_atlas_graph.py \
  --merged-dir ./merged \
  --output-dir ./autoprov_atlas_graph \
  --cee-model gpt-4o \
  --model-name llama3:70b

# Step 6: Graph generation (baseline)
cd ../../../KAIROS
python graph_gen_atlas.py --baseline

# Step 7: Graph generation (autoprov)
python graph_gen_atlas.py --autoprov \
  --cee gpt-4o --rule_generator llama3_70b --embedding mpnet

# Training
python AutoProv/KAIROS/learning_atlas.py --baseline
python AutoProv/KAIROS/learning_atlas.py --autoprov --cee gpt-4o --rule_generator llama3_70b --embedding mpnet

# Evaluation
python AutoProv/KAIROS/eval_atlas.py
```
Note: `graph_gen_atlas.py` writes to `AutoProv/KAIROS/ATLAS_artifacts/`.  
`learning_atlas.py` and `eval_atlas.py` default to `AutoProv/BIGDATA/KAIROS_artifacts/ATLAS_artifacts/`.  
Either pass `--artifacts_dir/--artifacts_root` to match, or symlink/copy the output folder.

## 3) MAGIC

### Eval scripts
```
python AutoProv/MAGIC/eval_theia.py
python AutoProv/MAGIC/eval_atlas.py
```

### End-to-end: THEIA
```
# Step 1-2: Generate and apply rules (see KAIROS THEIA steps above)
# Then proceed with graph generation:

# Graph generation (baseline)
python AutoProv/MAGIC/graph_gen_theia.py --baseline \
  --dataset_path AutoProv/BIGDATA/DARPA-E3

# Graph generation (autoprov)
python AutoProv/MAGIC/graph_gen_theia.py --autoprov \
  --autoprov_graph_path AutoProv/BIGDATA/ExtractedProvGraph \
  --embedding mpnet

# Training
python AutoProv/MAGIC/learning_theia.py --baseline
python AutoProv/MAGIC/learning_theia.py --autoprov --embedding mpnet

# Evaluation
python AutoProv/MAGIC/eval_theia.py
```

### End-to-end: ATLAS
```
# Step 0-5: Generate candidates, rules, and apply rules (see KAIROS ATLAS steps above)
# Then proceed with graph generation:

# Graph generation (baseline)
python AutoProv/MAGIC/graph_gen_atlas.py --baseline \
  --atlas_graph_dir AutoProv/rule_generator/ATLAS/original_atlas_graph

# Graph generation (autoprov)
python AutoProv/MAGIC/graph_gen_atlas.py --autoprov \
  --cee gpt-4o --rule_generator llama3_70b --embedding mpnet

# Training
python AutoProv/MAGIC/learning_atlas.py --baseline
python AutoProv/MAGIC/learning_atlas.py --autoprov --cee gpt-4o --rule_generator llama3_70b --embedding mpnet

# Evaluation
python AutoProv/MAGIC/eval_atlas.py
```

## 4) OCR_APT

### Eval script
```
python AutoProv/OCR_APT/eval_theia.py
```

### End-to-end: THEIA
```
# Step 1-2: Generate and apply rules (see KAIROS THEIA steps above)
# Then proceed with graph generation:

# Graph generation (baseline)
python AutoProv/OCR_APT/graph_gen_theia.py --baseline \
  --dataset_path AutoProv/BIGDATA/DARPA-E3

# Graph generation (autoprov)
python AutoProv/OCR_APT/graph_gen_theia.py --autoprov \
  --extracted_graph_path AutoProv/BIGDATA/ExtractedProvGraph \
  --embedding mpnet --pca_dim 128

# Training
python AutoProv/OCR_APT/learning_theia.py --baseline --epochs 100
python AutoProv/OCR_APT/learning_theia.py --autoprov --embedding mpnet --epochs 50

# Evaluation
python AutoProv/OCR_APT/eval_theia.py
```

## 5) Artifacts output locations (defaults)
- KAIROS: `AutoProv/BIGDATA/KAIROS_artifacts/` (THEIA/FIVEDIRECTIONS);  
  ATLAS graph gen outputs to `AutoProv/KAIROS/ATLAS_artifacts/` unless redirected.
- MAGIC: `AutoProv/BIGDATA/MAGIC_artifacts/`
- OCR_APT: `AutoProv/BIGDATA/OCR_APT_artifacts/`

# Auto-Prov
