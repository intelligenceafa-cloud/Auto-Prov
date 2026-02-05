# MAGIC

This guide explains how to run the MAGIC on AutoProv and Baseline graphs

1. [Displaying Results](#displaying-results)
2. [End-to-End Pipeline](#end-to-end-pipeline)
   - [THEIA Dataset](#theia-dataset)
   - [ATLAS Dataset](#atlas-dataset)

---

## Displaying Results

### THEIA Results

To display evaluation results for THEIA (baseline and AutoProv):

```bash
cd AutoProv/MAGIC
python eval_theia.py
```

---

### ATLAS Results

To display evaluation results for ATLAS (baseline and AutoProv):

```bash
cd AutoProv/MAGIC
python eval_atlas.py
```

**Default behavior:**
- Evaluates both baseline and AutoProv models
- Uses default paths: `AutoProv/BIGDATA/MAGIC_artifacts/ATLAS_artifacts/`
- AutoProv defaults: `embedding=mpnet`, `cee=gpt-4o`, `rule_generator=llama3_70b`, `llmfets-model=llama3:70b`

**Customize AutoProv evaluation:**

```bash
# Specify different embedding/model
python eval_atlas.py --embedding roberta --cee gpt-4o --rule_generator llama3_70b --llmfets-model llama3:70b

# Skip baseline evaluation
python eval_atlas.py --skip_baseline

# Skip AutoProv evaluation
python eval_atlas.py --skip_llmlabel

# Use custom artifacts directory
python eval_atlas.py --artifacts_root /path/to/artifacts --llmlabel_dir /path/to/llmlabel/artifacts
```

---

## End-to-End Pipeline

### THEIA Dataset

#### Step 1: Graph Generation

**Baseline mode** (raw log processing):
```bash
cd AutoProv/MAGIC
python graph_gen_theia.py --baseline --dataset_path ../BIGDATA/DARPA-E3/
```

**AutoProv mode** (CSV processing with LLM embeddings):
```bash
cd AutoProv/MAGIC
python graph_gen_theia.py --autoprov \
    --autoprov_graph_path ../BIGDATA/ExtractedProvGraph/ \
    --embedding mpnet
```

---

#### Step 2: Model Training

**Baseline mode:**
```bash
cd AutoProv/MAGIC
python learning_theia.py --baseline --epochs 50
```

**AutoProv mode:**
```bash
cd AutoProv/MAGIC
python learning_theia.py --autoprov
```

---

#### Step 3: Evaluation

```bash
cd AutoProv/MAGIC
python eval_theia.py
```

This evaluates both baseline and AutoProv models and displays results tables.

---

### ATLAS Dataset

#### Step 1: Graph Generation

**Baseline mode** (original ATLAS graphs):
```bash
cd AutoProv/MAGIC
python graph_gen_atlas.py --baseline \
    --atlas_graph_dir ../rule_generator/ATLAS/original_atlas_graph
```

**AutoProv mode** (RuleLLM graphs with LLM embeddings):
```bash
cd AutoProv/MAGIC
python graph_gen_atlas.py --autoprov \
    --cee gpt-4o \
    --rule_generator llama3_70b \
    --embedding mpnet \
    --llmfets-model llama3:70b
```

**Process all folders** (if `--cee` and `--rule_generator` are omitted):
```bash
cd AutoProv/MAGIC
python graph_gen_atlas.py --autoprov --embedding mpnet --llmfets-model llama3:70b
```

---

#### Step 2: Model Training

**Baseline mode:**
```bash
cd AutoProv/MAGIC
python learning_atlas.py --baseline --epochs 50
```

**AutoProv mode:**
```bash
cd AutoProv/MAGIC
python learning_atlas.py --autoprov \
    --cee gpt-4o \
    --rule_generator llama3_70b \
    --embedding mpnet \
    --llmfets-model llama3:70b
```

---

#### Step 3: Evaluation

```bash
cd AutoProv/MAGIC
python eval_atlas.py
```

**With custom parameters:**
```bash
cd AutoProv/MAGIC
python eval_atlas.py \
    --embedding mpnet \
    --cee gpt-4o \
    --rule_generator llama3_70b \
    --llmfets-model llama3:70b
```

This evaluates both baseline and AutoProv models and displays results tables.

---

#### Step 4: Attack Graph Extraction (ATLAS Only)

Extract attack subgraphs from detected anomalous nodes:

```bash
cd AutoProv/MAGIC
python attack_graph_atlas.py \
    --output_dir ./attack_graphs
```

**Or specify artifacts directory directly:**
```bash
cd AutoProv/MAGIC
python attack_graph_atlas.py \
    --artifacts_dir AutoProv/BIGDATA/MAGIC_artifacts/ATLAS_artifacts/rulellm_llmlabel_mpnet/gpt-4o_llama3_70b/llama3_70b \
    --output_dir ./attack_graphs
```

---

## Complete Workflow Examples

### THEIA - Complete Pipeline

```bash
# 1. Generate graphs (baseline)
python graph_gen_theia.py --baseline --dataset_path ../BIGDATA/DARPA-E3/

# 2. Generate graphs (autoprov)
python graph_gen_theia.py --autoprov --autoprov_graph_path ../BIGDATA/ExtractedProvGraph/ --embedding mpnet

# 3. Train models
python learning_theia.py --baseline --epochs 50
python learning_theia.py --autoprov

# 4. Evaluate
python eval_theia.py
```

### ATLAS - Complete Pipeline

```bash
# 1. Generate graphs (baseline)
python graph_gen_atlas.py --baseline --atlas_graph_dir ../rule_generator/ATLAS/original_atlas_graph

# 2. Generate graphs (autoprov)
python graph_gen_atlas.py --autoprov --cee gpt-4o --rule_generator llama3_70b --embedding mpnet --llmfets-model llama3:70b

# 3. Train models
python learning_atlas.py --baseline --epochs 50
python learning_atlas.py --autoprov --cee gpt-4o --rule_generator llama3_70b --embedding mpnet --llmfets-model llama3:70b

# 4. Evaluate
python eval_atlas.py --embedding mpnet --cee gpt-4o --rule_generator llama3_70b --llmfets-model llama3:70b

# 5. Extract attack graphs (autoprov only)
python attack_graph_atlas.py --embedding mpnet --cee gpt-4o --rule_generator llama3_70b --llmfets-model llama3:70b --output_dir ./attack_graphs
```

