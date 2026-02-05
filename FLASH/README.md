# FLASH

This guide explains how to run FLASH on AutoProv and Baseline graphs for THEIA dataset.

1. [Displaying Results](#displaying-results)
2. [End-to-End Pipeline](#end-to-end-pipeline)

---

## Displaying Results

### THEIA Results

To display evaluation results for THEIA (baseline and AutoProv):

```bash
cd AutoProv/FLASH
python eval_theia.py
```

**Default behavior:**
- Evaluates both baseline and AutoProv models
- Uses default paths: `AutoProv/BIGDATA/FLASH_artifacts/`
- AutoProv defaults: `embedding=word2vec`, model 36 at epoch 50

---

## End-to-End Pipeline

### THEIA Dataset

#### Step 1: Graph Generation

**Baseline mode** (raw log processing):
```bash
cd AutoProv/FLASH
python graph_gen_theia.py --baseline --dataset_path ../BIGDATA/DARPA-E3/
```

**AutoProv mode** (CSV processing with LLM embeddings):
```bash
cd AutoProv/FLASH
python graph_gen_theia.py --autoprov
```

AutoProv mode uses default path: `../BIGDATA/ExtractedProvGraph/`

---

#### Step 2: Model Training

**Baseline mode:**
```bash
cd AutoProv/FLASH
python learning_theia.py --baseline --epochs 20
```

**AutoProv mode:**
```bash
cd AutoProv/FLASH
python learning_theia.py --autoprov --embedding word2vec
```

Supported embeddings: `word2vec`, `mpnet`, `minilm`, `roberta`, `distilbert`, `fasttext`

---

#### Step 3: Evaluation

```bash
cd AutoProv/FLASH
python eval_theia.py
```

This evaluates both baseline and AutoProv models and displays results tables with:
- AUC-ROC
- AUC-PR
- ADP (Attack Detection Precision)

---

## Complete Workflow Example

### THEIA - Complete Pipeline

```bash
# 1. Generate graphs (baseline)
python graph_gen_theia.py --baseline --dataset_path ../BIGDATA/DARPA-E3/

# 2. Generate graphs (autoprov)
python graph_gen_theia.py --autoprov

# 3. Train models
python learning_theia.py --baseline --epochs 20
python learning_theia.py --autoprov --embedding word2vec

# 4. Evaluate
python eval_theia.py
```

