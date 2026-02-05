# OCR_APT

This guide explains how to run OCR_APT (OCRGCN) on AutoProv and Baseline graphs for THEIA dataset.

1. [Displaying Results](#displaying-results)
2. [End-to-End Pipeline](#end-to-end-pipeline)

---

## Displaying Results

### THEIA Results

To display evaluation results for THEIA (baseline and AutoProv):

```bash
cd AutoProv/OCR_APT
python eval_theia.py
```

**Default behavior:**
- Evaluates both baseline and AutoProv models
- Uses default paths: `AutoProv/BIGDATA/OCR_APT_artifacts/`
- AutoProv defaults: `embedding=mpnet`, model 8 at epoch 50

---

## End-to-End Pipeline

### THEIA Dataset

#### Step 1: Graph Generation

**Baseline mode** (raw log processing with OCR-APT features):
```bash
cd AutoProv/OCR_APT
python graph_gen_theia.py --baseline --dataset_path ../BIGDATA/DARPA-E3/
```

**AutoProv mode** (CSV processing with LLM embeddings):
```bash
cd AutoProv/OCR_APT
python graph_gen_theia.py --autoprov \
    --autoprov_graph_path ../BIGDATA/ExtractedProvGraph/ \
    --embedding mpnet
```

---

#### Step 2: Model Training

**Baseline mode:**
```bash
cd AutoProv/OCR_APT
python learning_theia.py --baseline --epochs 100
```

**AutoProv mode:**
```bash
cd AutoProv/OCR_APT
python learning_theia.py --autoprov --embedding mpnet --epochs 50
```

---

#### Step 3: Evaluation

```bash
cd AutoProv/OCR_APT
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
python graph_gen_theia.py --autoprov --autoprov_graph_path ../BIGDATA/ExtractedProvGraph/ --embedding mpnet

# 3. Train models
python learning_theia.py --baseline --epochs 100
python learning_theia.py --autoprov --embedding mpnet --epochs 50

# 4. Evaluate
python eval_theia.py
```

