# ATLAS Log Clustering Pipeline


## Pipeline Overview

1. **micro-cluster-atlas.py** - Select k most dissimilar seeds per timestamp
2. **cluster-micro-cluster-atlas.py** - Perform continual clustering on seeds
3. **sample-atlas.py** - Sample candidates from clustering results
4. **candidate_seeds-atlas.py** - Process LLM candidate outputs

## Required Directory Structure

### Embeddings (`BIGDATA/ATLAS_embeddings/`)

**Single-host scenarios (S1, S3, S4):**
```
ATLAS_embeddings/
  S1/
    audit/
      2018-11-02 21:00:00_2018-11-02 22:00:00/
        embeddings.pkl  # List of numpy arrays
    dns/
    firefox/
```

### Original Log Data (`BIGDATA/ATLAS/`)

**Single-host:**
```
ATLAS/
  S1/
    audit/
      2018-11-02 21:00:00_2018-11-02 22:00:00/
        audit.pkl  # List of log strings
    dns/
    firefox/
```

**Note:** `embeddings.pkl` should contain a list of numpy arrays (one per log entry). `{log_type}.pkl` should contain a list of log strings.

## Usage

### Step 1: Seed Selection

Select k most dissimilar seeds per timestamp using Farthest Point Sampling:

```bash
python micro-cluster-atlas.py \
  --embedding roberta \
  --k 1000 \
  --embeddings_dir ../BIGDATA/ATLAS_embeddings/ \
  --output_dir ./sample-micro-cluster-atlas
```


**Output:** `sample-micro-cluster-atlas/{embedding}/{log_type}/seeds_*.pkl`

### Step 2: Continual Clustering

Perform continual clustering on seeds chronologically:

```bash
python cluster-micro-cluster-atlas.py \
  --embedding roberta \
  --method dbstream \
  --base_dir ./sample-micro-cluster-atlas \
  --embeddings_dir ../BIGDATA/ATLAS_embeddings/ \
  --output_dir ./continual-clusters-atlas \
  --clustering_threshold 0.6 \
  --fading_factor 0.005
```

**Output:** `continual-clusters-atlas/{embedding}/{log_type}/continual_clusters_*.json`

### Step 3: Candidate Sampling

Sample n items from clusters (all from first timestamp, new clusters from subsequent):

```bash
python sample-atlas.py \
  --embedding roberta \
  --n_samples 5 \
  --continual_clusters_dir ./continual-clusters-atlas \
  --original_data_dir ../BIGDATA/ATLAS/ \
  --candidates_dir ./candidates-atlas
```

**Output:** `candidates-atlas/{embedding}/{log_type}/candidate_ids.pkl`, `candidate.pkl`

### Step 4: Process Candidate Outputs

Process LLM-generated candidate-output.json into structured files:

```bash
python candidate_seeds-atlas.py \
  --embedding roberta \
  --log_type audit \
  --candidates_dir ./candidates-atlas
```

**Output:** `candidates-atlas/{embedding}/{log_type}/candidate_{edges,enames,vtypes}.pkl`

## Dependencies

- `river` - Streaming clustering algorithms
- `numpy` - Numerical operations
- `tqdm` - Progress bars

