# Rule Generator

## Input Data Structure

### THEIA Dataset
Store THEIA datasets at: `../../../BIGDATA/DARPA-E3/`
- Structure: `DARPA-E3/{THEIA|FIVEDIRECTIONS}/{timestamp_folder}/logs.pkl` or `logs.pkl.zip`
- Timestamp format: `YYYY-MM-DD HH:MM:SS_YYYY-MM-DD HH:MM:SS`
- Candidate examples: `{candidates_dir}/{DATASET}/{embedding}/candidate_{edges|enames|vtypes}.pkl`

### ATLAS Dataset
Store ATLAS datasets at: `../../../BIGDATA/ATLAS/`
- Structure: `ATLAS/{S1|S2|S3|S4}/{audit|dns|firefox}/{timestamp_folder}/{log_type}.pkl`
- Labels: `ATLAS/labels/{S1|S2|S3|S4}/malicious_labels.txt`
- Candidate examples: `{candidates_dir}/{embedding}/{cee}/{log_type}/candidate-output.json`

## Execution

### THEIA Dataset

#### 1. Generate Regex Patterns
```bash
cd THEIA
python rule-gen-llm.py \
    --dataset THEIA \
    --embedding mpnet \
    --file-type edges \
    --ollama-url <ollama_url> \
    --model-name llama3:70b \
    --candidates-dir ../../../clusterlogs_theia/candidates-simple \
    --save-dir ./rules
```

Run for each `--file-type` (edges, enames, vtypes).

#### 2. Apply Patterns to Extract Graphs
```bash
python apply-rules.py \
    --dataset THEIA \
    --embedding mpnet \
    --base-path ../../../BIGDATA/DARPA-E3 \
    --output-path ../../../BIGDATA/ExtractedProvGraph
```

### ATLAS Dataset

#### 1. Generate Regex Patterns
```bash
cd ATLAS/ablation
python rule-gen-llm-atlas.py \
    --cee <cee_name> \
    --model-name llama3:70b \
    --file-type edges \
    --ollama-url <ollama_url> \
    --embedding roberta \
    --log-type audit \
    --candidates-dir ../../../clusterlogs_atlas/candidates-atlas_ablation \
    --save-dir ./rules
```

Run for each `--file-type` (edges, enames, vtypes) and each `--log-type` (audit, dns, firefox).

#### 2. Apply Patterns to Extract Graphs
```bash
python apply_rules.py \
    --cee <cee_name> \
    --model-name llama3:70b \
    --dataset S1 \
    --log-type audit \
    --rules-dir ./rules \
    --atlas-dir ../../../BIGDATA/ATLAS \
    --output-dir ./Extracted_Graph
```

Run for each dataset (S1, S2, S3, S4) and log-type combination.

#### 3. Fix CSV Columns (DNS/Firefox only)
```bash
python fix_csv_columns.py \
    --extracted-graph-dir ./Extracted_Graph/ATLAS \
    --datasets S1 S2 S3 S4
```

#### 4. Merge and Label Graphs
```bash
python labelling.py \
    --extracted-graph-dir ./Extracted_Graph/ATLAS \
    --atlas-dir ../../../BIGDATA/ATLAS \
    --labels-dir ../../../BIGDATA/ATLAS/labels \
    --output-dir ./Extracted_Graph \
    --datasets S1 S2 S3 S4
```

#### 5. Create AutoProv Format Graphs
```bash
python create_autoprov_atlas_graph.py \
    --merged-dir ./merged \
    --output-dir ./autoprov_atlas_graph \
    --cee-model <cee_name> \
    --model-name llama3:70b
```

Or process multiple CPE/model combinations:
```bash
python create_autoprov_atlas_graph.py \
    --merged-dir ./merged \
    --output-dir ./autoprov_atlas_graph \
    --cee-models gpt-3.5-turbo_llama3_70b gpt-4o_llama3_70b
```
