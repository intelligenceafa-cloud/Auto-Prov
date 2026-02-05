# Rule Generator

## Dataset Storage

Store ATLAS datasets at: `../../../BIGDATA/ATLAS/`
- Structure: `ATLAS/{S1|S2|S3|S4}/{audit|dns|firefox}/{timestamp_folder}/{log_type}.pkl`
- Labels: `../../../BIGDATA/ATLAS/labels/{S1|S2|S3|S4}/malicious_labels.txt`
- Candidate examples: `../../../clusterlogs_atlas/candidates-atlas_ablation/{embedding}/{cee}/{log_type}/candidate-output.json`

## Execution Order

### 1. Generate Regex Patterns
```bash
cd ATLAS/ablation
python rule-gen-llm-atlas.py \
    --cee <cee_name> \
    --model-name <model_name> \
    --file-type <edges|enames|vtypes> \
    --ollama-url <ollama_url> \
    --embedding roberta \
    --log-type <audit|dns|firefox> \
    --candidates-dir ../../../clusterlogs_atlas/candidates-atlas_ablation \
    --save-dir ./rules
```

Run for each `--file-type` (edges, enames, vtypes) and each `--log-type` (audit, dns, firefox).

### 2. Apply Patterns to Extract Graphs
```bash
python apply_rules.py \
    --cee <cee_name> \
    --model-name <model_name> \
    --dataset <S1|S2|S3|S4> \
    --log-type <audit|dns|firefox> \
    --rules-dir ./rules \
    --atlas-dir ../../../BIGDATA/ATLAS \
    --output-dir ./Extracted_Graph
```

Run for each dataset and log-type combination.

### 3. Fix CSV Columns (DNS/Firefox only)
```bash
python fix_csv_columns.py \
    --extracted-graph-dir ./Extracted_Graph/ATLAS \
    --datasets S1 S2 S3 S4
```

### 4. Merge and Label Graphs
```bash
python labelling.py \
    --extracted-graph-dir ./Extracted_Graph/ATLAS \
    --atlas-dir ../../../BIGDATA/ATLAS \
    --labels-dir ../../../BIGDATA/ATLAS/labels \
    --output-dir ./Extracted_Graph \
    --datasets S1 S2 S3 S4
```

### 5. Create AutoProv Format Graphs
```bash
python create_autoprov_atlas_graph.py \
    --merged-dir ./merged \
    --output-dir ./autoprov_atlas_graph \
    --cee-model <cee_name> \
    --model-name <model_name>
```

Or process multiple CPE/model combinations:
```bash
python create_autoprov_atlas_graph.py \
    --merged-dir ./merged \
    --output-dir ./autoprov_atlas_graph \
    --cee-models gpt-3.5-turbo_llama3_70b gpt-4o_llama3_70b
```

