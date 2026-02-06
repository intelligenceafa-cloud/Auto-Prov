# Functional Features Pipeline

Extract LLM type/functionality features from entity names, build behavioral profiles from provenance graphs, and classify unknown entities with the behavioral classifier. Run all commands from this directory (`AutoProv/functional_fets`).

---

## 1. Extract LLM features from an enames file

**Input:** A pickle file `enames_{dataset}.pkl` in this directory containing a list of entity names (paths/strings). Example: `enames_atlas.pkl`.

**Steps (run in order):**

```bash
# 1) Normalize enames and build maps (writes maps/summarized-enames_*.json, maps/ename-cluster-map_*.json)
python file-processing.py --dataset atlas

# 2) Classify each unique ename as VALID / INVALID / COMMAND-LINE (writes file_classification_results/{llm_name}/ename_validity_{dataset}.json)
python file-check.py --dataset_name atlas --llm_name llama3:70b --ollama_url <your_ollama_url>

# 3) For each valid ename, get Type + Functionality via LLM (writes llm-fets/{llm_name}/ename_fets_{dataset}.json)
# For Ollama models:
python feature-extraction.py --dataset atlas --llm_name llama3:70b --ollama_url <your_ollama_url>
# For OpenAI (gpt-4o), create config.yaml with OpenAI API key: token: "your-api-key"
python feature-extraction.py --dataset atlas --llm_name gpt-4o
```

- For step 2/3, use the same `--llm_name` (e.g. `llama3:70b`, `gpt-4o` for feature-extraction if using OpenAI).  
- `feature-extraction.py` reads validity from `file_classification_results/{llm_name}/ename_validity_{dataset}.json`.  
- Omit `--dataset` in step 1 to process every `enames_*.pkl` in the directory.
- **Important:** `--ollama_url` is required for Ollama models. For OpenAI (gpt-4o), create `config.yaml` with `token: "your-api-key"`.
- Use `--gpus` to specify GPU IDs (e.g., `--gpus 0,1,2`); omit to use all available GPUs.

---

## 2. Extract behavioral profile

**Input:** Provenance graph data (CSV or ATLAS zip) and the LLM features from section 1 (validity + `llm-fets`). Script reads `file_classification_results`, `llm-fets`, and optional `edge_type_validation_{dataset}.json` from this directory.

```bash
python behavioral-profile.py \
  --data-path /path/to/graph/csv/or/zip \
  --dataset atlas \
  --llmfets-model llama3:70b \
  --cee llama3:70b \
  --rule-generator qwen2:72b \
  --script-dir ./
```

- For ATLAS, `--cee` and `--rule-generator` are required.  
- Add `--causal` or `--timeoh` for causal or per-timestamp one-hop profiles.  
- Output: `behavioral-profiles/{llmfets-model}/` (or `.../causal/`, `.../timeoh/`) — typed/untyped nodes, enames, `.npz` matrices, metadata.

---

## 3. Classify unknown entities (behavioral classifier)

**Input:** Behavioral profiles from section 2 and ename embeddings. Build ename embeddings once from the profile enames:

```bash
# Build ename embeddings (reads typed/untyped enames from behavioral-profiles; writes ename-embeddings/ or ename-embeddings/{model}/)
python ename_embedding.py --dataset atlas --llmfets-model llama3:70b --embedding roberta --timeoh
```

Then run the classifier (1-NN on behavioral profiles, with ename-based tie-breaking):

```bash
python behavioral_classifier.py \
  --dataset atlas \
  --llmfets-model llama3:70b \
  --embedding roberta \
  --timeoh
```

- Use `--causal` or `--timeoh` to match the profile type used in section 2.  
- Output: `behavioral-profiles/.../untype2type_nodes_{dataset}.json`, `untype2type_nodes_functionality_{dataset}.json`, `untype2type_meta_{dataset}.json`.

---