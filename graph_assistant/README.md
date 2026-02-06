# Graph Assistant

Attack summarization pipeline for provenance graphs using LLM-based summarization and APT stage labeling.

## Quick Start

### Baseline Mode

Process attack graphs with sequential subgraph processing:

```bash
cd AutoProv/graph_assistant
python run_baseline.py \
    --dataset atlas \
    --model llama3:70b \
    --output ./output_baseline \
    --ollama_url <your_ollama_url> \
    --subgraph_size 5
```

### Arguments

- `--dataset`: Dataset to process (`atlas` or `theia`)
- `--model`: Ollama model name (default: `llama3:70b`)
- `--output`: Output directory (default: `./output_baseline`)
- `--ollama_url`: Ollama server URL (required)
- `--subgraph_size`: Maximum nodes per subgraph (`-1` = no splitting, `>0` = split if needed, `None` = no splitting)
- `--base_path`: Base path to repository (default: parent directory)
- `--apt_stages`: Path to MITRE ATT&CK stages JSON (default: `./data/mitre_attack_description.json`)

## Output Structure

```
output_baseline/
├── {dataset}/
│   ├── summaries/
│   │   ├── {attack_name}/
│   │   │   └── magic_subgraph_{idx}/
│   │   │       ├── summary.txt
│   │   │       └── labelled_summary.txt
│   │   └── pdf/
│   ├── parsed_data/
│   │   └── {attack_name}/
│   │       └── magic_subgraph_{idx}/
│   └── eval/
│       └── {attack_name}/
│           └── magic_subgraph_{idx}/
│               ├── hallucination_score.txt
│               └── {model}_judge.txt
└── evaluation_results_table.csv
```

## Direct Usage

Run the main pipeline directly:

```bash
cd AutoProv/graph_assistant/src
python main.py \
    --data <path_to_graph.json> \
    --output <output_dir> \
    --model llama3:70b \
    --apt_stages ../data/mitre_attack_description.json \
    --ollama_url <your_ollama_url> \
    --attack_name <attack_name> \
    --subgraph_size 5 \
    --dataset atlas \
    --magicsubgraph_idx 0 \
    --baseline_mode
```

## Evaluation

Evaluate summaries:

```bash
cd AutoProv/graph_assistant/src
python LLM_evaluation.py \
    --output <output_dir> \
    --summary <summary_file> \
    --labeled_summary <labeled_summary_file> \
    --node_class <node_class_file> \
    --APT_stages <apt_stages_file> \
    --attack_name <attack_name> \
    --subgraph_idx <idx> \
    --dataset atlas \
    --ollama_url <your_ollama_url> \
    --is_magic_subgraph
```

