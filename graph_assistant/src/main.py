
from pipeline import data_pipeline
from data_parsing import data_parser
from data_parsing import kairos_data_parser
from entity_enrichment import enricher
import argparse
import os
import json


def main():

    parser = argparse.ArgumentParser(description="main entry point for provenance attack story summarization")

    parser.add_argument("--data", help="provide the data file containing the provenance graph")
    parser.add_argument("--output", help="the output directory for the results")
    parser.add_argument("--model", help="the model you want to use")
    parser.add_argument("--apt_stages", help="the file with the apt stages")
    parser.add_argument("--attack_name", help="name of the attack (required if using subgraph_size)")
    parser.add_argument("--subgraph_size", type=int, default=None, help="maximum number of unique nodes per subgraph (None = no splitting, -1 = no splitting, >0 = split if needed)")
    parser.add_argument("--dataset", help="dataset name (e.g., 'atlas')")
    parser.add_argument("--magicsubgraph_idx", type=int, default=None, help="MAGIC subgraph index (from MAGIC output)")
    parser.add_argument("--baseline_mode", action="store_true", help="Use baseline mode (old sequential processing, no frontier subgraphs)")
    parser.add_argument("--use_baseline_context", action="store_true", help="Use baseline-style context (only previous summary) but keep topological processing order")
    parser.add_argument("--poisoning_mapping", default=None, help="Path to JSON file containing poisoning mapping (original -> poisoned names)")
    parser.add_argument("--summary_only", action="store_true", help="Generate summary only, skip APT labeling")
    parser.add_argument("--poisoned_enriched_base_dir", default=None, help="Base directory containing poisoned enriched files (optional)")
    parser.add_argument("--ollama_url", default=None, help="Ollama server URL")
    args = parser.parse_args()

    if not args.data:
        parser.print_help()
        exit()
    if not args.output:
        parser.print_help()
        exit()
    if not args.model:
        parser.print_help()
        exit()
    if not args.apt_stages:
        parser.print_help()
        exit()
   
    poisoning_mapping = None
    if args.poisoning_mapping and os.path.exists(args.poisoning_mapping):
        with open(args.poisoning_mapping, 'r') as f:
            poisoning_mapping = json.load(f)
    
    parser_obj = data_parser(
        args.data,
        args.output,
        attack_name=args.attack_name,
        subgraph_size=args.subgraph_size,
        magicsubgraph_idx=args.magicsubgraph_idx,
        poisoning_mapping=poisoning_mapping,
        poisoned_enriched_base_dir=args.poisoned_enriched_base_dir
    )
    entity_enricher = enricher(args.model, ollama_url=args.ollama_url)

    pipeline = data_pipeline(None, parser_obj, entity_enricher)
    pipeline.run(args.model, args.output, args.apt_stages, attack_name=args.attack_name, subgraph_size=args.subgraph_size, dataset=args.dataset, magicsubgraph_idx=args.magicsubgraph_idx, baseline_mode=args.baseline_mode, summary_only=args.summary_only, use_baseline_context=args.use_baseline_context, ollama_url=args.ollama_url)

if __name__ == "__main__":
    main()

