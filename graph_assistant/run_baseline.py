#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import json
import ast
import glob
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from LLM_evaluation import extract_all_stages_and_reasonings
except ImportError:
    def extract_all_stages_and_reasonings(text):
        lines = text.strip().split('\n')
        pairs = []
        current_stage = None
        
        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue
            
            if 'Stage:' in clean_line:
                clean_line_no_md = clean_line.replace('**', '').replace('*', '')
                if 'Stage:' in clean_line_no_md:
                    parts = clean_line_no_md.split(':', 1)
                    if len(parts) == 2:
                        current_stage = parts[1].strip()
            elif clean_line.startswith('**') and clean_line.endswith('**'):
                stage_line = clean_line.replace('**', '').strip()
                apt_stages = [
                    'Reconnaissance', 'Initial Access', 'Execution', 'Persistence',
                    'Privilege Escalation', 'Defense Evasion', 'Credential Access',
                    'Discovery', 'Lateral Movement', 'Collection', 'Command and Control',
                    'Exfiltration', 'Impact'
                ]
                for stage in apt_stages:
                    if stage.lower() in stage_line.lower() or stage_line.lower() in stage.lower():
                        current_stage = stage_line
                        break
            elif '**' in clean_line and 'Stage:' in clean_line:
                clean_line_no_md = clean_line.replace('**', '').replace('*', '')
                if 'Stage:' in clean_line_no_md:
                    parts = clean_line_no_md.split(':', 1)
                    if len(parts) == 2:
                        current_stage = parts[1].strip()
            elif 'Reasoning:' in clean_line and current_stage:
                reasoning = clean_line.split(':', 1)[1].strip() if ':' in clean_line else clean_line
                pairs.append((current_stage, reasoning))
                current_stage = None
        
        return pairs


def find_atlas_attack_graphs(base_path):
    graphs_dir = os.path.join(base_path, "MAGIC", "attack_graphs", "gpt-4o_llama3_70b")
    attack_graphs = []
    
    if not os.path.exists(graphs_dir):
        return attack_graphs
    
    pattern = os.path.join(graphs_dir, "**/subgraph_*.json")
    graph_files = glob.glob(pattern, recursive=True)
    
    for graph_file in sorted(graph_files):
        graph_path = Path(graph_file)
        attack_name = graph_path.parent.name
        
        filename = graph_path.name
        match = re.match(r'subgraph_(\d+)\.json', filename)
        if match:
            magicsubgraph_idx = int(match.group(1))
            attack_graphs.append((attack_name, graph_file, magicsubgraph_idx))
    
    return attack_graphs


def find_theia_attack_graphs(base_path):
    graphs_dir = os.path.join(base_path, "OCR_APT", "attack_graphs", "theia", "llmlabel_mpnet")
    attack_graphs = []
    
    if not os.path.exists(graphs_dir):
        return attack_graphs
    
    pattern = os.path.join(graphs_dir, "**/subgraph_*.json")
    graph_files = glob.glob(pattern, recursive=True)
    
    for graph_file in sorted(graph_files):
        graph_path = Path(graph_file)
        attack_name = graph_path.parent.name
        
        filename = graph_path.name
        match = re.match(r'subgraph_(\d+)\.json', filename)
        if match:
            magicsubgraph_idx = int(match.group(1))
            attack_graphs.append((attack_name, graph_file, magicsubgraph_idx))
    
    return attack_graphs


def run_main_pipeline_baseline(graph_path, output_dir, model, apt_stages_path, attack_name=None, subgraph_size=None, dataset=None, magicsubgraph_idx=None, ollama_url=None):
    script_path = os.path.join(os.path.dirname(__file__), "src", "main.py")
    
    dataset_output_dir = os.path.join(output_dir, dataset.lower()) if dataset else output_dir
    
    cmd = [
        sys.executable,
        script_path,
        "--data", graph_path,
        "--output", dataset_output_dir,
        "--model", model,
        "--apt_stages", apt_stages_path
    ]
    
    if attack_name:
        cmd.extend(["--attack_name", attack_name])
    if subgraph_size is not None:
        cmd.extend(["--subgraph_size", str(subgraph_size)])
    if dataset:
        cmd.extend(["--dataset", dataset])
    if magicsubgraph_idx is not None:
        cmd.extend(["--magicsubgraph_idx", str(magicsubgraph_idx)])
    cmd.extend(["--baseline_mode"])
    if ollama_url:
        cmd.extend(["--ollama_url", ollama_url])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        return False


def run_evaluation_subgraph(attack_name, subgraph_idx, output_dir, apt_stages_path, dataset=None, magicsubgraph_idx=None, is_magic_subgraph=False, ollama_url=None):
    script_path = os.path.join(os.path.dirname(__file__), "src", "LLM_evaluation.py")
    
    dataset_output_dir = os.path.join(output_dir, dataset.lower()) if dataset else output_dir
    
    if is_magic_subgraph:
        summary_file = os.path.join(dataset_output_dir, "summaries", attack_name, f"magic_subgraph_{subgraph_idx}", "summary.txt")
        labeled_summary_file = os.path.join(dataset_output_dir, "summaries", attack_name, f"magic_subgraph_{subgraph_idx}", "labelled_summary.txt")
        node_class_file = os.path.join(dataset_output_dir, "parsed_data", attack_name, f"magic_subgraph_{subgraph_idx}", "malicious_nodes.txt")
        edges_file = None
        parsed_data_dir = os.path.join(dataset_output_dir, "parsed_data", attack_name, f"magic_subgraph_{subgraph_idx}")
        if os.path.exists(parsed_data_dir):
            edges_file_path = os.path.join(parsed_data_dir, "malicious_edges.txt")
            if os.path.exists(edges_file_path):
                edges_file = edges_file_path
            else:
                for fname in os.listdir(parsed_data_dir):
                    if fname.startswith("malicious_edges_") and fname.endswith(".txt"):
                        edges_file = os.path.join(parsed_data_dir, fname)
                        break
    else:
        summary_file = os.path.join(dataset_output_dir, "summaries", attack_name, f"magic_subgraph_{magicsubgraph_idx}", f"split_{subgraph_idx}", "summary.txt")
        labeled_summary_file = os.path.join(dataset_output_dir, "summaries", attack_name, f"magic_subgraph_{magicsubgraph_idx}", f"split_{subgraph_idx}", "labelled_summary.txt")
        node_class_file = os.path.join(dataset_output_dir, "parsed_data", attack_name, f"magic_subgraph_{magicsubgraph_idx}", f"split_{subgraph_idx}", "malicious_nodes.txt")
        edges_file = None
        parsed_data_dir = os.path.join(dataset_output_dir, "parsed_data", attack_name, f"magic_subgraph_{magicsubgraph_idx}", f"split_{subgraph_idx}")
        if os.path.exists(parsed_data_dir):
            edges_file_path = os.path.join(parsed_data_dir, "malicious_edges.txt")
            if os.path.exists(edges_file_path):
                edges_file = edges_file_path
            else:
                for fname in os.listdir(parsed_data_dir):
                    if fname.startswith("malicious_edges_") and fname.endswith(".txt"):
                        edges_file = os.path.join(parsed_data_dir, fname)
                        break
    
    for file_path, name in [
        (summary_file, "summary"),
        (labeled_summary_file, "labeled_summary"),
        (node_class_file, "node_class")
    ]:
        if not os.path.exists(file_path):
            return False
    
    cmd = [
        sys.executable,
        script_path,
        "--output", dataset_output_dir,
        "--summary", summary_file,
        "--labeled_summary", labeled_summary_file,
        "--node_class", node_class_file,
        "--APT_stages", apt_stages_path,
        "--attack_name", attack_name,
        "--subgraph_idx", str(subgraph_idx)
    ]
    if dataset:
        cmd.extend(["--dataset", dataset])
    if magicsubgraph_idx is not None:
        cmd.extend(["--magicsubgraph_idx", str(magicsubgraph_idx)])
    if is_magic_subgraph:
        cmd.extend(["--is_magic_subgraph"])
    if edges_file and os.path.exists(edges_file):
        cmd.extend(["--edges_file", edges_file])
    if ollama_url:
        cmd.extend(["--ollama_url", ollama_url])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        return False


def parse_evaluation_results_subgraph(attack_name, subgraph_idx, output_dir, dataset=None, magicsubgraph_idx=None, is_magic_subgraph=False):
    results = {
        'hallucination_score': None,
        'correctness_score': None,
        'judge_responses': {}
    }
    
    dataset_output_dir = os.path.join(output_dir, dataset.lower()) if dataset else output_dir
    
    if is_magic_subgraph:
        eval_dir = os.path.join(dataset_output_dir, "eval", attack_name, f"magic_subgraph_{subgraph_idx}")
    else:
        eval_dir = os.path.join(dataset_output_dir, "eval", attack_name, f"magic_subgraph_{magicsubgraph_idx}", f"split_{subgraph_idx}")
    
    models = ["qwen2:72b", "deepseek-r1:32b", "gemma2:27b"]
    import re
    for model in models:
        model_safe_name = model.replace(":", "_")
        judge_file = os.path.join(eval_dir, f"{model_safe_name}_judge.txt")
        if os.path.exists(judge_file):
            try:
                with open(judge_file, 'r') as f:
                    content = f.read().strip()
                    responses = []
                    for line in content.split('\n'):
                        line = line.strip().upper()
                        matches = re.findall(r'\b(YES|NO)\b', line)
                        if matches:
                            responses.extend(matches)
                    if responses:
                        results['judge_responses'][model] = responses
            except Exception as e:
                pass
    
    hallucination_file = os.path.join(eval_dir, "hallucination_score.txt")
    if os.path.exists(hallucination_file):
        try:
            with open(hallucination_file, 'r') as f:
                content = f.read().strip()
                try:
                    import json
                    scores = json.loads(content)
                    results['hallucination_score'] = scores.get('hallucination_score')
                    results['correctness_score'] = scores.get('correctness_score')
                except (json.JSONDecodeError, ValueError):
                    try:
                        val = float(content)
                        results['hallucination_score'] = val
                        results['correctness_score'] = 1 - val if val is not None else None
                    except ValueError:
                        try:
                            val = ast.literal_eval(content)
                            val = float(val)
                            results['hallucination_score'] = val
                            results['correctness_score'] = 1 - val if val is not None else None
                        except:
                            pass
        except Exception:
            pass
    
    return results


def count_incorrectly_labeled_stages(output_dir, attack_name, all_magic_indices, dataset="atlas"):
    dataset_output_dir = os.path.join(output_dir, dataset.lower())
    models = ["qwen2:72b", "deepseek-r1:32b", "gemma2:27b"]
    
    stage_judgments = defaultdict(lambda: defaultdict(lambda: {}))
    
    for magicsubgraph_idx in all_magic_indices:
        labeled_summary_file = os.path.join(
            dataset_output_dir, "summaries", attack_name,
            f"magic_subgraph_{magicsubgraph_idx}", "labelled_summary.txt"
        )
        
        if not os.path.exists(labeled_summary_file):
            continue
        
        try:
            with open(labeled_summary_file, 'r') as f:
                data = json.load(f)
                labeled_summary = data.get("output", "")
            
            extracted_stages = extract_all_stages_and_reasonings(labeled_summary)
            
            if not extracted_stages:
                continue
            
            for model in models:
                model_safe_name = model.replace(":", "_")
                judge_file = os.path.join(
                    dataset_output_dir, "eval", attack_name,
                    f"magic_subgraph_{magicsubgraph_idx}",
                    f"{model_safe_name}_judge.txt"
                )
                
                if not os.path.exists(judge_file):
                    continue
                
                try:
                    with open(judge_file, 'r') as f:
                        content = f.read().strip()
                        responses = []
                        for line in content.split('\n'):
                            line = line.strip().upper()
                            matches = re.findall(r'\b(YES|NO)\b', line)
                            if matches:
                                responses.extend(matches)
                    
                    for i, (stage_name, _) in enumerate(extracted_stages):
                        if i < len(responses):
                            stage_judgments[stage_name][magicsubgraph_idx][model] = responses[i]
                except Exception:
                    continue
        
        except Exception:
            continue
    
    incorrect_stages = set()
    all_stages = set(stage_judgments.keys())
    
    for stage_name in all_stages:
        is_incorrect = False
        for magicsubgraph_idx in stage_judgments[stage_name]:
            model_responses = stage_judgments[stage_name][magicsubgraph_idx]
            no_count = sum(1 for resp in model_responses.values() if resp == 'NO')
            if no_count >= 2:
                is_incorrect = True
                break
        
        if is_incorrect:
            incorrect_stages.add(stage_name)
    
    return len(incorrect_stages), len(all_stages), sorted(incorrect_stages)


def create_results_table(results_dict, output_file):
    import csv
    
    models = ["qwen2:72b", "deepseek-r1:32b", "gemma2:27b"]
    
    rows = []
    headers = ['Attack Name', 'Hallucination Score', 'Correctness Score', 'Stage Correctness']
    for model in models:
        headers.append(model)
    
    for attack_name, results in sorted(results_dict.items()):
        stage_correctness = results.get('stage_correctness')
        if stage_correctness is not None:
            correct_str = f"{stage_correctness:.4f}"
        else:
            incorrect_stages = results.get('incorrect_stages')
            total_stages = results.get('total_stages')
            if incorrect_stages is not None and total_stages is not None and total_stages > 0:
                stage_correctness = (total_stages - incorrect_stages) / total_stages
                correct_str = f"{stage_correctness:.4f}"
            else:
                correct_str = 'N/A'
        
        halluc_score = results.get('hallucination_score', 'N/A')
        if isinstance(halluc_score, float):
            halluc_score = f"{halluc_score:.4f}"
        
        correctness_score_val = results.get('correctness_score')
        if isinstance(correctness_score_val, float):
            correctness_score = f"{correctness_score_val:.4f}"
        else:
            correctness_score = 'N/A'
        
        row = [
            attack_name,
            halluc_score,
            correctness_score,
            correct_str
        ]
        
        for model in models:
            score = results.get(model)
            if score is not None:
                row.append(f"{score:.4f}")
            else:
                row.append('N/A')
        rows.append(row)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run BASELINE attack summarization pipeline (old version without frontier subgraphs)"
    )
    
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["atlas", "theia"],
        help="Dataset to process (atlas or theia)"
    )
    
    parser.add_argument(
        "--model",
        default="llama3:70b",
        help="Ollama model to use for summarization (default: llama3:70b)"
    )
    
    parser.add_argument(
        "--output",
        default="./output_baseline",
        help="Output directory (default: ./output_baseline)"
    )
    
    parser.add_argument(
        "--base_path",
        default=None,
        help="Base path to the repository (default: parent directory of this script)"
    )
    
    parser.add_argument(
        "--apt_stages",
        default=None,
        help="Path to MITRE ATT&CK stages JSON file (default: ./data/mitre_attack_description.json)"
    )
    
    parser.add_argument(
        "--subgraph_size",
        type=int,
        default=None,
        help="Maximum number of unique nodes per subgraph (-1 = no splitting, >0 = split if needed, None = no splitting)"
    )
    
    parser.add_argument(
        "--ollama_url",
        default=None,
        help="Ollama server URL"
    )
    
    args = parser.parse_args()
    
    if args.base_path is None:
        args.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.apt_stages is None:
        args.apt_stages = os.path.join(os.path.dirname(__file__), "data", "mitre_attack_description.json")
    
    if not os.path.exists(args.apt_stages):
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    dataset_output_dir = os.path.join(args.output, args.dataset.lower())
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    pdf_dir = os.path.join(dataset_output_dir, "summaries", "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    
    if args.dataset == "atlas":
        attack_graphs = find_atlas_attack_graphs(args.base_path)
    elif args.dataset == "theia":
        attack_graphs = find_theia_attack_graphs(args.base_path)
    else:
        sys.exit(1)
    
    if not attack_graphs:
        sys.exit(1)
    
    all_results = {}
    
    attack_magic_indices = defaultdict(list)
    for attack_name, graph_path, magicsubgraph_idx in attack_graphs:
        attack_magic_indices[attack_name].append(magicsubgraph_idx)
    
    for attack_name, graph_path, magicsubgraph_idx in tqdm(attack_graphs, desc="Processing attacks"):
        success = run_main_pipeline_baseline(
            graph_path, args.output, args.model, args.apt_stages,
            attack_name=attack_name, subgraph_size=args.subgraph_size, 
            dataset=args.dataset, magicsubgraph_idx=magicsubgraph_idx, ollama_url=args.ollama_url
        )
        if not success:
            all_results[f"{attack_name}_magic{magicsubgraph_idx}"] = {'error': 'Pipeline failed'}
            continue
        
        if args.subgraph_size == -1:
            run_evaluation_subgraph(attack_name, magicsubgraph_idx, args.output, args.apt_stages, dataset=args.dataset, is_magic_subgraph=True, ollama_url=args.ollama_url)
            
            results = parse_evaluation_results_subgraph(attack_name, magicsubgraph_idx, args.output, dataset=args.dataset, is_magic_subgraph=True)
            if results:
                all_results[f"{attack_name}_magic{magicsubgraph_idx}"] = results
            else:
                all_results[f"{attack_name}_magic{magicsubgraph_idx}"] = {'error': 'Could not parse results'}
        elif args.subgraph_size:
            concatenated_summary_file = os.path.join(dataset_output_dir, "summaries", attack_name, f"magic_subgraph_{magicsubgraph_idx}", "summary.txt")
            concatenated_labeled_summary_file = os.path.join(dataset_output_dir, "summaries", attack_name, f"magic_subgraph_{magicsubgraph_idx}", "labelled_summary.txt")
            
            if os.path.exists(concatenated_summary_file) and os.path.exists(concatenated_labeled_summary_file):
                run_evaluation_subgraph(attack_name, magicsubgraph_idx, args.output, args.apt_stages, dataset=args.dataset, is_magic_subgraph=True, ollama_url=args.ollama_url)
                
                results = parse_evaluation_results_subgraph(attack_name, magicsubgraph_idx, args.output, dataset=args.dataset, is_magic_subgraph=True)
                if results:
                    all_results[f"{attack_name}_magic{magicsubgraph_idx}"] = results
                else:
                    all_results[f"{attack_name}_magic{magicsubgraph_idx}"] = {'error': 'Could not parse results'}
            else:
                run_evaluation_subgraph(attack_name, magicsubgraph_idx, args.output, args.apt_stages, dataset=args.dataset, is_magic_subgraph=True, ollama_url=args.ollama_url)
                
                results = parse_evaluation_results_subgraph(attack_name, magicsubgraph_idx, args.output, dataset=args.dataset, is_magic_subgraph=True)
                if results:
                    all_results[f"{attack_name}_magic{magicsubgraph_idx}"] = results
                else:
                    all_results[f"{attack_name}_magic{magicsubgraph_idx}"] = {'error': 'Could not parse results'}
        else:
            all_results[f"{attack_name}_magic{magicsubgraph_idx}"] = {'error': 'Single graph processing not supported with new structure'}
            continue
    
    for attack_name in attack_magic_indices:
        magic_indices = attack_magic_indices[attack_name]
        incorrect_count, total_stages, incorrect_list = count_incorrectly_labeled_stages(
            args.output, attack_name, magic_indices, dataset=args.dataset
        )
        for idx in magic_indices:
            key = f"{attack_name}_magic{idx}"
            if key in all_results and isinstance(all_results[key], dict):
                all_results[key]['incorrect_stages'] = incorrect_count
                all_results[key]['total_stages'] = total_stages
    
    aggregated_results = {}
    models = ["qwen2:72b", "deepseek-r1:32b", "gemma2:27b"]
    
    for key, results in all_results.items():
        if not isinstance(results, dict) or 'error' in results:
            continue
        
        if '_magic' in key:
            attack_name = key.rsplit('_magic', 1)[0]
        else:
            continue
        
        if attack_name not in aggregated_results:
            aggregated_results[attack_name] = {
                'hallucination_scores': [],
                'correctness_scores': [],
                'judge_responses': {model: [] for model in models},
                'incorrect_stages': None,
                'total_stages': None
            }
        
        if results.get('hallucination_score') is not None:
            aggregated_results[attack_name]['hallucination_scores'].append(results['hallucination_score'])
        if results.get('correctness_score') is not None:
            aggregated_results[attack_name]['correctness_scores'].append(results['correctness_score'])
        
        judge_responses = results.get('judge_responses', {})
        for model in models:
            if model in judge_responses:
                aggregated_results[attack_name]['judge_responses'][model].extend(judge_responses[model])
        
        if results.get('incorrect_stages') is not None:
            aggregated_results[attack_name]['incorrect_stages'] = results['incorrect_stages']
        if results.get('total_stages') is not None:
            aggregated_results[attack_name]['total_stages'] = results['total_stages']
    
    final_results = {}
    for attack_name, agg_data in aggregated_results.items():
        halluc_avg = (sum(agg_data['hallucination_scores']) / len(agg_data['hallucination_scores']) 
                     if agg_data['hallucination_scores'] else None)
        correct_avg = (sum(agg_data['correctness_scores']) / len(agg_data['correctness_scores']) 
                      if agg_data['correctness_scores'] else None)
        
        judge_scores = {}
        for model in models:
            responses = agg_data['judge_responses'][model]
            if responses:
                yes_count = sum(1 for r in responses if r == 'YES')
                score = yes_count / len(responses) if responses else None
                judge_scores[model] = score
            else:
                judge_scores[model] = None
        
        incorrect_stages = agg_data['incorrect_stages']
        total_stages = agg_data['total_stages']
        if incorrect_stages is not None and total_stages is not None and total_stages > 0:
            stage_correctness = (total_stages - incorrect_stages) / total_stages
        else:
            stage_correctness = None
        
        final_results[attack_name] = {
            'hallucination_score': halluc_avg,
            'correctness_score': correct_avg,
            'judge_responses': {model: agg_data['judge_responses'][model] for model in models},
            'incorrect_stages': incorrect_stages,
            'total_stages': total_stages,
            'stage_correctness': stage_correctness
        }
        
        for model in models:
            final_results[attack_name][model] = judge_scores[model]
    
    table_file = os.path.join(dataset_output_dir, "evaluation_results_table.csv")
    create_results_table(final_results, table_file)
    
    summaries_dir = os.path.join(dataset_output_dir, "summaries")
    pdf_dir = os.path.join(dataset_output_dir, "summaries", "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    
    try:
        pdf_gen_path = os.path.join(os.path.dirname(__file__), 'src', 'pdf_generator.py')
        if os.path.exists(pdf_gen_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("pdf_generator", pdf_gen_path)
            pdf_generator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pdf_generator)
            
            pdf_generator.generate_pdfs_for_summaries(summaries_dir, pdf_dir)
    except ImportError as e:
        pass
    except Exception as e:
        pass


if __name__ == "__main__":
    main()

