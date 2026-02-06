import os
import ollama
import argparse
import json
import re
import ast
import sys
from collections import Counter

models = ["qwen2:72b", "deepseek-r1:32b","gemma2:27b"]

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ollama_client = None

def set_ollama_client(ollama_url=None):
    global ollama_client
    if ollama_url:
        ollama_client = ollama.Client(host=ollama_url)
    else:
        ollama_client = ollama.Client()

def create_LLM_eval_prompt(mitre_reasoning, model_reasoning):
    system_prompt = """
    You are an expert in cybersecurity, the English language, and an expert evaluator in cybersecurity and the MITRE ATT&CK framework.
    """
    user_prompt = f"""
    Below, you are given two pieces of reasoning:
    - Model Reasoning: {model_reasoning}
    - MITRE Reference Reasoning: {mitre_reasoning}

    Carefully compare the Model Reasoning to the MITRE Reference Reasoning.

    Task:
    Does the Model Reasoning align with the MITRE Reference Reasoning for the model to categorize this attack under this APT stage?
    Respond with ONLY "YES" or "NO".
    
    Format your answer as:
    YES/NO
    
    Guidelines:
    1. ONLY OUTPUT YES or NO.
    
    """
    return system_prompt, user_prompt

def hallucination_eval(model_summary, nodes):
    nodes_in_summary = re.findall(r'"([^"]*)"', model_summary)
    correct = 0
    incorrect = 0
    for item in nodes_in_summary:
        if item in nodes:
            correct += 1
        else:
            incorrect += 1

    if len(nodes_in_summary) == 0:
        return 0.0
    
    correctness = correct / len(nodes_in_summary)
    hallucination_score = 1 - correctness
    return hallucination_score


def extract_all_nodes_from_edges(edges_file_path):
    all_nodes = set()
    try:
        with open(edges_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if '-->' in line:
                    parts = line.split('-->')
                    if len(parts) == 2:
                        source_part = parts[0].strip()
                        target_part = parts[1].strip()
                        if '(' in target_part:
                            target_part = target_part.split('(')[0].strip()
                        if '--' in source_part:
                            source = source_part.rsplit('--', 1)[0].strip()
                        else:
                            source = source_part
                        all_nodes.add(source)
                        all_nodes.add(target_part)
    except Exception as e:
        pass
    
    return all_nodes


def calculate_entity_coverage(summary_text, graph_nodes):
    entities_in_summary = set(re.findall(r'"([^"]*)"', summary_text))
    
    graph_nodes_set = set(graph_nodes) if not isinstance(graph_nodes, set) else graph_nodes
    
    covered_nodes = entities_in_summary.intersection(graph_nodes_set)
    
    total_nodes = len(graph_nodes_set)
    covered_count = len(covered_nodes)
    
    if total_nodes == 0:
        coverage_score = 0.0
    else:
        coverage_score = covered_count / total_nodes
    
    return {
        'coverage_score': coverage_score,
        'covered_nodes': list(covered_nodes),
        'total_nodes': total_nodes,
        'covered_count': covered_count
    }


def query_model(system_prompt, user_prompt, model):
    global ollama_client
    if ollama_client is None:
        set_ollama_client()
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    
    max_attempts = 3
    last_exception = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = ollama_client.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": 0,
                    "num_predict": 4096
                }
            )
            return response["message"]["content"]
        except Exception as e:
            last_exception = e
            if attempt < max_attempts:
                pass
            else:
                raise last_exception
    
    raise last_exception

def create_output_directory(output):
    eval_dir = output + "/eval"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

            

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


def pull_models():
    global ollama_client
    if ollama_client is None:
        set_ollama_client()
    for model in models:
        ollama_client.pull(model)


def calculate_agreement_score(label_lists):
    if not label_lists:
        return {'overall_agreement': 0.0, 'per_position_agreement': []}

    n_lists = len(label_lists)
    min_length = min(len(lst) for lst in label_lists)

    per_position_agreement = []

    for i in range(min_length):
        column = [lst[i] for lst in label_lists if i < len(lst)]

        counts = Counter(column)
        max_freq = max(counts.values()) if counts else 0

        agreement_i = max_freq / len(column)
        per_position_agreement.append(agreement_i)

    overall_agreement = sum(per_position_agreement) / len(per_position_agreement) if per_position_agreement else 0.0

    return {
        'overall_agreement': overall_agreement,
        'per_position_agreement': per_position_agreement
    }


def remove_think_tags(text):
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned.strip())
    return cleaned



def main():
    parser = argparse.ArgumentParser(description="using Ollama to evaluate LLM summarys")

    parser.add_argument("--node_class", help="provide the file that contains the node classification data (should be in the output directory generated by pipeline).", required = False)
    parser.add_argument("--labeled_summary", help="provide the file containing your labeled summary", required = False)
    parser.add_argument("--summary", help="original summary")
    parser.add_argument("--APT_stages", help="provide the file that contains the APT stages", required = False)
    parser.add_argument("--output", help= "output directory (should already include dataset level)")
    parser.add_argument("--attack_name", help="name of the attack (for subgraph-based evaluation)")
    parser.add_argument("--subgraph_idx", type=int, help="subgraph index (for subgraph-based evaluation)")
    parser.add_argument("--dataset", help="dataset name (e.g., 'atlas')")
    parser.add_argument("--magicsubgraph_idx", type=int, default=None, help="MAGIC subgraph index (if splitting was used)")
    parser.add_argument("--is_magic_subgraph", action="store_true", help="If True, subgraph_idx is the MAGIC subgraph index (no splitting)")
    parser.add_argument("--edges_file", help="path to edges file for entity coverage calculation (optional)", required=False)
    parser.add_argument("--ollama_url", default=None, help="Ollama server URL")

    args = parser.parse_args()
    
    set_ollama_client(args.ollama_url)

    if args.is_magic_subgraph:
        eval_dir = os.path.join(args.output, "eval", args.attack_name, f"magic_subgraph_{args.subgraph_idx}")
        os.makedirs(eval_dir, exist_ok=True)
        unique_output_path = f"{args.attack_name}_magic_subgraph_{args.subgraph_idx}"
        use_subgraph_structure = True
    elif args.attack_name is not None and args.subgraph_idx is not None:
        eval_dir = os.path.join(args.output, "eval", args.attack_name, f"magic_subgraph_{args.magicsubgraph_idx}", f"split_{args.subgraph_idx}")
        os.makedirs(eval_dir, exist_ok=True)
        unique_output_path = f"{args.attack_name}_magic_subgraph_{args.magicsubgraph_idx}_split_{args.subgraph_idx}"
        use_subgraph_structure = True
    else:
        eval_dir = os.path.join(args.output, "eval")
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        split_data = args.summary.replace(".txt", "").split("_")
        if len(split_data) >= 6:
            unique_output_path = split_data[-6] + "_" + split_data[-5] + "_" + split_data[-4] + "_" + split_data[-3] + "_" + split_data[-2] + "_" + split_data[-1]
        else:
            unique_output_path = "_".join(split_data[-6:])
        use_subgraph_structure = False

    with open(args.APT_stages, "r") as f:
       stages = f.readlines()
       
    stages = "".join(stages)
    stages_dict = json.loads(stages)
    stage_names = list(stages_dict.keys())
    
    stage_normalization = {
        "Data Exfiltration": "Exfiltration",
        "Data exfiltration": "Exfiltration",
        "data exfiltration": "Exfiltration",
    }
    
    def normalize_stage_name(stage):
        if stage in stage_normalization:
            return stage_normalization[stage]
        if stage in stages_dict:
            return stage
        for valid_stage in stage_names:
            if stage.lower() == valid_stage.lower():
                return valid_stage
        return None
   
    with open(args.node_class, "r") as f:
        nodes = [line.strip() for line in f.readlines() if line.strip()]
    
    with open(args.summary, "r") as f:
        data = f.read()
        data_json = json.loads(data)
    
    pull_models()
    
    from tqdm import tqdm
    
    hallucination_result = hallucination_eval(data_json["output"], nodes)

    coverage_results = {}
    
    malicious_coverage = calculate_entity_coverage(data_json["output"], nodes)
    coverage_results['malicious_coverage'] = malicious_coverage['coverage_score']
    coverage_results['malicious_covered_count'] = malicious_coverage['covered_count']
    coverage_results['malicious_total_nodes'] = malicious_coverage['total_nodes']
    coverage_results['malicious_covered_nodes'] = malicious_coverage['covered_nodes']
    
    if args.edges_file and os.path.exists(args.edges_file):
        all_graph_nodes = extract_all_nodes_from_edges(args.edges_file)
        overall_coverage = calculate_entity_coverage(data_json["output"], all_graph_nodes)
        coverage_results['overall_coverage'] = overall_coverage['coverage_score']
        coverage_results['overall_covered_count'] = overall_coverage['covered_count']
        coverage_results['overall_total_nodes'] = overall_coverage['total_nodes']
        coverage_results['overall_covered_nodes'] = overall_coverage['covered_nodes']
    else:
        coverage_results['overall_coverage'] = None
        coverage_results['overall_covered_count'] = None
        coverage_results['overall_total_nodes'] = None
        coverage_results['overall_covered_nodes'] = None

    with open(args.labeled_summary, 'r') as f:
        data = f.read()
        labeled_summary = json.loads(data)["output"]
    
    extracted_data = extract_all_stages_and_reasonings(labeled_summary)
    all_model_outputs = []

    coverage_file = os.path.join(eval_dir, "entity_coverage.txt") if use_subgraph_structure else os.path.join(eval_dir, f"{unique_output_path}_entity_coverage.txt")
    with open(coverage_file, "w") as f:
        json.dump(coverage_results, f, indent=2)

    try:
        for model in tqdm(models, desc="Evaluating with judge models", leave=False):
            model_output_raw = []
            model_output_clean = []
            
            if not use_subgraph_structure:
                with open(f"{args.output}/eval/{model}_LLM_eval.txt", "w") as f:
                    for item in tqdm(extracted_data, desc=f"  {model}", leave=False, disable=len(extracted_data) < 3):
                        stage = item[0]
                        LLM_reasoning = item[1]
                        
                        normalized_stage = normalize_stage_name(stage)
                        if normalized_stage is None:
                            model_output_raw.append(f"SKIPPED: Unknown stage '{stage}'")
                            model_output_clean.append("UNKNOWN")
                            f.write(f"SKIPPED: Unknown stage '{stage}'\n")
                            continue
                        
                        mitre_reasoning = stages_dict[normalized_stage]
                        system_prompt, user_prompt = create_LLM_eval_prompt(mitre_reasoning, LLM_reasoning)
                        try:
                            result = query_model(system_prompt, user_prompt, model)
                            result = remove_think_tags(result)
                            
                            result_upper = result.strip().upper()
                            matches = re.findall(r'\b(YES|NO)\b', result_upper)
                            clean_answer = matches[0] if matches else None
                            
                            model_output_raw.append(result)
                            if clean_answer:
                                model_output_clean.append(clean_answer)
                            else:
                                model_output_clean.append("UNKNOWN")
                            
                            f.write(result)
                        except Exception as e:
                            error_msg = f"ERROR: Failed to get response from {model} for stage '{stage}' after retries: {e}"
                            model_output_raw.append(error_msg)
                            model_output_clean.append("UNKNOWN")
                            f.write(error_msg + "\n")
            else:
                for item in tqdm(extracted_data, desc=f"  {model}", leave=False, disable=len(extracted_data) < 3):
                    stage = item[0]
                    LLM_reasoning = item[1]
                    
                    normalized_stage = normalize_stage_name(stage)
                    if normalized_stage is None:
                        model_output_raw.append(f"SKIPPED: Unknown stage '{stage}'")
                        model_output_clean.append("UNKNOWN")
                        continue
                    
                    mitre_reasoning = stages_dict[normalized_stage]
                    system_prompt, user_prompt = create_LLM_eval_prompt(mitre_reasoning, LLM_reasoning)
                    try:
                        result = query_model(system_prompt, user_prompt, model)
                        result = remove_think_tags(result)
                        
                        result_upper = result.strip().upper()
                        matches = re.findall(r'\b(YES|NO)\b', result_upper)
                        clean_answer = matches[0] if matches else None
                        
                        model_output_raw.append(result)
                        if clean_answer:
                            model_output_clean.append(clean_answer)
                        else:
                            model_output_clean.append("UNKNOWN")
                    except Exception as e:
                        error_msg = f"ERROR: Failed to get response from {model} for stage '{stage}' after retries: {e}"
                        model_output_raw.append(error_msg)
                        model_output_clean.append("UNKNOWN")
                
                model_safe_name = model.replace(":", "_")
                if use_subgraph_structure:
                    judge_file = os.path.join(eval_dir, f"{model_safe_name}_judge.txt")
                else:
                    judge_file = os.path.join(eval_dir, f"{unique_output_path}_{model_safe_name}_judge.txt")
                with open(judge_file, "w") as f:
                    for result in model_output_raw:
                        f.write(result + "\n")
                
                all_model_outputs.append(model_output_clean)

        agreement_scores = calculate_agreement_score(all_model_outputs)

        agreement_file = os.path.join(eval_dir, "LLM_agreement_scores.txt") if use_subgraph_structure else os.path.join(eval_dir, f"{unique_output_path}_LLM_agreement_scores.txt")
        with open(agreement_file, "w") as f:
            f.write(str(agreement_scores))
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        agreement_file = os.path.join(eval_dir, "LLM_agreement_scores.txt") if use_subgraph_structure else os.path.join(eval_dir, f"{unique_output_path}_LLM_agreement_scores.txt")
        with open(agreement_file, "w") as f:
            f.write(str({"error": str(e)}))

    correctness_result = 1 - hallucination_result if hallucination_result is not None else None
    
    hallucination_file = os.path.join(eval_dir, "hallucination_score.txt") if use_subgraph_structure else os.path.join(eval_dir, f"{unique_output_path}_hallucination_score.txt")
    with open(hallucination_file, "w") as f:
        scores = {
            "hallucination_score": hallucination_result,
            "correctness_score": correctness_result
        }
        f.write(json.dumps(scores))
   


if __name__ == "__main__":
    main()
