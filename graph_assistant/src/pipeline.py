from data_parsing import data_parser
from data_parsing import kairos_data_parser
from ollama_attack_summarization import summarizer
from entity_enrichment import enricher 
import json
import os
from tqdm import tqdm

class data_pipeline:
    def __init__(self, summarizer, parser, enricher):
        self.enricher = enricher
        self.summarizer = summarizer
        self.parser = parser

    def run(self, model, output, apt_stages, attack_name=None, subgraph_size=None, dataset=None, magicsubgraph_idx=None, baseline_mode=False, summary_only=False, use_baseline_context=False, ollama_url=None):
        use_subgraphs = (subgraph_size is not None) and (attack_name is not None)
        
        if use_subgraphs:
            self.run_with_subgraphs(model, output, apt_stages, attack_name, dataset, magicsubgraph_idx, baseline_mode=baseline_mode, summary_only=summary_only, use_baseline_context=use_baseline_context, ollama_url=ollama_url)
        else:
            self.run_single_graph(model, output, apt_stages, dataset, summary_only=summary_only, ollama_url=ollama_url)
    
    def run_single_graph(self, model, output, apt_stages, dataset=None, summary_only=False, ollama_url=None):
        self.create_output_directories(output, dataset)
        
        malicious_edges, malicious_nodes, malicious_nodes_func, unique_output_path = self.parser.parse()

        suplement_node_info = []
        for i in tqdm(range(len(malicious_nodes)), desc="Enriching entities"):
            node = malicious_nodes[i]
            functionality = malicious_nodes_func[i]
            response = self.enricher.validate_node(node)
            if response == "NO":
                suplement_node_info.append((node, functionality))

        if not suplement_node_info:
            suplement_node_info = None

        with open(apt_stages,"r") as f:
            stages = f.readlines()

        stages = "".join(stages)
        stages_dict = json.loads(stages)
        apt_stages  = list(stages_dict.keys())

        self.summarizer = summarizer(malicious_edges, malicious_nodes, model, output, apt_stages, ollama_url=ollama_url)
        summary, labelled_summary = self.summarizer.summarize(suplement_node_info)

        with open(f"{output}/summaries/summary_{unique_output_path}.txt", "w") as f:
            f.write(json.dumps(summary))

        if not summary_only:
            with open(f"{output}/summaries/labelled_summary_{unique_output_path}.txt", "w") as f:
                f.write(json.dumps(labelled_summary))
    
    def _topological_sort_subgraphs(self, frontier_data):
        if not frontier_data or "frontier_subgraphs" not in frontier_data:
            return []
        
        frontier_subgraphs = frontier_data["frontier_subgraphs"]
        subgraph_names = list(frontier_subgraphs.keys())
        
        dependencies = {}
        for subgraph_name in subgraph_names:
            dependencies[subgraph_name] = frontier_subgraphs[subgraph_name].get("connected_from", [])
        
        in_degree = {name: len(deps) for name, deps in dependencies.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            queue.sort()
            current = queue.pop(0)
            result.append(current)
            
            for subgraph_name in subgraph_names:
                if current in frontier_subgraphs[subgraph_name].get("connected_from", []):
                    in_degree[subgraph_name] -= 1
                    if in_degree[subgraph_name] == 0:
                        queue.append(subgraph_name)
        
        remaining = [name for name in subgraph_names if name not in result]
        result.extend(remaining)
        
        return result
    
    def run_with_subgraphs(self, model, output, apt_stages, attack_name, dataset=None, magicsubgraph_idx=None, baseline_mode=False, summary_only=False, use_baseline_context=False, ollama_url=None):
        self.create_output_directories_for_subgraphs(output, attack_name, dataset, magicsubgraph_idx)
        
        frontier_data = None
        if not baseline_mode:
            frontier_json_path = os.path.join(output, "frontier_subgraphs", attack_name, f"magic_subgraph_{magicsubgraph_idx}", "frontier_subgraphs.json")
            if os.path.exists(frontier_json_path):
                with open(frontier_json_path, 'r') as f:
                    frontier_data = json.load(f)
        
        subgraph_results = self.parser.split_and_parse(attack_name)
        
        with open(apt_stages, "r") as f:
            stages = f.readlines()
        stages = "".join(stages)
        stages_dict = json.loads(stages)
        apt_stages_list = list(stages_dict.keys())
        
        is_split = len(subgraph_results) > 1
        
        if baseline_mode:
            processing_order = [f"FSubgraph_{idx}" for idx, _ in enumerate(subgraph_results)]
        elif frontier_data and is_split:
            processing_order = self._topological_sort_subgraphs(frontier_data)
        else:
            processing_order = [f"FSubgraph_{idx}" for idx, _ in enumerate(subgraph_results)]
        
        summaries = {}
        all_summaries_text = []
        all_summary_inputs = []
        
        for order_idx, fsubgraph_name in enumerate(tqdm(processing_order, desc=f"Processing {attack_name} subgraphs")):
            try:
                fsubgraph_idx = int(fsubgraph_name.replace("FSubgraph_", ""))
            except ValueError:
                continue
            
            if fsubgraph_idx >= len(subgraph_results):
                continue
            
            subgraph_idx, malicious_edges, malicious_nodes, malicious_nodes_func = subgraph_results[fsubgraph_idx]
            
            suplement_node_info = []
            for i in tqdm(range(len(malicious_nodes)), desc=f"  Enriching {fsubgraph_name}", leave=False, disable=len(malicious_nodes) < 10):
                node = malicious_nodes[i]
                functionality = malicious_nodes_func[i]
                response = self.enricher.validate_node(node)
                if response == "NO":
                    suplement_node_info.append((node, functionality))
            
            if not suplement_node_info:
                suplement_node_info = None
            
            connected_summaries = {}
            frontier_nodes = None
            connected_to = None
            connected_from = None
            previous_summary = None
            
            if baseline_mode:
                if fsubgraph_idx > 0:
                    prev_fsubgraph_name = f"FSubgraph_{fsubgraph_idx - 1}"
                    if prev_fsubgraph_name in summaries:
                        previous_summary = summaries[prev_fsubgraph_name]
            elif use_baseline_context:
                if order_idx > 0:
                    prev_fsubgraph_name = processing_order[order_idx - 1]
                    if prev_fsubgraph_name in summaries:
                        previous_summary = summaries[prev_fsubgraph_name]
            else:
                if frontier_data and fsubgraph_name in frontier_data.get("frontier_subgraphs", {}):
                    fsubgraph_info = frontier_data["frontier_subgraphs"][fsubgraph_name]
                    frontier_nodes = fsubgraph_info.get("frontier_nodes")
                    connected_to = fsubgraph_info.get("connected_to", [])
                    connected_from = fsubgraph_info.get("connected_from", [])
                    
                    for connected_name in connected_from:
                        if connected_name in summaries:
                            connected_summaries[connected_name] = summaries[connected_name]
            
            self.summarizer = summarizer(malicious_edges, malicious_nodes, model, output, apt_stages_list, ollama_url=ollama_url)
            if baseline_mode or use_baseline_context:
                summary, _ = self.summarizer.summarize(
                    suplement_node_info,
                    previous_summary=previous_summary
                )
            else:
                summary, _ = self.summarizer.summarize(
                    suplement_node_info,
                    connected_summaries=connected_summaries if connected_summaries else None,
                    frontier_nodes=frontier_nodes,
                    connected_to=connected_to,
                    connected_from=connected_from
                )
            
            summary_text = summary["output"]
            if summary_text.startswith("Summary:"):
                summary_text = summary_text.replace("Summary:", "", 1).strip()
            
            summaries[fsubgraph_name] = summary_text
            all_summaries_text.append(summary_text)
            all_summary_inputs.append(summary["input"])
            
            parsed_data_dir = os.path.join(output, "parsed_data", attack_name, f"magic_subgraph_{magicsubgraph_idx}", f"split_{subgraph_idx}")
            os.makedirs(parsed_data_dir, exist_ok=True)
            
            with open(os.path.join(parsed_data_dir, "malicious_edges.txt"), "w") as f:
                for item in malicious_edges:
                    f.write(item + "\n")
            
            with open(os.path.join(parsed_data_dir, "malicious_nodes.txt"), "w") as f:
                for item in malicious_nodes:
                    f.write(item + "\n")
        
        concatenated_summary_text = "\n\n".join(all_summaries_text)
        concatenated_summary_input = "\n\n".join(all_summary_inputs)
        
        concatenated_summary = {
            "input": concatenated_summary_input,
            "output": concatenated_summary_text
        }
        
        all_nodes = []
        all_nodes_func = []
        for _, _, malicious_nodes, malicious_nodes_func in subgraph_results:
            all_nodes.extend(malicious_nodes)
            all_nodes_func.extend(malicious_nodes_func)
        
        node_func_map = {}
        for i, node in enumerate(all_nodes):
            if node not in node_func_map:
                node_func_map[node] = all_nodes_func[i]
        all_nodes_unique = list(node_func_map.keys())
        
        suplement_node_info_concatenated = []
        for i in tqdm(range(len(all_nodes_unique)), desc="  Enriching for concatenated summary", leave=False, disable=len(all_nodes_unique) < 10):
            node = all_nodes_unique[i]
            functionality = node_func_map[node]
            response = self.enricher.validate_node(node)
            if response == "NO":
                suplement_node_info_concatenated.append((node, functionality))
        
        if not suplement_node_info_concatenated:
            suplement_node_info_concatenated = None
        
        magic_summaries_dir = os.path.join(output, "summaries", attack_name, f"magic_subgraph_{magicsubgraph_idx}")
        os.makedirs(magic_summaries_dir, exist_ok=True)
        
        with open(os.path.join(magic_summaries_dir, "summary.txt"), "w") as f:
            f.write(json.dumps(concatenated_summary))
        
        if not summary_only:
            concatenated_summarizer = summarizer([], all_nodes_unique, model, output, apt_stages_list, ollama_url=ollama_url)
            label_prompt = concatenated_summarizer.attack_stage_label_prompt()
            labelled_summary = concatenated_summarizer.query_model(label_prompt, concatenated_summary_text)
            
            labelled_results = {
                "input": concatenated_summary_text,
                "output": labelled_summary
            }
            
            with open(os.path.join(magic_summaries_dir, "labelled_summary.txt"), "w") as f:
                f.write(json.dumps(labelled_results))
        
        magic_parsed_data_dir = os.path.join(output, "parsed_data", attack_name, f"magic_subgraph_{magicsubgraph_idx}")
        os.makedirs(magic_parsed_data_dir, exist_ok=True)
        
        with open(os.path.join(magic_parsed_data_dir, "malicious_nodes.txt"), "w") as f:
            for node in all_nodes_unique:
                f.write(node + "\n")

    def create_output_directories(self, output, dataset=None):
        parsed_data_dir = os.path.join(output, "parsed_data")
        summary_dir = os.path.join(output, "summaries")

        if not os.path.exists(parsed_data_dir):
            os.makedirs(parsed_data_dir)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
    
    def create_output_directories_for_subgraphs(self, output, attack_name, dataset=None, magicsubgraph_idx=None):
        parsed_data_dir = os.path.join(output, "parsed_data", attack_name)
        summary_dir = os.path.join(output, "summaries", attack_name)
        eval_dir = os.path.join(output, "eval", attack_name)
        
        os.makedirs(parsed_data_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)


