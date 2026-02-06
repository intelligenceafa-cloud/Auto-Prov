import os
import ollama
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class summarizer:
    
    def __init__(self, graph_data, node_classification_data, model, output, apt_stages, ollama_url=None):
        self.graph_data = graph_data
        self.node_classification_data = node_classification_data
        self.model = model
        self.output = output
        self.apt_stages = apt_stages
        if ollama_url:
            self.ollama_client = ollama.Client(host=ollama_url)
        else:
            self.ollama_client = ollama.Client()
    def summarize(self, sumplement_node_info, previous_summary=None, connected_summaries=None, frontier_nodes=None, connected_to=None, connected_from=None):

        system_prompt = self.construct_prompt(
            self.node_classification_data, 
            sumplement_node_info, 
            previous_summary=previous_summary,
            connected_summaries=connected_summaries,
            frontier_nodes=frontier_nodes,
            connected_to=connected_to,
            connected_from=connected_from
        )

        graph_data = "\n".join(self.graph_data)
        summary  = self.query_model(system_prompt, graph_data)
        summarize_results = {
            "input": graph_data,
            "output": summary
        }

        label_prompt = self.attack_stage_label_prompt()

        labelled_summary  = self.query_model(label_prompt, summary)

        labelled_results = {
            "input": summary,
            "output": labelled_summary
        }

        return summarize_results, labelled_results

    def attack_stage_label_prompt(self):
        system_prompt = f"""

    You are an expert in cybersecurity, the English language, and system provenance graphs. You will be given an attack summary derived from a provenance graph. Your task is to compare the fundamental nature of the events described in the summary to the APT stages from the provided list. For each stage that is clearly evident in the summary, identify it and explain your reasoning clearly.


    Here are the APT stages to consider:
    {self.apt_stages}


    Task:

    List only the stages that are clearly evident in the attack summary.

    For each identified stage, provide an explanation specifically referencing system entities (processes, files, network connections, IP addresses, domains, websites etc.) mentioned in the summary that connect the events to the identified APT stage.

    Use a structured format:

    Stage: [Name of stage]
    Reasoning: [Your explanation]

        Guidelines:
    1. You do not need to find all the stages, only those supported by the text.
    2. Think carefully and provide step-by-step reasoning in your explanations.
        """
        return system_prompt

    def construct_prompt(self, node_classification_data, suplement_node_info, previous_summary=None, connected_summaries=None, frontier_nodes=None, connected_to=None, connected_from=None):
        system_prompt = f"""You are an expert in cybersecurity, english language, and system provenance graphs. You will be given a provenance graph as input where each line represents an edge within the graph with the following structure: [node] --[edge type]--> [node]. The edges are ordered chronologically (from earliest to latest timestamp), representing the temporal sequence of events in the attack.

When the same edge (same source node, target node, and action type) appears consecutively in the sequence, it is shown with a count in parentheses, e.g., "A --read--> B (x3)" means this edge occurred 3 times in sequence. This indicates repeated or ongoing operations, such as persistent connections, repeated file access, or multiple queries. Single occurrences do not show a count.

These are the nodes within the graph that have been identified as malicious in a Python list: {self.node_classification_data}

The rest of the nodes that aren't listed above can be either malicious or benign.

Below is the provided node information to supplement your understanding, it is in a tuple containing (node_name, node_functionality):

{suplement_node_info}"""
        
        if frontier_nodes or connected_to or connected_from:
            system_prompt += "\n\nGRAPH STRUCTURE:"
            system_prompt += "\nThis subgraph is part of a larger attack graph with the following connections:"
            
            if connected_from:
                system_prompt += f"\n- Connects FROM: {', '.join(connected_from)}"
            if connected_to:
                system_prompt += f"\n- Connects TO: {', '.join(connected_to)}"
            
            if frontier_nodes:
                incoming = frontier_nodes.get('incoming', [])
                outgoing = frontier_nodes.get('outgoing', [])
                all_frontiers = incoming + outgoing
                if all_frontiers:
                    system_prompt += f"\n- IMPORTANT: The following nodes are connection points (frontiers) to other parts of the attack: {all_frontiers}. These nodes are critical for understanding how events in this subgraph relate to the overall attack flow."
        
        if connected_summaries:
            system_prompt += "\n\nCONTEXT FROM CONNECTED SUBGRAPHS:"
            for subgraph_name, summary_text in connected_summaries.items():
                if summary_text:
                    clean_text = summary_text
                    if clean_text.startswith("Summary:"):
                        clean_text = clean_text.replace("Summary:", "", 1).strip()
                    system_prompt += f"\n\nSummary from {subgraph_name}:"
                    system_prompt += f"\n{clean_text}"
            
            system_prompt += "\n\nWhen writing your summary, you may reference relevant events from these connected subgraphs if they help explain the current activities. However, focus primarily on describing what is happening in THIS subgraph."
        
        elif previous_summary:
            previous_text = previous_summary
            if previous_text.startswith("Summary:"):
                previous_text = previous_text.replace("Summary:", "", 1).strip()
            
            system_prompt += f"""

Previous Context:
The following is a summary of the previous part of this attack graph:
{previous_text}

This current subgraph continues from where the previous part left off. When writing your summary, you may reference events from the previous summary if they help explain the current activities. However, focus primarily on describing what is happening in THIS subgraph."""
        
        system_prompt += f"""

Task:

Your task is to produce a detailed block of text that narrates what is happening in this graph WITHOUT infering any APT stages, step by step.

The graph data may only represent a portion of the attacker's overall operation. construct the most coherent narrative possible from the provided graph.

DO NOT include the following apt stages in your summary.
    {self.apt_stages}

Use a structured format:

Summary: [Your detailed narrative here as a continuous block of text, in ONE paragraph]

 Guidelines:
1. DO NOT include any labels or subheadings within the summary text itself.
2. DO NOT explain what you are doing. Simply produce the final narrative.
3. DO NOT infer any APT stages.
4. CRITICAL: Surround any referenced nodes (processes, files, IP addresses, domains, websites, file paths, registry keys, etc.) within double quotes. This is essential for accurate evaluation. Every entity mentioned in the summary must be quoted.
5. The summary text should be a continuous block of text (ONE paragraph) without bullet points, numbered lists, or structured formatting. Write in natural flowing prose.
   
    """
        return system_prompt

    def query_model(self, system_prompt, user_prompt):
        messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]

        response = self.ollama_client.chat(
            model=self.model,
            messages=messages,
            options={
            "temperature": 0,
            "num_predict": 4096
            }
        )
        return response["message"]["content"]
