import os
import sys
import argparse as argparse_early

def parse_args_early():
    parser = argparse_early.ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids (e.g. 0,1,2); default empty = use all available")
    args, _ = parser.parse_known_args()
    return args.gpus

gpus = parse_args_early()
if gpus and gpus.strip():
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import pickle
import json
import re

from tqdm import tqdm
import torch
import yaml
import requests

from openai import OpenAI

cache_dir = "./llm_cache"

huggingface_token = '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'

def parse_args():
    parser = argparse_early.ArgumentParser(description="KAIROS Graph creation")
    parser.add_argument("--dataset", type=str, default="fivedirections", help="Select dataset")
    parser.add_argument("--os_type", type=str, default="windows", help="Select os type linux/windows")
    parser.add_argument("--llm_name", type=str, default="llama3:70b", help="Select llm name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Specify temperature")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Specify max tokens")
    parser.add_argument("--ollama_url", type=str, default=None, help="URL for Ollama API (required when using Ollama models)")

    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")

    return parser.parse_args()


def extract_entries(text):
    entries = []
    candidate_blocks = re.split(r'(?=(?:Filename|File/Directory/URL|File/Directory|Command):\s*)', text, flags=re.IGNORECASE)
    final_entries = []
    
    for block in candidate_blocks:
        if not any(pattern in block.lower() for pattern in ["filename:", "file/directory/url:", "file/directory:", "command:"]):
            continue
        
        entry = {}
        start_pos = text.find(block)
        line_index = text[:start_pos].count('\n') + 1
        
        parts = block.split("|")
        for part in parts:
            part = part.strip()
            m = re.match(r"([^:]+):\s*(.*)", part)
            if m:
                key = m.group(1).strip().lower()
                value = m.group(2).strip()
                entry[key] = value.split("###")[0].strip().replace("", "").replace('---',"")
        
        has_entity_key = any(key in entry for key in ['filename', 'file/directory/url', 'file/directory', 'command'])
        has_type_or_functionality = any(key in entry for key in ['type', 'functionality'])
        
        if has_entity_key and has_type_or_functionality:
            entry['line_index'] = line_index
            entries.append(entry)

        for e in entries:
            flag = 0
            for key, value in e.items():
                if type(value) == int:
                    continue
                if '<' in value and '>' in value:
                    flag = 1
                    break
            if flag == 0 and e not in final_entries:
                final_entries.append(e)

    return final_entries


def classify_files_with_gpt4o(os_type, file_name, entity_type):

    with open("./config.yaml") as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)

    if entity_type == "COMMAND-LINE":
        prompt = (
            f"Analyze the following command line invocation from {os_type} operating system logs and determine its type. You will need to think through this carefully step-by-step.\n\n"
            f"Command: {file_name}\n\n"
            "Think through the following steps carefully before forming your final conclusion:\n\n"
            "1. First, examine the executable and its arguments. What does this tell you about the command's purpose?\n\n"
            "2. Analyze the command structure and parameters. Does it follow standard command line conventions?\n\n"
            "3. Consider the context and implications in terms of system operations. What would this command typically be used for?\n\n" 
            "4. Based on your analysis in steps 1-3, PROVIDE A DETAILED EXPLANATION of what the command does, its purpose, and how it operates in the system. Don't include any parts of the command - describe the general functionality comprehensively with technical details, and describe each technical term.\n"
            "5. Finally, SUMMARIZE the extracted functionality into a concise but VERY DETAILED `Type` label for the command. IMPORTANT: If the command is not a known valid command, is ambiguous, or appears suspicious, assign 'NO LABEL.'\n\n"
            "**STRICT OUTPUT FORMAT (Do not add extra text or line breaks):**\n"
            "Command: <command> | Type: <specific_detailed_type_label> | Functionality: <detailed_explanation_of_functionality>\n\n"
            "If uncertain or the command is not a recognized valid command, output:\n"
            "Command: <command> | Type: NO LABEL | Functionality: NONE\n"
        )
    else:
        prompt = (
            f"Analyze the following file/directory from {os_type} operating system logs or URL/website and determine its type. You will need to think through this carefully step-by-step.\n\n"
            f"File/Directory/URL: {file_name}\n\n"
            "Think through the following steps carefully before forming your final conclusion:\n\n"
            "1. First, determine if this is a URL/website (contains http://, https://, www., or domain patterns) or a file/directory path. If it's a URL, analyze the protocol (http/https), domain, path structure, and what type of web service or resource it represents. If it's a file/directory, examine the file extension (if any) or directory patterns.\n\n"
            "2. Analyze the naming pattern. For URLs: examine the domain, subdomain, path segments, and query parameters. For files/directories: check if it follows standard naming conventions for system files, applications, libraries, or other known patterns.\n\n"
            "3. Consider the context and implications in terms of system operations. For URLs: what type of network resource or web service does this represent? For files/directories: what would this file/directory typically be used for?\n\n" 
            "4. Based on your analysis in steps 1-3, PROVIDE A DETAILED EXPLANATION of what the entity does, its purpose, and how it operates in the system. Don't include any parts of the entity name - describe the general functionality comprehensively with technical details, and describe each technical term.\n"
            "5. Finally, SUMMARIZE the extracted functionality into a concise but VERY DETAILED `Type` label for the entity. IMPORTANT: If the entity is not a known valid file, directory, or URL type, is ambiguous, or appears suspicious, assign 'NO LABEL.'\n\n"
            "**STRICT OUTPUT FORMAT (Do not add extra text or line breaks):**\n"
            "File/Directory/URL: <entity_name> | Type: <specific_detailed_type_label> | Functionality: <detailed_explanation_of_functionality>\n\n"
            "If uncertain or the entity is not a recognized valid file, directory, or URL, output:\n"
            "File/Directory/URL: <entity_name> | Type: NO LABEL | Functionality: NONE\n"
        )   
    client = OpenAI(api_key = config_yaml['token'])
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a highly accurate entity classification model for files, directories, and network resources (URLs)."},
            {"role": "user", "content": prompt}
        ],
        temperature = 0,
        max_tokens = 4096
    )
    
    return response.choices[0].message.content


def classify_files_with_ollama(os_type, file_name, model_name, ollama_url, max_tokens, temperature, do_sample, entity_type):
    if entity_type == "COMMAND-LINE":
        prompt = (
            f"Analyze the following command line invocation from {os_type} operating system logs and determine its type. You will need to think through this carefully step-by-step.\n\n"
            f"Command: {file_name}\n\n"
            "Think through the following steps carefully before forming your final conclusion:\n\n"
            "1. First, examine the executable and its arguments. What does this tell you about the command's purpose?\n\n"
            "2. Analyze the command structure and parameters. Does it follow standard command line conventions?\n\n"
            "3. Consider the context and implications in terms of system operations. What would this command typically be used for?\n\n" 
            "4. Based on your analysis in steps 1-3, PROVIDE A DETAILED EXPLANATION of what the command does, its purpose, and how it operates in the system. Don't include any parts of the command - describe the general functionality comprehensively with technical details, and describe each technical term.\n"
            "5. Finally, SUMMARIZE the extracted functionality into a concise but VERY DETAILED `Type` label for the command. IMPORTANT: If the command is not a known valid command, is ambiguous, or appears suspicious, assign 'NO LABEL.'\n\n"
            "**STRICT OUTPUT FORMAT (Do not add extra text or line breaks):**\n"
            "Command: <command> | Type: <specific_detailed_type_label> | Functionality: <detailed_explanation_of_functionality>\n\n"
            "If uncertain or the command is not a recognized valid command, output:\n"
            "Command: <command> | Type: NO LABEL | Functionality: NONE\n"
        )
    else:
        prompt = (
            f"Analyze the following file/directory from {os_type} operating system logs or URL/website and determine its type. You will need to think through this carefully step-by-step.\n\n"
            f"File/Directory/URL: {file_name}\n\n"
            "Think through the following steps carefully before forming your final conclusion:\n\n"
            "1. First, determine if this is a URL/website (contains http://, https://, www., or domain patterns) or a file/directory path. If it's a URL, analyze the protocol (http/https), domain, path structure, and what type of web service or resource it represents. If it's a file/directory, examine the file extension (if any) or directory patterns.\n\n"
            "2. Analyze the naming pattern. For URLs: examine the domain, subdomain, path segments, and query parameters. For files/directories: check if it follows standard naming conventions for system files, applications, libraries, or other known patterns.\n\n"
            "3. Consider the context and implications in terms of system operations. For URLs: what type of network resource or web service does this represent? For files/directories: what would this file/directory typically be used for?\n\n" 
            "4. Based on your analysis in steps 1-3, PROVIDE A DETAILED EXPLANATION of what the entity does, its purpose, and how it operates in the system. Don't include any parts of the entity name - describe the general functionality comprehensively with technical details, and describe each technical term.\n"
            "5. Finally, SUMMARIZE the extracted functionality into a concise but VERY DETAILED `Type` label for the entity. IMPORTANT: If the entity is not a known valid file, directory, or URL type, is ambiguous, or appears suspicious, assign 'NO LABEL.'\n\n"
            "**STRICT OUTPUT FORMAT (Do not add extra text or line breaks):**\n"
            "File/Directory/URL: <entity_name> | Type: <specific_detailed_type_label> | Functionality: <detailed_explanation_of_functionality>\n\n"
            "If uncertain or the entity is not a recognized valid file, directory, or URL, output:\n"
            "File/Directory/URL: <entity_name> | Type: NO LABEL | Functionality: NONE\n"
        )


    messages = [
        {"role": "system", "content": "You are a highly accurate entity classification model for files, directories, and network resources (URLs)."},
        {"role": "user", "content": prompt}
    ]
    
    data = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature if do_sample else 0.0,
            "num_predict": max_tokens
        }
    }
    
    try:
        response = requests.post(f"{ollama_url}/api/chat", json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "")
        
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"


def load_validity_results(dataset, llm_name):
    validity_file = f"./file_classification_results/{llm_name}/ename_validity_{dataset.lower()}.json"
    
    if not os.path.exists(validity_file):
        return {}
    try:
        with open(validity_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except (json.JSONDecodeError, Exception) as e:
        return {}

def load_existing_feature_results(feature_file):
    if not os.path.exists(feature_file):
        return {}
    try:
        with open(feature_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except (json.JSONDecodeError, Exception) as e:
        return {}

def save_feature_results(feature_file, results):
    try:
        with open(feature_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        pass

if __name__ == "__main__":

    args = parse_args()
    dataset = args.dataset.upper()

    os_type = args.os_type.upper()
    llm_name = args.llm_name
    ollama_url = args.ollama_url

    if llm_name != 'gpt-4o' and not ollama_url:
        raise ValueError("--ollama_url is required when using Ollama models (not gpt-4o)")

    validity_results = load_validity_results(dataset, llm_name)
    if not validity_results:
        exit(1)
    valid_entities = []
    for entity, classification in validity_results.items():
        if classification in ["VALID", "COMMAND-LINE"]:
            valid_entities.append(entity)
    if not valid_entities:
        exit(1)
    output_dir = os.path.join("./llm-fets", llm_name)
    feature_file = os.path.join(output_dir, f"ename_fets_{dataset.lower()}.json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    existing_features = load_existing_feature_results(feature_file)
    remaining_entities = []
    for entity in valid_entities:
        if entity not in existing_features:
            remaining_entities.append(entity)
    if not remaining_entities:
        exit(0)
    for entity in tqdm(remaining_entities, desc="Extracting Features"):
        entity_type = validity_results[entity]
        if llm_name == 'gpt-4o':
            response_txt = classify_files_with_gpt4o(os_type, entity, entity_type)
        else:
            response_txt = classify_files_with_ollama(
                os_type=os_type,
                file_name=entity,
                model_name=llm_name,
                ollama_url=ollama_url,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                entity_type=entity_type
            )
        
        response = extract_entries(response_txt)
        if len(response) > 0 and len(response[0]) > 0:
            entry = response[0]
            entity_features = {
                "Type": entry.get('type', 'NO LABEL'),
                "Functionality": entry.get('functionality', 'NONE')
            }
        else:
            entity_features = {
                "Type": "NO LABEL",
                "Functionality": "NONE"
            }
        existing_features[entity] = entity_features
        save_feature_results(feature_file, existing_features)
