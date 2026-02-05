#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import json
import re
import requests
from tqdm import tqdm
import argparse
import glob
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="File/Directory Classification using LLM")
    parser.add_argument("--llm_name", type=str, default="llama3:70b", help="Select llm name for Ollama")
    parser.add_argument("--dataset_name", type=str, default="fivedirections", help="Select dataset name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Specify temperature")
    parser.add_argument("--max_tokens", type=int, default=500, help="Specify max tokens")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="URL for Ollama API")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    
    return parser.parse_args()


def extract_classification_result(text):
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if '|' in line and 'Classification:' in line:
            parts = line.split('|')
            for part in parts:
                part = part.strip()
                if part.startswith('Classification:'):
                    classification = part.split(':', 1)[1].strip().upper()
                    if 'COMMAND-LINE' in classification or 'COMMANDLINE' in classification:
                        return 'COMMAND-LINE'
                    elif 'INVALID' in classification:
                        return 'INVALID'
                    elif 'VALID' in classification:
                        return 'VALID'
    
    text_upper = text.upper()
    if 'COMMAND-LINE' in text_upper or 'COMMANDLINE' in text_upper:
        return 'COMMAND-LINE'
    elif 'VALID' in text_upper and 'INVALID' not in text_upper:
        return 'VALID'
    elif 'INVALID' in text_upper:
        return 'INVALID'
    
    return 'INVALID'

def classify_word_with_ollama(word, model_name, ollama_url, max_tokens, temperature, do_sample):
    prompt = (
        f"Analyze the following text and determine if it represents a valid file system object (file or directory) or network resource (URL/website).\n\n"
        f"Text to analyze: {word}\n\n"
        "Classification rules:\n\n"
        "**VALID**: If the text appears to be:\n"
        "- A file path or filename (has extensions like .dll, .exe, etc. or clear file patterns)\n"
        "- A directory, registry path or folder name (no file extension, contains typical directory patterns)\n"
        "- A URL, website link, or web address (http://, https://, www., domain names, etc.)\n"
        "- Any recognizable file system path or object\n\n"
        "**COMMAND-LINE**: If the text appears to be:\n"
        "- A command line invocation or process execution (executable with arguments, parameters, and options)\n"
        "- A complete command with flags, parameters, and arguments\n\n"
        "**INVALID**: If the text is:\n"
        "- Abstract terms, keywords, or placeholder text\n"
        "- IDs, or identifiers\n"
        "- Not representing a file system object or network resource\n\n"
        "**STRICT OUTPUT FORMAT (single line only):**\n"
        "Word: <original_word> | Classification: <VALID|INVALID|COMMAND-LINE> | Explanation: <brief_reason>\n\n"
    )
    
    messages = [
        {"role": "system", "content": "You are a highly accurate entity classifier. Classify text as VALID (file system objects or network resources), INVALID, or COMMAND-LINE entities."},
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

def load_words_from_cluster_map(cluster_map_file):
    words = []
    
    if not os.path.exists(cluster_map_file):
        return words
    try:
        with open(cluster_map_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            unique_values = set()
            for value in data.values():
                if isinstance(value, str) and value.strip():
                    unique_values.add(value.strip())
            
            for value in unique_values:
                words.append({
                    'word': value,
                    'source_file': os.path.basename(cluster_map_file),
                    'original_output': value
                })
        else:
            return words
    except (json.JSONDecodeError, Exception) as e:
        return words
    return words

def load_existing_validity_results(validity_file):
    if not os.path.exists(validity_file):
        return {}
    try:
        with open(validity_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except (json.JSONDecodeError, Exception) as e:
        return {}

def save_validity_results(validity_file, results):
    try:
        with open(validity_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        pass

if __name__ == "__main__":
    args = parse_args()
    
    llm_name = args.llm_name
    ollama_url = args.ollama_url
    dataset_name = args.dataset_name.lower()
    
    cluster_map_file = f"./maps/ename-cluster-map_{dataset_name.lower()}.json"
    output_dir = os.path.join("./file_classification_results", llm_name)
    validity_file = os.path.join(output_dir, f"ename_validity_{dataset_name.lower()}.json")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    word_data = load_words_from_cluster_map(cluster_map_file)
    if not word_data:
        exit(1)
    unique_words = {}
    for word_entry in word_data:
        word = word_entry['word']
        if word not in unique_words:
            unique_words[word] = word_entry
    word_data = list(unique_words.values())
    existing_results = load_existing_validity_results(validity_file)
    remaining_words = []
    for word_entry in word_data:
        word = word_entry['word']
        if word not in existing_results:
            remaining_words.append(word_entry)
    if not remaining_words:
        exit(0)
    for word_entry in tqdm(remaining_words, desc="Classifying words"):
        word = word_entry['word']
        response_text = classify_word_with_ollama(
            word=word,
            model_name=llm_name,
            ollama_url=ollama_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample
        )
        
        classification = extract_classification_result(response_text)
        existing_results[word] = classification
        save_validity_results(validity_file, existing_results)
