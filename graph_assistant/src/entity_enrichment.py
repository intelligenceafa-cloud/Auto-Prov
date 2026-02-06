
import json
import ollama
import ast

class enricher():

    def __init__(self, model, ollama_url=None):
        self.model = model
        if ollama_url:
            self.ollama_client = ollama.Client(host=ollama_url)
        else:
            self.ollama_client = ollama.Client()

    def validate_node(self, node):
        prompt = f"""
        
        You have been given a system entity (eg., processes, files, network sockets) from logs. Based on your knowledge, do you have information about the functionality of this system entity?

        Respond with ONLY "YES" or "NO" nothing else.

        system entity: {node}    

    """
        return self.query_model(prompt, node, self.model)
    
    def query_model(self, system_prompt, user_prompt, model):
        messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
        
        response = self.ollama_client.chat(
            model=model, 
            messages=messages,
            options={
            "temperature": 0,
            "num_predict": 4096
            }
        )
        return response["message"]["content"]
    
    def enrich(self, node):
        pass


def main():
    agent = Classification_agent("llama3.1:8b")
    
    
    with open("data_parsing/results/malicious_theia_node_classification_subgraph_3.txt", 'r', encoding='utf-8') as f:
        malicious_nodes = f.readline().split(":",1)[1]
        malicious_nodes_list = ast.literal_eval(malicious_nodes)

    malicious_nodes_list.append("boo.exe")
    for item in malicious_nodes_list:
        pass

if __name__ == "__main__":
    main()

