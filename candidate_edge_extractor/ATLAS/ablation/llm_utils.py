from openai import OpenAI
import os
from llm_backend import call_llm


def llm_desc_cont(llm_model, key, log_report, id_eg, unique_entities_examples, log_type, model_type="openai", ollama_url=None):

    messages = [     
    {
    "role": "system",
    "content": '''You are an expert in cybersecurity, english language, and system provenance graphs. You will be provided with a  '''+log_type+''' LOGS. Your task is to understand its structure and meaning to write a paragraph that records all the system-level services and applications names (along with their unique IDs within brackets (...)), file names/paths (along with their unique IDs within brackets (...)), registry keys (along with their unique IDs within brackets (...)), and hosts/users (along with their unique IDs within brackets (...)) present in the logs, and the only event names happening between them (along with their unique IDs within brackets (...)). These information will be later used to create a provenance graph. Take a deep breath and think step by step.\n\n
    (Step 1) Go to the log and deeply understand the relations that are important to construct system provenance graphs. Then write a detailed paragraph describing all the activities happening in that log, where each sentence should talk about only one activity. The paragraphs should also include all the system-level service and application names, file names/paths, registry keys, and hosts/users name and along with their unique IDs only. Do not include any other IDs. Each log may have multiple interactions, therefore, indicate all relevant hierarchial relationships/interactions between the entities in separate sentences.\n
    (Step 2) Each line in that paragraph should be in active voice. Ensure the paragraph to include only the information of the relevant (1) system-level service names within quotes "..." and their unique IDs within round brackets (...), (2) system-level application names within quotes "..." and their unique IDs within round brackets (...), (3)  file names/paths, registry keys, and hosts/users names within quotes "..." and their unique IDs within round brackets (...), (4) different types of source/destination IP Addresses and port number within quotes "...". Do not include action/event/access IDs or any other IDs not useful for system provenance graphs.\n
    (Step 3) Search for the words describing the action/event/access types in the paragraph. Put within curly brackets {...} in the paragraph. Do not put them within quotes "...".\n
    (Step 4) Search for the unique subject and object IDs in the paragraph. Put them within round brackets (...) in the paragraph. Do not put them within quotes "...".\n
    (Step 5) Search for the words describing the "type" of system-level services or applications or files or registry keys or IP addresses in the paragraph. MAKE SURE to put them within square brackets [...] in the paragraph. Do not put them within quotes "...".\n
    (Step 6) Strictly follow the given output format to display your final output paragraph.\n\n
    Output Format:\n\n### Final Output\n\n[PARAGRAPH]\n\nparagraph...\n\n\n
    Additionally, display the output of each step in a very concise way.'''+'''\n\nHere are some examples of the entity names:\n\n'''+unique_entities_examples+'''\n\nHere are some examples of the IDs:\n\n'''+id_eg
    },
    { "role": "user","content": "LOGS: \n\n"+ log_report},]

    ans = call_llm(
        model_type=model_type,
        model_name=llm_model,
        messages=messages,
        api_key=key,
        ollama_url=ollama_url,
        temperature=0,
        max_tokens=4096,
        seed=400
    )

    return ans

def llm_desc(llm_model, key, log_report, log_type, model_type="openai", ollama_url=None):

    messages = [     
{
"role": "system",
"content": '''You are an expert in cybersecurity, english language, and system provenance graphs. You will be provided with a  '''+log_type+''' LOG. Your task is to understand its structure and meaning to write a paragraph that records all the system-level services and applications names (along with their unique IDs within brackets (...)), file names/paths (along with their unique IDs within brackets (...)), registry keys (along with their unique IDs within brackets (...)), and hosts/users (along with their unique IDs within brackets (...)) present in the log, and the only action/event/access names happening between them (do not include action/event/access IDs). These information will be later used to create a provenance graph, therefore, do not miss the relevant IDs. Take a deep breath and think step by step.\n\n
(Step 1) Go to the log and deeply understand the relations that are important to construct system provenance graphs. Then, write a detailed paragraph describing all the activities happening in that log. The paragraphs should also include all the system-level service and application names, file names/paths, registry keys, and hosts/users, along with their unique IDs only. Indicate all relevant hierarchial relationships/interactions between applications. Do not include any other IDs.\n
(Step 2) Each line in that paragraph should be in active voice. Ensure the paragraph to include only the information of the relevant (1) system-level service names within quotes "..." and their unique IDs within round brackets (...) beside them, (2) system-level application names within quotes "..." and their unique IDs within round brackets (...) beside them, (3)  file names/paths, registry keys, and host/user names within quotes "...", and their unique IDs within round brackets (...) beside them, (4) different types of source/destination IP Addresses and port number within quotes "...". Do not include action/event/access IDs or any other IDs not useful for system provenance graphs.\n
(Step 3) Search for the words describing the action/event/access types in the paragraph. Put within curly brackets {...} in the paragraph. Do not put them within quotes "...".\n
(Step 4) Search for the unique subject and object IDs in the paragraph. Put them within round brackets (...) in the paragraph. Do not put them within quotes "...".\n
(Step 5) Search for the words describing the "type" of system-level services or applications or files or registry keys or IP addresses in the paragraph. MAKE SURE to put put them within square brackets [...] in the paragraph. Do not put them within quotes "...".\n
(Step 6) Strictly follow the given output format to display your final output paragraph.\n\n
Output Format:\n\n### Final Output\n\n[PARAGRAPH]\n\nparagraph...\n\n\n
Additionally, display the output of each step in a very concise way.'''
},
{ "role": "user","content": "LOGS: \n\n"+ log_report},]

    ans = call_llm(
        model_type=model_type,
        model_name=llm_model,
        messages=messages,
        api_key=key,
        ollama_url=ollama_url,
        temperature=0,
        max_tokens=4096,
        seed=400
    )

    return ans

def llm_entitytypExt(llm_model, key, logs, desc, log_type, model_type="openai", ollama_url=None):

    messages = [

{
"role": "system",
"content": '''You are an expert in cybersecurity, system provenance graphs, and english language. You will be given the following as input:\n 
(1) a  '''+log_type+''' LOG,\n
(2) a paragraph describing the activities happening in the  '''+log_type+''' LOG,\n 
Your task is to extract the entity types of each of the ids and the different types of IP addresses. These information will be later used to create a provenance graph. Take a deep breath and let's do step by step.\n\n
(Step 1) Deeply understand the structure and meaning of the LOG, and the meaning of the PARAGRAPH given to you as input.\n
(Step 2) Identify all unique IDs present in round brackets (...). Store these IDs in list L, ensuring no duplicate entries. Then, extract any source/local and destination/remote IP addresses, and store them in a list L. There can be different types of IP addresses. Ignore event IDs.\n
(Step 3) Read the LOG and the PARAGRAPH again and find if any "entity ID" is missed in the list L. If the "enitity ID" of any file, application or registry key is missed, extract the "enitity ID" (that can be useful for provenance graphs) from the LOG and add it to the list L. Do not ever include any entity name (even if the entity name has any ID in it) in this list L.\n
(Step 4) Read the LOG and the PARAGRAPH again and for each "entity ID" in list L, write the entity-type of its corresponding entity-name written within [...] in the PARAGRAPH or written in the LOG, to create a provenance graph. If any provenance graph related entity-type is not present or is blank, write NONE. Please note, different "entity IDs" in the list may have the same entity-type.\n 
(Step 5) Strictly follow the given output format to display your final output.\n\n\n
Output Format:\n\n### Final Output\n\n[ENTITY TYPES]\n\n
"ID-1" = "entity-type-1"\n
"ID-2" = "entity-type-2"\n
"ID-3" = "NONE"\n
"IP:port-1" = "source address"\n
"IP:port-2" = "destination address"\n
...\n
\n\nAdditionally, display the output of each step.\n\n'''
},
{"role": "user","content": "[LOG]"+logs+"\n\n"+desc},

    ]

    ans = call_llm(
        model_type=model_type,
        model_name=llm_model,
        messages=messages,
        api_key=key,
        ollama_url=ollama_url,
        temperature=0,
        max_tokens=4096,
        seed=400
    )

    return ans


def llm_summarize_desc(llm_model, key, log_reports, model_type="openai", ollama_url=None):

   
    messages = [     
    {
    "role": "system",
    "content": "Summarize the following paragraphs into a single cohesive paragraph without missing any information:\n"},
    { "role": "user","content": "PARAGRAPHS: \n\n"+log_reports},]

    ans = call_llm(
        model_type=model_type,
        model_name=llm_model,
        messages=messages,
        api_key=key,
        ollama_url=ollama_url,
        temperature=0,
        max_tokens=4096,
        seed=400
    )

    return ans


def llm_GlobalEntity(llm_model, key, desc, eid_input, logs, log_type, model_type="openai", ollama_url=None):
    messages = [

{
"role": "system",
"content": '''You are an expert in cybersecurity, system provenance graphs, and English language. You will be given a [PARAGRAPH] describing the activities happening in a ''' + log_type + ''' LOG, entity IDs from the LOG that are having activity, and the LOG. Take a deep breath and let's do this step by step.\n\n
  (Step 1) Carefully parse each line of the [PARAGRAPH] and the [LOG]. Identify all unique IDs present in round brackets (...) or given to you as input [Entity IDs]. Store these IDs in list L, ensuring no duplicate entries. Then, extract any source/local and destination/remote IP addresses, and store them in a list L. There can be different types of IP addresses. Ignore event IDs.\n
  (Step 2) For each entity in list L, read the [PARAGRAPH] and the [LOG] to identify **hierarchical or transitory relationships** (e.g., parent-child relationships between processes or between services and files). Create pairs of IDs from the [Entity IDs] in the format (Higher-Level-ID, Lower-Level-ID) to represent hierarchical relationships.\n
  (Step 3) Identify if any two entities (IDs) in list L have an explicit interaction described in the [PARAGRAPH] and the [LOG]. For each pair of interacting entities, represent their relationship in the format (ID-1, ID-2). Do not ever include system-level service or application names or file paths from the pairings.\n
(Step 4) For each "IP address" in the list L, make pairs of the IPs as (IP:port-1, IP:port-2) if there is a connection between them in the [PARAGRAPH]. There can be different types of IP addresses.\n
(Step 5) For each "IP address" in the list L and for each "id" in the list L, make pairs of the ID and the IP as (IP:port-1, ID-1) or (ID-1, IP:port-1) if they have a relation or an event happening between them in the [PARAGRAPH] and the [LOG]. There can be different types of IP addresses. Do not ever add the system-level service or application names or file paths or user names in the pairs.\n
(Step 6) Write all the "unique pairs" that you created. If no provenance graph meaningful pairs are possible, display [NO MEANINGFUL PAIRS POSSIBLE], or else, go to (Step 7). \n
(Step 7) Only display the unique pairs in the following format but make sure not to write terms such as "ID", "IP", etc. at the beginning. Keep the A: [...] arrays empty. Do not display anything else.\n\n\n
  Output Format:\n\n### Final Output\n\n[RELATED ENTITY IDs and IP Addresses]\n\n
  (ID-1, ID-2)  A: [...]\n
  (ID-1, ID-3)  A: [...]\n
  (ID-2, ID-5)  A: [...]\n
  (ID-6, ID-3)  A: [...]\n
  (ID-3, ID-4)  A: [...]\n
  (ID-4, ID-5)  A: [...]\n
  (ID-6, IP:port-5) A: [...]\n
  (IP:port-1, IP:port-2)  A: [...]\n
  (IP:port-1, IP:port-3)  A: [...]\n
  (IP:port-1, IP:port-4)  A: [...]\n
  (ID-7, IP:port-1) A: [...]\n...
  \n\nAdditionally, display the output of each step.'''
},
{"role": "user","content": desc+'\n\n[Entity IDs]\n'+eid_input+'\n\n[LOG]\n\n'+logs},

    ]

    ans = call_llm(
        model_type=model_type,
        model_name=llm_model,
        messages=messages,
        api_key=key,
        ollama_url=ollama_url,
        temperature=0,
        max_tokens=4096,
        seed=400
    )

    return ans


def llm_entityNames(llm_model, key, logs, desc, pairs, unique_entities_examples,  log_type, model_type="openai", ollama_url=None):

    messages = [

{
"role": "system",
"content": '''You are an expert in cybersecurity and english language. You will be given the following as input:\n 
(1) a  '''+log_type+''' LOG,\n
(2) a paragraph describing the activities happening in the  '''+log_type+''' LOG,\n 
(3) pairs of entity IDs as (ID-1, ID-2),\n 
(4) pairs of IP addresses as (IP:port-1, IP:port-2)\n 
(5) pairs of the entity ID and the IP as (IP:port-1, ID-1) or (ID-1, IP:port-1).\n\n
Your task is to ignore the IP addresses and display the full entity names which includes system-level services and applications names, file names, file paths, full website URLs of each entity ID if they are in the [PARAGRAPH]. These information will be later used to create a provenance graph. Take a deep breath and let's do step by step.\n\n
(Step 1) Deeply understand the structure and meaning of each of the LOG, and the meaning of the PARAGRAPH given to you as input.\n
(Step 2) Find the full entity-names written within quotes "..." in the PARAGRAPH and store them in a list L. The full enity-names that you find should only be system-level service names, file names, file paths, full website URLs, and application names. Do not store words that represent entity types, event types, and action types.\n
(Step 3) From the given pairs, store unique "entity IDS" in a list K.\n
(Step 4) Read the LOG and the PARAGRAPH again and for each "entity ID" in list K, write their corresponding full entity-name or from the list L to create a provenance graph. If any entity-name is not present or is blank, write NONE. If the entity-name from L represent a entity type or event type, write NONE. Please note, different "entity IDs" in the list K may have the same entity-name.\n 
(Step 5) Strictly follow the given output format to display your final output.\n\n\n
Output Format:\n\n### Final Output\n\n[ENTITY NAMES/PARENT IDS]\n\n
"ID-1" = "entity-name-1"\n
"ID-2" = "entity-name-2"\n
"ID-3" = "entity-name-1"\n
"ID-4" = "entity-name-2"\n
"ID-5" = NONE\n
"ID-6" = "entity-name-6"\n
"ID-7" = NONE\n
"ID-8" = "entity-name-3"\n
"ID-9" = NONE\n
...\n
\n\nAdditionally, display the output of each step.\n\n
Here are some examples of the entity names:\n\n'''+unique_entities_examples
},
{"role": "user","content": "[LOG]"+logs+"\n\n"+desc+"\n\n[RELATED ENTITY ID PAIRS]\n\n"+pairs},

    ]

    ans = call_llm(
        model_type=model_type,
        model_name=llm_model,
        messages=messages,
        api_key=key,
        ollama_url=ollama_url,
        temperature=0,
        max_tokens=4096,
        seed=400
    )

    return ans


def llm_entityNames_init(llm_model, key, logs, desc, pairs, log_type, model_type="openai", ollama_url=None):

    messages = [

{
"role": "system",
"content": '''You are an expert in cybersecurity and english language. You will be given the following as input:\n 
(1) a  '''+log_type+''' LOG,\n
(2) a paragraph describing the activities happening in the  '''+log_type+''' LOG,\n 
(3) pairs of entity IDs as (ID-1, ID-2),\n 
(4) pairs of IP addresses as (IP:port-1, IP:port-2)\n 
(5) pairs of the entity ID and the IP as (IP:port-1, ID-1) or (ID-1, IP:port-1).\n\n
Your task is to ignore the IP addresses and display the full entity names which includes system-level services and applications names, file names, file paths, full website URLs of each entity ID if they are in the [PARAGRAPH]. These information will be later used to create a provenance graph. Take a deep breath and let's do step by step.\n\n
(Step 1) Deeply understand the structure and meaning of the LOG, and the meaning of the PARAGRAPH given to you as input.\n
(Step 2) Find the full entity-names written within quotes "..." in the PARAGRAPH and store them in a list L. The full enity-names that you find should only be system-level service names, file names, file paths, full website URLs, and application names. Do not store words that represent entity types, event types, and action types.\n
(Step 3) From the given pairs, store unique "entity IDS" in a list K.\n
(Step 4) Read the LOG and the PARAGRAPH again and for each "entity ID" in list K, write their corresponding full entity-name or from the list L to create a provenance graph. If any entity-name is not present or is blank, write NONE. If the entity-name from L represent a entity type or event type, write NONE. Please note, different "entity IDs" in the list K may have the same entity-name.\n 
(Step 5) Strictly follow the given output format to display your final output.\n\n\n
Output Format:\n\n### Final Output\n\n[ENTITY NAMES/PARENT IDS]\n\n
"ID-1" = "entity-name-1"\n
"ID-2" = "entity-name-2"\n
"ID-3" = "entity-name-1"\n
"ID-4" = "entity-name-2"\n
"ID-5" = NONE\n
"ID-6" = "entity-name-6"\n
"ID-7" = NONE\n
"ID-8" = "entity-name-3"\n
"ID-9" = NONE\n
...\n
\n\nAdditionally, display the output of each step.\n\n'''
},
{"role": "user","content": "[LOGS]"+logs+"\n\n"+desc+"\n\n[RELATED ENTITY ID PAIRS]\n\n"+pairs},

    ]

    ans = call_llm(
        model_type=model_type,
        model_name=llm_model,
        messages=messages,
        api_key=key,
        ollama_url=ollama_url,
        temperature=0,
        max_tokens=4096,
        seed=400
    )

    return ans


def llm_GlobalEntity_contEg(llm_model, key, desc, eid_input, logs, eg, log_type, model_type="openai", ollama_url=None):

    messages = [
                 
    
{
"role": "system",
"content": '''You are an expert in cybersecurity, English language, and system provenance graphs. You will be given a [PARAGRAPH] describing the activities happening in a ''' + log_type + ''' LOG, entity IDs from the LOG that are having activity, and the LOG. Take a deep breath and let's do this step by step.\n\n
  (Step 1) Carefully parse each line of the [PARAGRAPH] and the [LOG]. Identify all unique IDs present in round brackets (...) or given to you as input [Entity IDs]. Store these IDs in list L, ensuring no duplicate entries. Then, extract any source/local and destination/remote IP addresses, and store them in a list L. There can be different types of IP addresses. Ignore event IDs.\n
  (Step 2) For each entity in list L, read the [PARAGRAPH] and the [LOG] to identify **hierarchical or transitory relationships** (e.g., parent-child relationships between hosts and other entities, processes or between other services, and files).  Create pairs of IDs from the [Entity IDs] in the format (Higher-Level-ID, Lower-Level-ID) to represent hierarchical relationships.\n
  (Step 3) Identify if any two entities (IDs) in list L have an explicit interaction described in the [PARAGRAPH] and the [LOG]. For each pair of interacting entities (IDs), represent their relationship in the format (ID-1, ID-2). Do not ever include system-level service or application names or file paths from the pairings.\n
(Step 4) For each "IP address" in the list L, make pairs of the IPs as (IP:port-1, IP:port-2) if there is a connection between them in the [PARAGRAPH] and the [LOG]. There can be different types of IP addresses.\n
(Step 5) For each "IP address" in the list L and for each "id" in the list L, make pairs of the ID and the IP as (IP:port-1, ID-1) or (ID-1, IP:port-1) if they have a relation or an event happening between them in the [PARAGRAPH] and the [LOG]. There can be different types of IP addresses. Do not ever add the system-level service or application names or file paths or user names in the pairs.\n
(Step 6) Write all the "unique pairs" that you created.\n
(Step 7) Only display the unique pairs in the following format but make sure not to write terms such as "ID", "IP", etc. at the beginning. Keep the A: [...] arrays empty. Do not display anything else.\n\n\n
  Output Format:\n\n### Final Output\n\n[RELATED ENTITY IDs and IP Addresses]\n\n
  (ID-1, ID-2)  A: [...]\n
  (ID-1, ID-3)  A: [...]\n
  (ID-2, ID-5)  A: [...]\n
  (ID-6, ID-3)  A: [...]\n
  (ID-3, ID-4)  A: [...]\n
  (ID-4, ID-5)  A: [...]\n
  (ID-6, IP:port-5) A: [...]\n
  (IP:port-1, IP:port-2)  A: [...]\n
  (IP:port-1, IP:port-3)  A: [...]\n
  (IP:port-1, IP:port-4)  A: [...]\n
  (ID-7, IP:port-1) A: [...]\n...\n\n\nHere are some examples of the outputs that you have to produce:\n\n'''+eg+'''
\n\n\nAdditionally, display the output of each step.'''
},

{"role": "user","content": desc+'\n\n[Entity IDs]\n'+eid_input+'\n\n[LOG]\n\n'+logs},
    ]

    ans = call_llm(
        model_type=model_type,
        model_name=llm_model,
        messages=messages,
        api_key=key,
        ollama_url=ollama_url,
        temperature=0,
        max_tokens=4096,
        seed=400
    )

    return ans


def llm_GlobalAct(llm_model, key, log_report, desc, fillup, log_type, model_type="openai", ollama_url=None):

    messages = [

{
"role": "system",
"content": '''You are an expert in cybersecurity and system provenance graphs. You will be given the following as input:\n 
(1) a  '''+log_type+''' LOG,\n
(2) a paragraph describing the activities happening in the '''+log_type+''' LOG,\n 
(3) pairs of entity IDs as (ID-1, ID-2),\n 
(5) pairs of IP addresses as (IP:port-1, IP:port-2)\n 
(6) pairs of the entity ID and the IP as (IP:port-1, ID-1) or (ID-1, IP:port-1).\n\n
Your task is to read each of the logs, understand its format, and find the action/event/access types that between the elements in the pairs given to you as input along with their directions, to create a provenance graph. Take a deep breath and let's do step by step.\n\n
(Step 1) Analyze all the log and understand all the operations that are happening.\n
(Step 2) Read the [LOGS] to fill all the ""A: [...]"" arrays with the exact name of the file/system/network level action/event/access types between the elements in its corresponding pairs.    \n**NO REPHRASING**, use the exact wording of the types found in the [LOGS] and DO NOT MISS THE RELEVANT TYPES.    \n**IF YOU CANNOT FIND** a '''+log_type+''' file/system/network level action/event/access type relevant for provenance graphs, to reduce ambiguity write all of them as "NO LABEL", and do not rephrase or make up action/event/access types.\n
(Step 3) Also, read the [PARAGRAPH] and specify the direction of the action/event/access types that is between the elements in each of the pairs. If it is from left to right then write {D=->} and if it is from right to left then write {D=<-}.\n
(Step 4) Also, read the [LOGS] and write the timestamp of each edge within (timestamp=...). If no timestamp is written in the [LOGS], just (timestamp=...).\n
(Step 5) Only display the output in the following format and nothing else.\n\n\n
Output Format:\n\n[ENTITY ID PAIRS and ACTIONS]\n\n 
(ID-1, ID-2)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-1, ID-2)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-1, ID-3)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-1, ID-3)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-2, ID-5)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-6, ID-3)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-3, ID-4)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-4, ID-5)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-6, IP:port-5) A: [...] {D=-> or D=<-} (timestamp=...)\n
(IP:port-1, IP:port-2)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(IP:port-1, IP:port-3)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(IP:port-1, IP:port-4)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-7, IP:port-1) A: [...] {D=-> or D=<-} (timestamp=...)\n
...'''},
{"role": "user","content": "[LOGS] \n\n" +log_report+"\n\n"+desc+"\n\n[RELATED ENTITY ID PAIRS]\n\n"+fillup},

    ]

    ans = call_llm(
        model_type=model_type,
        model_name=llm_model,
        messages=messages,
        api_key=key,
        ollama_url=ollama_url,
        temperature=0,
        max_tokens=4096,
        seed=400
    )

    return ans


def llm_GlobalAct_cont(llm_model, key, desc, pairs, log_report, unique_edges, log_type, model_type="openai", ollama_url=None):                  

    messages = [

{
"role": "system",
"content": '''You are an expert in cybersecurity and system provenance graphs. You will be given the following as input:\n 
(1) a  '''+log_type+''' LOGS,\n
(2) a paragraph describing the activities happening in the '''+log_type+''' LOGS,\n 
(3) pairs of entity IDs as (ID-1, ID-2),\n 
(5) pairs of IP addresses as (IP:port-1, IP:port-2),\n 
(6) pairs of the entity ID and the IP as (IP:port-1, ID-1) or (ID-1, IP:port-1).\n\n
Your task is to read each of the logs, understand its format, and find the action/event/access types that occur between the elements in the pairs given to you as input along with their directions, to create a provenance graph. Take a deep breath and let's do step by step.\n\n
(Step 1) Analyze all the logs and understand all the operations that are happening.\n
(Step 2) Read the [LOGS] to fill all the ""A: [...]"" arrays with the exact name of the file/system/network level action/event/access types between the elements in its corresponding pairs.    \n**NO REPHRASING**, use the exact wording of the types found in the [LOGS] and DO NOT MISS THE RELEVANT TYPES.    \n**IF YOU CANNOT FIND** a '''+log_type+''' file/system/network level action/event/access type relevant for provenance graphs, to reduce ambiguity write all of them as "NO LABEL", and do not rephrase or make up action/event/access types.\n
(Step 3) Also, read the [PARAGRAPH] and specify the direction of the action/event/access types that is occuring between the elements in each of the pairs. If it is from left to right then write {D=->} and if it is from right to left then write {D=<-}.\n
(Step 4) Also, read the [LOGS] and write the timestamp of each edge within (timestamp=...). If no timestamp is written in the [LOGS], just (timestamp=...).\n
(Step 5) Only display the output in the following format and nothing else.\n\n\n
Output Format:\n\n[ENTITY ID PAIRS and ACTIONS]\n\n 
(ID-1, ID-2)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-1, ID-3)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-2, ID-5)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-6, ID-3)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-3, ID-4)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-4, ID-5)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-6, IP:port-5) A: [...] {D=-> or D=<-} (timestamp=...)\n
(IP:port-1, IP:port-2)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(IP:port-1, IP:port-3)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(IP:port-1, IP:port-4)  A: [...] {D=-> or D=<-} (timestamp=...)\n
(ID-7, IP:port-1) A: [...] {D=-> or D=<-} (timestamp=...)\n
...\n\n
Here are some Examples:\n\n'''+unique_edges},


{"role": "user","content": "[LOGS] \n\n" +log_report+"\n\n"+desc+"\n\n[RELATED ENTITY ID PAIRS]\n\n"+pairs},


    ]

    ans = call_llm(
        model_type=model_type,
        model_name=llm_model,
        messages=messages,
        api_key=key,
        ollama_url=ollama_url,
        temperature=0,
        max_tokens=4096,
        seed=400
    )

    return ans


def llm_check_id_pairs(llm_model, key, log_report, log_type, model_type="openai", ollama_url=None):
    messages = [
        {
            "role": "system",
            "content": '''We will be constructing a provenance graph. Are there any unique numeric identifiers in this event (like PIDs or file IDs, etc) that can be directly used to connect nodes in the provenance graph? Think step by step before answering and just answer in yes or no:'''
        },
        {
            "role": "user",
            "content": f"[{log_type} LOG]\n\n{log_report}"
        }
    ]
    
    ans = call_llm(
        model_type=model_type,
        model_name=llm_model,
        messages=messages,
        api_key=key,
        ollama_url=ollama_url,
        temperature=0,
        max_tokens=100,
        seed=400
    )
    
    return ans
