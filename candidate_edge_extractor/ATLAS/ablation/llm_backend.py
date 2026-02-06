import requests
from typing import List, Dict, Optional, Union


class MockResponse:
    class Choice:
        class Message:
            def __init__(self, content: str):
                self.content = content

        def __init__(self, content: str):
            self.message = self.Message(content)

    def __init__(self, content: str):
        self.choices = [self.Choice(content)]


def convert_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    prompt_parts = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role == 'system':
            prompt_parts.append(f"System: {content}")
        elif role == 'user':
            prompt_parts.append(f"User: {content}")
        elif role == 'assistant':
            prompt_parts.append(f"Assistant: {content}")
        else:
            prompt_parts.append(content)
    return "\n\n".join(prompt_parts)


def call_llm(
    model_type: str,
    model_name: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    ollama_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    seed: Optional[int] = None
):
    if model_type == "openai":
        return _call_openai(model_name, messages, api_key, temperature, max_tokens, seed)
    elif model_type == "ollama":
        if not ollama_url:
            raise ValueError("ollama_url is required when model_type is 'ollama'")
        return _call_ollama(model_name, messages, ollama_url, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Must be 'openai' or 'ollama'")


def _call_openai(
    model_name: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
    seed: Optional[int]
):
    from openai import OpenAI
    if not api_key:
        raise ValueError("OpenAI API key is required for OpenAI models")
    client = OpenAI(api_key=api_key)
    kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        kwargs["seed"] = seed
    response = client.chat.completions.create(**kwargs)
    return response


def _call_ollama(
    model_name: str,
    messages: List[Dict[str, str]],
    ollama_url: str,
    temperature: float,
    max_tokens: int
) -> MockResponse:
    prompt = convert_messages_to_prompt(messages)
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    try:
        response = requests.post(f"{ollama_url}/api/generate", json=data, timeout=300)
        response.raise_for_status()
        result = response.json()
        content = result.get("response", "")
        return MockResponse(content)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Cannot connect to Ollama at {ollama_url}. Make sure Ollama is running.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error calling Ollama API: {e}")
