import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def ask_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return "⚠️ AI Unavailable: Unable to connect to local Ollama instance. Please ensure Ollama is running or configure a remote API endpoint."
