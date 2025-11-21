import requests
import json

def call_ollama_local(prompt: str, model: str, url: str):
    """
    Accepts plain text prompt and returns merged streaming output.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        full_response = ""

        for line in response.iter_lines():
            if not line:
                continue

            try:
                data = json.loads(line.decode("utf-8"))
            except:
                continue

            if "response" in data:
                full_response += data["response"]

            if data.get("done", False):
                break

        return full_response.strip()

    except Exception as e:
        return f"[ERROR] {str(e)}"
