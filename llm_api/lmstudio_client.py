import requests

class LlmStudioClient:
    def __init__(self, base_url="http://localhost:1234/v1/chat/completions"):
        self.url = base_url

    def ask(self, prompt, model="google/gemma-3-12b"):
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(self.url, headers=headers, json=data)

        return response.json()["choices"][0]["message"]["content"]