# from openai import OpenAI
# from anthropic import Anthropic
# from ollama import Client
import os, utils, asyncio, json, requests
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
# OPENAI_MODEL = os.getenv("OPENAI_MODEL")
# ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# openai_client = OpenAI(api_key=OPENAI_API_KEY)
# anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
# ollama_client = Client(
#     host="https://ollama.com",
#     headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
# )

async def call_openai(prompt: str, text_format: Dict, model: str = OPENAI_MODEL, max_tokens: int = 800) -> Dict[str, Any]:
    def _call():
        return openai_client.responses.parse(
            model=model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=max_tokens,
            temperature=0.0,
            text_format=text_format
        )

    try:
        resp = await asyncio.get_event_loop().run_in_executor(None, _call)
        text = resp.output_parsed
        return {"model": f"openai:{model}", "text": text, "raw": resp}
    except Exception as e:
        return {
            "model": "error",
            "summary": "",
            "article": {
                "bias": "Unknown",
                "credibility": "Unknown",
                "notes": f"OpenAI model call failed: {r}"
            },
            "publication": {
                "source_of_funding": None,
                "location": None
            },
            "raw": None
        }

async def call_anthropic(prompt: str, model: str = ANTHROPIC_MODEL, max_tokens: int = 800) -> dict:
    def _call():
        return anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

    try:
        resp = await asyncio.get_event_loop().run_in_executor(None, _call)
        text = resp.content[0].text if resp.content else ""
        return {"model": f"anthropic:{model}", "text": text, "raw": resp}
    except Exception as e:
        return {
            "model": "error",
            "summary": "",
            "article": {
                "bias": "Unknown",
                "credibility": "Unknown",
                "notes": f"Anthropic model call failed: {r}"
            },
            "publication": {
                "source_of_funding": None,
                "location": None
            },
            "raw": None
        }

async def call_ollama(prompt: str, format: Dict, model: str = OLLAMA_MODEL) -> dict:
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            format=format
        )

        text = resp['message']['content']

        return {"model": f"ollama:{model}", "text": text, "raw": resp}
    
    except Exception as e:
        return {
            "model": "error",
            "summary": "",
            "article": {
                "bias": "Unknown",
                "credibility": "Unknown",
                "notes": f"Ollama model call failed."
            },
            "publication": {
                "source_of_funding": None,
                "location": None
            },
            "raw": None
        }

async def call_openrouter(prompt: str, format:dict, model: list[str]) -> dict:
    try:
        url = "https://openrouter.ai/api/v1/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "models": model,
            "prompt": prompt,
            'provider': {
                'require_parameters': True,
            },
            "response_format": {
                "type": "json_object",
            }
        }
        # print(payload)
        response = requests.post(url, headers=headers, json=payload)
        # print(response.json())

        text = response.json()["choices"][0]["text"]
        
        return {"model": response.json()["model"], "text": text, "raw": response.json()}
    except Exception as e:
        return ("Error:", e)