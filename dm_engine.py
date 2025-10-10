import os, requests
from dotenv import load_dotenv
import time 
import re
load_dotenv()

PROVIDER = os.getenv('PROVIDER', 'groq').lower()
MODEL_NAME = os.getenv('MODEL_NAME', 'llama-3.1-8b-instant')

class DMEngine:
    def __init__(self, use_semantic_rag: bool = False):
       
        self.use_semantic_rag = use_semantic_rag
        self.provider = PROVIDER
        self.model = MODEL_NAME
        self.session = requests.Session()
        self._validate_env()

    def _validate_env(self):
        if self.provider == 'groq':
            if not os.getenv('GROQ_API_KEY'):
                raise RuntimeError('GROQ_API_KEY missing. Put it in .env or Streamlit secrets.')
        elif self.provider == 'openai':
            if not os.getenv('OPENAI_API_KEY'):
                raise RuntimeError('OPENAI_API_KEY missing.')
        else:
            raise RuntimeError("PROVIDER must be 'groq' or 'openai'")

  

    def chat(self, messages):
        if self.provider == 'groq':
            url = 'https://api.groq.com/openai/v1/chat/completions'
            headers = {
                'Authorization': f"Bearer {os.getenv('GROQ_API_KEY')}",
                'Content-Type': 'application/json'
            }
        else:
            url = 'https://api.openai.com/v1/chat/completions'
            headers = {
                'Authorization': f"Bearer {os.getenv('OPENAI_API_KEY')}",
                'Content-Type': 'application/json'
            }

        max_tokens = int(os.getenv("MAX_TOKENS", "350"))  
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': 0.9,
            'max_tokens': max_tokens,
        }

        
        for attempt in range(3):
            r = self.session.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code == 429:
                
                retry_after = r.headers.get("Retry-After")
                wait_s = 10
                if retry_after:
                    try:
                        wait_s = int(retry_after)
                    except Exception:
                        pass
                else:
                    m = re.search(r"try again in ([0-9.]+)s", r.text)
                    if m:
                        try:
                            wait_s = float(m.group(1))
                        except Exception:
                            pass
                time.sleep(min(max(wait_s, 5), 20))  
                continue

            if r.status_code >= 400:
                raise RuntimeError(f"{r.status_code} {r.text}")

            data = r.json()
            return data['choices'][0]['message']['content']

        
        raise RuntimeError("Rate limit: too many tokens/min. Please wait a moment and try again.")



