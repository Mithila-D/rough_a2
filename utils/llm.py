"""
LLM backend — supports Ollama (local Llama 3) with Gemini as fallback.

Priority:
  1. Ollama (local, free, no quota)  — set LLM_BACKEND=ollama  (default)
  2. Gemini                          — set LLM_BACKEND=gemini + GEMINI_API_KEY

Ollama setup (one-time, on your machine):
  1. Install  : https://ollama.com/download
  2. Pull model: ollama pull llama3
  3. Start     : ollama serve          (it auto-starts on most installs)
  The default endpoint is http://localhost:11434 — no key needed.
"""

import os
import time
import re
from dotenv import load_dotenv

load_dotenv()

BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()


# ── Shared retry wrapper ──────────────────────────────────────────────────────
class _RetryLLM:
    def __init__(self, base_llm, max_retries=3, base_wait=10):
        self._llm        = base_llm
        self._max_retries = max_retries
        self._base_wait  = base_wait

    def invoke(self, prompt):
        for attempt in range(self._max_retries + 1):
            try:
                return self._llm.invoke(prompt)
            except Exception as e:
                msg = str(e)
                is_quota = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                if is_quota and attempt < self._max_retries:
                    wait = self._base_wait * (2 ** attempt)
                    m = re.search(r"retryDelay.*?(\d+)s", msg)
                    if m:
                        wait = max(int(m.group(1)) + 2, wait)
                    print(f"[LLM] 429 quota — waiting {wait}s (retry {attempt+1}/{self._max_retries})")
                    time.sleep(wait)
                    continue
                raise


# ── Ollama backend ────────────────────────────────────────────────────────────
def _build_ollama():
    from langchain_ollama import ChatOllama
    model = os.getenv("OLLAMA_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"[LLM] Using Ollama  model={model}  url={base_url}")
    base = ChatOllama(model=model, base_url=base_url, temperature=0)
    return _RetryLLM(base, max_retries=1, base_wait=2)


# ── Gemini backend ────────────────────────────────────────────────────────────
def _build_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_BACKEND=gemini but no GEMINI_API_KEY found in environment")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    print(f"[LLM] Using Gemini  model={model}")
    base = ChatGoogleGenerativeAI(model=model, temperature=0,
                                   google_api_key=api_key, max_output_tokens=1024)
    return _RetryLLM(base, max_retries=4, base_wait=15)


# ── Build ─────────────────────────────────────────────────────────────────────
if BACKEND == "gemini":
    llm = _build_gemini()
else:
    try:
        llm = _build_ollama()
    except Exception as e:
        print(f"[LLM] Ollama unavailable ({e}), falling back to Gemini")
        llm = _build_gemini()
