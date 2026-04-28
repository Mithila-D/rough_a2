"""
LLM backend — Azure OpenAI primary, with optional Ollama/Gemini fallbacks.

This project uses Azure OpenAI by default. Put your credentials in a .env file
inside the `.env` folder (or root .env). Supported environment variables:

    AZURE_OPENAI_KEY            - Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT       - Azure OpenAI endpoint (https://<your-resource>.openai.azure.com/)
    AZURE_OPENAI_API_VERSION    - API version (e.g., 2023-05-15) (optional)
    AZURE_OPENAI_DEPLOYMENT     - Deployment / model name (e.g., gpt-4o-mini)

Fallbacks: set `LLM_BACKEND=ollama` or `LLM_BACKEND=gemini` if needed.
"""

import os
import time
import re
from dotenv import load_dotenv

load_dotenv()
# Also attempt to load .env/.env if present (user keeps creds in .env folder)
from pathlib import Path
env_path = Path(__file__).resolve().parents[1] / ".env" / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))

# Enforce Azure-only deployment for LLMs
env_backend = os.getenv("LLM_BACKEND")
if env_backend and env_backend.lower() != "azure":
    raise RuntimeError("Only Azure OpenAI backend is supported in this deployment. Remove LLM_BACKEND or set it to 'azure'.")

BACKEND = "azure"


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


# ── Azure OpenAI backend ────────────────────────────────────────────────────
def _build_azure():
    try:
        import openai
    except Exception as e:
        raise RuntimeError("Azure OpenAI support requires the `openai` package") from e

    api_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", None)
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_DEPLOYMENT")

    if not api_key or not endpoint or not deployment:
        raise RuntimeError("Azure OpenAI backend selected but AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, or AZURE_OPENAI_DEPLOYMENT is missing in environment")

    # Configure OpenAI client for Azure
    openai.api_type = "azure"
    openai.api_key = api_key
    openai.api_base = endpoint.rstrip("/")
    if api_version:
        openai.api_version = api_version

    class AzureWrapper:
        def __init__(self, deployment):
            self.deployment = deployment

        def invoke(self, prompt: str):
            # Use ChatCompletions for chat-capable deployments
            resp = openai.ChatCompletion.create(
                engine=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0,
            )
            content = resp.choices[0].message.content

            class R:
                pass

            r = R()
            r.content = content
            return r

    base = AzureWrapper(deployment)
    print(f"[LLM] Using Azure OpenAI deployment={deployment} endpoint={endpoint}")
    return _RetryLLM(base, max_retries=4, base_wait=10)


# ── Build ─────────────────────────────────────────────────────────────────────
try:
    llm = _build_azure()
except Exception as e:
    raise RuntimeError(f"Failed to initialise Azure OpenAI backend: {e}") from e
