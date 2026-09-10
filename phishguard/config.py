import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
TRACE_DIR = ROOT / "traces"
ESCALATION_LOG = ROOT / "escalations.jsonl"
SEED_PATTERNS = DATA_DIR / "seed_patterns.jsonl"
MCP_SERVER = ROOT / "mcp_server.py"

CHROMA_COLLECTION = "phishing_patterns"
RETRIEVAL_K = 5

PROVIDER = os.getenv("PHISHGUARD_PROVIDER", "echo").strip().lower()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2").strip()
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

TOOL_TIMEOUT = 8.0

# Risk formula weights. Sum to 1.0; see README.
W_RETRIEVAL = 0.30
W_TOOLS = 0.35
W_CONTENT = 0.25
W_INJECTION = 0.10

INJECTION_RISK_FLOOR = 0.6
CONFIDENCE_ESCALATION_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.7
SUSPICIOUS_THRESHOLD = 0.4


def ensure_dirs() -> None:
    TRACE_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
