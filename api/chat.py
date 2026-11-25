import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PRIMARY_DATASET = ROOT_DIR / "combined_flacourtia_dataset.jsonl"
FALLBACK_DATASET = ROOT_DIR / "flacourtia_montana_dataset.jsonl"

GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b"),
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
TOP_K = 8

SYSTEM_PROMPT = (
    "You are a helpful community assistant for the 'Conservation of Flacourtia montana' "
    "student project at Siddaganga Institute of Technology (SIT), Tumakuru. "
    "LANGUAGE RULE: Reply ONLY in the requested language (English, Hindi, or Kannada). "
    "FORMAT RULE: Write in clean, natural short paragraphs and simple bullet points (using •). "
    "NEVER use Markdown tables (| ... |), HTML tags (<br>), # headings, or ** asterisks. "
    "ACCURACY RULE: Answer using ONLY the provided context passages. When reading botanical keys "
    "comparing Flacourtia montana with F. indica or F. latifolia, match each trait carefully to the exact species. "
    "If the context does not contain the answer, say so honestly in the requested language. "
    "CREDITS RULE: Mention the source title or organization briefly in your answer."
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_records() -> list[dict]:
    path = PRIMARY_DATASET if PRIMARY_DATASET.exists() else FALLBACK_DATASET
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


RECORDS = load_records()
DOCS = [tokenize(f"{r.get('topic', '')} {r.get('title', '')} {r.get('text', '')}") for r in RECORDS]
DOC_FREQS: Counter[str] = Counter()
for _doc in DOCS:
    for _term in set(_doc):
        DOC_FREQS[_term] += 1
AVG_DL = sum(len(d) for d in DOCS) / max(len(DOCS), 1)
N_DOCS = len(DOCS)


def bm25_search(query: str, top_k: int = TOP_K) -> list[dict]:
    if not RECORDS:
        return []
    q_terms = tokenize(query)
    scores: list[tuple[float, int]] = []
    for idx, doc in enumerate(DOCS):
        if not doc:
            continue
        tf = Counter(doc)
        doc_len = len(doc)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            df = DOC_FREQS[term]
            idf = math.log((N_DOCS - df + 0.5) / (df + 0.5) + 1.0)
            freq = tf[term]
            denom = freq + 1.5 * (1 - 0.75 + 0.75 * (doc_len / AVG_DL))
            score += idf * (freq * 2.5) / denom
        if score > 0:
            rec_id = str(RECORDS[idx].get("id", ""))
            if RECORDS[idx].get("origin") == "curated" or rec_id.startswith(("proj_", "bot_")):
                score *= 3.5
            scores.append((score, idx))
    scores.sort(reverse=True)
    return [RECORDS[idx] for _, idx in scores[:top_k]]


def post_json(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_groq(messages: list[dict], api_key: str) -> str:
    last_err = None
    for model in GROQ_MODELS:
        try:
            data = post_json(
                "https://api.groq.com/openai/v1/chat/completions",
                {"model": model, "messages": messages, "temperature": 0.2},
                {"Authorization": f"Bearer {api_key}"},
            )
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Groq error: {last_err}")


def call_gemini(prompt: str, system_prompt: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    data = post_json(
        url,
        {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        },
        {},
    )
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def translate_to_english_keywords(question: str, groq_key: str, gemini_key: str) -> str:
    prompt = (
        "Fix any spelling typos, translate to English if needed, and output 5-10 concise "
        "search keywords/synonyms to match a Flacourtia montana conservation project dataset "
        "(e.g. for 'participents' include 'team members students names'). Return only keywords:\n\n"
        + question
    )
    if groq_key:
        return call_groq([{"role": "user", "content": prompt}], groq_key)
    if gemini_key:
        return call_gemini(prompt, "Translate concisely.", gemini_key)
    return question


def build_credits(passages: list[dict]) -> list[dict]:
    seen = set()
    credits = []
    for p in passages:
        meta = p.get("metadata", {})
        title = (
            p.get("source_title")
            or meta.get("source_title")
            or p.get("title")
            or p.get("topic")
            or "Botanical / Project Source"
        )
        url = p.get("source_url") or meta.get("source_url") or ""
        if (title, url) not in seen:
            seen.add((title, url))
            credits.append({"title": title, "url": url})
    return credits


def answer_question(question: str, language: str) -> dict:
    greetings = {"hi", "hello", "hey", "namaste", "namaskara", "नमस्ते", "ನಮಸ್ಕಾರ", "help"}
    if question.strip().lower().rstrip("!.,?") in greetings:
        welcome = {
            "English": "Namaskara! Ask me anything about Flacourtia montana (Mountain Sweet Thorn), its medicinal uses, or our 10-week conservation project at SIT Tumakuru.",
            "Hindi": "नमस्कार! आप मुझसे Flacourtia montana, इसके उपयोगों या हमारे 10-सप्ताह के संरक्षण प्रोजेक्ट के बारे में कुछ भी पूछ सकते हैं।",
            "Kannada": "ನಮಸ್ಕಾರ! Flacourtia montana ಮರ, ಅದರ ಉಪಯೋಗಗಳು ಅಥವಾ ನಮ್ಮ 10 ವಾರಗಳ ಸಂರಕ್ಷಣಾ ಯೋಜನೆಯ ಬಗ್ಗೆ ಏನನ್ನಾದರೂ ಕೇಳಿ.",
        }
        return {"answer": welcome.get(language, welcome["English"]), "credits": []}

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    search_query = question
    if groq_key or gemini_key:
        try:
            search_query = translate_to_english_keywords(question, groq_key, gemini_key)
        except Exception:
            search_query = question

    matches = bm25_search(search_query, top_k=TOP_K)

    q_toks = set(tokenize(search_query)) | set(tokenize(question))
    pinned_ids = set()
    if q_toks & {"project", "team", "member", "members", "student", "students", "participant", "participants", "participents", "who", "name", "names", "guide", "college", "sit", "where"}:
        pinned_ids.update({"proj_001", "proj_004"})
    if q_toks & {"medicinal", "medicine", "ayurveda", "use", "uses", "fruit", "fruits", "food", "eat", "edible", "nutrition"}:
        pinned_ids.add("bot_005")
    if q_toks & {"identify", "identification", "leaf", "leaves", "thorn", "thorns", "bark", "flower", "flowers", "height", "tree"}:
        pinned_ids.update({"bot_002", "bot_003"})
    if pinned_ids:
        pinned_recs = [r for r in RECORDS if r.get("id") in pinned_ids and r not in matches]
        matches = (pinned_recs + matches)[:TOP_K]

    if not matches:
        no_match = {
            "English": "I couldn't find an exact match in our dataset. Try asking about Flacourtia montana's uses, identification, habitat, or our 10-week SIT Tumakuru project!",
            "Hindi": "मुझे डेटासेट में इससे जुड़ी सटीक जानकारी नहीं मिली। कृपया Flacourtia montana के उपयोग, पहचान या हमारे 10-सप्ताह के प्रोजेक्ट के बारे में पूछें।",
            "Kannada": "ನಮ್ಮ ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ಇದಕ್ಕೆ ನಿಖರವಾದ ಮಾಹಿತಿ ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು Flacourtia montana ಮರದ ಗುಣಲಕ್ಷಣಗಳು, ಉಪಯೋಗಗಳು ಅಥವಾ ನಮ್ಮ 10 ವಾರಗಳ ಯೋಜನೆಯ ಬಗ್ಗೆ ಕೇಳಿ.",
        }
        return {"answer": no_match.get(language, no_match["English"]), "credits": []}

    credits = build_credits(matches)
    context_blocks = [
        f"[{i}] {p.get('topic') or p.get('title') or 'N/A'} ({p.get('source_title') or p.get('title') or ''}):\n{p.get('text', '')}"
        for i, p in enumerate(matches, start=1)
    ]
    user_prompt = (
        "Context Passages:\n\n"
        + "\n\n---\n\n".join(context_blocks)
        + f"\n\nTarget Reply Language: {language}\nUser Question: {question}"
    )

    errors = []
    if groq_key:
        try:
            ans = call_groq(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                groq_key,
            )
            ans = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", ans)
            return {"answer": ans, "credits": credits}
        except Exception as exc:
            errors.append(str(exc))

    if gemini_key:
        try:
            ans = call_gemini(user_prompt, SYSTEM_PROMPT, gemini_key)
            ans = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", ans)
            return {"answer": ans, "credits": credits}
        except Exception as exc:
            errors.append(f"Gemini error: {exc}")

    snippets = "\n\n".join(f"• {m.get('text', '')[:320]}..." for m in matches[:3])
    reason = " | ".join(errors) if errors else "Add GROQ_API_KEY or GEMINI_API_KEY in Vercel Environment Variables"
    return {"answer": f"(Retrieval mode — {reason})\n\n{snippets}", "credits": credits}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        question = (body.get("question") or "").strip()
        language = (body.get("language") or "English").strip()

        if not question:
            self._send_json(400, {"error": "Question is required."})
            return

        result = answer_question(question, language)
        self._send_json(200, result)

    def _send_json(self, status: int, payload: dict):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
