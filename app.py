"""Hallucination Detector Layer
Extracts factual claims from an LLM answer, checks each against web sources,
and colour-codes the answer by confidence. Runs fully local via Ollama.
"""
import html
import json
import re
import uuid
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Hallucination Detector", page_icon="🔍", layout="centered")
st.markdown(f"<style>{Path(__file__).with_name('style.css').read_text()}</style>", unsafe_allow_html=True)

# ───────────────────────── prompts ─────────────────────────
EXTRACT_PROMPT = """Extract the verifiable factual claims from the text below.
Rules: each claim must be atomic and self-contained (no pronouns), and skip opinions.
For each claim also return "quote": the EXACT substring of the text it came from.
Return ONLY JSON: {{"claims": [{{"claim": "...", "quote": "..."}}]}}

TEXT:
{text}"""

VERIFY_PROMPT = """You are a strict fact-checker. Judge the CLAIM using ONLY the EVIDENCE.
- "supported": evidence clearly confirms it
- "contradicted": evidence clearly conflicts with it
- "unverifiable": evidence is missing, off-topic or inconclusive
Return ONLY JSON: {{"verdict": "supported|contradicted|unverifiable",
"confidence": 0.0-1.0, "reason": "one short sentence", "source": <evidence number or null>}}

CLAIM: {claim}

EVIDENCE:
{evidence}"""


# ───────────────────────── helpers ─────────────────────────
def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def extract_claims(llm, text: str) -> list[dict]:
    data = parse_json(llm.invoke(EXTRACT_PROMPT.format(text=text)).content)
    claims = [c for c in data.get("claims", []) if isinstance(c, dict) and c.get("claim")]
    return claims[:15]


def gather_evidence(claim: str, k: int, embed_model: str | None) -> list[dict]:
    """Web search, then (optionally) re-rank snippets with Chroma embeddings."""
    try:
        results = DuckDuckGoSearchAPIWrapper().results(claim, max_results=k + 3)
    except Exception:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    docs = []
    for r in results:
        body = f"{r.get('title', '')}. {r.get('snippet', '')}"
        for chunk in splitter.split_text(body):
            docs.append(Document(page_content=chunk, metadata={"title": r.get("title", ""), "link": r.get("link", "")}))
    if not docs:
        return []
    if embed_model:
        try:
            store = Chroma.from_documents(docs, OllamaEmbeddings(model=embed_model),
                                          collection_name=f"c_{uuid.uuid4().hex}")
            docs = store.similarity_search(claim, k=k)
            store.delete_collection()
        except Exception:
            docs = docs[:k]
    else:
        docs = docs[:k]
    return [{"text": d.page_content, **d.metadata} for d in docs]


def verify_claim(llm, claim: str, evidence: list[dict]) -> dict:
    if not evidence:
        return {"verdict": "unverifiable", "confidence": 0.0, "reason": "No web evidence found.", "source": None}
    ev = "\n".join(f"[{i + 1}] {e['text']}" for i, e in enumerate(evidence))
    out = parse_json(llm.invoke(VERIFY_PROMPT.format(claim=claim, evidence=ev)).content)
    verdict = out.get("verdict", "unverifiable")
    if verdict not in {"supported", "contradicted", "unverifiable"}:
        verdict = "unverifiable"
    try:
        conf = max(0.0, min(1.0, float(out.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return {"verdict": verdict, "confidence": conf, "reason": out.get("reason", ""), "source": out.get("source")}


def classify(r: dict) -> tuple[str, str, float]:
    """-> (css level, label, trust score 0-1)"""
    v, c = r["verdict"], r["confidence"]
    if v == "supported":
        return ("ok", f"Supported · {c:.0%}", c) if c >= 0.7 else ("warn", f"Weak support · {c:.0%}", c * 0.7)
    if v == "contradicted":
        return "bad", f"Contradicted · {c:.0%}", 0.0
    return "warn", "Unverifiable", 0.4


def highlight(answer: str, claims: list[dict]) -> str:
    low, spans = answer.lower(), []
    for c in claims:
        q = (c.get("quote") or "").strip()
        s = low.find(q.lower()) if q else -1
        if s >= 0:
            spans.append((s, s + len(q), c))
    spans.sort(key=lambda x: x[0])
    out, pos = [], 0
    for s, e, c in spans:
        if s < pos:
            continue
        out.append(html.escape(answer[pos:s]))
        tip = html.escape(f"{c['label']} — {c['reason']}", quote=True)
        out.append(f'<mark class="hl {c["level"]}" title="{tip}">{html.escape(answer[s:e])}</mark>')
        pos = e
    out.append(html.escape(answer[pos:]))
    return "".join(out).replace("\n", "<br>")


# ───────────────────────── sidebar ─────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    mode = st.radio("Mode", ["Verify pasted answer", "Ask local LLM, then verify"])
    chat_model = st.text_input("Ollama chat model", "llama3.1")
    use_embed = st.toggle("Re-rank evidence with embeddings", value=True)
    embed_model = st.text_input("Embedding model", "nomic-embed-text") if use_embed else None
    k = st.slider("Evidence snippets per claim", 2, 6, 3)
    st.caption("Requires Ollama running locally.\n\n`ollama pull llama3.1`\n\n`ollama pull nomic-embed-text`")

# ───────────────────────── main UI ─────────────────────────
st.markdown(
    '<div class="greeting"><div class="hello">Hello, Developer</div>'
    '<div class="sub">Which answer should we fact-check?</div></div>',
    unsafe_allow_html=True,
)

placeholder = ("Paste any LLM answer here…" if mode.startswith("Verify")
               else "Ask a question — I'll answer it, then check my own answer…")
with st.form("prompt", clear_on_submit=False):
    user_text = st.text_area("input", height=140, placeholder=placeholder)
    go = st.form_submit_button("Check ✨")

if go:
    if not user_text.strip():
        st.warning("Enter some text first.")
        st.stop()

    llm_json = ChatOllama(model=chat_model, temperature=0, format="json")
    answer, question = user_text.strip(), None

    try:
        if mode.startswith("Ask"):
            question = answer
            with st.spinner("Generating answer…"):
                answer = ChatOllama(model=chat_model, temperature=0.7).invoke(question).content

        with st.status("Running hallucination check…", expanded=True) as status:
            st.write("Extracting factual claims…")
            claims = extract_claims(llm_json, answer)
            if not claims:
                status.update(label="No verifiable claims found", state="complete")
                st.info("The text contains no checkable factual claims.")
                st.stop()
            bar = st.progress(0.0)
            for i, c in enumerate(claims):
                st.write(f"Checking claim {i + 1}/{len(claims)}: {c['claim']}")
                ev = gather_evidence(c["claim"], k, embed_model)
                c.update(verify_claim(llm_json, c["claim"], ev))
                c["evidence"] = ev
                c["level"], c["label"], c["trust"] = classify(c)
                bar.progress((i + 1) / len(claims))
            status.update(label="Verification complete", state="complete", expanded=False)
    except Exception as e:
        st.error(f"Could not reach Ollama or the model failed: {e}")
        st.stop()

    # ── results ──
    if question:
        st.markdown(f'<div class="msg-user">{html.escape(question)}</div>', unsafe_allow_html=True)

    n = len(claims)
    ok = sum(c["level"] == "ok" for c in claims)
    warn = sum(c["level"] == "warn" for c in claims)
    bad = sum(c["level"] == "bad" for c in claims)
    trust = sum(c["trust"] for c in claims) / n

    st.markdown(
        f'<div class="score-row">'
        f'<div class="score-card"><div class="num">{trust:.0%}</div><div class="lbl">Overall trust score</div></div>'
        f'<div class="score-card"><div class="num" style="color:#81c995">{ok}</div><div class="lbl">Supported</div></div>'
        f'<div class="score-card"><div class="num" style="color:#fdd663">{warn}</div><div class="lbl">Uncertain</div></div>'
        f'<div class="score-card"><div class="num" style="color:#f28b82">{bad}</div><div class="lbl">Contradicted</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="legend">'
        '<span class="chip"><span class="dot ok"></span>High confidence</span>'
        '<span class="chip"><span class="dot warn"></span>Uncertain / unverifiable</span>'
        '<span class="chip"><span class="dot bad"></span>Likely hallucination</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="msg-ai"><div class="spark">✦</div><div class="body">{highlight(answer, claims)}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Claim breakdown")
    for c in claims:
        src = ""
        idx = c.get("source")
        if isinstance(idx, int) and 1 <= idx <= len(c["evidence"]):
            e = c["evidence"][idx - 1]
            src = f'<div class="src"><a href="{html.escape(e["link"], quote=True)}" target="_blank">🔗 {html.escape(e["title"] or e["link"])}</a></div>'
        elif c["evidence"]:
            links = "".join(
                f'<div class="src"><a href="{html.escape(e["link"], quote=True)}" target="_blank">🔗 {html.escape(e["title"] or e["link"])}</a></div>'
                for e in c["evidence"][:2])
            src = links
        st.markdown(
            f'<div class="claim {c["level"]}"><div class="top"><div class="text">{html.escape(c["claim"])}</div>'
            f'<span class="badge {c["level"]}">{html.escape(c["label"])}</span></div>'
            f'<div class="reason">{html.escape(str(c["reason"]))}</div>{src}</div>',
            unsafe_allow_html=True,
        )

    st.download_button(
        "Download report (JSON)",
        json.dumps({"answer": answer, "trust_score": trust, "claims": claims}, indent=2, default=str),
        file_name="hallucination_report.json",
        mime="application/json",
    )