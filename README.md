# 🔍 Hallucination Detector Layer

A developer tool that sits in front of any LLM answer, extracts its factual claims, checks each one against web sources, and **color-codes the answer by confidence**.

![Screenshot](screenshot.png)

## How it works

```
LLM answer → extract atomic claims → web search per claim → re-rank evidence → verify → color-coded answer
```

1. **Claim extraction:** an LLM splits the answer into atomic, verifiable claims.
2. **Evidence retrieval:** each claim is searched on the web (DuckDuckGo).
3. **Evidence re-ranking (optional):** snippets are chunked and ranked by embedding similarity with Chroma.
4. **Verification:** the LLM judges each claim using only the retrieved evidence and returns a verdict, a confidence score and a short reason.
5. **Color-coded output:** the original answer is highlighted, and every claim gets a card with its source link.

| Color | Meaning |
|-------|---------|
| 🟢 Green | Supported by sources with high confidence |
| 🟡 Amber | Weak support or unverifiable |
| 🔴 Red | Contradicted by sources, a likely hallucination |

An overall **trust score** summarizes the answer, and the full report can be downloaded as JSON.

## Features

- Gemini-style dark UI built with Streamlit and custom CSS
- Two modes: **verify a pasted answer**, or **ask an LLM and verify its own answer**
- Switchable providers: **Anthropic (Claude)**, **OpenAI**, or **Ollama** (fully local)
- Hover over any highlight to see the verdict and reason
- Source links for each claim
- Downloadable JSON report

## Tech stack

Streamlit · LangChain · Chroma · DuckDuckGo Search · Anthropic / OpenAI / Ollama

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/YOUR-USERNAME/hallucination-detector.git
cd hallucination-detector

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Configure a provider

**Cloud (Anthropic or OpenAI):** create a `.env` file in the project folder:

```
ANTHROPIC_API_KEY=your-key-here
# or
OPENAI_API_KEY=your-key-here
```

You can also paste the key into the sidebar at runtime.

**Local (Ollama):** no key needed.

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### Run

```bash
streamlit run app.py
```

## Try it

Paste this into **Verify pasted answer** mode:

> The Eiffel Tower is located in Paris and was completed in 1889. The Great Wall of China is visible from the Moon with the naked eye. Python was created by Guido van Rossum and first released in 1995.

The Eiffel Tower claim should come back green, while the Great Wall and Python 1995 claims should be flagged red.

## Project structure

```
├── app.py             # Streamlit app and verification pipeline
├── style.css          # Gemini-style theme
├── requirements.txt
└── .env               # API keys (not committed)
```

## Limitations

- Verdicts depend on web snippets, not full pages, so some true claims may show as *unverifiable*.
- Small local models can be inconsistent at structured JSON output. Larger or hosted models are more reliable.
- This tool reduces the risk of hallucinations but does not guarantee correctness. Always review important claims manually.

## Roadmap

- Fetch and parse full web pages instead of snippets
- Optional Tavily / SerpAPI search backends
- Batch mode for evaluating many answers
- Browser extension for checking chatbot answers in place

## License

MIT
