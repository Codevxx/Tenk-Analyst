"""
10-K Analyst: a Streamlit RAG chatbot over the fiscal 2025 Form 10-K filings
of Alphabet, Amazon and Microsoft.

Started from the course starter chat_with_pdf_gemini_with_history.py in
JHU-CDHAI/Chatbot-AIEB (BU.520.710 AI Essentials for Business, Johns Hopkins
Carey Business School, Summer 2026). Run with:

    streamlit run financial_rag_chatbot.py

Differences from the starter:

 1. Retrieval query. The starter passed the whole prompt template (persona,
    chat history and question) to the retriever. Here FAISS is searched with
    the user's question only; the system prompt and history go to the model.
 2. Company-aware retrieval. Every chunk is tagged with its company, detected
    from the PDF filename. A question that names one company searches only
    that company's chunks; any other question searches each company
    separately and merges the results, so one filing cannot crowd the others
    out of the context window.
 3. Chunking. Default chunk size 1200 characters with 200 overlap (starter:
    500 and 50), both adjustable in the sidebar.
 4. Model switcher. Gemini (cloud) or Ollama (local) models selectable in the
    sidebar, with the response time shown under each answer.
 5. System prompt. Grounding rule with a fixed refusal sentence, exact figures
    with units and fiscal period, the fiscal-year difference between the three
    companies, step-by-step arithmetic, (Company, p. X) citations, and no
    forecasts or investment advice.
 6. Disk-cached FAISS index under .faiss_cache/, keyed by the PDF list, chunk
    settings and embedding model. "Rebuild index" deletes the current one.
 7. Default temperature 0.1 (starter: 0.5).

Setup: put the 10-K PDFs in a PDFs/ folder next to this script (filenames
must contain the company name or ticker), or use the in-app uploader; set
GOOGLE_API_KEY in the environment or paste it in the sidebar. See README.md.
"""

import os
import re
import glob
import time
import shutil
import hashlib
import tempfile

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDF_DIR = "PDFs"           # drop the three 10-K PDFs here
CACHE_DIR = ".faiss_cache" # cached FAISS indexes live here (gitignore this)

# Company detection from filenames and from question text.
COMPANY_PATTERNS = {
    "Alphabet (Google)": ("alphabet", "google", "goog"),
    "Amazon": ("amazon", "amzn"),
    "Microsoft": ("microsoft", "msft"),
}

# If a question contains any of these, treat it as a multi-company question
# and retrieve chunks from every company.
COMPARISON_HINTS = (
    "compare", "comparison", "rank", "ranking", "each company", "all three",
    "which company", "which of", "highest", "lowest", "most", "least",
    "versus", " vs", "between", "sum", "across",
)

GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "Custom…"]
OLLAMA_MODELS = ["mistral", "llama3.1:8b", "deepseek-r1:8b", "Custom…"]
EMBEDDING_OPTIONS = [
    "models/embedding-001 (Gemini)",
    "models/text-embedding-004 (Gemini)",
    "nomic-embed-text (Ollama, local, slow)",
]

SYSTEM_PROMPT = """You are a meticulous financial analyst assistant. You answer questions \
about the latest 10-K filings of Alphabet (Google), Amazon, and Microsoft using ONLY the \
document excerpts provided below.

Rules you must always follow:
1. GROUNDING: Use only the excerpts. If they do not contain the answer, say: \
"I don't have enough information in the provided documents to answer this." \
Never fill gaps with outside knowledge or guesses.
2. NUMBERS: Quote figures exactly as reported, with units (millions vs. billions), \
currency, and the fiscal year or period. Never round silently and never convert \
units without showing the conversion.
3. FISCAL YEARS: These companies use different fiscal calendars: Microsoft's fiscal \
year ends June 30, while Amazon's and Alphabet's end December 31. When comparing \
across companies, state each figure's fiscal period explicitly.
4. CALCULATIONS: For ratios, differences, or rankings: first list the input numbers \
taken from the excerpts (with citations), then show the arithmetic step by step, \
then give the result.
5. NO FORECASTS, NO INVESTMENT ADVICE: 10-K filings report the past; they contain \
no predictions of future profits and no investment recommendations. If asked for \
either, say so plainly, and offer the relevant historical figures instead.
6. CITATIONS: After each key fact, cite the company and page in parentheses, \
e.g. (Amazon, p. 45).
7. If the excerpts conflict with each other or appear incomplete, say so plainly \
rather than papering over it."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def company_from_filename(filename: str) -> str:
    """Infer the company from the PDF filename; fall back to the filename."""
    low = filename.lower()
    for company, keys in COMPANY_PATTERNS.items():
        if any(k in low for k in keys):
            return company
    return os.path.splitext(os.path.basename(filename))[0]


def companies_in_question(question: str) -> list:
    """Which of the known companies does the question explicitly mention?"""
    low = question.lower()
    return [
        company
        for company, keys in COMPANY_PATTERNS.items()
        if any(k in low for k in keys)
    ]


def tag_documents(docs, filename: str) -> None:
    company = company_from_filename(filename)
    for d in docs:
        d.metadata["company"] = company
        d.metadata["source_file"] = filename


def load_documents(local_paths, uploaded_files):
    """Load PDFs from ./PDFs if present, otherwise from the uploader."""
    documents = []
    if local_paths:
        for path in local_paths:
            docs = PyPDFLoader(path).load()
            tag_documents(docs, os.path.basename(path))
            documents.extend(docs)
    else:
        tmp = tempfile.mkdtemp()
        for uf in uploaded_files:
            path = os.path.join(tmp, uf.name)
            with open(path, "wb") as fh:
                fh.write(uf.getbuffer())
            docs = PyPDFLoader(path).load()
            tag_documents(docs, uf.name)
            documents.extend(docs)
    return documents


def settings_fingerprint(file_sig, chunk_size, chunk_overlap, emb_name) -> str:
    raw = repr((sorted(file_sig), int(chunk_size), int(chunk_overlap), emb_name))
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def make_embeddings(emb_name: str, api_key: str):
    if emb_name.startswith("models/"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=emb_name, google_api_key=api_key or None)
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=emb_name)


def make_llm(provider: str, model_name: str, temperature: float, api_key: str):
    if provider.startswith("Gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature, google_api_key=api_key or None
        )
    from langchain_ollama import ChatOllama
    # num_ctx raised so 12 retrieved chunks fit in local models' context.
    return ChatOllama(model=model_name, temperature=temperature, num_ctx=8192)


def build_vector_store(chunks, embeddings):
    """Embed in batches with a progress bar (3 x 10-K = several hundred chunks)."""
    batch = 100
    vs = None
    prog = st.progress(0.0, text="Embedding chunks…")
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        if vs is None:
            vs = FAISS.from_documents(part, embeddings)
        else:
            vs.add_documents(part)
        done = min(i + batch, len(chunks))
        prog.progress(done / len(chunks), text=f"Embedding chunks… {done}/{len(chunks)}")
    prog.empty()
    return vs


def retrieve(vs, question, companies_all, k_per, smart: bool):
    """Company-aware retrieval (item 2 in the module docstring)."""

    def search(query, k, flt=None):
        try:
            return vs.max_marginal_relevance_search(
                query, k=k, fetch_k=max(20, k * 5), filter=flt
            )
        except Exception:
            return vs.similarity_search(query, k=k, filter=flt)

    if not smart:
        # Starter-style behaviour: one global search (kept so the two modes can be compared).
        return search(question, k_per * max(1, len(companies_all)))

    mentioned = companies_in_question(question)
    low = question.lower()
    is_multi = len(mentioned) != 1 or any(h in low for h in COMPARISON_HINTS)
    targets = companies_all if (is_multi or not mentioned) else mentioned

    docs = []
    for company in targets:
        docs.extend(search(question, k_per, {"company": company}))

    # De-duplicate (same chunk can surface twice).
    seen, unique = set(), []
    for d in docs:
        key = (d.metadata.get("source_file"), d.metadata.get("page"), d.page_content[:80])
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique or search(question, k_per * max(1, len(companies_all)))


def page_label(doc) -> str:
    md = doc.metadata
    if md.get("page_label"):
        return str(md["page_label"])
    p = md.get("page")
    return str(p + 1) if isinstance(p, int) else "?"


def format_context(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(
            f"[Excerpt {i}: {d.metadata.get('company', '?')}, page {page_label(d)}]\n"
            f"{d.page_content}"
        )
    return "\n\n".join(parts)


def format_history(messages, max_turns=6, max_chars=600) -> str:
    lines = []
    for m in messages[-max_turns:]:
        role = "User" if m["role"] == "user" else "Assistant"
        text = m["content"]
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


def build_prompt(question, context, history_text) -> str:
    return f"""{SYSTEM_PROMPT}

Conversation so far (context only; the excerpts always take priority):
<history>
{history_text if history_text else "(none)"}
</history>

Document excerpts:
<context>
{context}
</context>

Question: {question}

Answer:"""


def clean_llm_text(text: str) -> str:
    """Strip <think>…</think> blocks emitted by reasoning models (e.g. deepseek-r1)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="10-K Analyst", layout="wide")
st.title("10-K Analyst: Alphabet, Amazon, Microsoft")
st.caption(
    "Upgraded from the course starter: company-aware retrieval, fixed retrieval query, "
    "analyst prompt, model switcher, cached index."
)

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")

    provider = st.radio("LLM provider", ["Gemini (cloud)", "Ollama (local)"])
    if provider.startswith("Gemini"):
        model_pick = st.selectbox("Gemini model", GEMINI_MODELS, index=0)
    else:
        model_pick = st.selectbox("Ollama model", OLLAMA_MODELS, index=0)
    model_name = (
        st.text_input("Custom model name", value="")
        if model_pick == "Custom…"
        else model_pick
    )

    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.1, 0.05,
        help="Keep low for factual number retrieval (starter default was 0.5).",
    )

    emb_choice = st.selectbox(
        "Embedding model", EMBEDDING_OPTIONS, index=0,
        help="Changing this rebuilds the index.",
    )
    emb_name = emb_choice.split(" ")[0]

    st.divider()
    smart_retrieval = st.checkbox(
        "Per-company retrieval (smart mode)", value=True,
        help="Comparison questions fetch top-k chunks from EACH company. "
        "Untick to demo the starter-style single global search.",
    )
    k_per_company = st.slider("Chunks per company (k)", 2, 8, 4)

    st.divider()
    chunk_size = st.number_input(
        "Chunk size (chars)", 300, 4000, 1200, 100,
        help="Starter used 500, which is too small for financial tables. Changing rebuilds the index.",
    )
    chunk_overlap = st.number_input("Chunk overlap (chars)", 0, 600, 200, 50)

    st.divider()
    needs_gemini_key = provider.startswith("Gemini") or emb_name.startswith("models/")
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if needs_gemini_key and not api_key:
        api_key = st.text_input(
            "Gemini API key", type="password",
            help="Free at https://aistudio.google.com/apikey. Never hardcode it in the script.",
        )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    col1, col2 = st.columns(2)
    rebuild_clicked = col1.button("Rebuild index")
    if col2.button("Clear chat"):
        st.session_state.messages = []

# ---- Guards ----
if not model_name:
    st.warning("Enter a model name in the sidebar.")
    st.stop()
if needs_gemini_key and not api_key:
    st.warning("A Gemini API key is required for the selected model/embeddings. "
               "Paste it in the sidebar or set GOOGLE_API_KEY before launching.")
    st.stop()

# ---- Document source ----
local_pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
uploaded_files = None
if local_pdfs:
    st.caption(
        f"Using {len(local_pdfs)} PDF(s) from ./{PDF_DIR}: "
        + ", ".join(os.path.basename(p) for p in local_pdfs)
    )
else:
    uploaded_files = st.file_uploader(
        "Upload the three 10-K PDFs (filenames should contain the company name)",
        type=["pdf"], accept_multiple_files=True,
    )
if not local_pdfs and not uploaded_files:
    st.info(f"Drop the three 10-K PDFs into ./{PDF_DIR} next to this script, or upload them above.")
    st.stop()

# ---- Build or load the index ----
if local_pdfs:
    file_sig = [(os.path.basename(p), os.path.getsize(p)) for p in local_pdfs]
else:
    file_sig = [(f.name, f.size) for f in uploaded_files]

fp = settings_fingerprint(file_sig, chunk_size, chunk_overlap, emb_name)
cache_path = os.path.join(CACHE_DIR, fp)

if rebuild_clicked:
    shutil.rmtree(cache_path, ignore_errors=True)
    st.session_state.pop("index_key", None)
    st.session_state.pop("vector_store", None)

if st.session_state.get("index_key") != fp:
    try:
        embeddings = make_embeddings(emb_name, api_key)
        vs = None
        if os.path.isdir(cache_path):
            try:
                vs = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
            except TypeError:  # older langchain without that kwarg
                vs = FAISS.load_local(cache_path, embeddings)
            except Exception:
                vs = None
        if vs is None:
            with st.spinner("Reading PDFs…"):
                documents = load_documents(local_pdfs, uploaded_files)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
            )
            chunks = splitter.split_documents(documents)
            vs = build_vector_store(chunks, embeddings)
            os.makedirs(CACHE_DIR, exist_ok=True)
            vs.save_local(cache_path)
            st.success(f"Indexed {len(chunks)} chunks (cached for next time).")
        st.session_state.vector_store = vs
        try:
            st.session_state.companies = sorted(
                {d.metadata.get("company", "?") for d in vs.docstore._dict.values()}
            )
        except Exception:
            st.session_state.companies = list(COMPANY_PATTERNS.keys())
        st.session_state.index_key = fp
    except Exception as e:
        st.error(f"Index build failed: {e}")
        st.stop()

st.caption(
    "Index ready · companies: " + ", ".join(st.session_state.companies)
    + f" · chunks of {int(chunk_size)}/{int(chunk_overlap)} · embeddings: {emb_name}"
)

# ---- Chat ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask about the 10-Ks…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    vs = st.session_state.vector_store
    companies = st.session_state.companies

    t0 = time.time()
    with st.spinner("Retrieving + thinking…"):
        docs = retrieve(vs, question, companies, k_per_company, smart_retrieval)
        context = format_context(docs)
        history_text = format_history(st.session_state.messages[:-1])
        prompt = build_prompt(question, context, history_text)
        try:
            llm = make_llm(provider, model_name, temperature, api_key)
            resp = llm.invoke(prompt)
            answer = clean_llm_text(getattr(resp, "content", str(resp)))
        except Exception as e:
            answer = f"LLM call failed: {e}"
    elapsed = time.time() - t0

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"{model_name} · {len(docs)} chunks retrieved · {elapsed:.1f}s")
        with st.expander("Retrieved excerpts (first 700 characters of each chunk the model saw)"):
            for i, d in enumerate(docs, 1):
                st.markdown(
                    f"**{i}. {d.metadata.get('company', '?')}, page {page_label(d)}** · "
                    f"`{d.metadata.get('source_file', '')}`"
                )
                snippet = d.page_content[:700]
                st.text(snippet + ("…" if len(d.page_content) > 700 else ""))
                st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})
