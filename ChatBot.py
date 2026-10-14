import hashlib
import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-3.8-flash"
TOP_K = 4
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_PAGES = 150
REFUSAL = "i couldn't find that in the document"

# Models are tried in this order. Add a spare key to Tokens.env (or the app
# secrets) and it is picked up automatically - no code change needed.
KEY_SLOTS = [
    ("Groq", "GROQ_API_KEY"),
    ("Gemini", "GEMINI_API_KEY"),
    ("Groq (spare)", "GROQ_API_KEY_2"),
    ("Gemini (spare)", "GEMINI_API_KEY_2"),
]

SAMPLE_QUESTIONS = [
    ("📝", "Summarize this document in five bullet points."),
    ("📌", "What are the key rules or policies mentioned?"),
    ("📅", "List any dates, deadlines or amounts mentioned."),
]

STEPS = [
    ("📤", "Upload", "Add one or more PDF files in the sidebar."),
    ("✂️", "Chunk", "The app reads every page and splits the text into small, overlapping chunks."),
    ("🧮", "Embed", "Each chunk is turned into numbers (an embedding) and stored in a FAISS search index."),
    ("🔎", "Retrieve", "When you ask a question, the app finds the chunks that match it best."),
    ("🤖", "Answer", "An AI model reads only those chunks and writes the answer. Sources are shown."),
    ("⏳", "Be patient",
     "Answers usually appear in a few seconds, but can take 1-2 minutes when "
     "the document is large or the AI service is busy."),
]

AVATARS = {"user": "🙋", "assistant": "🤖"}

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions about a document. "
     "Answer the question using ONLY the context below. "
     "If the answer is not in the context, say: \"I couldn't find that in the document.\" "
     "Do not make things up.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📄",
    layout="wide",
)

load_dotenv("Tokens.env")


# ---------------------------------------------------------------- helpers
def get_key(name):
    """Read a key from Streamlit secrets (deployment) or Tokens.env (local)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_models():
    """Return [(label, model), ...] for every key that is available."""
    models = []
    for label, secret_name in KEY_SLOTS:
        key = get_key(secret_name)
        if not key:
            continue
        if label.startswith("Groq"):
            model = ChatGroq(
                model=GROQ_MODEL, temperature=0.2, api_key=key,
                timeout=30, max_retries=1)
        else:
            model = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL, temperature=0.2, google_api_key=key,
                timeout=30, max_retries=1)
        models.append((label, model))
    if not models:
        raise RuntimeError(
            "No API keys found. Add GROQ_API_KEY and/or GEMINI_API_KEY "
            "to Tokens.env (local) or to the app secrets (deployment).")
    return models


def answer_question(context, question):
    """Try each model in turn. Returns (answer, label of the model that answered)."""
    last_error = None
    for label, model in MODELS:
        try:
            chain = PROMPT | model | RunnableLambda(lambda m: m.text)
            answer = chain.invoke({"context": context, "question": question})
            return answer, label
        except Exception as err:
            last_error = err
    raise last_error


def files_key(files):
    digest = hashlib.md5()
    for f in files:
        digest.update(f.name.encode())
        digest.update(f.getvalue())
    return digest.hexdigest()


def build_index(files, log):
    """Read the PDFs, split them into chunks and build a FAISS index."""
    docs = []
    file_info = []
    log("📖 Reading PDF pages...")
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
        finally:
            os.remove(tmp_path)
        for page in pages:
            page.metadata["file"] = f.name
        docs.extend(pages)
        file_info.append({"name": f.name, "pages": len(pages)})

    if len(docs) > MAX_PAGES:
        raise ValueError(
            f"These files have {len(docs)} pages. Please upload at most "
            f"{MAX_PAGES} pages in total.")
    if not any(d.page_content.strip() for d in docs):
        raise ValueError(
            "No readable text was found. Scanned (image-only) PDFs are not supported.")

    log("✂️ Splitting the text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    log("🧮 Creating embeddings and the search index...")
    vectorstore = FAISS.from_documents(chunks, load_embeddings())

    stats = {
        "files": file_info,
        "pages": len(docs),
        "chunks": len(chunks),
        "words": sum(len(d.page_content.split()) for d in docs),
        "preview": [c.page_content[:300] for c in chunks[:3]],
    }
    return vectorstore, stats


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def render_sources(sources):
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            with st.container(border=True):
                st.caption(f"Source {i} · 📄 {s['file']} · page {s['page']}")
                st.write(s["text"])


def chat_as_text(messages):
    lines = ["# Chat export", ""]
    for m in messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"**{who}:** {m['content']}")
        lines.append("")
    return "\n".join(lines)


def reset_document():
    st.session_state.index_key = None
    st.session_state.vectorstore = None
    st.session_state.stats = None
    st.session_state.messages = []


# ---------------------------------------------------------------- state
for key, default in {
    "messages": [], "pending": None,
    "index_key": None, "vectorstore": None, "stats": None,
}.items():
    st.session_state.setdefault(key, default)

try:
    MODELS = load_models()
except RuntimeError as err:
    st.error(f"🔑 {err}")
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📁 Your document")
    uploaded = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Text-based PDFs only, up to {MAX_PAGES} pages in total.",
    )
    files = uploaded or []

    if not files:
        if st.session_state.index_key is not None:
            reset_document()
    else:
        new_key = files_key(files)
        if new_key != st.session_state.index_key:
            error = None
            result = None
            with st.status("⚙️ Processing your document...", expanded=True) as status:
                try:
                    result = build_index(files, status.write)
                except ValueError as err:
                    error = str(err)
                except Exception:
                    error = "Sorry, this file could not be read. Try another PDF."
                if error:
                    status.update(label="Could not process the document", state="error")
                else:
                    status.update(label="Document ready!", state="complete", expanded=False)
            if error:
                st.error(f"⚠️ {error}")
                st.stop()
            st.session_state.vectorstore, st.session_state.stats = result
            st.session_state.index_key = new_key
            st.session_state.messages = []
            st.toast("Document indexed. Ask away!", icon="✅")

    ready = st.session_state.vectorstore is not None

    if ready:
        s = st.session_state.stats
        st.success(f"✅ Ready: {len(s['files'])} file(s), {s['pages']} pages, {s['chunks']} chunks")
    else:
        st.info("👆 Upload a PDF to get started.")

    st.divider()
    st.subheader("💡 Try asking")
    for i, (icon, sample) in enumerate(SAMPLE_QUESTIONS):
        if st.button(f"{icon} {sample}", key=f"sample_{i}", disabled=not ready):
            st.session_state.pending = sample

    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider(
        "🔢 Chunks to retrieve", 2, 8, TOP_K,
        help="How many pieces of the document are given to the AI for each question.")
    show_sources = st.toggle("📚 Show sources", value=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Clear", disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        st.download_button(
            "⬇️ Save",
            data=chat_as_text(st.session_state.messages),
            file_name="chat.md",
            mime="text/markdown",
            disabled=not st.session_state.messages,
        )

    st.divider()
    st.caption("🔑 AI models ready: " + ", ".join(label for label, _ in MODELS))
    st.caption("Built with LangChain · FAISS · Streamlit")

# ---------------------------------------------------------------- main page
st.title("📄 Document Q&A Chatbot")
st.caption("Upload a PDF and chat with it. Answers come only from your document, with sources. ✨")

tab_chat, tab_how, tab_doc = st.tabs(["💬 Chat", "🧭 How it works", "📊 Document"])

# ---- chat tab: history
with tab_chat:
    if not st.session_state.messages:
        if ready:
            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                st.markdown(
                    "👋 **Hi!** Your document is ready. Ask me anything about it, "
                    "or pick a sample question from the sidebar.")
        else:
            st.info("👈 Upload a PDF in the sidebar to start chatting. "
                    "New here? Open the **🧭 How it works** tab.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
            st.markdown(msg["content"])
            if show_sources and msg.get("sources"):
                render_sources(msg["sources"])
            if msg.get("elapsed"):
                st.caption(f"⏱️ {msg['elapsed']:.1f} s · 🤖 {msg['model']}")

# ---- new question (the chat box stays pinned at the bottom of the page)
question = st.chat_input("💬 Ask a question about your document...", disabled=not ready)
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None

if question and ready:
    st.session_state.messages.append({"role": "user", "content": question})
    with tab_chat:
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(question)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            try:
                with st.spinner("🤔 Thinking... this can take a few seconds, "
                                "longer if the AI service is busy."):
                    start = time.time()
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": top_k})
                    found = retriever.invoke(question)
                    answer, used = answer_question(format_docs(found), question)
                    elapsed = time.time() - start
            except Exception as err:
                st.error("😕 Sorry, I could not get an answer. The AI service may be "
                         "busy or out of free quota. Please try again in a minute.")
                st.caption(f"Error type: {type(err).__name__}")
            else:
                st.markdown(answer)
                sources = []
                if REFUSAL not in answer.lower():
                    sources = [
                        {
                            "file": d.metadata.get("file", "document"),
                            "page": d.metadata.get("page", 0) + 1,
                            "text": d.page_content[:400],
                        }
                        for d in found
                    ]
                    if show_sources:
                        render_sources(sources)
                st.caption(f"⏱️ {elapsed:.1f} s · 🤖 {used}")
                if used != MODELS[0][0]:
                    st.toast(f"The main model was busy, so {used} answered.", icon="🔁")
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "elapsed": elapsed, "model": used,
                })

# ---- how it works tab
with tab_how:
    st.subheader("How this app works")
    for start_index, row in ((1, STEPS[:3]), (4, STEPS[3:])):
        columns = st.columns(3)
        for n, (column, (icon, title, text)) in enumerate(zip(columns, row), start_index):
            with column.container(border=True):
                st.markdown(f"#### {n}. {icon} {title}")
                st.write(text)
    st.info("⏳ **Please be patient:** the first answer after an upload can take longer, "
            "and answers can take up to 1-2 minutes when the document is large or "
            "the AI service is busy.")

# ---- document tab (filled last so the counters are up to date)
with tab_doc:
    if not ready:
        st.info("📄 Upload a PDF to see its details here.")
    else:
        stats = st.session_state.stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📁 Files", len(stats["files"]))
        m2.metric("📄 Pages", stats["pages"])
        m3.metric("🧩 Chunks", stats["chunks"])
        m4.metric("📝 Words", f"{stats['words']:,}")

        st.subheader("Files")
        for info in stats["files"]:
            st.write(f"📄 **{info['name']}** · {info['pages']} pages")

        asked = sum(1 for m in st.session_state.messages if m["role"] == "user")
        times = [m["elapsed"] for m in st.session_state.messages if m.get("elapsed")]
        st.subheader("This session")
        q1, q2 = st.columns(2)
        q1.metric("💬 Questions asked", asked)
        q2.metric("⏱️ Average answer time", f"{sum(times) / len(times):.1f} s" if times else "-")

        with st.expander("🔍 Preview the first chunks"):
            for i, text in enumerate(stats["preview"], 1):
                with st.container(border=True):
                    st.caption(f"Chunk {i}")
                    st.write(text)
