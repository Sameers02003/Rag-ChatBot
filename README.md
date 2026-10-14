# 📄 Document Q&A Chatbot (RAG)

Upload a PDF and ask questions about it. The chatbot answers **only from your document** and shows the sources it used.

🔗 **Live app:** 

## ✨ Features
- 📤 Upload one or more PDFs in the app (no document is hard-coded)
- 🔎 Retrieval-Augmented Generation: answers are grounded in your file
- 📚 Sources shown for every answer (file and page)
- 🤖 Hosted AI models with automatic backup: Groq first, then Gemini
- 🧭 "How it works" tab and a "Document" tab with live stats
- ⚙️ Adjustable number of retrieved chunks, chat download

## 🧠 How it works
1. **Upload** a PDF in the sidebar.
2. **Chunk:** the text is split into 800-character chunks with 150 overlap.
3. **Embed:** each chunk is converted to a vector (`all-MiniLM-L6-v2`, runs locally) and stored in a FAISS index.
4. **Retrieve:** your question is matched to the top-k most similar chunks.
5. **Answer:** an LLM reads only those chunks and writes the answer.
   If the answer is not in the document, it says so.
6. **Be patient:** answers usually take a few seconds, but can take around 80–120 seconds
   when the document is large or the AI service is busy.

## 🛠 Tech stack
LangChain · FAISS · Sentence-Transformers · Groq (`openai/gpt-oss-20b`) · Google Gemini · Streamlit

## 🚀 Run locally
```bash
pip install -r requirements.txt
```
Create a file named `Tokens.env` (see `.env.example`) with your own free API keys:
```
GROQ_API_KEY=...
GEMINI_API_KEY=...
```
Then start the app:
```bash
python -m streamlit run ChatBot.py
```

## 🔑 API keys and backup
The app tries the models in order: Groq → Gemini → spare keys
(`GROQ_API_KEY_2`, `GEMINI_API_KEY_2`). If one key hits its free limit or expires,
the next one answers automatically. Keys are never stored in the code or in GitHub.

## ☁️ Deployment
Deployed on Streamlit Community Cloud. Keys are added in the app's **Secrets** settings.

## ⚠️ Limitations
- Works with text-based PDFs only (scanned images are not supported)
- Up to 150 pages in total per session
- Free API plans have rate limits

## 📁 Project files
| File | Purpose |
|---|---|
| `ChatBot.py` | Streamlit app |
| `rag_chatbot.ipynb` | Notebook: step-by-step build and experiments |
| `.streamlit/config.toml` | Fonts and theme |
| `requirements.txt` | Dependencies |
| `tests/` | Retriever test |
| `.env.example` | Template for API keys |
