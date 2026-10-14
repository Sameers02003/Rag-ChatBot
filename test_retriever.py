import glob

import pytest
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDFS = [p for p in glob.glob("**/*.pdf", recursive=True) if "venv" not in p]


@pytest.mark.skipif(not PDFS, reason="no sample PDF found in the project")
def test_retriever_returns_chunks():
    pages = PyPDFLoader(PDFS[0]).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150).split_documents(pages)
    store = FAISS.from_documents(
        chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    found = store.as_retriever(search_kwargs={"k": 4}).invoke("leave policy")
    assert len(found) == 4
    assert all(d.page_content.strip() for d in found)
