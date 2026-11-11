import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def load_and_chunk_pdf(pdf_path: str) -> list[Document]:
    """
    Loads a technical PDF and splits it
    into optimized chunks for technical context.
    """
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    # Text splitter optimized to avoid breaking technical sentences or tables
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"[INFO] Successfully loaded {pdf_path}: {len(chunks)} chunks generated.")
    return chunks

def setup_hybrid_retrievers(chunks: list[Document]):
    """
    Sets up both BM25 (keyword-based) and Vector (semantic) retrievers
    for the Hybrid Search pipeline.
    """
    # 1. BM25 Retriever (Keyword search)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3 
    
    # 2. Vector Retriever (Semantic search)
    print("[INFO] Initializing local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Stores locally vectors in memory
    vectorstore = Chroma.from_documents(chunks, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    print("[INFO] Hybrid retrievers initialized successfully.")
    return bm25_retriever, vector_retriever