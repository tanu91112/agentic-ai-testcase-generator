# RAG-Based Test Case Generator (Autonomous QA Agent)


## Overview
An AI-powered QA automation system that uses Retrieval-Augmented Generation (RAG) 
to generate structured test cases directly from project documentation and 
convert them into executable Selenium scripts.

This project implements an end-to-end autonomous QA pipeline designed to:
- Ingest requirement documents and UI HTML
- Build a semantic knowledge base using vector embeddings
- Generate documentation-grounded test cases
- Automatically produce Selenium (Python) automation scripts

---

## Key Features
- Document ingestion (txt, md, pdf)
- Semantic retrieval using MiniLM embeddings
- ChromaDB vector storage
- Local LLM integration (Flan-T5)
- Structured JSON test case generation
- Automated Selenium script creation
- Streamlit-based interactive UI
- FastAPI backend architecture

---

## Architecture
User Upload 
→ Document Parsing 
→ Embedding Generation 
→ Vector Store (ChromaDB) 
→ Semantic Retrieval 
→ Local LLM (Flan-T5) 
→ Structured Test Cases 
→ Selenium Script Generation

---

## Tech Stack
- Python 3.10+
- FastAPI
- Streamlit
- Transformers (Flan-T5)
- SentenceTransformers (MiniLM)
- ChromaDB
- Selenium

---

## Setup Instructions

### 1. Clone Repository
git clone https://github.com/tanu91112/RAG-Test-Case-Generator-main.git  
cd RAG-Test-Case-Generator-main  

### 2. Create Virtual Environment
python -m venv venv  
venv\Scripts\activate  
pip install -r requirements.txt  

### 3. Run Backend
uvicorn api:app --reload  

### 4. Run Frontend
streamlit run ui.py  

---

## How It Works
1. Upload requirement documents and HTML files.
2. Build a semantic knowledge base using vector embeddings.
3. Provide a generation prompt.
4. Retrieve relevant contextual chunks.
5. Generate structured test cases using the local LLM.
6. Convert selected test cases into Selenium scripts.

---

## Design Decisions
- MiniLM chosen for efficient embedding generation.
- ChromaDB selected for lightweight local vector storage.
- Flan-T5 used as local LLM to avoid external API dependency.
- Modular architecture for easy backend swapping.

---

## Future Improvements
- Dockerized deployment
- Scalable vector indexing
- Multi-user authentication
- Cloud deployment (AWS/GCP)

---

## Maintainer
Tanu Chandravanshi  
AI & Machine Learning Student  
GitHub: https://github.com/tanu91112
