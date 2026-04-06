# 🏥 Healthcare Policy Assistant

An AI-powered RAG (Retrieval-Augmented Generation) application that answers 
questions about healthcare policy documents using natural language.

Built by a Healthcare Payor Data Architect transitioning into AI Engineering.

---

## 💡 What It Does

- Upload any healthcare policy PDF (benefits manual, coverage guidelines, prior auth policies)
- Ask questions in plain English
- Get accurate answers grounded in the actual document
- No hallucinations — AI only answers from the document content

---

## 🏗️ Architecture
PDF Document
↓
Text Extraction (PyPDF)
↓
Chunking (LangChain Text Splitter)
↓
Embeddings (OpenAI)
↓
Vector Storage (ChromaDB)
↓
User Question → Similarity Search → GPT-3.5 → Answer

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| LangChain | AI orchestration framework |
| OpenAI GPT-3.5 | Language model |
| ChromaDB | Local vector database |
| Streamlit | Web UI |
| PyPDF | PDF parsing |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

1. Clone the repo
   git clone https://github.com/YOUR_USERNAME/healthcare-rag-assistant.git
   cd healthcare-rag-assistant

2. Create and activate virtual environment
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux

3. Install dependencies
   pip install -r requirements.txt

4. Set up environment variables
   Create a .env file in the root folder:
   OPENAI_API_KEY=your-key-here

5. Add your PDF
   Place your healthcare policy PDF in the root folder
   Rename it to policy.pdf

6. Build the vector database
   python main.py

7. Run the app
   streamlit run app.py

---

## 💬 Example Questions You Can Ask

- What does this policy cover?
- Are there any deductibles or copayments?
- What services are not covered?
- How do I file a claim?
- What is the prior authorization process?

---

## 🗺️ Roadmap

- [x] PDF ingestion and chunking
- [x] Vector database with ChromaDB
- [x] RAG pipeline with LangChain
- [x] Streamlit chat UI
- [ ] Support multiple PDF uploads
- [ ] Claims denial prediction model
- [ ] Prior auth automation agent
- [ ] Deploy to AWS

---

## 👤 Author

**Prakash** — Healthcare Payor Data Architect → AI Engineer
- Domain: Healthcare Payor (Claims, Benefits, Prior Auth)
- Background: Data Architecture, ETL, Informatica
- Transition: Building AI systems for healthcare

---

## 📄 License
MIT