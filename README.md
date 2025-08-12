#  InfoScribe — RAG Ollama PDF Chatbot

**InfoScribe** is a user-friendly **Retrieval-Augmented Generation (RAG)** application built with **Streamlit** and powered by **Ollama**.  
It lets you upload PDFs, build a searchable knowledge library, and chat with an AI assistant that answers **strictly from your documents** — no guessing, no hallucinations.

---

## ✨ Features
- **📄 Upload & Manage PDFs** — Store your documents in a local library.
- **🔄 Replace or Append Mode** — Rebuild the library from scratch or add to it.
- **⚡ Vector Search with FAISS** — Fast, semantic retrieval of the most relevant chunks.
- **🛡 Strict Context Control** — Answers are generated only from retrieved document chunks.
- **📂 Session Management** — Create, open, and delete named chat sessions for different topics.
- **🧰 Utilities in Chat** — Save conversation as `chat.md` or clear history without deleting the session.
- **🤝 Polite & Friendly** — Responds warmly to greetings, while sticking to the document content for factual answers.

---

## 🛠 How It Works
1. **Build Library**:  
   - Extracts text from PDFs using PyMuPDF or PyPDF.
   - Splits text into smaller chunks for better search accuracy.
   - Generates vector embeddings and stores them in a FAISS index.
   
2. **Ask Questions**:  
   - Your question is converted into an embedding.
   - The app searches the FAISS index for the most relevant chunks based on your question.

3. **Get Answers**:  
   - The LLM receives only the retrieved chunks as **context**.
   - It responds using **only** that context, ensuring accuracy and preventing hallucinations.

---

## 💡 Tech Stack
- [Streamlit](https://streamlit.io/) — for the user interface.
- [Ollama](https://ollama.ai/) — for local LLM inference.
- [FAISS](https://github.com/facebookresearch/faiss) — for vector search and storage.
- [LangChain](https://www.langchain.com/) — for chaining prompts and retrieval.
- [PyMuPDF](https://pymupdf.readthedocs.io/) / [PyPDF](https://pypdf.readthedocs.io/) — for PDF text extraction.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/infoscribe.git
cd infoscribe
