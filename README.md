# InfoScribe — RAG Ollama PDF Chatbot with Quiz & QnA

**InfoScribe2.o** is a user-friendly **Retrieval-Augmented Generation (RAG)** application built with **Streamlit** and powered by **Ollama**. It lets you upload PDFs, build a searchable knowledge library, and chat with an AI assistant that answers **strictly from your documents** — no guessing, no hallucinations.

---

## ✨ Features
- **📄 Upload & Manage PDFs** — Store your documents in a local library.
- **🔄 Replace or Append Mode** — Rebuild the library from scratch or add to it.
- **⚡ Vector Search with FAISS** — Fast, semantic retrieval of relevant chunks.
- **🛡 Strict Context Control** — Answers only from retrieved document chunks.
- **📂 Session Management** — Create, open, and delete named chat sessions.
- **🧩 Quiz & QnA Module** — Generate quizzes (MCQ, True/False, Short Answer) or QnA flashcards from your PDFs.
- **📊 Score & Export** — View detailed quiz reports and export results/QnA as CSV.
- **🤝 Polite & Friendly** — Responds warmly to greetings, while sticking to document content.

---

## 🛠 How It Works
1. **Build Library**  
   - Extracts text from PDFs using PyMuPDF or PyPDF.
   - Splits text into smaller chunks for accurate retrieval.
   - Generates vector embeddings and stores them in a FAISS index.

2. **Ask Questions**  
   - Your query is converted to an embedding.
   - The FAISS index retrieves the most relevant chunks.

3. **Get Answers**  
   - The LLM receives only retrieved chunks as **context**.
   - Responds **only** from this context.

4. **Generate Quizzes & QnA**  
   - Choose scope (All docs, specific docs, or a topic).
   - Generate topic-specific quizzes (MCQ / True-False / Short Answer) or QnA flashcards.
   - Submit quizzes to see score, explanations, and sources.
   - Export quiz results and QnA sets to CSV.

---

## 💡 Tech Stack
- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.ai/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://www.langchain.com/)
- [PyMuPDF](https://pymupdf.readthedocs.io/) / [PyPDF](https://pypdf.readthedocs.io/)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/infoscribe.git
cd infoscribe
```

### 2. Install Dependencies
```bash
pip install streamlit langchain-ollama langchain-community pypdf pymupdf faiss-cpu
```

### 3. Start Ollama
```bash
ollama run llama3.2:1b
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 📚 Usage
1. **Build the Library** — Upload PDFs and click **Build Index**.
2. **Chat** — Ask questions; answers come only from your PDFs.
3. **Sessions** — Manage separate chats for different topics.
4. **Quiz & QnA** — Generate quizzes or QnA flashcards from your library:
   - Scope: All docs | Specific docs | Topic keywords
   - Question Types: MCQ | True/False | Short Answer
   - Export results/QnA as CSV

---

## 📊 Example Quiz Flow
- Select **TOPIC** = "Quantum Mechanics"
- Choose **MCQ**, 5 questions
- Generate quiz → Answer questions → Submit → View score & explanations
- Export CSV for records

---

## 🔒 Context Control
If the answer is not in your documents, the assistant will respond:
```
I don't know. I couldn't find anything relevant in the provided documents.
```

---


