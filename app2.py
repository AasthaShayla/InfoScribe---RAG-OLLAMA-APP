import os, io, shutil, json, uuid, datetime, re, random
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

USE_PYMUPDF = True
try:
    import fitz  
except Exception:
    USE_PYMUPDF = False
    from pypdf import PdfReader

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

# ====================== PAGE CONFIGURATION ======================
st.set_page_config(
    page_title="InfoScribe 2.0 — Professional RAG Suite", 
    layout="wide", 
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# ====================== PROFESSIONAL STYLING ======================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables for Professional Theme */
    :root {
        --primary-600: #1E40AF;
        --primary-500: #3B82F6;
        --primary-100: #DBEAFE;
        --success-600: #059669;
        --success-100: #D1FAE5;
        --warning-600: #D97706;
        --warning-100: #FED7AA;
        --error-600: #DC2626;
        --error-100: #FEE2E2;
        --gray-900: #111827;
        --gray-800: #1F2937;
        --gray-700: #374151;
        --gray-600: #4B5563;
        --gray-500: #6B7280;
        --gray-400: #9CA3AF;
        --gray-300: #D1D5DB;
        --gray-200: #E5E7EB;
        --gray-100: #F3F4F6;
        --gray-50: #F9FAFB;
    }
    
    /* Global Styles */
    .main > div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Styling */
    .app-header {
        background: linear-gradient(135deg, var(--primary-600) 0%, var(--primary-500) 100%);
        color: white;
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-radius: 0 0 20px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(30, 64, 175, 0.1);
    }
    
    .app-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .app-header .subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.9;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Professional Cards */
    .pro-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--gray-200);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        color: var(--gray-900);
    }
    
    .pro-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .pro-card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--gray-200);
    }
    
    .pro-card-icon {
        font-size: 1.5rem;
        margin-right: 0.75rem;
        color: var(--gray-700);
    }
    
    .pro-card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--gray-900) !important;
        margin: 0;
    }
    
    .pro-card h3, .pro-card p, .pro-card div {
        color: var(--gray-900) !important;
    }
    
    .pro-card h2 {
        color: var(--primary-600) !important;
    }
    
    /* Navigation Styles */
    .nav-container {
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--gray-200);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-success {
        background-color: var(--success-100);
        color: var(--success-600);
    }
    
    .status-warning {
        background-color: var(--warning-100);
        color: var(--warning-600);
    }
    
    .status-error {
        background-color: var(--error-100);
        color: var(--error-600);
    }
    
    /* Chat Messages */
    .stChatMessage {
        max-width: 1000px;
        margin: 1rem auto;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* File Display */
    .file-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1rem;
        background: var(--gray-50);
        border: 1px solid var(--gray-200);
        border-radius: 8px;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    
    .file-item:hover {
        background: var(--gray-100);
        border-color: var(--gray-300);
    }
    
    .file-info {
        display: flex;
        align-items: center;
    }
    
    .file-icon {
        font-size: 1.25rem;
        margin-right: 0.75rem;
        color: var(--error-600);
    }
    
    .file-name {
        font-weight: 500;
        color: var(--gray-900);
    }
    
    /* Quiz Styles */
    .quiz-question {
        background: var(--gray-50);
        border-left: 4px solid var(--primary-500);
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: var(--gray-900) !important;
    }
    
    .quiz-question h3, .quiz-question p {
        color: var(--gray-900) !important;
    }
    
    .quiz-options {
        margin: 1rem 0;
    }
    
    .quiz-feedback {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .quiz-correct {
        background: var(--success-100);
        border-color: var(--success-600);
        color: var(--success-600);
    }
    
    .quiz-incorrect {
        background: var(--error-100);
        border-color: var(--error-600);
        color: var(--error-600);
    }
    
    /* Progress Bar */
    .progress-container {
        background: var(--gray-200);
        border-radius: 10px;
        height: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, var(--primary-600), var(--primary-500));
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .app-header {
            padding: 1.5rem 1rem;
        }
        
        .app-header h1 {
            font-size: 2rem;
        }
        
        .pro-card {
            margin: 1rem 0.5rem;
            padding: 1rem;
        }
    }
    
    /* Custom Streamlit Component Overrides */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    .stButton > button {
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====================== CONSTANTS AND INITIALIZATION ======================
INDEX_DIR = "vs_faiss"         
INDEX_META = os.path.join(INDEX_DIR, "_meta.json")
DATA_DIR  = "library_pdfs"        
SESS_DIR  = "sessions"            
SESS_LIST_JSON = os.path.join(SESS_DIR, "_sessions.json")
QUIZ_DIR = "quiz_data"
ANALYTICS_DIR = "analytics"

# Create directories
for directory in [DATA_DIR, SESS_DIR, QUIZ_DIR, ANALYTICS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ====================== SYSTEM PROMPTS ======================
QNA_SYSTEM_PROMPT = """You are a precise academic research assistant specializing in document analysis.

CRITICAL RULES (NEVER BREAK):
- Answer ONLY using information from the provided Context section
- If the Context doesn't contain the answer, respond EXACTLY: "I cannot find this information in the provided documents."
- Never use external knowledge, world knowledge, or speculation
- Always cite specific sources with page numbers when available
- Maintain an academic, precise tone
- Quote relevant passages when appropriate

Format your responses with:
1. Direct answer from the documents
2. Source citations: [Source: filename, Page: X]
3. Relevant quotes if they support your answer

Context: {context}

Question: {input}"""

CHAT_SYSTEM_PROMPT = """You are a friendly, knowledgeable conversation partner with access to the user's document library.

Your personality:
- Be conversational, engaging, and helpful
- Use the provided documents as context when relevant, but don't restrict yourself to them only
- Share broader knowledge when appropriate to enrich the conversation
- Maintain a warm, human-like personality
- Reference documents naturally when they add value to the discussion

When referencing documents: "Based on your documents and my knowledge..."
When going beyond documents: "From my understanding..." or "Generally speaking..."

Remember: You have access to the user's document library, but you're not limited to it. Use it as context to provide richer, more personalized responses.

Available Context: {context}

User: {input}"""

QUIZ_GENERATION_PROMPT = """Generate educational quiz questions based on the provided content.

Create {num_questions} questions with the following distribution:
- 60% Multiple choice (4 options each)
- 25% Short answer (1-2 sentences)
- 15% Fill-in-the-blank

For each question, provide:
1. Question text
2. Correct answer
3. Brief explanation
4. Difficulty level (Easy/Medium/Hard)
5. Source reference (page/section)

Format as JSON:
{{
  "questions": [
    {{
      "type": "multiple_choice|short_answer|fill_blank",
      "question": "Question text",
      "options": ["A", "B", "C", "D"] (for multiple choice only),
      "correct_answer": "Correct answer",
      "explanation": "Why this is correct",
      "difficulty": "Easy|Medium|Hard",
      "source": "Page/section reference"
    }}
  ]
}}

Content: {content}"""

ANSWER_EVALUATION_PROMPT = """You are a STRICT quiz answer evaluator. Your task is to determine if a student's answer is correct or acceptable.

Question: {question}
Correct Answer: {correct_answer}
Student Answer: {user_answer}
Question Type: {question_type}

STRICT EVALUATION RULES:
- For multiple_choice: Student answer must EXACTLY match the correct answer (case insensitive)
- For short_answer: Student answer must contain the key concepts from the correct answer. Be strict - partial or vague answers should be marked INCORRECT
- For fill_blank: Student answer must be the same word or a direct synonym. Do not accept unrelated words

IMPORTANT: Be STRICT in your evaluation. Only mark as CORRECT if the student clearly demonstrates knowledge of the correct answer. When in doubt, mark as INCORRECT.

Examples:
- If correct answer is "photosynthesis" and student writes "plant process", mark INCORRECT
- If correct answer is "Paris" and student writes "France", mark INCORRECT
- If correct answer is "mitochondria" and student writes "powerhouse", mark CORRECT only if it's clearly referring to the same concept

Respond with EXACTLY one word: "CORRECT" or "INCORRECT"

Evaluation:"""

# ====================== SESSION STATE INITIALIZATION ======================
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "page": "home",
        "current_session_id": None,
        "current_section": "home",
        "quiz_state": {
            "active_quiz": None,
            "current_question": 0,
            "score": 0,
            "answers": [],
            "start_time": None
        },
        "user_stats": {
            "total_questions_asked": 0,
            "total_quizzes_taken": 0,
            "average_quiz_score": 0.0,
            "documents_processed": 0
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# ====================== UTILITY FUNCTIONS ======================
def sanitize_filename(fname: str) -> str:
    """Clean filename for safe storage"""
    base = os.path.basename(fname)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)

def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    try:
        dt = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return timestamp_str

def get_library_stats():
    """Get comprehensive library statistics"""
    pdf_files = list(iter_pdf_paths())
    total_files = len(pdf_files)
    
    # Check if index exists
    index_exists = os.path.isdir(INDEX_DIR) and os.path.exists(INDEX_META)
    
    # Get total size
    total_size = 0
    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            total_size += os.path.getsize(pdf_path)
    
    total_size_mb = round(total_size / (1024 * 1024), 2)
    
    # Get index info
    index_info = load_index_meta() if index_exists else None
    
    return {
        "total_files": total_files,
        "total_size_mb": total_size_mb,
        "index_exists": index_exists,
        "index_info": index_info,
        "last_built": index_info.get("built_at") if index_info else None
    }

# ====================== SESSION MANAGEMENT ======================
def _session_path(session_id: str) -> str:
    return os.path.join(SESS_DIR, f"{session_id}.json")

def _load_session_list():
    if os.path.exists(SESS_LIST_JSON):
        try:
            with open(SESS_LIST_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_session_list(lst):
    with open(SESS_LIST_JSON, "w", encoding="utf-8") as f:
        json.dump(lst, f, indent=2, ensure_ascii=False)

def load_chat_log(session_id: str):
    path = _session_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle legacy format
            if isinstance(data, list):
                return data
            
            # New format with metadata
            return data.get("messages", [])
        except Exception:
            return []
    return []

def save_chat_log(session_id: str, chat_log, section="chat", metadata=None):
    path = _session_path(session_id)
    
    # Enhanced session format with metadata
    session_data = {
        "messages": chat_log,
        "section": section,
        "created": datetime.datetime.now().isoformat(),
        "last_modified": datetime.datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

def get_session_metadata(session_id: str):
    """Get session metadata"""
    path = _session_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("metadata", {})
        except:
            return {}
    return {}

# ====================== PDF PROCESSING ======================
def save_uploaded_pdfs(files):
    saved = []
    for f in files:
        name = sanitize_filename(f.name)
        path = os.path.join(DATA_DIR, name)
        with open(path, "wb") as w:
            w.write(f.read())
        saved.append(path)
    return saved

def iter_pdf_paths():
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.lower().endswith(".pdf"):
            yield os.path.join(DATA_DIR, fn)

def pdf_to_page_texts_from_bytes(file_bytes: bytes):
    out = []
    if USE_PYMUPDF:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc, start=1):
            txt = page.get_text("text") or ""
            out.append((txt, i))
    else:
        reader = PdfReader(io.BytesIO(file_bytes))
        for i, page in enumerate(reader.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            out.append((txt, i))
    return out

def pdf_to_page_texts_from_path(path: str):
    with open(path, "rb") as r:
        return pdf_to_page_texts_from_bytes(r.read())

def chunk_pages(pages, filename: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = []
    for text, page_no in pages:
        if not text or not text.strip():
            continue
        for chunk in splitter.split_text(text):
            docs.append(Document(
                page_content=chunk, 
                metadata={"source": filename, "page": page_no}
            ))
    return docs

def _format_docs(docs):
    lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        lines.append(f"[S{i}] {src} (p.{page})\n{d.page_content}\n\n")
    return "\n".join(lines)

# ====================== LLM AND EMBEDDING FUNCTIONS ======================
@st.cache_resource(show_spinner=False)
def get_embeddings(base_url: str, model: str):
    return OllamaEmbeddings(model=model, base_url=base_url)

@st.cache_resource(show_spinner=False)
def get_llm(base_url: str, model: str, temperature: float):
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)

def save_index_meta(embed_model: str, base_url: str):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INDEX_META, "w", encoding="utf-8") as f:
        json.dump({
            "embed_model": embed_model, 
            "base_url": base_url, 
            "built_at": datetime.datetime.now().isoformat()
        }, f)

def load_index_meta():
    if not os.path.exists(INDEX_META):
        return None
    try:
        with open(INDEX_META, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def clear_library_files_and_index():
    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)

def is_greeting(text: str) -> bool:
    """Check if text is a simple greeting"""
    t = (text or "").strip().lower()
    return t in {
        "hi", "hello", "hey", "yo", "hola", "namaste", "bonjour", "howdy", 
        "sup", "hey there", "hii", "good morning", "good afternoon", "good evening"
    } or re.fullmatch(r"(hi|hello|hey)[!. ]*", t or "") is not None

# ====================== QUIZ FUNCTIONS ======================
def generate_quiz_questions(content: str, num_questions: int = 5):
    """Generate quiz questions from content using LLM - Pure AI-based approach"""
    return create_llm_quiz(content, num_questions)

def create_llm_quiz(content: str, num_questions: int = 5):
    """Create quiz questions using purely LLM-based generation with robust error handling"""
    try:
        # Get LLM instance
        llm = get_llm(base_url, llm_model, 0.3)  # Lower temperature for more consistent JSON
        
        # Create the quiz generation prompt
        quiz_prompt = ChatPromptTemplate.from_template(QUIZ_GENERATION_PROMPT)
        
        # Generate quiz using LLM
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Generate response
                response = (quiz_prompt | llm).invoke({
                    "content": content[:4000],  # Limit content to avoid token limits
                    "num_questions": num_questions
                })
                
                # Extract content from response
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Try to parse JSON
                quiz_data = parse_quiz_json(response_text)
                if quiz_data and "questions" in quiz_data:
                    questions = quiz_data["questions"][:num_questions]
                    return [validate_question(q) for q in questions if validate_question(q)]
                
            except Exception as e:
                st.warning(f"Quiz generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # If all attempts fail, create a simple backup question set
                    return create_simple_backup_quiz(content, num_questions)
                continue
        
        # If we get here, all attempts failed
        return create_simple_backup_quiz(content, num_questions)
        
    except Exception as e:
        st.error(f"Quiz generation error: {str(e)}")
        return create_simple_backup_quiz(content, num_questions)

def parse_quiz_json(response_text: str):
    """Parse JSON from LLM response with error handling"""
    try:
        # Clean the response text
        response_text = response_text.strip()
        
        # Find JSON content between curly braces
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_text = response_text[start_idx:end_idx]
            return json.loads(json_text)
        else:
            # Try to parse the entire response
            return json.loads(response_text)
            
    except json.JSONDecodeError as e:
        # Try to fix common JSON issues
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.replace('```json', '').replace('```', '')
            return json.loads(cleaned.strip())
        except:
            st.warning(f"Failed to parse quiz JSON: {str(e)}")
            return None

def validate_question(question):
    """Validate and fix question structure"""
    if not isinstance(question, dict):
        return None
    
    # Required fields
    required_fields = ['type', 'question', 'correct_answer', 'explanation']
    for field in required_fields:
        if field not in question or not question[field]:
            return None
    
    # Set default difficulty if missing
    if 'difficulty' not in question:
        question['difficulty'] = 'Medium'
    
    # Validate question types (removed true_false)
    valid_types = ['multiple_choice', 'short_answer', 'fill_blank']
    if question['type'] not in valid_types:
        question['type'] = 'short_answer'  # Default fallback
    
    # Ensure multiple choice questions have options
    if question['type'] == 'multiple_choice':
        if 'options' not in question or not isinstance(question['options'], list) or len(question['options']) < 2:
            # Convert to short answer if options are invalid
            question['type'] = 'short_answer'
    
    return question

def create_simple_backup_quiz(content: str, num_questions: int = 3):
    """Create simple backup questions when LLM fails"""
    questions = []
    
    # Extract sentences from content
    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 50]
    if not sentences:
        sentences = [s.strip() for s in content.split('\n') if len(s.strip()) > 30]
    
    # Create simple questions (removed true_false)
    for i in range(min(num_questions, len(sentences))):
        sentence = sentences[i][:200]  # Limit sentence length
        
        if i % 2 == 0:  # Short answer
            words = sentence.split()
            if len(words) > 6:
                question_part = ' '.join(words[:int(len(words)/2)])
                answer_part = ' '.join(words[int(len(words)/2):int(len(words)/2)+3])
                questions.append({
                    "type": "short_answer",
                    "question": f"Complete this sentence: '{question_part}...'",
                    "correct_answer": answer_part,
                    "explanation": f"The completion is: '{answer_part}'",
                    "difficulty": "Medium"
                })
        else:  # Fill in blank
            words = sentence.split()
            if len(words) > 4:
                blank_idx = min(2, len(words) - 2)
                blank_word = words[blank_idx].strip('.,!?;:')
                if len(blank_word) > 3:
                    blanked_sentence = sentence.replace(words[blank_idx], "______", 1)
                    questions.append({
                        "type": "fill_blank",
                        "question": f"Fill in the blank: {blanked_sentence}",
                        "correct_answer": blank_word,
                        "explanation": f"The missing word is '{blank_word}'.",
                        "difficulty": "Easy"
                    })
    
    return questions[:num_questions] if questions else [{
        "type": "short_answer",
        "question": "What is the main topic discussed in the content?",
        "correct_answer": "Content analysis",
        "explanation": "This is a general question about the document content.",
        "difficulty": "Easy"
    }]

def evaluate_answer_with_llm(question: str, correct_answer: str, user_answer: str, question_type: str):
    """Use LLM to evaluate if the user's answer is correct"""
    try:
        # Get LLM instance with very low temperature for consistent evaluation
        llm = get_llm(base_url, llm_model, 0.05)
        
        # Create evaluation prompt
        eval_prompt = ChatPromptTemplate.from_template(ANSWER_EVALUATION_PROMPT)
        
        # Get evaluation
        response = (eval_prompt | llm).invoke({
            "question": question,
            "correct_answer": correct_answer,
            "user_answer": user_answer,
            "question_type": question_type
        })
        
        # Extract response text
        if hasattr(response, 'content'):
            evaluation = response.content.strip().upper()
        else:
            evaluation = str(response).strip().upper()
        
        # Debug logging for problematic responses
        if evaluation not in ["CORRECT", "INCORRECT"]:
            st.warning(f"LLM gave unexpected response: '{evaluation}'. Using fallback evaluation.")
            return fallback_answer_check(user_answer, correct_answer, question_type)
        
        # More strict checking - only return True if explicitly CORRECT
        return evaluation == "CORRECT"
        
    except Exception as e:
        st.warning(f"LLM evaluation failed, using fallback: {str(e)}")
        # Fallback to simple matching for critical errors
        return fallback_answer_check(user_answer, correct_answer, question_type)

def fallback_answer_check(user_answer: str, correct_answer: str, question_type: str):
    """Simple fallback answer checking when LLM fails"""
    if not user_answer or not correct_answer:
        return False
    
    user_clean = user_answer.lower().strip()
    correct_clean = correct_answer.lower().strip()
    
    if question_type == 'multiple_choice':
        return user_clean == correct_clean
    else:
        # For short answer and fill blank, use strict substring matching
        # More strict than before to avoid false positives
        return user_clean == correct_clean or (
            len(user_clean) > 2 and len(correct_clean) > 2 and (
                user_clean in correct_clean or correct_clean in user_clean
            )
        )

def save_quiz_result(session_id: str, quiz_data: dict):
    """Save quiz results for analytics"""
    quiz_file = os.path.join(QUIZ_DIR, f"{session_id}_quiz_results.json")
    results = []
    
    if os.path.exists(quiz_file):
        try:
            with open(quiz_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except:
            results = []
    
    results.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "score": quiz_data.get("score", 0),
        "total_questions": quiz_data.get("total_questions", 0),
        "time_taken": quiz_data.get("time_taken", 0),
        "difficulty": quiz_data.get("difficulty", "Medium")
    })
    
    with open(quiz_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

# ====================== SIDEBAR SETTINGS ======================
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    
    # LLM Settings
    base_url = st.text_input("🌐 Ollama Base URL", value="http://localhost:11434")
    llm_model = st.text_input("🤖 LLM Model", value="llama3.2:1b")
    embed_model = st.text_input("📊 Embedding Model", value="nomic-embed-text")
    
    # RAG Settings
    st.markdown("#### 🔍 RAG Parameters")
    top_k = st.slider("Top-K Retrieve", 1, 30, 6)
    max_dist = st.slider("Max Distance", 0.0, 2.0, 0.7, 0.05, 
                        help="Lower = closer matches")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    stream_answers = st.toggle("Stream Responses", value=True)
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        st.markdown("#### System Prompts")
        custom_qna_prompt = st.toggle("Custom QNA Prompt", value=False)
        custom_chat_prompt = st.toggle("Custom Chat Prompt", value=False)
        
        if custom_qna_prompt:
            qna_prompt_custom = st.text_area("QNA System Prompt", 
                                           value=QNA_SYSTEM_PROMPT, height=100)
        
        if custom_chat_prompt:
            chat_prompt_custom = st.text_area("Chat System Prompt", 
                                            value=CHAT_SYSTEM_PROMPT, height=100)

# ====================== MAIN HEADER ======================
st.markdown("""
<div class="app-header">
    <h1>🎓 InfoScribe 2.0</h1>
    <div class="subtitle">
        Professional RAG Suite • Upload Documents • Ask Questions • Take Quizzes • Chat with AI
    </div>
</div>
""", unsafe_allow_html=True)

# ====================== NAVIGATION ======================
st.markdown('<div class="nav-container">', unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["🏠 Home", "❓ QNA", "💬 Chat", "🎯 Quiz"],
    icons=["house", "question-circle", "chat-dots", "trophy"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0", "background-color": "transparent"},
        "icon": {"color": "#1E40AF", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "font-weight": "500",
            "text-align": "center",
            "margin": "0 4px",
            "padding": "12px 16px",
            "border-radius": "8px",
            "background-color": "transparent",
            "color": "#4B5563"
        },
        "nav-link-selected": {
            "background-color": "#1E40AF",
            "color": "white",
            "box-shadow": "0 2px 4px rgba(30, 64, 175, 0.3)"
        },
    }
)

st.markdown('</div>', unsafe_allow_html=True)

# Update session state based on selection
if selected == "🏠 Home":
    st.session_state.page = "home"
elif selected == "❓ QNA":
    st.session_state.page = "qna"
elif selected == "💬 Chat":
    st.session_state.page = "chat"
elif selected == "🎯 Quiz":
    st.session_state.page = "quiz"

# ====================== HOME SECTION ======================
if st.session_state.page == "home":
    # Dashboard Overview
    stats = get_library_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="pro-card">
            <div class="pro-card-header">
                <div class="pro-card-icon">📚</div>
                <div class="pro-card-title">Documents</div>
            </div>
            <h2 style="margin:0; color: var(--primary-600);">{stats['total_files']}</h2>
            <p style="margin:0; color: var(--gray-600);">{stats['total_size_mb']} MB</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        index_status = "✅ Ready" if stats['index_exists'] else "❌ Not Built"
        status_class = "status-success" if stats['index_exists'] else "status-error"
        st.markdown(f"""
        <div class="pro-card">
            <div class="pro-card-header">
                <div class="pro-card-icon">🔍</div>
                <div class="pro-card-title">Index Status</div>
            </div>
            <div class="status-indicator {status_class}">{index_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sessions = _load_session_list()
        st.markdown(f"""
        <div class="pro-card">
            <div class="pro-card-header">
                <div class="pro-card-icon">💬</div>
                <div class="pro-card-title">Sessions</div>
            </div>
            <h2 style="margin:0; color: var(--primary-600);">{len(sessions)}</h2>
            <p style="margin:0; color: var(--gray-600);">Active chats</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_questions = st.session_state.user_stats['total_questions_asked']
        st.markdown(f"""
        <div class="pro-card">
            <div class="pro-card-header">
                <div class="pro-card-icon">🎯</div>
                <div class="pro-card-title">Activity</div>
            </div>
            <h2 style="margin:0; color: var(--primary-600);">{total_questions}</h2>
            <p style="margin:0; color: var(--gray-600);">Questions asked</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Library Management
    st.markdown("""
    <div class="pro-card">
        <div class="pro-card-header">
            <div class="pro-card-icon">📦</div>
            <h3 class="pro-card-title">Library Management</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Build Mode Selection
    mode = st.radio("Build Mode", ["🔄 Replace (fresh build)", "➕ Append (add to existing)"], horizontal=True)
    
    # File Upload
    files = st.file_uploader("📄 Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    
    # Action Buttons
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        build_clicked = st.button("🚀 Build Library Index", type="primary", use_container_width=True)
    
    with col2:
        clear_clicked = st.button("🗑️ Clear Library", use_container_width=True)
    
    # Current Library Display
    existing = list(iter_pdf_paths())
    if existing:
        st.markdown("#### 📋 Current Library")
        for pdf_path in existing:
            filename = os.path.basename(pdf_path)
            file_size = round(os.path.getsize(pdf_path) / 1024, 1)  # KB
            
            st.markdown(f"""
            <div class="file-item">
                <div class="file-info">
                    <div class="file-icon">📄</div>
                    <div class="file-name">{filename}</div>
                </div>
                <div style="color: var(--gray-500); font-size: 0.875rem;">{file_size} KB</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Handle Actions
    if clear_clicked:
        clear_library_files_and_index()
        st.success("✅ Library cleared successfully!")
        st.rerun()
    
    if build_clicked:
        if mode.startswith("🔄"):
            clear_library_files_and_index()
        
        if files:
            save_uploaded_pdfs(files)
        
        pdf_paths = list(iter_pdf_paths())
        if not pdf_paths:
            st.warning("⚠️ No PDFs found. Please upload files first.")
        else:
            with st.spinner("🔄 Building search index..."):
                all_docs, total_pages = [], 0
                progress_bar = st.progress(0)
                
                for i, path in enumerate(pdf_paths):
                    progress_bar.progress((i + 1) / len(pdf_paths))
                    pages = pdf_to_page_texts_from_path(path)
                    total_pages += len(pages)
                    all_docs.extend(chunk_pages(pages, filename=os.path.basename(path)))
                
                if not all_docs:
                    st.error("❌ No extractable text found in PDFs.")
                    st.stop()
                
                emb = get_embeddings(base_url, embed_model)
                if os.path.isdir(INDEX_DIR):
                    shutil.rmtree(INDEX_DIR)
                vs = FAISS.from_documents(all_docs, emb)
                vs.save_local(INDEX_DIR)
                save_index_meta(embed_model, base_url)
            
            st.success(f"🎉 Library built successfully!")
            st.info(f"📊 Processed: {len(pdf_paths)} files, {total_pages} pages, {len(all_docs)} chunks")
            
            # Update user stats
            st.session_state.user_stats['documents_processed'] = len(pdf_paths)
    
    # Session Management
    st.markdown("""
    <div class="pro-card">
        <div class="pro-card-header">
            <div class="pro-card-icon">💬</div>
            <h3 class="pro-card-title">Session Management</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    sess_list = _load_session_list()
    
    # Create New Session
    col1, col2 = st.columns([3, 1])
    with col1:
        new_name = st.text_input("Session Name", placeholder="e.g., Python Programming - Chapter 3")
    with col2:
        section_type = st.selectbox("Section", ["QNA", "Chat", "Quiz"])
    
    if st.button("➕ Create New Session", type="primary"):
        if not (os.path.isdir(INDEX_DIR) and any(iter_pdf_paths())):
            st.warning("⚠️ Please build your library first.")
        else:
            new_id = str(uuid.uuid4())
            sess_list.append({
                "id": new_id,
                "name": new_name.strip() or f"Session {new_id[:8]}",
                "section": section_type.lower(),
                "created": datetime.datetime.now().isoformat()
            })
            _save_session_list(sess_list)
            st.session_state.current_session_id = new_id
            st.session_state.page = section_type.lower()
            st.success("✅ Session created!")
            st.rerun()
    
    # Existing Sessions
    if sess_list:
        st.markdown("#### 📂 Existing Sessions")
        for session in reversed(sess_list[-10:]):  # Show last 10
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{session['name']}**")
                st.caption(f"Created: {format_timestamp(session['created'])}")
            
            with col2:
                section_emoji = {"qna": "❓", "chat": "💬", "quiz": "🎯"}.get(session.get('section', 'chat'), "💬")
                st.write(f"{section_emoji} {session.get('section', 'chat').upper()}")
            
            with col3:
                if st.button("Open", key=f"open_{session['id']}", type="secondary"):
                    st.session_state.current_session_id = session['id']
                    st.session_state.page = session.get('section', 'chat')
                    st.rerun()
            
            with col4:
                if st.button("🗑️", key=f"delete_{session['id']}", help="Delete session"):
                    try:
                        os.remove(_session_path(session['id']))
                    except:
                        pass
                    sess_list = [s for s in sess_list if s["id"] != session['id']]
                    _save_session_list(sess_list)
                    st.rerun()
            
            st.divider()

# ====================== QNA SECTION ======================
elif st.session_state.page == "qna":
    st.markdown("### ❓ Questions & Answers")
    st.markdown("Ask specific questions and get precise answers based strictly on your uploaded documents.")
    
    # Check library status
    library_ready = os.path.isdir(INDEX_DIR) and any(iter_pdf_paths())
    if not library_ready:
        st.warning("📚 Your library is empty. Please go to **Home** and build your library first.")
        if st.button("🏠 Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()
    
    # Session management for QNA
    sess_list = _load_session_list()
    qna_sessions = [s for s in sess_list if s.get('section') == 'qna']
    
    if not qna_sessions:
        st.info("💡 No QNA sessions found. Create one from the Home page.")
        if st.button("🏠 Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()
    
    # Session selector
    if not st.session_state.current_session_id:
        st.session_state.current_session_id = qna_sessions[-1]['id']
    
    current_session = next((s for s in qna_sessions if s['id'] == st.session_state.current_session_id), None)
    if not current_session:
        st.session_state.current_session_id = qna_sessions[-1]['id']
        current_session = qna_sessions[-1]
    
    # Session controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        session_names = {s['name']: s['id'] for s in qna_sessions}
        selected_name = st.selectbox("Select QNA Session", options=list(session_names.keys()),
                                   index=list(session_names.values()).index(st.session_state.current_session_id))
        if session_names[selected_name] != st.session_state.current_session_id:
            st.session_state.current_session_id = session_names[selected_name]
            st.rerun()
    
    with col2:
        if st.button("🏠 Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    
    with col3:
        with st.popover("⚙️"):
            if st.button("🧹 Clear Chat", use_container_width=True):
                save_chat_log(st.session_state.current_session_id, [], "qna")
                st.rerun()
            
            if st.button("📥 Export", use_container_width=True):
                messages = load_chat_log(st.session_state.current_session_id)
                if messages:
                    md_content = f"# QNA Session: {current_session['name']}\n\n"
                    for msg in messages:
                        prefix = "**Question:**" if msg['role'] == 'user' else "**Answer:**"
                        md_content += f"{prefix} {msg['content']}\n\n"
                    
                    st.download_button("💾 Download QNA.md", 
                                     data=md_content.encode(), 
                                     file_name=f"qna_{current_session['name']}.md",
                                     mime="text/markdown")
    
    # Display chat history
    chat_log = load_chat_log(st.session_state.current_session_id)
    for msg in chat_log:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])
    
    # QNA input
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        # Add user question to chat
        with st.chat_message("user"):
            st.write(query)
        
        chat_log.append({"role": "user", "content": query})
        save_chat_log(st.session_state.current_session_id, chat_log, "qna")
        
        # Handle greeting
        if is_greeting(query):
            reply = "Hello! 👋 I'm here to answer questions based on your documents. What would you like to know?"
            with st.chat_message("assistant"):
                st.write(reply)
            chat_log.append({"role": "assistant", "content": reply})
            save_chat_log(st.session_state.current_session_id, chat_log, "qna")
            st.rerun()
        
        # Process QNA query
        with st.spinner("🔍 Searching documents..."):
            # Get embeddings and search
            emb = get_embeddings(base_url, embed_model)
            vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
            results = vs.similarity_search_with_score(query, k=max(top_k, 8))
            
            # Filter by distance
            filtered = [(d, dist) for (d, dist) in results if dist is not None and dist <= max_dist]
            
            if not filtered:
                reply = "I cannot find this information in the provided documents."
                with st.chat_message("assistant"):
                    st.write(reply)
                chat_log.append({"role": "assistant", "content": reply})
                save_chat_log(st.session_state.current_session_id, chat_log, "qna")
                st.stop()
            
            # Get relevant documents
            docs = [d for (d, _) in filtered[:top_k]]
            context = _format_docs(docs)
            
            # Generate response using strict QNA prompt
            llm = get_llm(base_url, llm_model, temperature)
            qna_prompt = ChatPromptTemplate.from_template(QNA_SYSTEM_PROMPT)
            
            if stream_answers:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    response_text = ""
                    
                    for chunk in (qna_prompt | llm).stream({"context": context, "input": query}):
                        text = getattr(chunk, "content", "") if hasattr(chunk, "content") else str(chunk)
                        if text:
                            response_text += text
                            placeholder.write(response_text)
                    
                    # Show sources
                    with st.expander("📚 Sources"):
                        for i, doc in enumerate(docs, 1):
                            meta = doc.metadata or {}
                            st.markdown(f"**Source {i}:** {meta.get('source', '?')} (Page {meta.get('page', '?')})")
                            st.markdown(f"_{doc.page_content[:200]}..._")
                            st.divider()
            else:
                result = (qna_prompt | llm).invoke({"context": context, "input": query})
                response_text = getattr(result, "content", str(result))
                
                with st.chat_message("assistant"):
                    st.write(response_text)
                    
                    # Show sources
                    with st.expander("📚 Sources"):
                        for i, doc in enumerate(docs, 1):
                            meta = doc.metadata or {}
                            st.markdown(f"**Source {i}:** {meta.get('source', '?')} (Page {meta.get('page', '?')})")
                            st.markdown(f"_{doc.page_content[:200]}..._")
                            st.divider()
            
            # Save response
            chat_log.append({"role": "assistant", "content": response_text})
            save_chat_log(st.session_state.current_session_id, chat_log, "qna")
            
            # Update stats
            st.session_state.user_stats['total_questions_asked'] += 1

# ====================== CHAT SECTION ======================
elif st.session_state.page == "chat":
    st.markdown("### 💬 Conversational AI")
    st.markdown("Have friendly, natural conversations about your documents. I'll chat with you about the content you've uploaded!")
    
    # Session management for Chat
    sess_list = _load_session_list()
    chat_sessions = [s for s in sess_list if s.get('section') == 'chat']
    
    if not chat_sessions:
        st.info("💡 No chat sessions found. Create one from the Home page.")
        if st.button("🏠 Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()
    
    # Session selector
    if not st.session_state.current_session_id:
        st.session_state.current_session_id = chat_sessions[-1]['id']
    
    current_session = next((s for s in chat_sessions if s['id'] == st.session_state.current_session_id), None)
    if not current_session:
        st.session_state.current_session_id = chat_sessions[-1]['id']
        current_session = chat_sessions[-1]
    
    # Session controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        session_names = {s['name']: s['id'] for s in chat_sessions}
        selected_name = st.selectbox("Select Chat Session", options=list(session_names.keys()),
                                   index=list(session_names.values()).index(st.session_state.current_session_id))
        if session_names[selected_name] != st.session_state.current_session_id:
            st.session_state.current_session_id = session_names[selected_name]
            st.rerun()
    
    with col2:
        if st.button("🏠 Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    
    with col3:
        with st.popover("⚙️"):
            if st.button("🧹 Clear Chat", use_container_width=True):
                save_chat_log(st.session_state.current_session_id, [], "chat")
                st.rerun()
            
            if st.button("📥 Export", use_container_width=True):
                messages = load_chat_log(st.session_state.current_session_id)
                if messages:
                    md_content = f"# Chat Session: {current_session['name']}\n\n"
                    for msg in messages:
                        prefix = "**You:**" if msg['role'] == 'user' else "**AI:**"
                        md_content += f"{prefix} {msg['content']}\n\n"
                    
                    st.download_button("💾 Download Chat.md", 
                                     data=md_content.encode(), 
                                     file_name=f"chat_{current_session['name']}.md",
                                     mime="text/markdown")
    
    # Display chat history
    chat_log = load_chat_log(st.session_state.current_session_id)
    for msg in chat_log:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])
    
    # Chat input
    query = st.chat_input("Let's have a conversation...")
    
    if query:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(query)
        
        chat_log.append({"role": "user", "content": query})
        save_chat_log(st.session_state.current_session_id, chat_log, "chat")
        
        # Process chat query
        with st.spinner("💭 Thinking..."):
            context = ""
            library_ready = os.path.isdir(INDEX_DIR) and any(iter_pdf_paths())
            
            # Get document context if available and relevant
            if library_ready:
                try:
                    emb = get_embeddings(base_url, embed_model)
                    vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
                    results = vs.similarity_search_with_score(query, k=3)  # Fewer results for chat
                    
                    relevant_docs = [(d, dist) for (d, dist) in results if dist is not None and dist <= max_dist]
                    if relevant_docs:
                        docs = [d for (d, _) in relevant_docs]
                        context = _format_docs(docs)
                except:
                    context = ""
            
            # Generate response using conversational prompt
            llm = get_llm(base_url, llm_model, temperature)
            chat_prompt = ChatPromptTemplate.from_template(CHAT_SYSTEM_PROMPT)
            
            if stream_answers:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    response_text = ""
                    
                    for chunk in (chat_prompt | llm).stream({"context": context, "input": query}):
                        text = getattr(chunk, "content", "") if hasattr(chunk, "content") else str(chunk)
                        if text:
                            response_text += text
                            placeholder.write(response_text)
                    
                    final_response = response_text
            else:
                result = (chat_prompt | llm).invoke({"context": context, "input": query})
                final_response = getattr(result, "content", str(result))
                
                with st.chat_message("assistant"):
                    st.write(final_response)
            
            # Save response
            chat_log.append({"role": "assistant", "content": final_response})
            save_chat_log(st.session_state.current_session_id, chat_log, "chat")

# ====================== QUIZ SECTION ======================
elif st.session_state.page == "quiz":
    st.markdown("### 🎯 Interactive Quiz")
    st.markdown("Test your knowledge with AI-generated quizzes based on your documents.")
    
    # Check library status
    library_ready = os.path.isdir(INDEX_DIR) and any(iter_pdf_paths())
    if not library_ready:
        st.warning("📚 Your library is empty. Please go to **Home** and build your library first.")
        if st.button("🏠 Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()
    
    # Session management for Quiz
    sess_list = _load_session_list()
    quiz_sessions = [s for s in sess_list if s.get('section') == 'quiz']
    
    if not quiz_sessions:
        st.info("💡 No quiz sessions found. Create one from the Home page.")
        if st.button("🏠 Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()
    
    # Initialize quiz state if needed
    if st.session_state.quiz_state['active_quiz'] is None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="pro-card">
                <div class="pro-card-header">
                    <div class="pro-card-icon">🚀</div>
                    <h3 class="pro-card-title" style="color: var(--gray-900) !important;">Auto-Generated Quiz</h3>
                </div>
                <p style="color: var(--gray-700) !important;">Let AI create a quiz from your entire document library.</p>
            </div>
            """, unsafe_allow_html=True)
            
            num_questions = st.slider("Number of questions", 3, 15, 5)
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
            
            if st.button("🎲 Generate Auto Quiz", type="primary", use_container_width=True):
                with st.spinner("🤖 Generating quiz questions..."):
                    # Get random content from documents
                    try:
                        emb = get_embeddings(base_url, embed_model)
                        vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
                        
                        # Get diverse content
                        sample_queries = ["main topic", "important concept", "key information", "definition", "explanation"]
                        all_docs = []
                        
                        for query in sample_queries:
                            results = vs.similarity_search(query, k=2)
                            all_docs.extend(results)
                        
                        # Create content for quiz generation
                        content = "\n\n".join([doc.page_content for doc in all_docs[:10]])
                        
                        # Generate quiz
                        quiz_questions = generate_quiz_questions(content, num_questions)
                        
                        if quiz_questions:
                            st.session_state.quiz_state = {
                                "active_quiz": quiz_questions,
                                "current_question": 0,
                                "score": 0,
                                "answers": [],
                                "start_time": datetime.datetime.now().isoformat(),
                                "total_questions": len(quiz_questions),
                                "difficulty": difficulty
                            }
                            st.rerun()
                        else:
                            st.error("Failed to generate quiz questions. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Error generating quiz: {e}")
        
        with col2:
            st.markdown("""
            <div class="pro-card">
                <div class="pro-card-header">
                    <div class="pro-card-icon">🎯</div>
                    <h3 class="pro-card-title" style="color: var(--gray-900) !important;">Topic-Specific Quiz</h3>
                </div>
                <p style="color: var(--gray-700) !important;">Request a quiz on specific topics from your documents.</p>
            </div>
            """, unsafe_allow_html=True)
            
            topic_query = st.text_input("Quiz topic", placeholder="e.g., machine learning basics, chapter 3")
            topic_questions = st.slider("Questions", 3, 10, 5, key="topic_questions")
            
            if st.button("🔍 Generate Topic Quiz", type="secondary", use_container_width=True):
                if not topic_query:
                    st.warning("Please enter a topic for the quiz.")
                else:
                    with st.spinner(f"🎯 Creating quiz about '{topic_query}'..."):
                        try:
                            emb = get_embeddings(base_url, embed_model)
                            vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
                            
                            # Search for topic-specific content
                            results = vs.similarity_search(topic_query, k=8)
                            content = "\n\n".join([doc.page_content for doc in results])
                            
                            # Generate topic-specific quiz
                            quiz_questions = generate_quiz_questions(content, topic_questions)
                            
                            if quiz_questions:
                                st.session_state.quiz_state = {
                                    "active_quiz": quiz_questions,
                                    "current_question": 0,
                                    "score": 0,
                                    "answers": [],
                                    "start_time": datetime.datetime.now().isoformat(),
                                    "total_questions": len(quiz_questions),
                                    "difficulty": "Medium",
                                    "topic": topic_query
                                }
                                st.rerun()
                            else:
                                st.error("Failed to generate quiz questions for this topic. Please try a different topic.")
                                
                        except Exception as e:
                            st.error(f"Error generating topic quiz: {e}")
    
    else:
        # Quiz is active
        quiz_data = st.session_state.quiz_state
        current_q = quiz_data['current_question']
        questions = quiz_data['active_quiz']
        total_questions = len(questions)
        
        # Progress bar
        progress = (current_q) / total_questions
        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress * 100}%"></div>
        </div>
        <div style="text-align: center; margin: 1rem 0;">
            Question {current_q + 1} of {total_questions} | Score: {quiz_data['score']}/{current_q if current_q > 0 else 1}
        </div>
        """, unsafe_allow_html=True)
        
        if current_q < len(questions):
            question = questions[current_q]
            
            # Display current question
            st.markdown(f"""
            <div class="quiz-question">
                <h3 style="color: var(--gray-900) !important; margin-bottom: 1rem;">Question {current_q + 1}</h3>
                <p style="font-size: 1.1rem; margin: 1rem 0; color: var(--gray-800) !important; line-height: 1.6;">{question['question']}</p>
                <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                    <span style="background: var(--primary-100); color: var(--primary-600); padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.875rem;">
                        {question.get('difficulty', 'Medium')}
                    </span>
                    <span style="background: var(--gray-200); color: var(--gray-700); padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.875rem;">
                        {question['type'].replace('_', ' ').title()}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Handle different question types
            user_answer = None
            
            if question['type'] == 'multiple_choice' and 'options' in question:
                user_answer = st.radio("Select your answer:", question['options'], key=f"q_{current_q}")
            
            elif question['type'] == 'short_answer':
                user_answer = st.text_input("Your answer:", key=f"q_{current_q}")
            
            elif question['type'] == 'fill_blank':
                user_answer = st.text_input("Fill in the blank:", key=f"q_{current_q}")
            
            # Submit answer button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_clicked = st.button("Submit Answer", type="primary", use_container_width=True)
            
            if submit_clicked and user_answer:
                # Use LLM-based answer evaluation for more accurate checking
                with st.spinner("🤔 Evaluating your answer..."):
                    correct = evaluate_answer_with_llm(
                        question=question['question'],
                        correct_answer=question['correct_answer'],
                        user_answer=user_answer,
                        question_type=question['type']
                    )
                
                # Update score
                if correct:
                    quiz_data['score'] += 1
                
                # Save answer
                quiz_data['answers'].append({
                    'question': question['question'],
                    'user_answer': user_answer,
                    'correct_answer': question['correct_answer'],
                    'is_correct': correct,
                    'explanation': question.get('explanation', '')
                })
                
                # Show feedback
                if correct:
                    st.markdown(f"""
                    <div class="quiz-feedback quiz-correct">
                        <h4>✅ Correct!</h4>
                        <p>{question.get('explanation', 'Good job!')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="quiz-feedback quiz-incorrect">
                        <h4>❌ Not quite right</h4>
                        <p><strong>Correct answer:</strong> {question['correct_answer']}</p>
                        <p>{question.get('explanation', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Move to next question
                quiz_data['current_question'] += 1
                st.session_state.quiz_state = quiz_data
                
                # Auto-advance after short delay
                if quiz_data['current_question'] < len(questions):
                    if st.button("Next Question ➡️", type="primary"):
                        st.rerun()
                else:
                    if st.button("View Results 🎉", type="primary"):
                        st.rerun()
        
        else:
            # Quiz completed - Show results
            quiz_data = st.session_state.quiz_state
            score = quiz_data['score']
            total = quiz_data['total_questions']
            percentage = (score / total) * 100
            
            # Calculate time taken
            start_time = datetime.datetime.fromisoformat(quiz_data['start_time'])
            end_time = datetime.datetime.now()
            time_taken = (end_time - start_time).total_seconds() / 60  # minutes
            
            # Show results
            st.markdown("## 🎉 Quiz Completed!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="pro-card" style="text-align: center;">
                    <h2 style="color: var(--primary-600); margin: 0;">{score}/{total}</h2>
                    <p style="margin: 0;">Questions Correct</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                color = "var(--success-600)" if percentage >= 80 else "var(--warning-600)" if percentage >= 60 else "var(--error-600)"
                st.markdown(f"""
                <div class="pro-card" style="text-align: center;">
                    <h2 style="color: {color}; margin: 0;">{percentage:.1f}%</h2>
                    <p style="margin: 0;">Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="pro-card" style="text-align: center;">
                    <h2 style="color: var(--gray-600); margin: 0;">{time_taken:.1f}m</h2>
                    <p style="margin: 0;">Time Taken</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance message
            if percentage >= 90:
                st.success("🌟 Excellent work! You've mastered this material!")
            elif percentage >= 80:
                st.success("👏 Great job! You have a solid understanding!")
            elif percentage >= 60:
                st.info("👍 Good effort! Consider reviewing the material.")
            else:
                st.warning("📚 Keep studying! Review the material and try again.")
            
            # Detailed results
            with st.expander("📊 Detailed Results"):
                for i, answer in enumerate(quiz_data['answers'], 1):
                    icon = "✅" if answer['is_correct'] else "❌"
                    st.markdown(f"""
                    **{icon} Question {i}:** {answer['question'][:100]}...
                    
                    **Your answer:** {answer['user_answer']}
                    
                    **Correct answer:** {answer['correct_answer']}
                    
                    **Explanation:** {answer['explanation']}
                    
                    ---
                    """)
            
            # Save quiz results
            if quiz_sessions:
                current_session_id = st.session_state.current_session_id or quiz_sessions[0]['id']
                save_quiz_result(current_session_id, {
                    "score": score,
                    "total_questions": total,
                    "percentage": percentage,
                    "time_taken": time_taken,
                    "difficulty": quiz_data.get('difficulty', 'Medium'),
                    "topic": quiz_data.get('topic', 'General')
                })
            
            # Update user stats
            st.session_state.user_stats['total_quizzes_taken'] += 1
            current_avg = st.session_state.user_stats['average_quiz_score']
            total_quizzes = st.session_state.user_stats['total_quizzes_taken']
            new_avg = ((current_avg * (total_quizzes - 1)) + percentage) / total_quizzes
            st.session_state.user_stats['average_quiz_score'] = new_avg
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Take Another Quiz", type="primary"):
                    st.session_state.quiz_state = {
                        "active_quiz": None,
                        "current_question": 0,
                        "score": 0,
                        "answers": [],
                        "start_time": None
                    }
                    st.rerun()
            
            with col2:
                if st.button("🏠 Back to Home"):
                    st.session_state.quiz_state = {
                        "active_quiz": None,
                        "current_question": 0,
                        "score": 0,
                        "answers": [],
                        "start_time": None
                    }
                    st.session_state.page = "home"
                    st.rerun()
            
            with col3:
                # Export quiz results
                results_text = f"""# Quiz Results - {quiz_data.get('topic', 'General Quiz')}

**Score:** {score}/{total} ({percentage:.1f}%)
**Time:** {time_taken:.1f} minutes
**Difficulty:** {quiz_data.get('difficulty', 'Medium')}
**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## Detailed Results

"""
                for i, answer in enumerate(quiz_data['answers'], 1):
                    icon = "✅" if answer['is_correct'] else "❌"
                    results_text += f"""### {icon} Question {i}
**Q:** {answer['question']}
**Your Answer:** {answer['user_answer']}
**Correct Answer:** {answer['correct_answer']}
**Explanation:** {answer['explanation']}

---

"""
                
                st.download_button("📥 Export Results",
                                 data=results_text.encode(),
                                 file_name=f"quiz_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                 mime="text/markdown")

# ====================== FOOTER ======================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--gray-500); padding: 2rem 0;">
    <p><strong>InfoScribe 2.0</strong> - Professional RAG Suite</p>
    <p>Powered by Ollama • Built with Streamlit • Enhanced with AI</p>
</div>
""", unsafe_allow_html=True)
                    
