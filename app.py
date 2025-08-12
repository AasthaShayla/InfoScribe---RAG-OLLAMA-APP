import os, io, shutil, json, uuid, datetime, re
import streamlit as st

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

# Here I am making my Browser , Giving it a heding and some css
st.set_page_config(page_title="Info Scribe — RAG (Ollama)", layout="wide", page_icon="💡")

st.markdown(
    """
    <style>
      .center-title {text-align:center;}
      .muted {opacity:0.85;}
      .stChatMessage {max-width: 1100px; margin-left:auto; margin-right:auto;}
      .file-row {display:flex;align-items:center;justify-content:space-between;padding:6px 10px;border:1px solid #2a2a2a;border-radius:8px;margin-bottom:6px}
      .nav {position:sticky; top:0; z-index:9; background:rgba(0,0,0,0.25); backdrop-filter: blur(6px); border-bottom:1px solid #2a2a2a; padding:8px 0;}
      .tiny {font-size: 12px; opacity: .8}
    </style>
    <div class=\"center-title\">
      <h1 style=\"margin-bottom:0.4rem;\">Info Scribe</h1>
      <p class=\"muted\" style=\"margin-top:0rem;\">Upload PDFs → Build Library → Chat (answers only from your library)</p>
    </div>
    """,
    unsafe_allow_html=True,
)


INDEX_DIR = "vs_faiss"         
INDEX_META = os.path.join(INDEX_DIR, "_meta.json")
DATA_DIR  = "library_pdfs"        
SESS_DIR  = "sessions"            
SESS_LIST_JSON = os.path.join(SESS_DIR, "_sessions.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SESS_DIR, exist_ok=True)

# This is again the part of UI the side bar settings , the parameter settings which can be tweeked by the user 
# The importance of these parameters I have mentioned in my supporting PDF file.
with st.sidebar:
    st.header("⚙️ Settings")
    base_url   = st.text_input("Ollama Base URL", value="http://localhost:11434")
    llm_model  = st.text_input("LLM model", value="llama3.2:1b")  
    embed_model= st.text_input("Embedding model", value="nomic-embed-text")

    top_k      = st.slider("Top‑K retrieve", 1, 30, 6)
    max_dist   = st.slider(
        "Max distance (lower = closer)", 0.0, 2.0, 0.7, 0.05,
        help="We use FAISS raw distances. Chunks with distance above this are ignored."
    )
    temperature= st.slider("LLM temperature", 0.0, 1.0, 0.1, 0.1)
    stream_answers = st.toggle("Stream responses", value=True) # copying a bit of iterfaces used in various other platforms

   # Here I am making my system response strict enough
    with st.expander("Prompt"): 
        default_prompt = (
            "You are a helpful study assistant."
            "STRONG RULES (never break):"
            "- Answer ONLY using the text in the section labeled 'Context'."
            "- If the answer is not contained in Context, respond EXACTLY: \"I don't know from the provided documents.\""
            "- Do not use prior conversation, world knowledge, or guesses."
            "- Do not invent citations or facts."
            "When answering from Context, keep it concise and stick to facts quoted or clearly implied by Context."
        )
        system_prompt = st.text_area("System prompt", default_prompt, height=180)


###################### Helper functions for session.
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

# chatlog I am saving in sessions folder made the dir = SESS_DIR (initialized above)
def load_chat_log(session_id: str):
    path = _session_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out = []
            for m in data:
                role = m.get("role") or m.get("type") or "assistant"
                content = m.get("content", "")
                if role in ("user", "human"): role = "user"
                elif role in ("ai", "assistant"): role = "assistant"
                out.append({"role": role, "content": content})
            return out
        except Exception:
            return []
    return []


def save_chat_log(session_id: str, chat_log):
    path = _session_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chat_log, f, indent=2, ensure_ascii=False)


if "page" not in st.session_state:
    st.session_state.page = "home"
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None


# In case filename throws an error , now it will not because of this function
def sanitize_filename(fname: str) -> str:
    base = os.path.basename(fname)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)

# Uploading the files in the Dir initialized above = DATA_DIR
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

# Below function will return (text , page_number) tupples
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

                      ########## Chunking Pages into Documents ##########################
# Below function splits text into 1500 char with 150 overlaps for retrieval
def chunk_pages(pages, filename: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=150,
        separators=["", "", " ", ""]
    )
    docs = []
    for text, page_no in pages:
        if not text or not text.strip():
            continue
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": filename, "page": page_no}))
    return docs

# Below function will then concatenate selected chunks with numbered souce tags for LLM input
def _format_docs(docs):
    lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "?"); page = meta.get("page", "?")
        lines.append(f"[S{i}] {src} (p.{page}){d.page_content}")
    return "".join(lines)


@st.cache_resource(show_spinner=False)
def get_embeddings(base_url: str, model: str):
    return OllamaEmbeddings(model=model, base_url=base_url)


@st.cache_resource(show_spinner=False)
def get_llm(base_url: str, model: str, temperature: float):
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def clear_library_files_and_index():
    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)

# Below function to optimize and skip RAG lookup for Greetings
def is_greeting(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {
        "hi","hello","hey","yo","hola","namaste","bonjour","howdy","sup","hey there","hii"
    } or re.fullmatch(r"(hi|hello|hey)[!. ]*", t or "") is not None


def save_index_meta(embed_model: str, base_url: str):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INDEX_META, "w", encoding="utf-8") as f:
        json.dump({"embed_model": embed_model, "base_url": base_url, "built_at": datetime.datetime.now().isoformat()}, f)


def load_index_meta():
    if not os.path.exists(INDEX_META):
        return None
    try:
        with open(INDEX_META, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

                     ######### Below again is the ui code of sticky navbar where I have made 2 tabs hOME TAB and Chat tab user can navigate between them
with st.container():
    st.markdown('<div class="nav">', unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 10])
    if c1.button("🏠 Home", type="secondary"):
        st.session_state.page = "home"; st.rerun()
    if c2.button("💬 Chat", type="secondary"):
        st.session_state.page = "chat"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


if st.session_state.page == "home":
    st.markdown("### Build Library")

    mode = st.radio("Build mode", ["Replace (fresh)", "Append (add)"], horizontal=True)
    files = st.file_uploader("Add PDF files", type=["pdf"], accept_multiple_files=True)

    colA, colB = st.columns([1, 1])
    build_clicked = colA.button("📦 Build Library (index)")
    clear_clicked = colB.button("🧹 Clear Library") # clears the pdf from ui and the FAISS

    existing = list(iter_pdf_paths())
    if existing:
        with st.expander(f"📄 {len(existing)} PDF(s) in Library"):
            for p in existing:
                st.markdown(f"<div class='file-row'>📄 {os.path.basename(p)}</div>", unsafe_allow_html=True)

    if clear_clicked:
        clear_library_files_and_index()
        st.toast("Library cleared (PDFs + index).", icon="🧽")
        st.rerun()

    if build_clicked:
        if mode.startswith("Replace"):
            clear_library_files_and_index()
        if files:
            save_uploaded_pdfs(files)

        pdf_paths = list(iter_pdf_paths())
        if not pdf_paths:
            st.warning("No PDFs found. Upload first.")
        else:
            with st.spinner("Indexing your documents…"):
                all_docs, total_pages = [], 0
                for path in pdf_paths:
                    pages = pdf_to_page_texts_from_path(path)
                    total_pages += len(pages)
                    all_docs.extend(chunk_pages(pages, filename=os.path.basename(path)))

                if not all_docs:
                    st.error("No extractable text found in PDFs."); st.stop()

                emb = get_embeddings(base_url, embed_model)
                if os.path.isdir(INDEX_DIR):
                    shutil.rmtree(INDEX_DIR)
                vs = FAISS.from_documents(all_docs, emb)
                vs.save_local(INDEX_DIR)
                save_index_meta(embed_model, base_url)

            st.success(f"Library built: {len(pdf_paths)} file(s), {total_pages} page(s), {len(all_docs)} chunks.")
            st.toast("Library built successfully 🎉", icon="✅")

    st.divider()

    st.markdown("### 🗂️ Start or open a session")
    sess_list = _load_session_list()

    new_name = st.text_input("New session name", placeholder="e.g., Quantum — Chapter 2")
    if st.button("Create & Open"):
        if not (os.path.isdir(INDEX_DIR) and any(iter_pdf_paths())):
            st.warning("Build your Library first.")
        else:
            new_id = str(uuid.uuid4())
            sess_list.append({
                "id": new_id,
                "name": new_name.strip() or f"Session {new_id[:6]}",
                "created": datetime.datetime.now().isoformat()
            })
            _save_session_list(sess_list)
            st.session_state.current_session_id = new_id
            st.session_state.page = "chat"
            st.toast("Session created.", icon="🆕"); st.rerun()

    if sess_list:
        name_to_id = {s["name"]: s["id"] for s in sess_list}
        open_name = st.selectbox("Open existing session", options=list(name_to_id.keys()))
        col1, col2 = st.columns([1,1])
        if col1.button("➡️ Open in Chat"):
            st.session_state.current_session_id = name_to_id[open_name]
            st.session_state.page = "chat"; st.rerun()
        if col2.button("🗑️ Delete Session"):
            sid = name_to_id[open_name]
           
            try:
                os.remove(_session_path(sid))
            except Exception:
                pass
            sess_list = [s for s in sess_list if s["id"] != sid]
            _save_session_list(sess_list)
            st.toast("Session deleted.", icon="🗑️"); st.rerun()


if st.session_state.page == "chat":
    st.markdown("### 💬 Chat")

 # 2 message propmt template with stricter system message and a human message
    llm = get_llm(base_url, llm_model, temperature)
    # I have used chat prompt template to make the template as suggested in the training 
    prompt = ChatPromptTemplate.from_messages([("system", "{sys}Context:{context}"),
        ("human", "{input}"),
    ])
   
    # chain = prompt | llm  # will use LCEL directly later


    library_ready = os.path.isdir(INDEX_DIR) and any(iter_pdf_paths())
    if not library_ready:
        st.warning("Your Library is empty. Go to Home and build it first.", icon="🧭")


    meta = load_index_meta()
    if meta:
        if (meta.get("embed_model") != embed_model) or (meta.get("base_url") != base_url):
            st.info(
                f"⚠️ Index built with embed_model=**{meta.get('embed_model')}** @ {meta.get('base_url')}."
                f"You're now set to **{embed_model}** @ {base_url}. Rebuild the index to avoid bad matches.",
                icon="⚠️"
            )
 # bELOW adds and deletes the session the conversion .json of that perticular session is deleted
    sess_list = _load_session_list()
    if not sess_list:
        st.info("No sessions yet. Create one from Home.")
    else:
        id_to_name = {s["id"]: s["name"] for s in sess_list}
        if not st.session_state.current_session_id:
            st.session_state.current_session_id = sess_list[-1]["id"] 
        current_id = st.session_state.current_session_id
        current_name = id_to_name.get(current_id, "(unnamed)")

        col_left, col_mid, col_right = st.columns([1.2, 3, 2])
        with col_left:
            if st.button("⬅️ Back to Home", type="secondary"):
                st.session_state.page = "home"; st.rerun()

        with col_mid:
            name_to_id = {s["name"]: s["id"] for s in sess_list}
            sel_name = st.selectbox(
                "Session",
                options=list(name_to_id.keys()),
                index=list(name_to_id.keys()).index(current_name) if current_name in name_to_id else 0
            )
            if name_to_id[sel_name] != current_id:
                st.session_state.current_session_id = name_to_id[sel_name]
                st.rerun()
           # Utilities Drop down as mentioned in my pdf allows user to get the conversation download as chat.md or clear the chat that will remove the history
        with col_right:
            with st.popover("⋯ Utilities"):
                st.caption("Chat tools")
                if st.button("🧹 Clear Chat", use_container_width=True):
                    save_chat_log(current_id, [])
                    st.toast("Session messages cleared.", icon="🧼"); st.rerun()
                if st.button("⬇️ Export Chat", use_container_width=True):
                    msgs = load_chat_log(current_id)
                    if msgs:
                        md_lines = [f"# Chat Transcript — {id_to_name.get(current_id, current_id)}", ""]
                        for m in msgs:
                            prefix = "**You:**" if m["role"] == "user" else "**Assistant:**"
                            md_lines.append(f"{prefix} {m['content']}"); md_lines.append("")
                        md_text = "".join(md_lines)
                        st.download_button("Download chat.md", data=md_text.encode("utf-8"), file_name="chat.md", mime="text/markdown")
                    else:
                        st.info("No messages yet.")

        chat_log = load_chat_log(current_id)
        for m in chat_log:
            st.chat_message("user" if m["role"] == "user" else "assistant").write(m["content"])


        query = st.chat_input("Type your question here…")
        if query:
            st.chat_message("user").write(query)
            chat_log.append({"role": "user", "content": query})
            save_chat_log(current_id, chat_log)

# If pure greeting skip the rag lookup 
            if is_greeting(query):
                reply = "Hello! 👋 How can I help you today?"
                st.chat_message("assistant").write(reply)
                chat_log.append({"role": "assistant", "content": reply})
                save_chat_log(current_id, chat_log)
                st.stop()


            status = st.empty()
            status.markdown("**🤔 Thinking...**")

            if not library_ready:
                reply = "Please build your Library first (Home tab) and then ask again."
                status.empty()
                st.chat_message("assistant").write(reply)
                chat_log.append({"role": "assistant", "content": reply})
                save_chat_log(current_id, chat_log)
                st.stop()

            emb = get_embeddings(base_url, embed_model)
            vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
            results = vs.similarity_search_with_score(query, k=max(top_k, 8))  

            filtered = [(d, dist) for (d, dist) in results if dist is not None and dist <= max_dist]

            if not filtered:
                status.empty()
                reply = "I don't know from the provided documents."
                st.chat_message("assistant").write(reply)
                chat_log.append({"role": "assistant", "content": reply})
                save_chat_log(current_id, chat_log)
                st.stop()

            docs = [d for (d, _) in filtered[:top_k]]
            context = _format_docs(docs)
 # 2 modes Streaming is turned on and its off (non-streaming)
            if stream_answers:
                placeholder = st.chat_message("assistant").empty()
                acc = []
                # Here I am using LCEL
                for chunk in (prompt | llm).stream({"sys": system_prompt, "input": query, "context": context}):
                    text = getattr(chunk, "content", "") if hasattr(chunk, "content") else str(chunk)
                    if text:
                        acc.append(text)
                        placeholder.write("".join(acc))
                final_answer = "".join(acc).strip()
            else:
                result = (prompt | llm).invoke({"sys": system_prompt, "input": query, "context": context})
                final_answer = (getattr(result, "content", "") if hasattr(result, "content") else str(result)).strip()
                st.chat_message("assistant").write(final_answer)

            status.empty()

            if not final_answer:
                final_answer = "I don't know from the provided documents."

            with st.expander("📎 Sources used"):
                for i, d in enumerate(docs, 1):
                    m = d.metadata or {}
                    st.markdown(f"**S{i}** — {m.get('source','?')} (p.{m.get('page','?')})")
                    st.markdown(f"<div class='tiny'>{d.page_content[:500]}...</div>", unsafe_allow_html=True)
                    st.markdown("—")

            chat_log.append({"role": "assistant", "content": final_answer})
            save_chat_log(current_id, chat_log)
