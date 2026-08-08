import json
import os
import time
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from groq import Groq
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

# =====================================================================
# 1. APPLICATION SETUP & PERMANENT MEMORY (LOCAL DISK PERSISTENCE)
# =====================================================================
st.set_page_config(
    page_title="Fast Project Finance AI Legal Redliner",
    page_icon="⚡",
    layout="wide",
)

PERSIST_DIR = os.path.abspath(os.path.join(os.getcwd(), "chroma_permanent_db"))
os.makedirs(PERSIST_DIR, exist_ok=True)

@st.cache_resource
def get_chroma_client():
    """Initializes and caches the persistent ChromaDB client pointing to local disk storage."""
    return chromadb.PersistentClient(path=PERSIST_DIR)

def get_collection():
    """Retrieves or creates the collection from the persistent disk client."""
    client = get_chroma_client()
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="hybrid_repository_playbook",
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# =====================================================================
# 2. WORD DOCUMENT PARSING & WORD-BY-WORD TRACK CHANGES GENERATOR
# =====================================================================

def extract_paragraphs_from_docx(docx_file):
    doc = Document(docx_file)
    paragraphs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text and len(text.split()) >= 4:
            paragraphs.append(text)
    return paragraphs

def create_redlined_docx(paragraph_pairs, author="AI Legal Agent"):
    """
    Generates DOCX with native track changes at the word level using difflib.
    Shows surgical inline deletions and insertions rather than full sentence replacements.
    """
    doc = Document()
    
    for orig_text, new_text, explanation, is_acceptable in paragraph_pairs:
        p = doc.add_paragraph()
        
        if is_acceptable or orig_text == new_text:
            p.add_run(orig_text)
        else:
            orig_words = orig_text.split()
            new_words = new_text.split()
            
            matcher = difflib.SequenceMatcher(None, orig_words, new_words)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    p.add_run(" " + " ".join(orig_words[i1:i2]))
                elif tag == 'delete':
                    del_text = " " + " ".join(orig_words[i1:i2])
                    clean_del = del_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    del_xml = (
                        f'<w:del {nsdecls("w")} w:author="{author}">'
                        f'<w:r><w:delText>{clean_del}</w:delText></w:r>'
                        f'</w:del>'
                    )
                    p._p.append(parse_xml(del_xml))
                elif tag == 'insert':
                    ins_text = " " + " ".join(new_words[j1:j2])
                    clean_ins = ins_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    ins_xml = (
                        f'<w:ins {nsdecls("w")} w:author="{author}">'
                        f'<w:r><w:t>{clean_ins}</w:t></w:r>'
                        f'</w:ins>'
                    )
                    p._p.append(parse_xml(ins_xml))
                elif tag == 'replace':
                    del_text = " " + " ".join(orig_words[i1:i2])
                    clean_del = del_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    del_xml = (
                        f'<w:del {nsdecls("w")} w:author="{author}">'
                        f'<w:r><w:delText>{clean_del}</w:delText></w:r>'
                        f'</w:del>'
                    )
                    p._p.append(parse_xml(del_xml))
                    
                    ins_text = " " + " ".join(new_words[j1:j2])
                    clean_ins = ins_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    ins_xml = (
                        f'<w:ins {nsdecls("w")} w:author="{author}">'
                        f'<w:r><w:t>{clean_ins}</w:t></w:r>'
                        f'</w:ins>'
                    )
                    p._p.append(parse_xml(ins_xml))

            # Add explanatory note as a formatted sub-paragraph
            comment_p = doc.add_paragraph()
            run = comment_p.add_run(f" 💡 [AI Legal Note: {explanation}]")
            run.font.italic = True
            run.font.size = 10

    output_path = "Redlined_Facility_Agreement.docx"
    doc.save(output_path)
    return output_path

# =====================================================================
# 3. HIGH-SPEED VECTOR RETRIEVAL & DUAL-KEY LLM ANALYSIS ENGINE
# =====================================================================

def batch_query_worker(collection, chunk_paras, chunk_start_idx):
    """Executes a single ChromaDB query batch on a thread without UI code."""
    matches = collection.query(
        query_texts=chunk_paras,
        n_results=1,
        include=["documents", "metadatas"]
    )
    
    batch_results = []
    for idx, p_text in enumerate(chunk_paras):
        ctx = "NO DIRECT REPOSITORY PRECEDENT FOUND."
        if matches and matches.get("documents") and matches["documents"][idx]:
            doc_str = matches["documents"][idx][0]
            src = matches["metadatas"][idx][0].get("source", "Repo")
            ctx = f"Precedent from [{src}]: \"{doc_str}\""
        
        batch_results.append({
            "id": chunk_start_idx + idx,
            "clause": p_text,
            "context": ctx
        })
    return batch_results, len(chunk_paras)


def run_fast_parallel_retrieval(collection, paragraphs, batch_size=25, max_workers=4, status_placeholder=None, progress_bar=None, log_area=None, logs_list=None):
    """
    Executes parallel retrieval and safely updates Streamlit UI on the MAIN thread as futures resolve.
    """
    total = len(paragraphs)
    prepared_items = [None] * total
    
    chunks = [
        (paragraphs[i : i + batch_size], i) 
        for i in range(0, total, batch_size)
    ]
    
    completed_clauses = 0
    total_chunks = len(chunks)
    completed_chunks = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(batch_query_worker, collection, chunk, start_idx): (chunk, start_idx)
            for chunk, start_idx in chunks
        }
        
        for future in as_completed(futures):
            batch_data, chunk_len = future.result()
            completed_clauses += chunk_len
            completed_chunks += 1
            
            for item in batch_data:
                prepared_items[item["id"]] = item

            # Update Streamlit UI safely from the MAIN thread
            if status_placeholder and progress_bar:
                pct = int((completed_clauses / total) * 20)
                progress_bar.progress(pct)
                status_placeholder.text(f"🔍 Parallel Search: Processed {completed_clauses}/{total} clauses ({completed_chunks}/{total_chunks} thread batches)...")
                
            if log_area and logs_list is not None:
                log_msg = f"[{time.strftime('%H:%M:%S')}] 🧵 Thread Batch {completed_chunks}/{total_chunks} retrieved ({completed_clauses}/{total} clauses)"
                logs_list.append(log_msg)
                log_area.text("\n".join(logs_list[-12:]))

    return prepared_items


def analyze_clause_batch(batch_items, custom_instruction, primary_key, secondary_key, model_strategy):
    """
    Analyzes clause batches using dual API key load balancing and backoff.
    """
    api_keys = [k for k in [primary_key, secondary_key] if k and k.strip()]
    if not api_keys:
        return []

    system_prompt = """
    You are an expert Legal & Contract Analyst performing minimal, surgical contract redlining.

    CRITICAL INSTRUCTION FOR REDLINING:
    - DO NOT REPHRASE OR REWRITE THE ENGLISH LANGUAGE OF THE CLAUSE.
    - PRESERVE ORIGINAL WORDING, SENTENCE STRUCTURE, AND PUNCTUATION AS MUCH AS POSSIBLE.
    - ONLY edit specific key terms (e.g., financial ratios, grace periods, party obligations, thresholds, covenants) to align the LEGAL POSITION with the provided PRECEDENT.
    - If a clause is acceptable or differs only in style/phrasing, MARK IT AS ACCEPTABLE and leave 'proposed_text' identical to the original clause.
    - If changes are needed, apply SURGICAL EDITS (changing only the specific words/numbers necessary) so that track changes in MS Word highlight only tiny, precise edits rather than deleting the entire sentence.

    EVALUATION RULES FOR EACH ITEM:
    1. Compare 'clause' against 'context' (Repository Precedent).
    2. If ACCEPTABLE or substantially equivalent in legal effect:
       - 'is_acceptable': true
       - 'proposed_text': EXACT ORIGINAL CLAUSE TEXT (do not change a single word)
       - 'explanation': "Matches repository precedent position."
    3. If UNACCEPTABLE / NEEDS LEGAL ADJUSTMENT:
       - 'is_acceptable': false
       - 'proposed_text': Original clause with MINIMAL target edits (change only numbers, durations, covenants, or legal terms to match precedent position).
       - 'explanation': Brief reason for the position change.
    4. If NO PRECEDENT IN CONTEXT: Evaluate against standard market project finance practice using the same minimal edit rules.

    YOU MUST RETURN A STRICT JSON OBJECT WITH A SINGLE KEY "results" CONTAINING AN ARRAY matching the order of input items:
    {
      "results": [
        {
          "id": 1,
          "is_acceptable": boolean,
          "proposed_text": "string",
          "explanation": "string"
        }
      ]
    }
    """
    
    formatted_input = []
    for item in batch_items:
        formatted_input.append({
            "id": item["id"],
            "clause": item["clause"],
            "repository_context": item["context"]
        })

    user_prompt = f"""
    DEAL-SPECIFIC OVERRIDE DIRECTIVE:
    {custom_instruction if custom_instruction else "None provided. Rely on repository precedent contexts."}

    REMINDER: PRESERVE ORIGINAL WORDING. Only change the specific words/values necessary to adjust the legal position.

    CLAUSES TO ANALYZE IN BATCH:
    {json.dumps(formatted_input, indent=2)}
    """

    if "llama-3.1-8b-instant" in model_strategy:
        models_to_try = ["llama-3.1-8b-instant"]
    else:
        models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    for key_idx, current_key in enumerate(api_keys):
        client = Groq(api_key=current_key)
        
        for model_name in models_to_try:
            for attempt in range(5):  # Up to 5 retries with exponential pause
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.0
                    )
                    data = json.loads(response.choices[0].message.content)
                    return data.get("results", [])
                    
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "rate_limit_exceeded" in err_msg:
                        if len(api_keys) > 1 and key_idx == 0:
                            print(f"⚠️ [Key 1 Rate Limit on {model_name}]: Switching to Secondary Key...", flush=True)
                            break  # Move to key 2 immediately
                        else:
                            wait_time = (attempt + 1) * 5  # Pauses for 5s, 10s, 15s, 20s, 25s
                            print(f"⚠️ [Rate Limit Hit on {model_name}]: Retrying in {wait_time}s...", flush=True)
                            time.sleep(wait_time)
                            continue
                    else:
                        print(f"❌ [Groq API Error]: {err_msg}", flush=True)
                        time.sleep(2)
                        break

    # Fallback if both keys and retries fail
    fallback_results = []
    for item in batch_items:
        fallback_results.append({
            "id": item["id"],
            "is_acceptable": True,
            "proposed_text": item["clause"],
            "explanation": "Skipped due to API rate limit limits."
        })
    return fallback_results

# =====================================================================
# 4. STREAMLIT UI & TABBED INTERFACE
# =====================================================================

st.title("⚡ Project Finance AI Legal Agent (Persistent Memory)")
st.caption("Surgical Legal Redliner with Multi-Threaded Retrieval & Dual-Key Load Balancing")

with st.sidebar:
    st.header("🔑 Configuration")
    primary_api_key = st.text_input("Primary Groq API Key", type="password")
    secondary_api_key = st.text_input(
        "Secondary Groq API Key (Optional Backup)", 
        type="password", 
        help="Adding a 2nd key doubles your rate limits and prevents rate-limit hangs."
    )
    
    st.divider()
    st.header("⚙️ Model Strategy")
    selected_model_mode = st.radio(
        "Select Processing Strategy:",
        options=[
            "Fastest & Highest Limit (llama-3.1-8b-instant)", 
            "High Accuracy (llama-3.3-70b-versatile with 8b Fallback)"
        ],
        index=0,
        help="Selecting 8B Instant directly avoids 70B daily quota blocks."
    )
    
    st.divider()
    st.header("🗄️ Permanent Repository")
    
    current_count = get_collection().count()
    st.metric("Indexed Clauses Saved on Disk", current_count)
    st.caption(f"💾 Storage Path:\n`{PERSIST_DIR}`")
    
    if st.button("🔄 Refresh Count"):
        st.rerun()
        
    if st.button("🗑️ Clear Permanent Memory", type="secondary"):
        client = get_chroma_client()
        client.delete_collection("hybrid_repository_playbook")
        st.success("Disk memory erased!")
        st.rerun()

tab_redline, tab_repository, tab_chat = st.tabs([
    "🚀 High-Speed Redliner", 
    "📚 Upload Repository Precedents", 
    "💬 Smart Chat Assistant"
])

# ---------------------------------------------------------------------
# TAB 1: HIGH-SPEED SURGICAL REDLINER
# ---------------------------------------------------------------------
with tab_redline:
    st.header("Analyze & Redline Facility Agreement")
    
    uploaded_draft = st.file_uploader("Upload New Document (.docx)", type=["docx"], key="target_doc")
    
    custom_instruction = st.text_area(
        "Optional Deal Directives / Overrides",
        placeholder="e.g., 'Ensure minimum DSCR covenant is set to 1.25x'."
    )
    
    if st.button("⚡ Fast Batch Redline Document", type="primary"):
        collection = get_collection()
        if not primary_api_key:
            st.error("Please enter at least your Primary Groq API Key in the sidebar.")
        elif not uploaded_draft:
            st.error("Please upload a .docx document.")
        elif collection.count() == 0:
            st.warning("Your permanent repository on disk is empty. Upload precedents in Tab 2 first.")
        else:
            paragraphs = extract_paragraphs_from_docx(uploaded_draft)
            total_paragraphs = len(paragraphs)
            st.info(f"Loaded document: {total_paragraphs} clauses identified.")
            
            start_time = time.time()
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            debug_box = st.expander("🔍 Live Batch Processing Logs", expanded=True)
            log_area = debug_box.empty()
            logs = []

            print(f"\n================ STARTING REDLINE PROCESS ({total_paragraphs} Clauses) ================", flush=True)
            
            # STEP 1: Multi-Threaded Local Vector Search (Main thread handles UI updates)
            logs.append(f"[{time.strftime('%H:%M:%S')}] ⚡ Launching 4 CPU threads for vector retrieval...")
            log_area.text("\n".join(logs))
            status_placeholder.text("Running thread-safe disk vector search...")
            
            vec_start = time.time()
            prepared_items = run_fast_parallel_retrieval(
                collection=collection, 
                paragraphs=paragraphs, 
                batch_size=25, 
                max_workers=4,
                status_placeholder=status_placeholder,
                progress_bar=progress_bar,
                log_area=log_area,
                logs_list=logs
            )
            vec_elapsed = round(time.time() - vec_start, 2)
            
            progress_bar.progress(20)
            logs.append(f"[{time.strftime('%H:%M:%S')}] ✅ Vector search complete in {vec_elapsed}s! Beginning LLM analysis...")
            log_area.text("\n".join(logs[-12:]))

            # STEP 2: Batched LLM Analysis with Throttle Pace
            BATCH_SIZE = 10  # 10 clauses per batch balances tokens and requests per minute
            redline_results = []
            total_items = len(prepared_items)
            
            for i in range(0, total_items, BATCH_SIZE):
                batch = prepared_items[i : i + BATCH_SIZE]
                current_c = min(i + BATCH_SIZE, total_items)
                
                batch_evals = analyze_clause_batch(
                    batch_items=batch, 
                    custom_instruction=custom_instruction, 
                    primary_key=primary_api_key, 
                    secondary_key=secondary_api_key, 
                    model_strategy=selected_model_mode
                )
                
                eval_map = {res["id"]: res for res in batch_evals}
                for item in batch:
                    res = eval_map.get(item["id"], {
                        "is_acceptable": True,
                        "proposed_text": item["clause"],
                        "explanation": "Matches repository precedent position."
                    })
                    
                    redline_results.append((
                        item["clause"],
                        res.get("proposed_text", item["clause"]),
                        res.get("explanation", "Matches repository precedent position."),
                        res.get("is_acceptable", True)
                    ))
                
                # Small 0.5s pause between API batches to keep RPM under control
                time.sleep(0.5)
                
                pct = 20 + int((current_c / total_items) * 80)
                progress_bar.progress(pct)
                status_placeholder.text(f"Analyzed {current_c}/{total_items} clauses ({pct}%)...")
                
                log_msg = f"[{time.strftime('%H:%M:%S')}] ✅ Processed Clauses {i+1} to {current_c} of {total_items} ({pct}%)"
                logs.append(log_msg)
                print(log_msg, flush=True)
                log_area.text("\n".join(logs[-12:]))
            
            elapsed = round(time.time() - start_time, 2)
            completion_msg = f"⚡ Completed redlining in {elapsed} seconds! Generating Word file..."
            print(f"================ {completion_msg} ================\n", flush=True)
            st.success(completion_msg)
            
            output_docx = create_redlined_docx(redline_results)
            with open(output_docx, "rb") as file_data:
                st.download_button(
                    label="📥 Download Redlined Document (.docx Track Changes)",
                    data=file_data,
                    file_name="Redlined_Facility_Agreement.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

# ---------------------------------------------------------------------
# TAB 2: REPOSITORY INDEXING
# ---------------------------------------------------------------------
with tab_repository:
    st.header("Upload Precedent Facility Agreements")
    st.info(f"Documents uploaded here are saved permanently on disk inside:\n`{PERSIST_DIR}`")
    
    precedent_files = st.file_uploader("Upload Agreements / Playbooks (.docx)", type=["docx"], accept_multiple_files=True)
    
    if precedent_files and st.button("📥 Save Permanently to Disk Repository"):
        collection = get_collection()
        prog_bar = st.progress(0)
        status_box = st.empty()
        
        debug_container = st.expander("🛠️ Real-Time Indexing Logs", expanded=True)
        log_console = debug_container.empty()
        logs = []
        
        total_files = len(precedent_files)
        total_indexed_clauses = 0
        
        for file_idx, file in enumerate(precedent_files):
            paragraphs = extract_paragraphs_from_docx(file)
            total_paras = len(paragraphs)
            
            if total_paras == 0:
                continue
                
            batch_size = 25
            for i in range(0, total_paras, batch_size):
                batch = paragraphs[i:i + batch_size]
                ids = [f"{file.name}_p{i+j}_{int(time.time()*1000)}" for j in range(len(batch))]
                metadatas = [{"source": file.name, "chunk": i+j} for j in range(len(batch))]
                
                collection.add(documents=batch, ids=ids, metadatas=metadatas)
                total_indexed_clauses += len(batch)
                
                current_clause_num = min(i + batch_size, total_paras)
                overall_pct = ((file_idx + (current_clause_num / total_paras)) / total_files)
                prog_bar.progress(min(overall_pct, 1.0))
                status_box.text(f"Indexing '{file.name}': Clause {current_clause_num}/{total_paras}...")
                
                log_line = f"[{time.strftime('%H:%M:%S')}] Saved to disk: Clauses {i+1} to {current_clause_num} from '{file.name}'"
                logs.append(log_line)
                print(log_line, flush=True)
                log_console.text("\n".join(logs[-12:]))
                
        st.success(f"Successfully saved {total_indexed_clauses} clauses across {total_files} file(s) permanently to disk!")
        time.sleep(1)
        st.rerun()

# ---------------------------------------------------------------------
# TAB 3: CHAT ASSISTANT
# ---------------------------------------------------------------------
with tab_chat:
    st.header("Project Finance Chat Assistant")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Ask me any question regarding your repository or general project finance practices."}
        ]
    
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if user_query := st.chat_input("Ask a question..."):
        collection = get_collection()
        active_key = primary_api_key or secondary_api_key
        if not active_key:
            st.error("Please enter a Groq API Key in the sidebar.")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
                
            context_str = "NO DOCUMENTS IN REPOSITORY."
            if collection.count() > 0:
                relevant_docs = collection.query(query_texts=[user_query], n_results=5, include=["documents", "metadatas"])
                if relevant_docs["documents"] and relevant_docs["documents"][0]:
                    context_blocks = []
                    for doc_text, meta in zip(relevant_docs["documents"][0], relevant_docs["metadatas"][0]):
                        source = meta.get("source", "Unknown Document")
                        context_blocks.append(f"From Precedent File [{source}]:\n\"{doc_text}\"")
                    context_str = "\n\n".join(context_blocks)

            chat_system_prompt = f"""
            You are an expert Project Finance Legal Assistant.
            
            1. REPOSITORY CHECK FIRST:
               - Examine REPOSITORY CONTEXT.
               - IF FOUND: Answer directly and cite the source file (e.g., "According to [Filename.docx]...").
                 
            2. FALLBACK TO GENERAL MARKET PRACTICE:
               - IF NOT FOUND: Begin response with:
                 "Your uploaded documents do not contain specific terms regarding this topic. However, in general project finance market practice, the following is typically followed:"
                 Then explain general market practice.

            RETRIEVED REPOSITORY CONTEXT:
            {context_str}
            """
            
            client = Groq(api_key=active_key)
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": chat_system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.2
                )
                answer = response.choices[0].message.content
            except Exception:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": chat_system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.2
                )
                answer = response.choices[0].message.content
                
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})