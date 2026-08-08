import json
import os
import re
import time
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from groq import Groq
import streamlit as st
from pinecone import Pinecone, ServerlessSpec

# =====================================================================
# 1. APPLICATION SETUP & PINECONE CLOUD VECTOR DB
# =====================================================================
st.set_page_config(
    page_title="Cloud Project Finance AI Legal Redliner",
    page_icon="⚡",
    layout="wide",
)

INDEX_NAME = "project-finance-playbook"
EMBED_MODEL = "multilingual-e5-large"  # Pinecone hosted embedding model (1024 dims)

@st.cache_resource
def init_pinecone(api_key):
    """Initializes Pinecone client and ensures serverless index exists with 1024 dimensions."""
    if not api_key:
        return None, None
    try:
        pc = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=1024,  # Exact dimension match for multilingual-e5-large
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        index = pc.Index(INDEX_NAME)
        return pc, index
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        return None, None

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
# 3. CLOUD VECTOR RETRIEVAL & LLM ENGINE
# =====================================================================

def query_pinecone_batch(pc, index, chunk_paras, chunk_start_idx):
    """Generates embeddings using Pinecone Hosted Inference and queries cloud vectors."""
    try:
        embeddings = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=chunk_paras,
            parameters={"input_type": "query"}
        )
        
        batch_results = []
        for idx, (p_text, emb) in enumerate(zip(chunk_paras, embeddings)):
            res = index.query(
                vector=emb["values"],
                top_k=1,
                include_metadata=True
            )
            
            ctx = "NO DIRECT REPOSITORY PRECEDENT FOUND."
            if res.get("matches") and len(res["matches"]) > 0:
                match = res["matches"][0]
                if match["score"] > 0.65:  # Relevance threshold
                    doc_str = match["metadata"].get("text", "")
                    src = match["metadata"].get("source", "Repo")
                    ctx = f"Precedent from [{src}]: \"{doc_str}\""
            
            batch_results.append({
                "id": chunk_start_idx + idx,
                "clause": p_text,
                "context": ctx
            })
        return batch_results, len(chunk_paras)
    except Exception as e:
        print(f"❌ Pinecone Query Error: {e}", flush=True)
        return [{
            "id": chunk_start_idx + idx,
            "clause": p_text,
            "context": "NO DIRECT REPOSITORY PRECEDENT FOUND."
        } for idx, p_text in enumerate(chunk_paras)], len(chunk_paras)


def run_parallel_pinecone_retrieval(pc, index, paragraphs, batch_size=10, max_workers=4, status_placeholder=None, progress_bar=None, log_area=None, logs_list=None):
    """Executes parallel retrieval against Pinecone Cloud DB."""
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
            executor.submit(query_pinecone_batch, pc, index, chunk, start_idx): (chunk, start_idx)
            for chunk, start_idx in chunks
        }
        
        for future in as_completed(futures):
            batch_data, chunk_len = future.result()
            completed_clauses += chunk_len
            completed_chunks += 1
            
            for item in batch_data:
                prepared_items[item["id"]] = item

            if status_placeholder and progress_bar:
                pct = int((completed_clauses / total) * 20)
                progress_bar.progress(pct)
                status_placeholder.text(f"🔍 Pinecone Cloud Search: {completed_clauses}/{total} clauses ({completed_chunks}/{total_chunks} batches)...")
                
            if log_area and logs_list is not None:
                log_msg = f"[{time.strftime('%H:%M:%S')}] ☁️ Cloud Batch {completed_chunks}/{total_chunks} retrieved ({completed_clauses}/{total} clauses)"
                logs_list.append(log_msg)
                log_area.text("\n".join(logs_list[-12:]))

    return prepared_items


def analyze_clause_batch(batch_items, custom_instruction, primary_key, secondary_key, model_strategy):
    """Analyzes clause batches using Groq LLM with dual-key failover and exponential backoff."""
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
            for attempt in range(3):
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
                            break
                        else:
                            wait_time = (attempt + 1) * 3
                            print(f"⚠️ [Rate Limit Hit on {model_name}]: Retrying in {wait_time}s...", flush=True)
                            time.sleep(wait_time)
                            continue
                    else:
                        print(f"❌ [Groq API Error]: {err_msg}", flush=True)
                        time.sleep(2)
                        break

    fallback_results = []
    for item in batch_items:
        fallback_results.append({
            "id": item["id"],
            "is_acceptable": True,
            "proposed_text": item["clause"],
            "explanation": "Skipped due to API rate limits."
        })
    return fallback_results

# =====================================================================
# 4. STREAMLIT UI & TABBED INTERFACE
# =====================================================================

st.title("☁️ Project Finance AI Legal Agent (Pinecone Cloud)")
st.caption("Surgical Redliner backed by Pinecone Serverless Vector Database & Groq")

# Read Secrets if available (Streamlit Cloud Secrets)
default_groq = st.secrets.get("GROQ_API_KEY", "") if "GROQ_API_KEY" in st.secrets else ""
default_pinecone = st.secrets.get("PINECONE_API_KEY", "") if "PINECONE_API_KEY" in st.secrets else ""

with st.sidebar:
    st.header("🔑 Credentials")
    pinecone_api_key = st.text_input("Pinecone API Key", value=default_pinecone, type="password")
    primary_api_key = st.text_input("Primary Groq API Key", value=default_groq, type="password")
    secondary_api_key = st.text_input("Secondary Groq API Key (Optional)", type="password")
    
    st.divider()
    st.header("⚙️ Model Strategy")
    selected_model_mode = st.radio(
        "Select Processing Strategy:",
        options=[
            "Fastest & Highest Limit (llama-3.1-8b-instant)", 
            "High Accuracy (llama-3.3-70b-versatile with 8b Fallback)"
        ],
        index=0
    )
    
    st.divider()
    st.header("☁️ Cloud Vector Storage")
    
    pc_client, pc_index = init_pinecone(pinecone_api_key)
    if pc_index:
        try:
            stats = pc_index.describe_index_stats()
            vector_count = stats.get("total_vector_count", 0)
            st.metric("Cloud Index Vector Count", vector_count)
        except Exception:
            st.caption("Unable to fetch stats.")
            
        if st.button("🗑️ Clear Cloud Memory"):
            try:
                pc_index.delete(delete_all=True)
                st.success("Cloud database erased!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing cloud database: {e}")

tab_redline, tab_repository, tab_chat = st.tabs([
    "🚀 High-Speed Redliner", 
    "📚 Upload Cloud Precedents", 
    "💬 Smart Chat Assistant"
])

# ---------------------------------------------------------------------
# TAB 1: REDLINER
# ---------------------------------------------------------------------
with tab_redline:
    st.header("Analyze & Redline Facility Agreement")
    
    uploaded_draft = st.file_uploader("Upload New Document (.docx)", type=["docx"], key="target_doc")
    custom_instruction = st.text_area("Optional Deal Directives / Overrides", placeholder="e.g., 'Ensure minimum DSCR covenant is set to 1.25x'.")
    
    if st.button("⚡ Fast Batch Redline Document", type="primary"):
        if not primary_api_key:
            st.error("Please enter your Primary Groq API Key.")
        elif not pinecone_api_key:
            st.error("Please enter your Pinecone API Key.")
        elif not uploaded_draft:
            st.error("Please upload a .docx document.")
        else:
            paragraphs = extract_paragraphs_from_docx(uploaded_draft)
            total_paragraphs = len(paragraphs)
            st.info(f"Loaded document: {total_paragraphs} clauses identified.")
            
            start_time = time.time()
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            debug_box = st.expander("🔍 Live Processing Logs", expanded=True)
            log_area = debug_box.empty()
            logs = []

            # STEP 1: Pinecone Cloud Vector Search
            logs.append(f"[{time.strftime('%H:%M:%S')}] ☁️ Querying Pinecone Serverless Cloud Database...")
            log_area.text("\n".join(logs))
            
            vec_start = time.time()
            prepared_items = run_parallel_pinecone_retrieval(
                pc=pc_client,
                index=pc_index, 
                paragraphs=paragraphs, 
                batch_size=10, 
                max_workers=4,
                status_placeholder=status_placeholder,
                progress_bar=progress_bar,
                log_area=log_area,
                logs_list=logs
            )
            vec_elapsed = round(time.time() - vec_start, 2)
            
            progress_bar.progress(20)
            logs.append(f"[{time.strftime('%H:%M:%S')}] ✅ Pinecone cloud lookup finished in {vec_elapsed}s! Starting LLM analysis...")
            log_area.text("\n".join(logs[-12:]))

            # STEP 2: Batched LLM Analysis
            BATCH_SIZE = 12
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
                
                time.sleep(0.3)  # Small pacing delay for RPM rate limits
                
                pct = 20 + int((current_c / total_items) * 80)
                progress_bar.progress(pct)
                status_placeholder.text(f"Analyzed {current_c}/{total_items} clauses ({pct}%)...")
                
                log_msg = f"[{time.strftime('%H:%M:%S')}] ✅ Processed Clauses {i+1} to {current_c} of {total_items} ({pct}%)"
                logs.append(log_msg)
                log_area.text("\n".join(logs[-12:]))
            
            elapsed = round(time.time() - start_time, 2)
            st.success(f"⚡ Completed redlining in {elapsed} seconds!")
            
            output_docx = create_redlined_docx(redline_results)
            with open(output_docx, "rb") as file_data:
                st.download_button(
                    label="📥 Download Redlined Document (.docx)",
                    data=file_data,
                    file_name="Redlined_Facility_Agreement.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

# ---------------------------------------------------------------------
# TAB 2: CLOUD REPOSITORY INDEXING
# ---------------------------------------------------------------------
with tab_repository:
    st.header("Upload Precedent Facility Agreements to Cloud")
    precedent_files = st.file_uploader("Upload Agreements (.docx)", type=["docx"], accept_multiple_files=True)
    
    if precedent_files and st.button("☁️ Save to Pinecone Cloud"):
        if not pinecone_api_key or not pc_client:
            st.error("Please provide a valid Pinecone API key.")
        else:
            prog_bar = st.progress(0)
            status_box = st.empty()
            debug_container = st.expander("🛠️ Real-Time Cloud Indexing Logs", expanded=True)
            log_console = debug_container.empty()
            logs = []
            
            total_files = len(precedent_files)
            total_indexed = 0
            
            for file_idx, file in enumerate(precedent_files):
                paragraphs = extract_paragraphs_from_docx(file)
                total_paras = len(paragraphs)
                if total_paras == 0:
                    continue
                
                batch_size = 10  # Kept small to stay safely below Pinecone 2MB limit
                clean_fname = re.sub(r'[^a-zA-Z0-9_]', '_', file.name)
                
                for i in range(0, total_paras, batch_size):
                    chunk = paragraphs[i : i + batch_size]
                    
                    try:
                        # 1. Embed via Pinecone Hosted Inference
                        embeddings = pc_client.inference.embed(
                            model=EMBED_MODEL,
                            inputs=chunk,
                            parameters={"input_type": "passage"}
                        )
                        
                        # 2. Structure Vectors safely
                        vectors_to_upsert = []
                        for j, (text, emb) in enumerate(zip(chunk, embeddings)):
                            v_id = f"{clean_fname}_p{i+j}_{int(time.time()*1000)}"
                            
                            # Trim text if clause is unusually large
                            metadata_text = text[:1500] if len(text) > 1500 else text
                            
                            vectors_to_upsert.append({
                                "id": v_id,
                                "values": emb["values"],
                                "metadata": {
                                    "text": metadata_text, 
                                    "source": file.name, 
                                    "chunk": i+j
                                }
                            })
                        
                        # 3. Upsert to Pinecone Cloud
                        pc_index.upsert(vectors=vectors_to_upsert)
                        total_indexed += len(chunk)
                        
                        current_clause = min(i + batch_size, total_paras)
                        overall_pct = ((file_idx + (current_clause / total_paras)) / total_files)
                        prog_bar.progress(min(overall_pct, 1.0))
                        status_box.text(f"Uploading '{file.name}': {current_clause}/{total_paras} clauses...")
                        
                        log_line = f"[{time.strftime('%H:%M:%S')}] Upserted to Cloud: Clauses {i+1} to {current_clause} from '{file.name}'"
                        logs.append(log_line)
                        log_console.text("\n".join(logs[-12:]))
                        
                    except Exception as err:
                        err_line = f"[{time.strftime('%H:%M:%S')}] ❌ Error upserting batch {i+1}-{i+len(chunk)}: {str(err)}"
                        logs.append(err_line)
                        log_console.text("\n".join(logs[-12:]))
                        time.sleep(1)
            
            st.success(f"Successfully processed {total_indexed} clauses to Pinecone Cloud Database!")
            time.sleep(1)
            st.rerun()

# ---------------------------------------------------------------------
# TAB 3: CHAT ASSISTANT
# ---------------------------------------------------------------------
with tab_chat:
    st.header("Project Finance Cloud Chat Assistant")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Ask me any question regarding your cloud repository precedents."}
        ]
    
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if user_query := st.chat_input("Ask a question..."):
        active_key = primary_api_key or secondary_api_key
        if not active_key or not pc_client:
            st.error("Please ensure Groq and Pinecone API Keys are provided.")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
                
            context_str = "NO DIRECT MATCHES FOUND IN CLOUD REPOSITORY."
            try:
                emb_res = pc_client.inference.embed(
                    model=EMBED_MODEL,
                    inputs=[user_query],
                    parameters={"input_type": "query"}
                )
                query_emb = emb_res[0]["values"]
                
                res = pc_index.query(vector=query_emb, top_k=5, include_metadata=True)
                
                context_blocks = []
                if res.get("matches"):
                    for m in res["matches"]:
                        if m["score"] > 0.6:
                            src = m["metadata"].get("source", "Unknown Document")
                            txt = m["metadata"].get("text", "")
                            context_blocks.append(f"From Precedent File [{src}]:\n\"{txt}\"")
                
                if context_blocks:
                    context_str = "\n\n".join(context_blocks)
            except Exception as e:
                print(f"Chat Context Search Error: {e}", flush=True)

            chat_system_prompt = f"""
            You are an expert Project Finance Legal Assistant.
            
            1. CLOUD REPOSITORY CHECK FIRST:
               - Examine CLOUD REPOSITORY CONTEXT.
               - IF FOUND: Answer directly and cite the source file (e.g., "According to [Filename.docx]...").
                 
            2. FALLBACK TO GENERAL MARKET PRACTICE:
               - IF NOT FOUND: Begin response with:
                 "Your uploaded cloud documents do not contain specific terms regarding this topic. However, in general project finance market practice:"
                 Then explain general market practice.

            RETRIEVED CLOUD CONTEXT:
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
