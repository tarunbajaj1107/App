import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from docx import Document
from docx.shared import Inches
from openai import OpenAI
import streamlit as st
from pinecone import Pinecone, ServerlessSpec

# =====================================================================
# 1. APPLICATION SETUP & PINECONE DB
# =====================================================================
st.set_page_config(
    page_title="NVIDIA AI Legal Reviewer (Word Bubble Comments)",
    page_icon="💬",
    layout="wide",
)

INDEX_NAME = "project-finance-playbook"
EMBED_MODEL = "multilingual-e5-large"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Model confirmed working on your account
NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"

@st.cache_resource
def init_pinecone(api_key):
    """Initializes Pinecone client and ensures serverless index exists."""
    if not api_key:
        return None, None
    try:
        pc = Pinecone(api_key=api_key)
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index(INDEX_NAME)
        return pc, index
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        return None, None

# =====================================================================
# 2. WORD DOCUMENT PARSING & NATIVE MARGIN COMMENT GENERATOR
# =====================================================================

def extract_paragraphs_from_docx(docx_file):
    doc = Document(docx_file)
    paragraphs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text and len(text.split()) >= 4:
            paragraphs.append(text)
    return paragraphs


def create_commented_docx(paragraph_results, author="AI Legal Reviewer"):
    """
    Generates a clean DOCX where suggestions and reasons are anchored as
    native MS Word sidebar comment bubbles attached to the paragraph runs.
    """
    doc = Document()

    for orig_text, suggested_text, explanation, is_acceptable in paragraph_results:
        p = doc.add_paragraph()
        run = p.add_run(orig_text)  # Document text remains completely clean

        if not is_acceptable and orig_text != suggested_text:
            full_comment_body = (
                f"💡 SUGGESTED REVISION:\n\"{suggested_text}\"\n\n"
                f"📌 PRECEDENT & REASON:\n{explanation}"
            )
            
            # Native python-docx (v1.2.0+) sidebar comment bubble insertion
            try:
                doc.add_comment(
                    runs=p.runs,
                    text=full_comment_body,
                    author=author,
                    initials="AI"
                )
            except AttributeError:
                # Fallback if library version is below 1.2.0
                p_sub = doc.add_paragraph()
                p_sub.paragraph_format.left_indent = Inches(0.3)
                r_tag = p_sub.add_run("💬 [AI Review Comment]: ")
                r_tag.bold = True
                r_body = p_sub.add_run(full_comment_body)
                r_body.font.italic = True

    output_path = "Reviewed_Facility_Agreement_Comments.docx"
    doc.save(output_path)
    return output_path

# =====================================================================
# 3. CLOUD VECTOR RETRIEVAL & NEMOTRON / LLAMA ENGINE
# =====================================================================

def query_pinecone_batch(pc, index, chunk_paras, chunk_start_idx):
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
                if match["score"] > 0.65:
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
                status_placeholder.text(f"🔍 Searching Precedents: {completed_clauses}/{total} clauses...")
                
            if log_area and logs_list is not None:
                log_msg = f"[{time.strftime('%H:%M:%S')}] ☁️ Batch {completed_chunks}/{total_chunks} retrieved ({completed_clauses}/{total} clauses)"
                logs_list.append(log_msg)
                log_area.text("\n".join(logs_list[-12:]))

    return prepared_items


def analyze_clause_batch_llm(batch_items, custom_instruction, nvidia_api_key):
    """Analyzes clause batches using meta/llama-3.1-8b-instruct via OpenAI SDK."""
    if not nvidia_api_key:
        st.error("❌ API Key is missing!")
        return []

    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=nvidia_api_key
    )

    system_prompt = """
    You are an expert Legal & Contract Analyst comparing contract clauses against precedent loan agreements.

    INSTRUCTIONS FOR COMMENT-BASED REVIEW:
    - Compare each input 'clause' against 'repository_context' (Precedent Agreement).
    - If a clause is UNACCEPTABLE or strays from precedent, provide a revised wording recommendation along with a clear reason.
    - If acceptable, mark 'is_acceptable': true.

    YOU MUST RETURN A STRICT JSON OBJECT WITH A SINGLE KEY "results" CONTAINING AN ARRAY:
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
    
    formatted_input = [
        {"id": item["id"], "clause": item["clause"], "repository_context": item["context"]}
        for item in batch_items
    ]

    user_prompt = f"DEAL DIRECTIVE: {custom_instruction}\nCLAUSES: {json.dumps(formatted_input)}"

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            raw_content = response.choices[0].message.content
            
            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```[a-zA-Z]*\n", "", raw_content)
                raw_content = re.sub(r"\n```$", "", raw_content)

            data = json.loads(raw_content)
            return data.get("results", [])
            
        except Exception as e:
            print(f"[DEBUG] Attempt {attempt+1} Failed: {str(e)}", flush=True)
            time.sleep((attempt + 1) * 2)

    return [
        {
            "id": item["id"],
            "is_acceptable": False,
            "proposed_text": item["clause"],
            "explanation": "Flagged for manual review due to API response timeout."
        }
        for item in batch_items
    ]

# =====================================================================
# 4. STREAMLIT UI & TABBED INTERFACE
# =====================================================================

st.title("💬 Legal Contract AI Auditor (Word Bubble Comments)")
st.caption("Automated Legal Review with Sidebar Comments backed by Pinecone & NVIDIA Llama 3.1 8B")

default_nvidia = st.secrets.get("NVIDIA_API_KEY", "") if "NVIDIA_API_KEY" in st.secrets else ""
default_pinecone = st.secrets.get("PINECONE_API_KEY", "") if "PINECONE_API_KEY" in st.secrets else ""

with st.sidebar:
    st.header("🔑 Credentials")
    nvidia_api_key = st.text_input("NVIDIA API Key", value=default_nvidia, type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", value=default_pinecone, type="password")
    
    st.divider()
    st.header("⚙️ Model Settings")
    st.info(f"**Active Model:** `{NVIDIA_MODEL}`\n\n**Output:** Word Sidebar Comment Bubbles")
    
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

tab_review, tab_repository, tab_chat = st.tabs([
    "💬 Sidebar Comment Reviewer", 
    "📚 Upload Precedent Agreements", 
    "🤖 Contract Chat Assistant"
])

# ---------------------------------------------------------------------
# TAB 1: WORD COMMENT REVIEWER
# ---------------------------------------------------------------------
with tab_review:
    st.header("Compare Draft against Precedent Agreements")
    
    uploaded_draft = st.file_uploader("Upload Target Facility Agreement (.docx)", type=["docx"], key="target_doc")
    custom_instruction = st.text_area("Optional Deal Directives / Overrides", placeholder="e.g., 'Ensure minimum DSCR covenant is set to 1.25x'.")
    
    if st.button("💬 Analyze Contract (Generate Sidebar Comments)", type="primary"):
        if not nvidia_api_key:
            st.error("Please enter your NVIDIA API Key.")
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

            # STEP 1: Vector Search
            logs.append(f"[{time.strftime('%H:%M:%S')}] ☁️ Fetching Precedents from Pinecone...")
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
            logs.append(f"[{time.strftime('%H:%M:%S')}] ✅ Precedents retrieved in {vec_elapsed}s! Evaluating with Llama 3.1 8B...")
            log_area.text("\n".join(logs[-12:]))

            # STEP 2: LLM Evaluation
            BATCH_SIZE = 10
            comment_results = []
            total_items = len(prepared_items)
            
            for i in range(0, total_items, BATCH_SIZE):
                batch = prepared_items[i : i + BATCH_SIZE]
                current_c = min(i + BATCH_SIZE, total_items)
                
                batch_evals = analyze_clause_batch_llm(
                    batch_items=batch, 
                    custom_instruction=custom_instruction, 
                    nvidia_api_key=nvidia_api_key
                )
                
                eval_map = {res["id"]: res for res in batch_evals}
                for item in batch:
                    res = eval_map.get(item["id"], {
                        "is_acceptable": True,
                        "proposed_text": item["clause"],
                        "explanation": "Matches repository precedent position."
                    })
                    
                    comment_results.append((
                        item["clause"],
                        res.get("proposed_text", item["clause"]),
                        res.get("explanation", "Matches repository precedent position."),
                        res.get("is_acceptable", True)
                    ))
                
                time.sleep(0.2)
                
                pct = 20 + int((current_c / total_items) * 80)
                progress_bar.progress(pct)
                status_placeholder.text(f"Evaluated {current_c}/{total_items} clauses ({pct}%)...")
                
                log_msg = f"[{time.strftime('%H:%M:%S')}] ✅ Evaluated Clauses {i+1} to {current_c} of {total_items} ({pct}%)"
                logs.append(log_msg)
                log_area.text("\n".join(logs[-12:]))
            
            elapsed = round(time.time() - start_time, 2)
            st.success(f"⚡ Completed document review in {elapsed} seconds!")
            
            output_docx = create_commented_docx(comment_results)
            with open(output_docx, "rb") as file_data:
                st.download_button(
                    label="📥 Download File with Margin Comment Bubbles (.docx)",
                    data=file_data,
                    file_name="Facility_Agreement_Bubble_Comments.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

# ---------------------------------------------------------------------
# TAB 2: REPOSITORY INDEXING
# ---------------------------------------------------------------------
with tab_repository:
    st.header("Upload Precedent Facility Agreements to Cloud")
    precedent_files = st.file_uploader("Upload Loan Agreements (.docx)", type=["docx"], accept_multiple_files=True)
    
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
                
                batch_size = 10
                clean_fname = re.sub(r'[^a-zA-Z0-9_]', '_', file.name)
                
                for i in range(0, total_paras, batch_size):
                    chunk = paragraphs[i : i + batch_size]
                    
                    try:
                        embeddings = pc_client.inference.embed(
                            model=EMBED_MODEL,
                            inputs=chunk,
                            parameters={"input_type": "passage"}
                        )
                        
                        vectors_to_upsert = []
                        for j, (text, emb) in enumerate(zip(chunk, embeddings)):
                            v_id = f"{clean_fname}_p{i+j}_{int(time.time()*1000)}"
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
            
            st.success(f"Successfully uploaded {total_indexed} clauses to Pinecone Cloud!")
            time.sleep(1)
            st.rerun()

# ---------------------------------------------------------------------
# TAB 3: CHAT ASSISTANT
# ---------------------------------------------------------------------
with tab_chat:
    st.header("Contract Chat Assistant")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Ask me any question regarding your precedent loan agreements."}
        ]
    
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if user_query := st.chat_input("Ask a question..."):
        if not nvidia_api_key or not pc_client:
            st.error("Please ensure NVIDIA and Pinecone API Keys are provided.")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
                
            context_str = "NO DIRECT MATCHES FOUND IN REPOSITORY."
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
                print(f"Chat Error: {e}", flush=True)

            chat_system_prompt = f"""
            You are an expert Legal Assistant.
            
            1. PRECEDENT CHECK FIRST:
               - Examine RETRIEVED CLOUD CONTEXT.
               - IF FOUND: Answer directly and cite the source precedent file.
                 
            2. FALLBACK TO GENERAL MARKET PRACTICE:
               - IF NOT FOUND: State that no exact precedent match was found, then explain standard market practice.

            RETRIEVED CLOUD CONTEXT:
            {context_str}
            """
            
            client = OpenAI(
                base_url=NVIDIA_BASE_URL,
                api_key=nvidia_api_key
            )
            
            try:
                response = client.chat.completions.create(
                    model=NVIDIA_MODEL,
                    messages=[
                        {"role": "system", "content": chat_system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.2
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"Error getting response: {str(e)}"
                
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
