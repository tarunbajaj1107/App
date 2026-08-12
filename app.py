import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from docx import Document
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from openai import OpenAI
import streamlit as st
from pinecone import Pinecone, ServerlessSpec

# =====================================================================
# 1. APPLICATION SETUP & PINECONE CLOUD VECTOR DB
# =====================================================================
st.set_page_config(
    page_title="NVIDIA Nemotron AI Legal Reviewer (Word Comments)",
    page_icon="🟢",
    layout="wide",
)

INDEX_NAME = "project-finance-playbook"
EMBED_MODEL = "multilingual-e5-large"  # Pinecone hosted embedding model (1024 dims)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-4-340b-instruct"

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
# 2. WORD DOCUMENT PARSING & NATIVE WORD COMMENTS GENERATOR
# =====================================================================

def extract_paragraphs_from_docx(docx_file):
    doc = Document(docx_file)
    paragraphs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text and len(text.split()) >= 4:
            paragraphs.append(text)
    return paragraphs

def add_native_comment(p, author, comment_text, comment_id):
    """
    Appends native MS Word comment XML elements to a paragraph.
    """
    # 1. Create or access comments.xml part
    part = p.part
    try:
        comments_part = part.package.parts[qn('w:comments')]
    except KeyError:
        # Create comments part if it doesn't exist
        comments_xml = OxmlElement('w:comments')
        comments_part = part.package.relate_to(
            comments_xml,
            'http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments'
        )

    # 2. Add comment element
    comment_elem = OxmlElement('w:comment')
    comment_elem.set(qn('w:id'), str(comment_id))
    comment_elem.set(qn('w:author'), author)
    comment_elem.set(qn('w:date'), time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    p_elem = OxmlElement('w:p')
    r_elem = OxmlElement('w:r')
    t_elem = OxmlElement('w:t')
    t_elem.text = comment_text
    r_elem.append(t_elem)
    p_elem.append(r_elem)
    comment_elem.append(p_elem)

    # Attach to root comments node
    comments_part._element.append(comment_elem)

    # 3. Wrap paragraph runs with Comment Range Start & End
    comment_range_start = OxmlElement('w:commentRangeStart')
    comment_range_start.set(qn('w:id'), str(comment_id))
    
    comment_range_end = OxmlElement('w:commentRangeEnd')
    comment_range_end.set(qn('w:id'), str(comment_id))

    comment_reference = OxmlElement('w:commentReference')
    comment_reference.set(qn('w:id'), str(comment_id))

    r_ref = OxmlElement('w:r')
    r_ref.append(comment_reference)

    p._p.insert(0, comment_range_start)
    p._p.append(comment_range_end)
    p._p.append(r_ref)


def create_commented_docx(paragraph_results, author="NVIDIA Nemotron Legal AI"):
    """
    Generates a clean DOCX where suggestions and explanations are placed
    in Word comments attached to paragraphs instead of inline track changes.
    """
    doc = Document()
    comment_counter = 0

    for orig_text, suggested_text, explanation, is_acceptable in paragraph_results:
        p = doc.add_paragraph(orig_text)

        if not is_acceptable and orig_text != suggested_text:
            comment_counter += 1
            full_comment_body = (
                f"💡 SUGGESTION / REVISION:\n\"{suggested_text}\"\n\n"
                f"📌 REASON & PRECEDENT:\n{explanation}"
            )
            try:
                add_native_comment(p, author, full_comment_body, comment_counter)
            except Exception as e:
                # Fallback to inline callout box if Word XML manipulation encounters a edge case
                p_sub = doc.add_paragraph()
                r = p_sub.add_run(f"💬 [Comment Suggestion]: {full_comment_body}")
                r.font.italic = True
                r.font.size = 10

    output_path = "Reviewed_Facility_Agreement_Comments.docx"
    doc.save(output_path)
    return output_path

# =====================================================================
# 3. CLOUD VECTOR RETRIEVAL & NEMOTRON LLM ENGINE
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


def analyze_clause_batch_nemotron(batch_items, custom_instruction, nvidia_api_key):
    """Analyzes clause batches using NVIDIA Nemotron via OpenAI SDK."""
    if not nvidia_api_key:
        return []

    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=nvidia_api_key
    )

    system_prompt = """
    You are NVIDIA Nemotron, an expert Legal & Contract Analyst comparing contract clauses against precedent loan agreements.

    INSTRUCTIONS FOR COMMENT-BASED REVIEW:
    - Compare each input 'clause' against 'repository_context' (Precedent Agreement).
    - If a clause is UNACCEPTABLE or strays from precedent, provide a revised wording recommendation along with a clear reason.
    - If acceptable, mark 'is_acceptable': true.

    EVALUATION RULES FOR EACH ITEM:
    1. If ACCEPTABLE or substantially equivalent in legal effect to repository context:
       - 'is_acceptable': true
       - 'proposed_text': EXACT ORIGINAL CLAUSE TEXT
       - 'explanation': "Matches precedent agreement standard."
    2. If UNACCEPTABLE / NEEDS LEGAL ADJUSTMENT:
       - 'is_acceptable': false
       - 'proposed_text': Clean, proposed replacement text for the clause.
       - 'explanation': Detailed reason describing why the revision is suggested based on the precedent agreement.

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

    CLAUSES TO ANALYZE IN BATCH:
    {json.dumps(formatted_input, indent=2)}
    """

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("results", [])
            
        except Exception as e:
            print(f"⚠️ [Nemotron API Error Attempt {attempt+1}]: {str(e)}", flush=True)
            time.sleep((attempt + 1) * 2)

    fallback_results = []
    for item in batch_items:
        fallback_results.append({
            "id": item["id"],
            "is_acceptable": True,
            "proposed_text": item["clause"],
            "explanation": "Skipped due to API connection timeouts."
        })
    return fallback_results

# =====================================================================
# 4. STREAMLIT UI & TABBED INTERFACE
# =====================================================================

st.title("🟢 NVIDIA Nemotron AI Legal Reviewer")
st.caption("Contract Auditor backed by NVIDIA Nemotron-4 LLM & Pinecone Serverless Vector DB")

default_nvidia = st.secrets.get("NVIDIA_API_KEY", "") if "NVIDIA_API_KEY" in st.secrets else ""
default_pinecone = st.secrets.get("PINECONE_API_KEY", "") if "PINECONE_API_KEY" in st.secrets else ""

with st.sidebar:
    st.header("🔑 Credentials")
    nvidia_api_key = st.text_input("NVIDIA API Key", value=default_nvidia, type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", value=default_pinecone, type="password")
    
    st.divider()
    st.header("⚙️ Model Settings")
    st.info(f"**LLM:** `{NVIDIA_MODEL}`\n\n**Output Mode:** Word Comments (No Redlines)")
    
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
    "💬 Word Comment Auditor", 
    "📚 Upload Precedent Agreements", 
    "🤖 Nemotron Chat Assistant"
])

# ---------------------------------------------------------------------
# TAB 1: WORD COMMENT REVIEWER
# ---------------------------------------------------------------------
with tab_review:
    st.header("Compare Draft against Precedent Agreements")
    
    uploaded_draft = st.file_uploader("Upload Target Facility Agreement (.docx)", type=["docx"], key="target_doc")
    custom_instruction = st.text_area("Optional Deal Directives / Overrides", placeholder="e.g., 'Ensure minimum DSCR covenant is set to 1.25x'.")
    
    if st.button("🟢 Analyze with Nemotron (Generate Comments)", type="primary"):
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

            # STEP 1: Pinecone Cloud Vector Search
            logs.append(f"[{time.strftime('%H:%M:%S')}] ☁️ Querying Pinecone Database for Precedents...")
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
            logs.append(f"[{time.strftime('%H:%M:%S')}] ✅ Precedents fetched in {vec_elapsed}s! Starting NVIDIA Nemotron evaluation...")
            log_area.text("\n".join(logs[-12:]))

            # STEP 2: Batched LLM Analysis via Nemotron
            BATCH_SIZE = 8
            comment_results = []
            total_items = len(prepared_items)
            
            for i in range(0, total_items, BATCH_SIZE):
                batch = prepared_items[i : i + BATCH_SIZE]
                current_c = min(i + BATCH_SIZE, total_items)
                
                batch_evals = analyze_clause_batch_nemotron(
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
                status_placeholder.text(f"Evaluated {current_c}/{total_items} clauses with Nemotron ({pct}%)...")
                
                log_msg = f"[{time.strftime('%H:%M:%S')}] ✅ Nemotron evaluated Clauses {i+1} to {current_c} of {total_items} ({pct}%)"
                logs.append(log_msg)
                log_area.text("\n".join(logs[-12:]))
            
            elapsed = round(time.time() - start_time, 2)
            st.success(f"⚡ Completed document review in {elapsed} seconds!")
            
            output_docx = create_commented_docx(comment_results)
            with open(output_docx, "rb") as file_data:
                st.download_button(
                    label="📥 Download Agreement with Word Comments (.docx)",
                    data=file_data,
                    file_name="Facility_Agreement_Nemotron_Comments.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

# ---------------------------------------------------------------------
# TAB 2: CLOUD REPOSITORY INDEXING
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
                        
                        # 3. Upsert to Pinecone
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
    st.header("NVIDIA Nemotron Contract Chat Assistant")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Ask me any question regarding your precedent loan agreements in Pinecone."}
        ]
    
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if user_query := st.chat_input("Ask a question about precedent clauses..."):
        if not nvidia_api_key or not pc_client:
            st.error("Please ensure NVIDIA and Pinecone API Keys are provided.")
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
            You are NVIDIA Nemotron, an expert Legal Assistant.
            
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
                answer = f"Error getting response from Nemotron: {str(e)}"
                
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
