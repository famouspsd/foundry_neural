import streamlit as st
import pandas as pd
import google.generativeai as genai
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from PIL import Image

# --- 1. THE NEURAL CORTEX (FAISS) ---
@st.cache_resource
def init_agent_memory():
    # Local model for high-speed search without API costs
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.IndexFlatL2(384) 
    return encoder, index

encoder, faiss_index = init_agent_memory()

# --- 2. AGENT CONFIG ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

# --- 3. AUTONOMOUS TOOLS ---
def autonomous_retrieval(query):
    if not os.path.exists('vector_vault.csv'):
        return "No local vault data. Using base intelligence."
    df = pd.read_csv('vector_vault.csv')
    query_vec = encoder.encode([query]).astype('float32')
    # Pull the 2 most relevant pieces of memory
    distances, indices = faiss_index.search(query_vec, k=2)
    return " ".join(df.iloc[indices[0]]['Context'].values)

# --- 4. THE DUAL-LOOP REASONING ENGINE ---
def run_autonomous_agent(user_goal, image=None):
    context = autonomous_retrieval(user_goal)
    
    # PHASE 1: REASONING & DRAFTING
    draft_prompt = f"""
    [SYSTEM: REASONING PHASE]
    Vault Context: {context}
    Mission: {user_goal}
    
    Task: Analyze the goal. Create a multi-step strategy. 
    Include: Direct Solution, Pros/Cons Table, and Strategic Advice.
    """
    
    try:
        if image:
            img = Image.open(image)
            first_pass = model.generate_content([draft_prompt, img]).text
        else:
            first_pass = model.generate_content(draft_prompt).text
            
        # PHASE 2: AUTONOMOUS SELF-CORRECTION
        correction_prompt = f"""
        [SYSTEM: CRITIQUE & CORRECTION PHASE]
        Original Mission: {user_goal}
        Draft Generated: {first_pass}
        
        Task: Review the draft for logic gaps or missed opportunities. 
        Ensure the 'Lead Scientist' advice is bold and actionable.
        Rewrite the final response to be perfect.
        """
        final_output = model.generate_content(correction_prompt).text
        return final_output
    except Exception as e:
        if "429" in str(e):
            return "🚨 Quota Limit Reached. Agent is cooling down for 60s."
        return f"Agent Error: {e}"

# --- 5. UI DESIGN ---
st.set_page_config(page_title="Foundry Neural: Autonomous", layout="wide", page_icon="🤖")

# --- SIDEBAR: MEMORY CONTROL ---
with st.sidebar:
    st.title("🧠 Neural Management")
    
    with st.expander("➕ Ingest New Knowledge", expanded=False):
        t = st.text_input("Topic")
        c = st.text_area("Context/Data")
        if st.button("SYNC TO VECTOR SPACE"):
            if t and c:
                vector = encoder.encode([c]).astype('float32')
                faiss_index.add(vector)
                pd.DataFrame({"Topic":[t],"Context":[c]}).to_csv('vector_vault.csv', mode='a', header=not os.path.exists('vector_vault.csv'), index=False)
                st.success(f"'{t}' integrated into cortex.")
    
    st.divider()
    
    if st.button("🗑️ RESET AGENT MEMORY"):
        if os.path.exists('vector_vault.csv'):
            os.remove('vector_vault.csv')
            st.rerun()

    st.divider()
    uploaded_img = st.file_uploader("📷 Visual Context (Vision)", type=["jpg","png","jpeg"])

# --- MAIN INTERFACE ---
st.title("🤖 Foundry Neural: Autonomous Agent")
st.caption("Status: Multi-Step Reasoning Enabled | FAISS Vector Cortex: Active")

user_input = st.text_input("Deploy Mission:", placeholder="What should we solve today?")

if st.button("DEPLOY AGENT") and user_input:
    # Use st.status for that "Agent-Like" feeling
    with st.status("Agent is Processing...", expanded=True) as status:
        st.write("🔍 Accessing Vector Memory...")
        st.write("📝 Drafting Initial Strategy...")
        st.write("⚖️ Performing Self-Correction Loop...")
        
        result = run_autonomous_agent(user_input, uploaded_img)
        status.update(label="Mission Accomplished!", state="complete")
    
    st.markdown("---")
    st.markdown(result)
    
    

            
    
            
