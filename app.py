import streamlit as st
import pandas as pd
import google.generativeai as genai
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from fpdf import FPDF

# --- 1. CORE ENGINES ---
st.set_page_config(page_title="Foundry Neural: Vector Labs", page_icon="🧬")

@st.cache_resource
def load_resources():
    # This turns your text into math (Vectors)
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    # FAISS Index (384 dimensions for this specific encoder)
    index = faiss.IndexFlatL2(384)
    return encoder, index

encoder, faiss_index = load_resources()

# --- 2. CONFIG & API ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('models/gemini-3-flash-preview')
else:
    st.error("Missing API Key in Secrets!")
    st.stop()

# --- 3. THE MEMORY VAULT FUNCTIONS ---
def get_vault_data():
    if os.path.exists('vault_vectors.csv'):
        return pd.read_csv('vault_vectors.csv')
    return pd.DataFrame(columns=["Topic", "Context"])

def sync_to_faiss(topic, context):
    # 1. Vectorize the text
    vector = encoder.encode([context]).astype('float32')
    # 2. Add to the FAISS index
    faiss_index.add(vector)
    # 3. Save a text backup
    df = get_vault_data()
    new_row = pd.DataFrame({"Topic": [topic], "Context": [context]})
    pd.concat([df, new_row], ignore_index=True).to_csv('vault_vectors.csv', index=False)

def vector_search(query, k=1):
    df = get_vault_data()
    if df.empty or faiss_index.ntotal == 0:
        return "General Intelligence Mode (No vault data found)."
    
    # Search the "Map" for the closest match
    query_vector = encoder.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_vector, k)
    
    # Get the best text match
    return df.iloc[indices[0][0]]['Context']

# --- 4. THE UI ---
st.title("🧬 Foundry Neural")

with st.sidebar:
    st.header(" Memory")
    t = st.text_input("Topic")
    c = st.text_area("Context/Rule")
    if st.button("SYNC TO FAISS"):
        sync_to_faiss(t, c)
        st.success("Memory Vectorized!")

# --- 5. ADVISOR LOGIC ---
user_query = st.text_input("What is the situation?", placeholder="Ask for advice...")

if st.button("ACTIVATE ADVISOR") and user_query:
    with st.spinner("Searching Vector Space..."):
        # Local search (doesn't use API quota!)
        relevant_context = vector_search(user_query)
        
        # Smart Prompt (Smaller = Safer for Quota)
        prompt = f"""
        Advisor Role: Strategic Peer.
        Vault Memory: {relevant_context}
        Inquiry: {user_query}
        
        Provide:
        1. Direct Answer.
        2. Pros/Cons Table.
        3. Lead Scientist Advice.
        """
        
        resp = model.generate_content(prompt)
        st.session_state.advice = resp.text

if 'advice' in st.session_state:
    st.markdown("---")
    st.markdown(st.session_state.advice)
    

            
    
            
