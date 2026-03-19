import streamlit as st
import pandas as pd
import google.generativeai as genai
from pypdf import PdfReader
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image

# --- CONFIG ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('models/gemini-3-flash-preview')
else:
    st.error("API Key not found!")
    st.stop()

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedder()

# --- FUNCTIONS ---
def load_vault():
    try:
        return pd.read_csv('vault.csv')
    except:
        # Create it if it doesn't exist
        df = pd.DataFrame(columns=["Topic", "Context"])
        df.to_csv('vault.csv', index=False)
        return df

def save_to_vault(topic, context):
    df = load_vault()
    new_row = pd.DataFrame({"Topic": [topic], "Context": [context]})
    updated_df = pd.concat([df, new_row], ignore_index=True)
    updated_df.to_csv('vault.csv', index=False)

def create_safe_pdf(text):
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, txt=clean_text)
    return pdf.output()

# --- SIDEBAR: DIRECT VAULT UPDATE ---
with st.sidebar:
    st.title("🛡️ Vault Management")
    
    with st.expander("➕ Add Direct Entry", expanded=True):
        new_topic = st.text_input("Topic (e.g., Advisor Logic)")
        new_context = st.text_area("Context (The actual data/rule)")
        if st.button("🚀 COMMIT TO VAULT"):
            if new_topic and new_context:
                save_to_vault(new_topic, new_context)
                st.success(f"Added '{new_topic}' to memory!")
            else:
                st.warning("Please fill both fields.")

    st.markdown("---")
    uploaded_img = st.file_uploader("Visual Input (Vision)", type=["jpg", "png", "jpeg"])

# --- MAIN HUB ---
st.title("Foundry Neural: Strategic Advisor")

query = st.text_input("What is on your mind?", placeholder="Ask for advice, a decision, or research...")

if st.button("ACTIVATE ADVISOR") and query:
    with st.spinner("Consulting the Vault..."):
        vault_df = load_vault()
        
        # Simple retrieval: search for relevant context
        context_str = "Standard Advisor Logic"
        if not vault_df.empty:
            # We use the first few rows as a base if semantic search isn't needed for simple advice
            context_str = " ".join(vault_df['Context'].tail(5).tolist())

        # The "Friendly Advisor" Prompt
        prompt = f"""
        You are the Foundry Neural Strategic Advisor. 
        Context from Vault: {context_str}
        
        User Inquiry: {query}
        
        Please provide your response in this format:
        1. Friendly, Direct Answer.
        2. Pros and Cons Table.
        3. 'Lead Scientist' Advice (Strategic next steps).
        """
        
        if uploaded_img:
            img = Image.open(uploaded_img)
            resp = model.generate_content([prompt, img])
        else:
            resp = model.generate_content(prompt)
            
        st.session_state.advice_out = resp.text

# --- OUTPUT & DOWNLOAD ---
if 'advice_out' in st.session_state:
    st.markdown("---")
    st.markdown(st.session_state.advice_out)
    
    try:
        pdf_bytes = create_safe_pdf(st.session_state.advice_out)
        st.download_button(
            label="📥 DOWNLOAD ADVICE REPORT",
            data=pdf_bytes,
            file_name="Strategic_Advice.pdf",
            mime="application/pdf",
            key="advice_download"
        )
    except:
        st.info("Generating PDF...")

            
    
            
