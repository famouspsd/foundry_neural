import streamlit as st
import pandas as pd
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# --- 1. CONFIG & THEME ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("API Key missing in Secrets.")

model = genai.GenerativeModel('models/gemini-3-flash-preview')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="Foundry Neural", page_icon="♾️", layout="wide")

# Replit-Inspired Custom CSS
st.markdown("""
    <style>
    /* Background and Global */
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    
    /* Input Area Styling */
    .stTextInput>div>div>input {
        background-color: #0d1117;
        color: white;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        font-size: 18px;
    }
    
    /* Main Button Styling */
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    /* Header Font */
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC ---
def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    return " ".join([page.extract_text() for page in reader.pages])

def load_vault():
    try: return pd.read_csv('vault.csv')
    except: return pd.DataFrame(columns=["Context", "Topic"])

def semantic_search(query, df):
    if df.empty: return "Foundational AI Ethics"
    query_vec = embed_model.encode([query])[0]
    vault_vecs = embed_model.encode(df['Context'].tolist())
    similarities = np.dot(vault_vecs, query_vec) / (np.linalg.norm(vault_vecs, axis=1) * np.linalg.norm(query_vec))
    return df.iloc[np.argmax(similarities)]['Context']

# --- 3. SIDEBAR (MANAGEMENT) ---
with st.sidebar:
    st.title("♾️ Foundry Lab")
    st.markdown("---")
    
    st.subheader("📂 Vault Storage")
    vault_df = load_vault()
    st.write(f"Knowledge Cells: {len(vault_df)}")
    if st.checkbox("Show Raw Vault"):
        st.dataframe(vault_df)
    
    st.markdown("---")
    st.subheader("📄 PDF Ingestion")
    uploaded_pdf = st.file_uploader("Drop research papers here", type="pdf")
    if uploaded_pdf and st.button("Start Ingest"):
        with st.spinner("Processing..."):
            text = get_pdf_text(uploaded_pdf)
            summary = model.generate_content(f"Technical summary for vault: {text[:4000]}").text
            new_row = pd.DataFrame({"Context": [summary], "Topic": ["Automated Ingest"]})
            pd.concat([vault_df, new_row]).to_csv('vault.csv', index=False)
            st.success("Integrated.")
            st.rerun()

# --- 4. MAIN INTERFACE (THE "REPLIT" HUB) ---
st.markdown("<br><br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.markdown("<h1 style='text-align: center;'>What do you want to innovate?</h1>", unsafe_allow_html=True)
    
    query = st.text_input("", placeholder="Describe an invention or ask a research question...", label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])
    with btn_col2:
        activate = st.button("Activate Reasoning")

    if activate and query:
        with st.spinner("Synthesizing..."):
            context = semantic_search(query, vault_df)
            prompt = f"SYSTEM: Lead Scientist. CONTEXT: {context}. INQUIRY: {query}. TASK: Framework + LaTeX math."
            response = model.generate_content(prompt)
            
            st.markdown("---")
            st.markdown("### 💡 Theoretical Output")
            st.markdown(response.text)
    
