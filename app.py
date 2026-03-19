import streamlit as st
import pandas as pd
import google.generativeai as genai
from pypdf import PdfReader
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image

# --- 1. CORE CONFIG ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")

model = genai.GenerativeModel('models/gemini-3-flash-preview')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="Foundry Neural Labs", page_icon="🔬", layout="wide")

# --- 2. THE "STANDARD-ONE" THEME (Dark Mode & UI) ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    .stTextInput>div>div>input { background-color: #0d1117; color: white; border: 1px solid #30363d; border-radius: 8px; padding: 15px; }
    .stButton>button { background-color: #238636; color: white; border-radius: 6px; border: none; font-weight: bold; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background-color: #2ea043; border: 1px solid #fff; }
    .css-1offfwp { font-family: 'Inter', sans-serif; }
    /* The "Plus" Button Styling for Uploads */
    .upload-text { font-size: 14px; color: #8b949e; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIC ENGINES ---
def load_vault():
    try: return pd.read_csv('vault.csv')
    except: return pd.DataFrame(columns=["Context", "Topic"])

def semantic_search(query, df):
    if df.empty: return "Universal Logic Foundations"
    query_vec = embed_model.encode([query])[0]
    vault_vecs = embed_model.encode(df['Context'].tolist())
    similarities = np.dot(vault_vecs, query_vec) / (np.linalg.norm(vault_vecs, axis=1) * np.linalg.norm(query_vec))
    return df.iloc[np.argmax(similarities)]['Context']

def create_safe_pdf(text):
    # Encoding fix: replace symbols that crash FPDF
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, txt=clean_text)
    return pdf.output()

# --- 4. SIDEBAR (THE PLUS BUTTONS & VAULT) ---
with st.sidebar:
    st.markdown("### ➕ Add to Lab")
    
    # Image/Vision Upload (The Plus Button functionality)
    st.markdown("<p class='upload-text'>Upload Diagrams or Math Notes</p>", unsafe_allow_html=True)
    uploaded_img = st.file_uploader("", type=["jpg", "png", "jpeg"], key="img_plus")
    if uploaded_img:
        st.image(uploaded_img, caption="Visual Context Active", use_column_width=True)
    
    st.markdown("---")
    
    # PDF Ingestion
    st.markdown("<p class='upload-text'>Ingest Research Papers (PDF)</p>", unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("", type="pdf", key="pdf_plus")
    if uploaded_pdf and st.button("🚀 SYNC TO VAULT"):
        with st.spinner("Processing Knowledge..."):
            reader = PdfReader(uploaded_pdf)
            text = " ".join([p.extract_text() for p in reader.pages])
            summary = model.generate_content(f"Dense summary for vault: {text[:4000]}").text
            df = load_vault()
            new_data = pd.DataFrame({"Context": [summary], "Topic": ["Document Ingest"]})
            pd.concat([df, new_data]).to_csv('vault.csv', index=False)
            st.success("Vault Updated.")

    st.markdown("---")
    if st.checkbox("Show Raw Vault Data"):
        st.dataframe(load_vault())

# --- 5. MAIN STAGE (RESEARCH HUB) ---
st.markdown("<br><br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 5, 1])

with col2:
    st.markdown("<h1 style='text-align: center;'>Foundry Neural Synthesis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e;'>Polymath Intelligence & Cross-Domain Optimization</p>", unsafe_allow_html=True)
    
    user_query = st.text_input("", placeholder="Enter your research inquiry...", label_visibility="collapsed")
    
    if st.button("ACTIVATE REASONING") and user_query:
        with st.spinner("Accessing Vault & Synthesizing..."):
            vault_df = load_vault()
            context = semantic_search(user_query, vault_df)
            
            # Combine Vision + Text if image exists
            if uploaded_img:
                img = Image.open(uploaded_img)
                response = model.generate_content([f"Vault Context: {context}\n\nTask: {user_query}\nFormat: Scientific Paper style with LaTeX.", img])
            else:
                response = model.generate_content(f"Vault Context: {context}\n\nTask: {user_query}\nFormat: Scientific Paper style with LaTeX.")
            
            st.session_state.research_output = response.text
            
            st.markdown("---")
            st.markdown("### 💡 Theoretical Framework")
            st.markdown(st.session_state.research_output)

    # Export Section
    if 'research_output' in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_bytes = create_safe_pdf(st.session_state.research_output)
        st.download_button(
            label="📥 DOWNLOAD RESEARCH REPORT (PDF)",
            data=pdf_bytes,
            file_name="Foundry_Neural_Research.pdf",
            mime="application/pdf"
    )
            
