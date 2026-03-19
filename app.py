import streamlit as st
import pandas as pd
import google.generativeai as genai
from pypdf import PdfReader
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image

# --- CONFIG & SECRETS ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('models/gemini-3-flash-preview')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="Foundry Neural Labs", page_icon="🔬", layout="wide")

# --- UI STYLING (REPLIT DARK) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    .stButton>button { background-color: #238636; color: white; border-radius: 6px; width: 100%; }
    h1, h2, h3 { color: #58a6ff; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCTIONS ---
def load_vault():
    try: return pd.read_csv('vault.csv')
    except: return pd.DataFrame(columns=["Context", "Topic"])

def semantic_search(query, df):
    if df.empty: return "General Logic"
    query_vec = embed_model.encode([query])[0]
    vault_vecs = embed_model.encode(df['Context'].tolist())
    similarities = np.dot(vault_vecs, query_vec) / (np.linalg.norm(vault_vecs, axis=1) * np.linalg.norm(query_vec))
    return df.iloc[np.argmax(similarities)]['Context']

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text)
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR: VAULT & VISION ---
with st.sidebar:
    st.title("♾️ Lab Controls")
    
    st.subheader("🖼️ Vision Analysis")
    uploaded_img = st.file_uploader("Upload Diagram/Notes", type=["jpg", "png", "jpeg"])
    
    st.subheader("📂 Vault Status")
    vault_df = load_vault()
    st.write(f"Knowledge Cells: {len(vault_df)}")
    
    st.subheader("📄 PDF Ingestion")
    uploaded_pdf = st.file_uploader("Digest Research Paper", type="pdf")
    if uploaded_pdf and st.button("INGEST"):
        # (PDF ingestion logic here...)
        st.success("Ingested.")

# --- MAIN HUB ---
st.markdown("<h1 style='text-align: center;'>Foundry Neural Strategic Synthesis</h1>", unsafe_allow_html=True)

query = st.text_input("", placeholder="Enter inquiry (e.g., 'Apply Game Theory to Grid Decay')...", label_visibility="collapsed")

if st.button("ACTIVATE REASONING") and query:
    with st.spinner("Synthesizing..."):
        context = semantic_search(query, vault_df)
        
        # Check if an image is uploaded for Vision support
        if uploaded_img:
            img = Image.open(uploaded_img)
            response = model.generate_content([f"Context: {context}\nInquiry: {query}", img])
        else:
            response = model.generate_content(f"CONTEXT: {context}\nINQUIRY: {query}")
        
        st.session_state.last_output = response.text
        st.markdown("### 💡 Theoretical Output")
        st.markdown(st.session_state.last_output)

# --- EXPORT BUTTON ---
if 'last_output' in st.session_state:
    st.download_button(
        label="📥 DOWNLOAD RESEARCH PDF",
        data=create_pdf(st.session_state.last_output),
        file_name="Foundry_Research_Export.pdf",
        mime="application/pdf"
    )
    def create_pdf(text):
    # Use 'latin-1' safe encoding by replacing non-standard characters
    # This prevents the UnicodeEncodingException
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12) # Helvetica is safer for standard exports
    pdf.multi_cell(0, 10, txt=clean_text)
    
    # Return as bytes
    return pdf.output()
    

    
