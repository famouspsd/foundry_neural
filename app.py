import streamlit as st
import pandas as pd

# --- EMERGENCY BOOT CHECK ---
st.set_page_config(page_title="Foundry Neural Labs", page_icon="🔬", layout="wide")

try:
    import google.generativeai as genai
    from pypdf import PdfReader
    from fpdf import FPDF
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from PIL import Image
except Exception as e:
    st.error(f"Initialization Failed. Missing library: {e}")
    st.stop()

# --- CONFIG ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('models/gemini-3-flash-preview')
    
else:
    st.error("API Key not found in Secrets!")
    st.stop()

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedder()

# --- UI THEME ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    .stButton>button { background-color: #238636; color: white; border-radius: 6px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- APP LOGIC ---
def load_vault():
    try: return pd.read_csv('vault.csv')
    except: return pd.DataFrame(columns=["Context", "Topic"])

def create_safe_pdf(text):
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, txt=clean_text)
    return pdf.output()

# --- SIDEBAR ---
with st.sidebar:
    st.title("➕ Lab Inputs")
    uploaded_img = st.file_uploader("Upload Image (Vision)", type=["jpg", "png", "jpeg"])
    st.markdown("---")
    uploaded_pdf = st.file_uploader("Ingest Research PDF", type="pdf")
    
    if uploaded_pdf and st.button("SYNC PDF"):
        reader = PdfReader(uploaded_pdf)
        text = " ".join([p.extract_text() for p in reader.pages])
        df = load_vault()
        new_row = pd.DataFrame({"Context": [text[:2000]], "Topic": ["Manual Ingest"]})
        pd.concat([df, new_row]).to_csv('vault.csv', index=False)
        st.success("Synced!")

# --- MAIN INTERFACE ---
st.title("Foundry Neural Synthesis")
query = st.text_input("Enter Research Inquiry", placeholder="Ask your lab...")

if st.button("ACTIVATE REASONING") and query:
    with st.spinner("Processing..."):
        vault_df = load_vault()
        # Simple context pull for stability
        context = vault_df['Context'].iloc[0] if not vault_df.empty else "General Knowledge"
        
        if uploaded_img:
            img = Image.open(uploaded_img)
            resp = model.generate_content([f"Vault: {context}\n\nTask: {query}", img])
        else:
            resp = model.generate_content(f"Vault: {context}\n\nTask: {query}")
        
        st.session_state.out = resp.text
        st.markdown(st.session_state.out)

if 'out' in st.session_state:
    st.download_button("📥 DOWNLOAD PDF", data=create_safe_pdf(st.session_state.out), file_name="research.pdf")
                                                          
    

            
