import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- RESEARCHER CONFIG ---
# This pulls your key from the Streamlit Cloud "Secrets" settings
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("API Key not found. Please set GEMINI_API_KEY in Streamlit Secrets.")

model = genai.GenerativeModel('models/gemini-3-flash-preview')

st.set_page_config(page_title="Foundry Neural | Research Lab", page_icon="🔬", layout="wide")

# Minimalist Typography UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #050505; color: #E0E0E0; }
    .stTextInput>div>div>input { background-color: #111; border: 1px solid #333; color: white; }
    .stButton>button { width: 100%; background-color: white; color: black; font-weight: 600; border-radius: 0px; }
    </style>
    """, unsafe_allow_html=True)

def load_vault():
    try:
        return pd.read_csv('vault.csv')
    except:
        return pd.DataFrame(columns=["Context", "Topic"])

def save_vault(df):
    df.to_csv('vault.csv', index=False)

st.title("🔬 Foundry Neural: Innovation Pipeline")
tab1, tab2 = st.tabs(["🚀 Execute Innovation", "📂 Research Archive"])

with tab2:
    st.header("Memory Consolidation")
    col1, col2 = st.columns([2, 1])
    with col1:
        new_context = st.text_area("New Knowledge/Hypothesis Cell:")
    with col2:
        new_topic = st.text_input("Domain:")
        if st.button("CONSOLIDATE TO VAULT"):
            if new_context:
                df = load_vault()
                new_row = pd.DataFrame({"Context": [new_context], "Topic": [new_topic]})
                df = pd.concat([df, new_row], ignore_index=True)
                save_vault(df)
                st.success("Knowledge Cell Integrated.")
                st.rerun()

    st.markdown("---")
    vault_df = load_vault()
    if not vault_df.empty:
        st.table(vault_df)

with tab1:
    st.header("Strategic Research Query")
    query = st.text_input("ENTER RESEARCH INQUIRY:", placeholder="How can we solve...")
    
    if st.button("ACTIVATE REASONING"):
        if query and not vault_df.empty:
            with st.spinner("Synthesizing Innovation..."):
                words = query.lower().split()
                match = vault_df[vault_df.apply(lambda r: any(w in str(r['Context']).lower() for w in words), axis=1)]
                context = match.iloc[0]['Context'] if not match.empty else "Global Optimization Foundations."

                prompt = f"""
                SYSTEM: You are the Lead Scientist at Foundry Neural Labs. 
                RESEARCHER ARCHIVE: {context}
                INQUIRY: {query}
                TASK: Propose a novel invention or framework to solve a global problem. Use LaTeX for math.
                """
                
                response = model.generate_content(prompt)
                st.subheader("💡 Proposed Innovation")
                st.markdown(response.text)
        else:
            st.warning("Archive empty or no query entered.")