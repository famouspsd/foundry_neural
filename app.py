import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import faiss
import os
from sentence_transformers import SentenceTransformer

# --- NEURAL ENGINE SETUP ---
@st.cache_resource
def init_cortex():
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.IndexFlatL2(384) # Local vector index for learning
    return encoder, index

encoder, faiss_index = init_cortex()
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# --- THE LEARNING & DECISION LOGIC ---
def deploy_thinking_agent(user_goal):
    # Recall relevant context from local memory
    context = "" 
    if os.path.exists('vector_vault.csv'):
        df = pd.read_csv('vector_vault.csv')
        query_vec = encoder.encode([user_goal]).astype('float32')
        _, indices = faiss_index.search(query_vec, k=3)
        context = " ".join(df.iloc[indices[0]]['Context'].values)

    # Enable 2026 Native Thinking Configuration
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        temperature=1.0 # High creativity for strategic thinking
    )

    prompt = f"[CONTEXT] {context}\n[MISSION] {user_goal}\nAnalyze, Think, and Decide."
    
    # Using the latest 2026 reasoning model
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt,
        config=config
    )
    
    # Process thoughts vs results
    thought_log = "".join([p.text for p in response.candidates[0].content.parts if p.thought])
    final_decision = "".join([p.text for p in response.candidates[0].content.parts if not p.thought])
    
    return thought_log, final_decision

# --- STREAMLIT UI ---
st.title("🤖 Foundry Neural: Thinking Agent")

mission = st.text_input("Deploy Mission:", placeholder="Enter a complex task...")

if st.button("ACTIVATE REASONING") and mission:
    with st.status("Agent is Thinking Deeply...", expanded=True) as status:
        thought, decision = deploy_thinking_agent(mission)
        
        with st.expander("🧠 View Chain of Thought"):
            st.write(thought)
            
        st.markdown("### 🎯 Final Strategic Decision")
        st.markdown(decision)
        
        # LEARNING: Automatically store the core logic of this decision
        pd.DataFrame({"Topic": ["Decision Log"], "Context": [decision[:250]]}).to_csv(
            'vector_vault.csv', mode='a', header=not os.path.exists('vector_vault.csv'), index=False
        )
        status.update(label="Mission Accomplished & Logged!", state="complete")
    
    
    

            
    
            
