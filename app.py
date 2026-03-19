
    import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import faiss
import os
from sentence_transformers import SentenceTransformer

# --- 1. THE CORTEX (FAISS + LOCAL ENCODER) ---
@st.cache_resource
def init_cortex():
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.IndexFlatL2(384)
    return encoder, index

encoder, faiss_index = init_cortex()

# --- 2. 2026 SDK CLIENT ---
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# --- 3. MEMORY TOOLS ---
def recall_memory(query):
    if not os.path.exists('vector_vault.csv'): return ""
    df = pd.read_csv('vector_vault.csv')
    query_vec = encoder.encode([query]).astype('float32')
    _, indices = faiss_index.search(query_vec, k=3)
    return " ".join(df.iloc[indices[0]]['Context'].values)

def learn_new_insight(topic, insight):
    # This is the "Learning" part of the loop
    vector = encoder.encode([insight]).astype('float32')
    faiss_index.add(vector)
    pd.DataFrame({"Topic":[topic],"Context":[insight]}).to_csv(
        'vector_vault.csv', mode='a', header=not os.path.exists('vector_vault.csv'), index=False
    )

# --- 4. THE DEEP THINKING ENGINE ---
def deploy_agent(user_goal):
    context = recall_memory(user_goal)
    
    # Enable Gemini 3's Native Thinking
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(include_thoughts=True),
        temperature=1.0 # Optimized for 2026 reasoning models
    )
    
    prompt = f"""
    [MISSION] {user_goal}
    [VAULT DATA] {context}
    [INSTRUCTION] Think deeply. Evaluate pros/cons. Decide on the best move.
    """
    
    response = client.models.generate_content(
        model='gemini-3-flash', # Using the latest 2026 reasoning model
        contents=prompt,
        config=config
    )
    
    # Extract thought and final answer
    thought_process = ""
    final_answer = ""
    
    for part in response.candidates[0].content.parts:
        if part.thought:
            thought_process += part.text
        else:
            final_answer += part.text
            
    return thought_process, final_answer

# --- 5. UI ---
st.title("🤖 Foundry Neural: Autonomous Agent v3")

user_mission = st.text_input("Deploy Mission:", placeholder="What should I analyze today?")

if st.button("EXECUTE MISSION") and user_mission:
    with st.status("Agent is Thinking...", expanded=True) as status:
        thought, answer = deploy_agent(user_mission)
        
        with st.expander("🧠 View Internal Reasoning (Chain of Thought)"):
            st.info(thought)
        
        st.markdown(answer)
        
        # SELF-LEARNING STEP: The agent saves its own conclusion to the vault
        learn_new_insight("Autonomous Conclusion", f"Task: {user_mission} | Result: {answer[:200]}")
        status.update(label="Mission Accomplished & Learned!", state="complete")
    
    

            
    
            
