import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import faiss
import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Callable
from sentence_transformers import SentenceTransformer
import requests

# ============================================
# CONFIGURATION
# ============================================
MEMORY_DIR = "neural_memory"
FAISS_INDEX_PATH = f"{MEMORY_DIR}/faiss_index.bin"
METADATA_PATH = f"{MEMORY_DIR}/metadata.pkl"
SESSION_HISTORY_PATH = f"{MEMORY_DIR}/session_history.json"

os.makedirs(MEMORY_DIR, exist_ok=True)

# ============================================
# TOOL DEFINITIONS (Agent can now ACT)
# ============================================
class ToolRegistry:
    """Extensible tool system for the agent"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
    
    def register(self, name: str, func: Callable, description: str):
        self.tools[name] = {"func": func, "description": description}
    
    def execute(self, tool_name: str, **kwargs):
        if tool_name in self.tools:
            return self.tools[tool_name]["func"](**kwargs)
        return f"Error: Tool '{tool_name}' not found"
    
    def get_tool_descriptions(self) -> str:
        return "\n".join([f"- {name}: {info['description']}" 
                         for name, info in self.tools.items()])

# Initialize tool registry
tool_registry = ToolRegistry()

# Example tools (add more as needed)
def web_search(query: str) -> str:
    """Simulated web search - replace with real API (SerpAPI, Bing, etc.)"""
    return f"[Web Search Result for: {query}]\nSimulated search result..."

def calculate(expression: str) -> str:
    """Safe mathematical calculation"""
    try:
        # Safe eval with limited scope
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

def save_to_memory(key: str, value: str) -> str:
    """Save important information to persistent memory"""
    timestamp = datetime.now().isoformat()
    entry = {
        "timestamp": timestamp,
        "key": key,
        "value": value,
        "type": "explicit_memory"
    }
    # Append to memory file
    memory_file = f"{MEMORY_DIR}/explicit_memories.json"
    memories = []
    if os.path.exists(memory_file):
        with open(memory_file, 'r') as f:
            memories = json.load(f)
    memories.append(entry)
    with open(memory_file, 'w') as f:
        json.dump(memories, f, indent=2)
    return f"Saved to memory: {key}"

# Register tools
tool_registry.register("web_search", web_search, "Search the internet for current information")
tool_registry.register("calculate", calculate, "Perform mathematical calculations")
tool_registry.register("save_memory", save_to_memory, "Save important facts to long-term memory")

# ============================================
# NEURAL ENGINE SETUP (Persistent Memory)
# ============================================
class NeuralCortex:
    """Enhanced cognitive engine with persistent vector memory"""
    
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()
    
    def _load_or_create_index(self):
        if os.path.exists(FAISS_INDEX_PATH):
            return faiss.read_index(FAISS_INDEX_PATH)
        return faiss.IndexFlatL2(self.dimension)
    
    def _load_metadata(self) -> List[Dict]:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'rb') as f:
                return pickle.load(f)
        return []
    
    def _save_state(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def recall(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant memories"""
        if self.index.ntotal == 0:
            return []
        
        query_vec = self.encoder.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata) and idx >= 0:
                memory = self.metadata[idx].copy()
                memory['relevance_score'] = float(1 / (1 + dist))
                results.append(memory)
        return results
    
    def learn(self, text: str, memory_type: str = "decision", 
              topic: str = "general", importance: float = 1.0):
        """Store new knowledge in vector memory"""
        embedding = self.encoder.encode([text]).astype('float32')
        self.index.add(embedding)
        
        memory_entry = {
            "id": len(self.metadata),
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "type": memory_type,
            "topic": topic,
            "importance": importance
        }
        self.metadata.append(memory_entry)
        self._save_state()
        return memory_entry
    
    def get_stats(self) -> Dict:
        return {
            "total_memories": self.index.ntotal,
            "memory_types": pd.DataFrame(self.metadata)['type'].value_counts().to_dict() if self.metadata else {}
        }

# Initialize cortex
@st.cache_resource
def init_cortex():
    return NeuralCortex()

cortex = init_cortex()
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ============================================
# SESSION MEMORY (Conversation Context)
# ============================================
class SessionMemory:
    """Short-term working memory for current session"""
    
    def __init__(self):
        self.history: List[Dict] = []
        self._load_history()
    
    def _load_history(self):
        if os.path.exists(SESSION_HISTORY_PATH):
            with open(SESSION_HISTORY_PATH, 'r') as f:
                self.history = json.load(f)
    
    def add_interaction(self, mission: str, thought: str, decision: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "mission": mission,
            "thought_summary": thought[:200] + "..." if len(thought) > 200 else thought,
            "decision_summary": decision[:200] + "..." if len(decision) > 200 else decision
        }
        self.history.append(entry)
        # Keep only last 20 interactions
        self.history = self.history[-20:]
        with open(SESSION_HISTORY_PATH, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get recent mission context"""
        if not self.history:
            return ""
        recent = self.history[-n:]
        context = "Recent missions:\n"
        for i, entry in enumerate(recent, 1):
            context += f"{i}. {entry['mission'][:100]}...\n"
        return context

session_memory = SessionMemory()

# ============================================
# ENHANCED THINKING AGENT
# ============================================
def deploy_thinking_agent(user_goal: str, use_tools: bool = True) -> Dict:
    """
    Enhanced cognitive agent with memory, tools, and structured reasoning
    """
    try:
        # --- MEMORY RECALL ---
        long_term_memories = cortex.recall(user_goal, k=3)
        recent_context = session_memory.get_recent_context(n=2)
        
        memory_context = ""
        if long_term_memories:
            memory_context = "Relevant past knowledge:\n" + "\n".join([
                f"- [{m['type']}] {m['text'][:150]}..." 
                for m in long_term_memories
            ])
        
        # --- TOOL AUGMENTATION ---
        tool_context = ""
        if use_tools:
            tool_context = f"""
Available tools to help accomplish this mission:
{tool_registry.get_tool_descriptions()}

If you need to use a tool, format your request as:
TOOL_CALL: tool_name|param1=value1|param2=value2
"""
        
        # --- PROMPT ENGINEERING ---
        system_prompt = f"""You are Foundry Neural, an advanced cognitive agent with memory and reasoning capabilities.

{memory_context}

{recent_context}

{tool_context}

MISSION: {user_goal}

Follow this reasoning process:
1. ANALYZE: Break down the problem and identify what you know vs. need to know
2. RECALL: Reference relevant memories above if applicable
3. REASON: Think step-by-step through the solution
4. DECIDE: Provide a clear, actionable final decision
5. REFLECT: Briefly note what you learned from this mission

Structure your response with clear headers for each phase."""

        # --- API CALL WITH BETTER CONFIG ---
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            temperature=0.7,  # Balanced: creative but consistent
            top_p=0.9,
            max_output_tokens=2048
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=system_prompt,
            config=config
        )
        
        # --- RESPONSE PROCESSING ---
        parts = response.candidates[0].content.parts
        
        thought_log = "".join([p.text for p in parts if getattr(p, 'thought', False)])
        final_output = "".join([p.text for p in parts if not getattr(p, 'thought', False)])
        
        # --- LEARNING: Store this interaction ---
        # Store the decision with full context for future retrieval
        cortex.learn(
            text=f"Mission: {user_goal}\nDecision: {final_output[:500]}",
            memory_type="mission_outcome",
            topic=user_goal[:50],
            importance=0.8
        )
        
        # Store in session history
        session_memory.add_interaction(user_goal, thought_log, final_output)
        
        return {
            "success": True,
            "thought_log": thought_log,
            "final_output": final_output,
            "memories_used": len(long_term_memories),
            "tools_available": len(tool_registry.tools)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "thought_log": "",
            "final_output": f"Neural engine encountered an error: {str(e)}"
        }

# ============================================
# STREAMLIT UI (Enhanced)
# ============================================
st.set_page_config(
    page_title="Foundry Neural | Cognitive Agent",
    page_icon="🧠",
    layout="wide"
)

# Sidebar with system stats
with st.sidebar:
    st.header("🧠 Neural Cortex Status")
    stats = cortex.get_stats()
    st.metric("Total Memories", stats['total_memories'])
    st.json(stats['memory_types'])
    
    st.divider()
    st.header("⚙️ Configuration")
    use_tools = st.toggle("Enable Tool Use", value=True)
    show_memories = st.toggle("Show Retrieved Memories", value=False)
    
    st.divider()
    if st.button("🗑️ Clear All Memory"):
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(METADATA_PATH):
            os.remove(METADATA_PATH)
        st.rerun()

# Main interface
st.title("🤖 Foundry Neural: Thinking Agent")
st.caption("Persistent Memory • Tool Augmentation • Chain-of-Thought Reasoning")

# Mission input with examples
col1, col2 = st.columns([3, 1])
with col1:
    mission = st.text_input(
        "Deploy Mission:", 
        placeholder="Enter a complex task... (e.g., 'Analyze the pros and cons of remote work for software teams')",
        key="mission_input"
    )
with col2:
    st.write("")
    st.write("")
    activate = st.button("🚀 ACTIVATE REASONING", use_container_width=True)

if activate and mission:
    with st.status("🧠 Neural pathways activating...", expanded=True) as status:
        
        # Show memory retrieval in real-time
        if show_memories:
            retrieved = cortex.recall(mission, k=3)
            if retrieved:
                st.info(f"📚 Retrieved {len(retrieved)} relevant memories")
                for mem in retrieved:
                    st.caption(f"[{mem['type']}] {mem['text'][:100]}...")
        
        # Execute thinking
        result = deploy_thinking_agent(mission, use_tools=use_tools)
        
        if result['success']:
            status.update(label="✅ Mission Accomplished & Learned!", state="complete")
            
            # Display chain of thought
            with st.expander("🧠 View Chain of Thought", expanded=True):
                if result['thought_log']:
                    st.markdown(result['thought_log'])
                else:
                    st.info("No explicit thought log available from this model configuration.")
                st.caption(f"Memories integrated: {result['memories_used']}")
            
            # Display final decision
            st.markdown("---")
            st.markdown("### 🎯 Final Strategic Decision")
            st.markdown(result['final_output'])
            
            # Feedback mechanism for learning
            col_good, col_bad = st.columns(2)
            with col_good:
                if st.button("👍 Good Response", key="good"):
                    cortex.learn(
                        text=f"High quality response for: {mission}",
                        memory_type="quality_feedback",
                        importance=1.0
                    )
                    st.toast("Feedback recorded for learning!")
            with col_bad:
                if st.button("👎 Needs Improvement", key="bad"):
                    st.toast("Noted for future improvement")
                    
        else:
            status.update(label="❌ Mission Failed", state="error")
            st.error(result['final_output'])

# Footer
st.divider()
st.caption(f"Foundry Neural v2.0 | Memories: {cortex.get_stats()['total_memories']} | Session interactions: {len(session_memory.history)}")

    
    
    

            
    
            
