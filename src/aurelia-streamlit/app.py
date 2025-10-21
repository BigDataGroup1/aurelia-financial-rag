import streamlit as st
import requests
import json
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="AURELIA - Financial Concept Generator",
    page_icon="📊",
    layout="wide"
)

# API URL - connects to your teammate's FastAPI
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.5rem 0;
        background-color: #d4edda;
        color: #155724;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 AURELIA</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Automated Financial Concept Note Generator</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url_input = st.text_input(
        "FastAPI Base URL",
        value=API_BASE_URL,
        help="Enter your deployed FastAPI service URL"
    )
    
    if api_url_input:
        API_BASE_URL = api_url_input.rstrip('/')
    
    st.divider()
    st.header("📚 Sample Concepts")
    sample_concepts = [
        "Duration",
        "Sharpe Ratio",
        "Black-Scholes",
        "Value at Risk",
        "Monte Carlo Simulation",
        "Beta",
        "CAPM"
    ]
    
    for concept in sample_concepts:
        if st.button(concept, key=f"sample_{concept}", use_container_width=True):
            st.session_state['selected_concept'] = concept

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []

# Main content
st.header("🔍 Query Concept")

# Get concept from session state if available
default_concept = st.session_state.get('selected_concept', '')

concept_input = st.text_input(
    "Enter Financial Concept",
    value=default_concept,
    placeholder="e.g., Duration, Sharpe Ratio, Black-Scholes",
    help="Enter the financial concept you want to learn about"
)

# Clear the session state after using it
if 'selected_concept' in st.session_state:
    del st.session_state['selected_concept']

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    query_button = st.button("🔎 Query Concept", type="primary", use_container_width=True)

with col_btn2:
    force_refresh = st.checkbox("Force Refresh", help="Bypass cache and regenerate note")

# Query endpoint handler
if query_button and concept_input:
    with st.spinner(f'🔍 Querying concept: **{concept_input}**...'):
        try:
            start_time = time.time()
            
            # Make POST request to /query endpoint
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={
                    "concept": concept_input,
                    "force_refresh": force_refresh
                },
                timeout=60
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the concept_note object
                concept_note = result.get('concept_note', {})
                
                # Add to history
                result['query_latency'] = latency
                st.session_state['query_history'].insert(0, result)
                
                # Display result
                st.success(f"✅ Concept note retrieved in {latency:.2f}s")
                
                # Display source and cache status
                source = concept_note.get('source', 'Unknown')
                
                col_badge1, col_badge2, col_badge3 = st.columns(3)
                with col_badge1:
                    st.markdown(f'<span class="source-badge">📄 Source: {source}</span>', unsafe_allow_html=True)
                
                with col_badge2:
                    if result.get('cached', False):
                        st.markdown('<span class="source-badge" style="background-color: #d1ecf1; color: #0c5460;">💾 Cached</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="source-badge" style="background-color: #fff3cd; color: #856404;">✨ Newly Generated</span>', unsafe_allow_html=True)
                
                with col_badge3:
                    confidence = concept_note.get('confidence', 0)
                    st.markdown(f'<span class="source-badge" style="background-color: #cce5ff; color: #004085;">🎯 Confidence: {confidence:.0%}</span>', unsafe_allow_html=True)
                
                st.divider()
                
                # Display the concept name
                st.markdown(f"## 📌 {concept_note.get('concept_name', concept_input)}")
                
                # Definition
                st.markdown("### 📝 Definition")
                st.info(concept_note.get('definition', 'No definition available'))
                
                # Key Components
                if concept_note.get('key_components'):
                    st.markdown("### 🔑 Key Components")
                    for idx, component in enumerate(concept_note['key_components'], 1):
                        st.markdown(f"**{idx}.** {component}")
                
                # Formula (if available)
                if concept_note.get('formula'):
                    st.markdown("### 🔢 Formula")
                    st.code(concept_note['formula'], language='latex')
                
                # Example
                if concept_note.get('example'):
                    st.markdown("### 💡 Example")
                    st.success(concept_note['example'])
                
                # Use Cases
                if concept_note.get('use_cases'):
                    st.markdown("### 🎯 Use Cases")
                    for idx, use_case in enumerate(concept_note['use_cases'], 1):
                        st.markdown(f"**{idx}.** {use_case}")
                
                # Related Concepts
                if concept_note.get('related_concepts'):
                    st.markdown("### 🔗 Related Concepts")
                    related = ", ".join(concept_note['related_concepts'])
                    st.info(f"**Related topics**: {related}")
                
                # Display citations if from PDF
                if concept_note.get('page_references') and source == 'fintbx.pdf':
                    st.markdown("### 📚 PDF References")
                    pages = ", ".join([f"Page {p}" for p in concept_note['page_references']])
                    st.success(f"**Cited from Financial Toolbox**: {pages}")
                
                # Additional metadata
                with st.expander("🔍 View Full Response Metadata"):
                    st.json({
                        "concept": concept_note.get('concept_name'),
                        "source": source,
                        "cached": result.get('cached'),
                        "confidence": confidence,
                        "retrieved_chunks": result.get('retrieved_chunks', 0),
                        "generation_time_ms": result.get('generation_time_ms', 0),
                        "query_latency_s": f"{latency:.3f}",
                        "timestamp": result.get('timestamp')
                    })
                
            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The API might be processing a large request. Try again.")
        except requests.exceptions.ConnectionError:
            st.error(f"🔌 Cannot connect to API at {API_BASE_URL}")
            st.info("**Make sure FastAPI is running!**")
            st.code("Terminal 1:\ncd BIG_DATA_PROJECT3/aurelia-financial-rag\npython -m uvicorn src.lab3.main:app --reload --host 0.0.0.0 --port 8000")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Query History Section
if st.session_state['query_history']:
    st.divider()
    st.header("📜 Recent Queries")
    
    for idx, query in enumerate(st.session_state['query_history'][:5]):
        concept_note = query.get('concept_note', {})
        with st.expander(f"🔹 {concept_note.get('concept_name', 'Unknown')} ({query.get('query_latency', 0):.2f}s)"):
            col_h1, col_h2 = st.columns([3, 1])
            
            with col_h1:
                st.markdown(f"**Definition**: {concept_note.get('definition', 'N/A')[:150]}...")
            
            with col_h2:
                st.metric("Source", concept_note.get('source', 'Unknown'))
                st.caption(f"{'✓ Cached' if query.get('cached') else 'Generated'}")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>AURELIA - Automated Financial Concept Note Generator</p>
        <p>Built with Streamlit | Powered by RAG & LLMs</p>
    </div>
""", unsafe_allow_html=True)