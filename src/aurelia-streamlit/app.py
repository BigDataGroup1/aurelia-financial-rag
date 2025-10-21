import streamlit as st
import requests
import json
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="AURELIA - Financial Concept Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL
API_BASE_URL = "http://localhost:8000"

# Modern vibrant CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Vibrant gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism effect for main content */
    .main .block-container {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1rem;
        margin-bottom: 2.5rem;
        font-weight: 500;
    }
    
    /* Styled input box */
    div[data-testid="stTextInput"] > div > div {
        background: white !important;
        border-radius: 12px !important;
        border: 2px solid #e0e7ff !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15) !important;
    }
    
    div[data-testid="stTextInput"] input {
        font-size: 1rem !important;
        padding: 0.8rem !important;
        border: none !important;
    }
    
    div[data-testid="stTextInput"] > div > div:focus-within {
        border: 2px solid #667eea !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    .concept-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25);
        border-color: #667eea;
    }
    
    .badge {
        display: inline-block;
        padding: 0.5rem 1.1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 0.4rem 0.3rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    
    .badge:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
    }
    
    .badge-pdf {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .badge-wiki {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .badge-cached {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .badge-new {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    
    .badge-confidence {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-left: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .component-item {
        background: linear-gradient(135deg, rgba(239, 246, 255, 0.9) 0%, rgba(219, 234, 254, 0.9) 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.6rem 0;
        border-left: 3px solid #3b82f6;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .component-item:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        transform: translateX(8px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .use-case-item {
        background: linear-gradient(135deg, rgba(254, 243, 199, 0.9) 0%, rgba(253, 230, 138, 0.9) 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.6rem 0;
        border-left: 3px solid #f59e0b;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .use-case-item:hover {
        background: linear-gradient(135deg, #fde68a 0%, #fcd34d 100%);
        transform: translateX(8px);
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);
    }
    
    .success-banner {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.1rem;
        border-radius: 12px;
        border-left: 4px solid #10b981;
        margin: 1.5rem 0;
        color: #065f46;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    
    .query-history-item {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px solid rgba(226, 232, 240, 0.8);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .query-history-item:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        transform: translateX(5px);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton>button {
        font-size: 1rem !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 AURELIA</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Automated Financial Concept Note Generator</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_url_input = st.text_input(
        "API",
        value=API_BASE_URL,
        help="FastAPI URL",
        label_visibility="collapsed"
    )
    
    if api_url_input:
        API_BASE_URL = api_url_input.rstrip('/')
    
    st.markdown("---")
    
    st.markdown("### 📚 Quick Select")
    
    sample_concepts = [
        ("📈", "Duration"),
        ("📊", "Sharpe Ratio"),
        ("🎯", "Black-Scholes"),
        ("⚠️", "Value at Risk"),
        ("🎲", "Monte Carlo"),
        ("💰", "Beta"),
        ("📉", "CAPM"),
        ("📐", "Yield Curve")
    ]
    
    cols = st.columns(2)
    for idx, (icon, concept) in enumerate(sample_concepts):
        with cols[idx % 2]:
            if st.button(f"{icon} {concept}", key=f"sample_{concept}", use_container_width=True):
                st.session_state['selected_concept'] = concept
    
    st.markdown("---")
    
    # Stats
    if 'query_history' in st.session_state and st.session_state['query_history']:
        st.markdown("### 📊 Stats")
        total = len(st.session_state['query_history'])
        cached = sum(1 for q in st.session_state['query_history'] if q.get('cached', False))
        wiki = sum(1 for q in st.session_state['query_history'] if q.get('concept_note', {}).get('source') == 'wikipedia')
        
        st.metric("Queries", total)
        st.metric("Cached", f"{cached}/{total}")
        if wiki > 0:
            st.metric("Wikipedia", wiki)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []

# Main query section
default_concept = st.session_state.get('selected_concept', '')

# Input box - full width with label
concept_input = st.text_input(
    "Search Financial Concept",
    value=default_concept,
    placeholder="Enter financial concept (e.g., Duration, Beta, Sharpe Ratio)",
    help="Type any financial concept to learn about it"
)

if 'selected_concept' in st.session_state:
    del st.session_state['selected_concept']

# Buttons row - perfectly aligned
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])

with col_btn1:
    query_button = st.button("🔍 Search Concept", type="primary", use_container_width=True)

with col_btn2:
    force_refresh = st.checkbox("🔄 Force Refresh", help="Bypass cache")

with col_btn3:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state['query_history'] = []
        st.rerun()

# Query handler
if query_button and concept_input:
    with st.spinner(f'🔍 Analyzing {concept_input}...'):
        try:
            start_time = time.time()
            
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
                concept_note = result.get('concept_note', {})
                
                result['query_latency'] = latency
                st.session_state['query_history'].insert(0, result)
                
                # Success banner
                st.markdown(f"""
                    <div class="success-banner">
                        <strong>✅ Success!</strong> Retrieved in <strong>{latency:.2f}s</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                # Badges
                source = concept_note.get('source', 'Unknown')
                cached = result.get('cached', False)
                confidence = concept_note.get('confidence', 0)
                
                badge_source_class = "badge-pdf" if source == 'fintbx.pdf' else "badge-wiki"
                badge_cache_class = "badge-cached" if cached else "badge-new"
                source_icon = "📄" if source == 'fintbx.pdf' else "🌐"
                cache_icon = "💾" if cached else "✨"
                
                st.markdown(f"""
                    <div style="text-align: center; margin: 1.5rem 0;">
                        <span class="badge {badge_source_class}">{source_icon} {source}</span>
                        <span class="badge {badge_cache_class}">{cache_icon} {('Cached' if cached else 'Fresh')}</span>
                        <span class="badge badge-confidence">🎯 {confidence:.0%}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Metrics row with labels inside boxes
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.markdown(f'''
                        <div class="metric-card">
                            <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.3rem;">⚡ LATENCY</div>
                            <div style="color: #1e293b; font-size: 1.8rem; font-weight: 700;">{latency:.2f}s</div>
                        </div>
                    ''', unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f'''
                        <div class="metric-card">
                            <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.3rem;">📦 CHUNKS</div>
                            <div style="color: #1e293b; font-size: 1.8rem; font-weight: 700;">{result.get('retrieved_chunks', 0)}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                
                with col_m3:
                    st.markdown(f'''
                        <div class="metric-card">
                            <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.3rem;">🎯 CONFIDENCE</div>
                            <div style="color: #1e293b; font-size: 1.8rem; font-weight: 700;">{confidence:.0%}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                
                with col_m4:
                    gen_time_s = result.get('generation_time_ms', 0) / 1000
                    st.markdown(f'''
                        <div class="metric-card">
                            <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.3rem;">⏱️ GEN TIME</div>
                            <div style="color: #1e293b; font-size: 1.8rem; font-weight: 700;">{gen_time_s:.2f}s</div>
                        </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Concept name
                st.markdown(f"## 📌 {concept_note.get('concept_name', concept_input)}")
                
                # Definition
                st.markdown('<div class="section-title">📝 Definition</div>', unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="concept-card">
                        {concept_note.get('definition', 'No definition available')}
                    </div>
                """, unsafe_allow_html=True)
                
                # Two column layout
                col_left, col_right = st.columns(2)
                
                with col_left:
                    # Key Components
                    if concept_note.get('key_components'):
                        st.markdown('<div class="section-title">🔑 Key Components</div>', unsafe_allow_html=True)
                        for idx, component in enumerate(concept_note['key_components'], 1):
                            st.markdown(f"""
                                <div class="component-item">
                                    <strong>{idx}.</strong> {component}
                                </div>
                            """, unsafe_allow_html=True)
                
                with col_right:
                    # Use Cases
                    if concept_note.get('use_cases'):
                        st.markdown('<div class="section-title">🎯 Use Cases</div>', unsafe_allow_html=True)
                        for idx, use_case in enumerate(concept_note['use_cases'], 1):
                            st.markdown(f"""
                                <div class="use-case-item">
                                    <strong>{idx}.</strong> {use_case}
                                </div>
                            """, unsafe_allow_html=True)
                
                # Formula
                if concept_note.get('formula'):
                    st.markdown('<div class="section-title">🔢 Formula</div>', unsafe_allow_html=True)
                    try:
                        st.latex(concept_note['formula'])
                    except:
                        st.code(concept_note['formula'], language='latex')
                
                # Example
                if concept_note.get('example'):
                    st.markdown('<div class="section-title">💡 Practical Example</div>', unsafe_allow_html=True)
                    st.info(concept_note['example'])
                
                # Related Concepts
                if concept_note.get('related_concepts'):
                    st.markdown('<div class="section-title">🔗 Related Topics</div>', unsafe_allow_html=True)
                    related = " • ".join(concept_note['related_concepts'])
                    st.success(f"**{related}**")
                
                # PDF References
                if concept_note.get('page_references') and source == 'fintbx.pdf':
                    st.markdown('<div class="section-title">📚 Citations</div>', unsafe_allow_html=True)
                    pages = ", ".join([f"Page {p}" for p in concept_note['page_references']])
                    st.success(f"**Financial Toolbox**: {pages}")
                
                # Metadata
                with st.expander("🔍 Technical Details"):
                    st.json({
                        "concept": concept_note.get('concept_name'),
                        "source": source,
                        "cached": cached,
                        "confidence": confidence,
                        "retrieved_chunks": result.get('retrieved_chunks', 0),
                        "generation_time_ms": result.get('generation_time_ms', 0),
                        "query_latency_s": f"{latency:.3f}",
                        "timestamp": result.get('timestamp')
                    })
            
            elif response.status_code == 400:
                # Handle relevance validation errors
                error_detail = response.json().get('detail', 'Invalid request')
                st.error(f"❌ {error_detail}")
                st.info("💡 This service only processes finance-related concepts. Try: Duration, Beta, Sharpe Ratio, Black-Scholes, CAPM, etc.")
                
            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error(f"🔌 Cannot connect to API at {API_BASE_URL}")
            st.info("💡 Make sure FastAPI is running in Terminal 1!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Query History
if st.session_state['query_history']:
    st.markdown("---")
    st.markdown("### 📜 Recent Queries")
    
    for idx, query in enumerate(st.session_state['query_history'][:5]):
        concept_note = query.get('concept_note', {})
        
        col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
        
        with col_h1:
            st.markdown(f"""
                <div class="query-history-item">
                    <strong style="color: #667eea;">🔹 {concept_note.get('concept_name', 'Unknown')}</strong><br>
                    <small style="color: #64748b;">{concept_note.get('definition', 'N/A')[:100]}...</small>
                </div>
            """, unsafe_allow_html=True)
        
        with col_h2:
            st.caption(f"⏱️ {query.get('query_latency', 0):.2f}s")
            st.caption(f"📄 {concept_note.get('source', 'Unknown')}")
        
        with col_h3:
            st.caption(f"🎯 {concept_note.get('confidence', 0):.0%}")
            st.caption(f"{'💾' if query.get('cached') else '✨'} {('Cached' if query.get('cached') else 'New')}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 1.5rem;'>
        <p style='font-size: 0.95rem;'>
            <strong style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.1rem;'>AURELIA</strong>
        </p>
        <p style='font-size: 0.85rem;'>
            Automated Financial Concept Note Generator<br>
            Built with Streamlit | Powered by RAG & LLMs
        </p>
    </div>
""", unsafe_allow_html=True)