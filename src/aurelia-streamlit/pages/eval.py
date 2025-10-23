"""
AURELIA Lab 5 - Evaluation Dashboard
"""
import streamlit as st
import requests
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Lab 5 Evaluation", page_icon="📊", layout="wide")

API_BASE_URL = "https://aurelia-financial-rag-475403.ue.r.appspot.com"

# Ground Truth
GROUND_TRUTH = {
    "duration": {"expected_source": "fintbx.pdf", "expected_pages": [22, 23, 24]},
    "sharpe ratio": {"expected_source": "fintbx.pdf", "expected_pages": [111]},
    "beta": {"expected_source": "fintbx.pdf", "expected_pages": [26, 27, 30, 32]},
    "internal rate of return": {"expected_source": "fintbx.pdf", "expected_pages": [11, 12]},
    "efficient frontier": {"expected_source": "fintbx.pdf", "expected_pages": [5, 98, 99]},
    "treasury bills": {"expected_source": "fintbx.pdf", "expected_pages": [25, 26, 27]},
}

TEST_CONCEPTS = ["Duration", "Sharpe Ratio", "Beta", "Internal Rate of Return"]

st.title("📊 Lab 5 - Evaluation Dashboard")
st.markdown("---")

# Controls
col1, col2 = st.columns([3, 1])

with col1:
    selected = st.multiselect("Select Concepts", TEST_CONCEPTS, default=TEST_CONCEPTS[:3])

with col2:
    force_refresh = st.checkbox("Force Refresh", value=True)

run_eval = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)

# Run evaluation
if run_eval and selected:
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    for i, concept in enumerate(selected):
        status.text(f"Testing {i+1}/{len(selected)}: {concept}...")
        
        try:
            start = time.time()
            
            resp = requests.post(
                f"{API_BASE_URL}/query",
                json={"concept": concept, "force_refresh": force_refresh},
                timeout=60
            )
            
            latency_ms = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                cn = data['concept_note']
                
                # Evaluate
                gt = GROUND_TRUTH.get(concept.lower(), {})
                source_match = cn['source'] == gt.get('expected_source', '')
                
                if cn.get('page_references') and gt.get('expected_pages'):
                    page_overlap = len(set(cn['page_references']) & set(gt['expected_pages']))
                    page_accuracy = page_overlap / len(gt['expected_pages'])
                else:
                    page_accuracy = 0.7 if cn['source'] == 'wikipedia' else 0
                
                quality = (0.5 * (100 if source_match else 50) + 0.5 * page_accuracy * 100)
                
                results.append({
                    'concept': concept,
                    'quality': quality,
                    'source': cn['source'],
                    'confidence': cn['confidence'],
                    'pages': cn.get('page_references'),
                    'chunks': data['retrieved_chunks'],
                    'latency_ms': latency_ms,
                    'cached': data['cached']
                })
            
        except Exception as e:
            results.append({'concept': concept, 'error': str(e)})
        
        progress.progress((i + 1) / len(selected))
    
    status.text("✅ Complete!")
    st.session_state['results'] = results
    time.sleep(0.5)
    st.rerun()

# Display results
if 'results' in st.session_state:
    results = st.session_state['results']
    successful = [r for r in results if 'error' not in r]
    
    if successful:
        # Summary
        st.markdown("## 📈 Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        avg_quality = np.mean([r['quality'] for r in successful])
        
        cached = [r for r in successful if r['cached']]
        fresh = [r for r in successful if not r['cached']]
        
        avg_cached = np.mean([r['latency_ms'] for r in cached]) if cached else 0
        avg_fresh = np.mean([r['latency_ms'] for r in fresh]) if fresh else 0
        
        with col1:
            st.metric("Quality Score", f"{avg_quality:.1f}/100")
        
        with col2:
            st.metric("Cached", f"{avg_cached:.0f}ms", delta=f"{len(cached)} queries")
        
        with col3:
            st.metric("Fresh", f"{avg_fresh:.0f}ms", delta=f"{len(fresh)} queries")
        
        with col4:
            speedup = avg_fresh / avg_cached if avg_cached > 0 else 0
            st.metric("Speedup", f"{speedup:.1f}x")
        
        # Table
        st.markdown("### 📋 Details")
        
        df = pd.DataFrame([
            {
                'Concept': r['concept'],
                'Quality': f"{r['quality']:.1f}",
                'Source': r['source'],
                'Pages': str(r['pages']) if r['pages'] else 'None',
                'Latency': f"{r['latency_ms']:.0f}ms",
                'Cached': '✓' if r['cached'] else '✗'
            }
            for r in successful
        ])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download
        st.markdown("---")
        
        json_data = json.dumps(results, indent=2, default=str)
        st.download_button(
            "📥 Download Results",
            data=json_data,
            file_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )