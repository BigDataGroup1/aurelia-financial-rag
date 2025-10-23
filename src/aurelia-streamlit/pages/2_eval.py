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
                
                status.text(f"✓ {concept}: {cn['source']} - {quality:.1f}/100")
            else:
                results.append({'concept': concept, 'error': f"HTTP {resp.status_code}"})
                status.text(f"✗ {concept}: Failed")
            
        except Exception as e:
            results.append({'concept': concept, 'error': str(e)})
            status.text(f"✗ {concept}: {str(e)}")
        
        progress.progress((i + 1) / len(selected))
    
    # Save to session state
    st.session_state['eval_results'] = results
    st.session_state['eval_timestamp'] = datetime.now()
    
    status.text("✅ Evaluation Complete! Results below ⬇️")
    progress.empty()
    
    # DON'T rerun - let results display immediately

# Display results - ALWAYS show if available
if 'eval_results' in st.session_state:
    st.markdown("---")
    st.success(f"📊 Showing results from: {st.session_state.get('eval_timestamp', 'Unknown').strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = st.session_state['eval_results']
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    if successful:
        # Summary Metrics
        st.markdown("## 📈 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics safely
        avg_quality = np.mean([r['quality'] for r in successful])
        
        cached_results = [r for r in successful if r.get('cached', False)]
        fresh_results = [r for r in successful if not r.get('cached', False)]
        
        avg_cached = np.mean([r['latency_ms'] for r in cached_results]) if cached_results else 0
        avg_fresh = np.mean([r['latency_ms'] for r in fresh_results]) if fresh_results else 0
        speedup = avg_fresh / avg_cached if avg_cached > 0 else 0
        
        with col1:
            st.metric("Quality Score", f"{avg_quality:.1f}/100")
        
        with col2:
            st.metric("Cached Latency", f"{avg_cached:.0f}ms", 
                     delta=f"{len(cached_results)} queries")
        
        with col3:
            st.metric("Fresh Latency", f"{avg_fresh:.0f}ms", 
                     delta=f"{len(fresh_results)} queries")
        
        with col4:
            if speedup > 0:
                st.metric("Speedup Factor", f"{speedup:.1f}x")
            else:
                st.metric("Speedup Factor", "N/A")
        
        st.markdown("---")
        
        # Detailed Table
        st.markdown("### 📋 Detailed Results")
        
        df = pd.DataFrame([
            {
                'Concept': r['concept'],
                'Quality': f"{r['quality']:.1f}",
                'Source': r['source'],
                'Confidence': f"{r['confidence']:.2f}",
                'Pages': str(r['pages']) if r['pages'] else 'None',
                'Chunks': r['chunks'],
                'Latency (ms)': f"{r['latency_ms']:.0f}",
                'Cached': '✓' if r['cached'] else '✗'
            }
            for r in successful
        ])
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Download Section
        st.markdown("### 📥 Download Results")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            # JSON download
            json_data = json.dumps({
                'timestamp': st.session_state['eval_timestamp'].isoformat(),
                'summary': {
                    'total_queries': len(results),
                    'successful': len(successful),
                    'failed': len(failed),
                    'avg_quality': avg_quality,
                    'avg_cached_ms': avg_cached,
                    'avg_fresh_ms': avg_fresh,
                    'speedup': speedup
                },
                'results': successful
            }, indent=2, default=str)
            
            st.download_button(
                "📊 Download JSON",
                data=json_data,
                file_name=f"lab5_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_d2:
            # CSV download
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                "📄 Download CSV",
                data=csv_data,
                file_name=f"lab5_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Show failed queries if any
    if failed:
        st.markdown("---")
        st.markdown("### ❌ Failed Queries")
        for r in failed:
            st.error(f"**{r['concept']}:** {r.get('error', 'Unknown error')}")
    
    # Clear results button
    st.markdown("---")
    if st.button("🔄 Clear Results & Run New Evaluation"):
        del st.session_state['eval_results']
        del st.session_state['eval_timestamp']
        st.rerun()

else:
    # Initial instructions
    st.info("""
    👆 **How to use:**
    1. Select concepts to evaluate
    2. Check "Force Refresh" to test fresh generation (not cached)
    3. Click "Run Evaluation"
    4. Results will appear below with download options
    
    **What gets evaluated:**
    - ✓ Response latency (cached vs fresh)
    - ✓ Data source (fintbx.pdf vs Wikipedia)
    - ✓ Page citation accuracy
    - ✓ Confidence scores
    """)