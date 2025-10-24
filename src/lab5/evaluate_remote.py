"""
Lab 5: Remote Evaluation Script
Evaluates the deployed App Engine AURELIA service
Works without local ChromaDB setup!
"""
import requests
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Your deployed API URL
API_URL = "https://aurelia-financial-rag-475403.ue.r.appspot.com"

print("="*70)
print("AURELIA LAB 5 - REMOTE EVALUATION")
print("="*70)
print()
print(f"📡 Testing deployed API: {API_URL}")
print()

# Test concepts
test_concepts = ["Duration", "Beta", "Sharpe Ratio"]

print(f"📋 Testing {len(test_concepts)} concepts:")
for concept in test_concepts:
    print(f"   - {concept}")
print()
print("🔄 Force refresh: True (testing fresh queries)")
print()
print("-"*70)
print("Running Evaluation...")
print("-"*70)
print()

start_time = time.time()

try:
    # First, let's test individual queries to see what's working
    results = []
    
    for i, concept in enumerate(test_concepts, 1):
        print(f"Testing {i}/{len(test_concepts)}: {concept}...", end=" ")
        
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "concept": concept,
                    "force_refresh": True
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                concept_note = data['concept_note']
                latency = data['generation_time_ms']
                cached = data['cached']
                chunks = data['retrieved_chunks']
                
                print(f"✓ {latency:.0f}ms (source: {concept_note['source']}, chunks: {chunks})")
                results.append({
                    'concept': concept,
                    'success': True,
                    'data': data
                })
            else:
                print(f"✗ Error {response.status_code}")
                results.append({
                    'concept': concept,
                    'success': False,
                    'error': response.text
                })
        
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({
                'concept': concept,
                'success': False,
                'error': str(e)
            })
    
    elapsed = time.time() - start_time
    
    print()
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print()
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✓ Successful: {len(successful)}/{len(test_concepts)}")
    print(f"✗ Failed: {len(failed)}/{len(test_concepts)}")
    print(f"⏱️  Total time: {elapsed:.1f}s")
    print()
    
    if successful:
        print("📊 Quality Analysis:")
        print("-"*70)
        
        # Analyze successful queries
        for result in successful:
            concept = result['concept']
            data = result['data']
            cn = data['concept_note']
            
            print(f"\n{concept}:")
            print(f"  Source: {cn['source']}")
            print(f"  Confidence: {cn['confidence']:.2f}")
            print(f"  Pages: {cn.get('page_references', 'None')}")
            print(f"  Chunks: {data['retrieved_chunks']}")
            print(f"  Latency: {data['generation_time_ms']:.0f}ms")
            print(f"  Cached: {data['cached']}")
        
        # Save results
        output_dir = Path(__file__).parent / "remote_results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"remote_eval_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'api_url': API_URL,
                'test_concepts': test_concepts,
                'results': results,
                'summary': {
                    'total': len(test_concepts),
                    'successful': len(successful),
                    'failed': len(failed),
                    'elapsed_seconds': elapsed
                }
            }, f, indent=2)
        
        print()
        print(f"✓ Results saved to: {output_file}")
    
    if failed:
        print()
        print("❌ Failed Queries:")
        for result in failed:
            print(f"   - {result['concept']}: {result.get('error', 'Unknown error')}")
    
    print()
    print("="*70)
    
    if len(successful) == len(test_concepts):
        print("✅ All queries successful! Your deployed API is working!")
        print()
        print("Next steps:")
        print("  1. Add /evaluate endpoint to main.py (Phase 4)")
        print("  2. Deploy updated backend")
        print("  3. Test full evaluation via API")
    else:
        print("⚠️  Some queries failed. Check your deployed service.")
    
    print("="*70)

except Exception as e:
    print(f"❌ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)