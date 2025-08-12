#!/usr/bin/env python3
"""
Test script for RAG tool functionality.
Run this to verify that the RAG tool is working correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_rag_tool():
    """Test the RAG tool initialization and basic functionality."""
    print("🧪 Testing RAG Tool Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Import and initialize
        print("1. Testing RAG tool import and initialization...")
        from services.rag_tool import RAGTool
        
        rag_tool = RAGTool()
        print(f"   ✅ RAG tool created successfully")
        
        # Test 2: Check status
        print("\n2. Checking RAG tool status...")
        status = rag_tool.get_status()
        print(f"   📊 Status: {status}")
        
        if not status.get("initialized", False):
            print(f"   ⚠️  RAG tool not initialized. This might be due to:")
            print(f"      - Missing dependencies: pip install sentence-transformers numpy scikit-learn PyPDF2")
            print(f"      - Missing documents folder: {status.get('documents_path', './documents')}")
            print(f"      - No documents in folder")
            return False
        
        print(f"   ✅ RAG tool initialized with {status.get('documents_indexed', 0)} documents")
        
        # Test 3: Debug search
        print("\n3. Testing debug search...")
        debug_result = rag_tool.debug_search("database connection troubleshooting")
        print(f"   🔍 Debug search results:")
        print(f"      - Documents indexed: {debug_result.get('documents_indexed', 0)}")
        print(f"      - Max similarity: {debug_result.get('max_similarity', 0):.3f}")
        print(f"      - Avg similarity: {debug_result.get('avg_similarity', 0):.3f}")
        print(f"      - Threshold: {debug_result.get('threshold', 0.5)}")
        
        if debug_result.get('top_results'):
            print(f"      - Top result similarity: {debug_result['top_results'][0]['similarity']:.3f}")
            print(f"      - Top result source: {debug_result['top_results'][0]['source_file']}")
        
        # Test 4: Actual search
        print("\n4. Testing actual search...")
        search_result = rag_tool.execute("How do I troubleshoot database connection issues?")
        
        print(f"   🔍 Search result:")
        print(f"      - Success: {search_result.get('success', False)}")
        print(f"      - Results found: {search_result.get('metadata', {}).get('results_found', 0)}")
        
        if search_result.get('success') and search_result.get('results'):
            results_preview = search_result['results'][:200] + "..." if len(search_result['results']) > 200 else search_result['results']
            print(f"      - Results preview: {results_preview}")
        
        if search_result.get('metadata', {}).get('error'):
            print(f"      - Error: {search_result['metadata']['error']}")
        
        # Test 5: Check documents folder
        print("\n5. Checking documents folder...")
        docs_path = Path(status.get("documents_path", "./documents"))
        if docs_path.exists():
            doc_files = list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.txt")) + list(docs_path.rglob("*.pdf"))
            print(f"   📁 Documents folder exists: {docs_path}")
            print(f"   📄 Found {len(doc_files)} document files:")
            for doc_file in doc_files[:5]:  # Show first 5
                print(f"      - {doc_file.name}")
            if len(doc_files) > 5:
                print(f"      - ... and {len(doc_files) - 5} more")
        else:
            print(f"   ❌ Documents folder does not exist: {docs_path}")
            print(f"   💡 Create it with: mkdir -p {docs_path}")
        
        # Test 6: Check cache folder
        print("\n6. Checking cache folder...")
        cache_file = Path(rag_tool.embeddings_cache_file)
        cache_dir = cache_file.parent
        
        if cache_dir.exists():
            print(f"   📁 Cache directory exists: {cache_dir}")
            if cache_file.exists():
                cache_size = cache_file.stat().st_size
                print(f"   💾 Cache file exists: {cache_file.name} ({cache_size:,} bytes)")
            else:
                print(f"   📝 Cache file will be created: {cache_file.name}")
        else:
            print(f"   📁 Cache directory will be created: {cache_dir}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created cache directory")
        
        print("\n" + "=" * 50)
        print("✅ RAG tool test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install dependencies: pip install sentence-transformers numpy scikit-learn PyPDF2")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking RAG dependencies...")
    dependencies = [
        "sentence_transformers",
        "numpy", 
        "sklearn",
        "PyPDF2"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep.replace("_", "-"))
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep}")
            missing.append(dep)
    
    if missing:
        print(f"\n💡 Install missing dependencies:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies available!")
    return True

if __name__ == "__main__":
    print("🤖 Agent-AIOps RAG Tool Test")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    print()
    
    # Test RAG functionality
    if test_rag_tool():
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)
