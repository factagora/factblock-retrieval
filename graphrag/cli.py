#!/usr/bin/env python3
"""
GraphRAG Command Line Interface

Simple CLI to test GraphRAG with your own investment questions.
"""

import os
import sys
import argparse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.vector_retriever import GraphVectorRetriever
from graphrag.cypher_retriever import TextToCypherRetriever

def vector_search(query: str, k: int = 5, graph_expansion: bool = True):
    """Perform vector search"""
    print(f"🔍 Vector Search: '{query}'")
    print("=" * 60)
    
    try:
        retriever = GraphVectorRetriever()
        results = retriever.search(query, k=k, include_graph_expansion=graph_expansion)
        
        if not results:
            print("❌ No results found")
            return
        
        for i, result in enumerate(results):
            factblock = result["factblock"]
            print(f"\n{i+1}. Score: {result['combined_score']:.3f} ({result['source']})")
            print(f"   📝 {factblock['claim']}")
            
            if factblock.get('impact_level'):
                print(f"   📈 Impact: {factblock['impact_level']}")
            
            if factblock.get('affected_sectors'):
                sectors = ', '.join(factblock['affected_sectors'])
                print(f"   🏭 Sectors: {sectors}")
            
            if result.get("relationship_path"):
                insights = [rel.get("investment_insight", "") for rel in result["relationship_path"] if rel.get("investment_insight")]
                if insights:
                    print(f"   💡 Insight: {insights[0]}")
        
        retriever.close()
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")

def cypher_search(query: str):
    """Perform natural language to Cypher search"""
    print(f"💬 Natural Language Search: '{query}'")
    print("=" * 60)
    
    try:
        retriever = TextToCypherRetriever()
        result = retriever.search(query)
        
        print(f"🔍 Query Type: {result['query_type']}")
        print(f"📊 Found: {result['result_count']} results")
        print(f"📝 Parameters: {result['parameters']}")
        
        # Show generated Cypher
        print(f"\n💻 Generated Cypher:")
        cypher_lines = result['cypher_query'].strip().split('\n')
        for line in cypher_lines:
            if line.strip():
                print(f"   {line.strip()}")
        
        # Show results
        if result['results']:
            print(f"\n📋 Results:")
            for i, res in enumerate(result['results'][:5]):  # Show top 5
                if 'f' in res:  # FactBlock result
                    factblock = res['f']
                    print(f"\n{i+1}. 📝 {factblock.get('claim', 'No claim')}")
                    if factblock.get('impact_level'):
                        print(f"   📈 Impact: {factblock['impact_level']}")
                    if factblock.get('affected_sectors'):
                        sectors = ', '.join(factblock['affected_sectors'])
                        print(f"   🏭 Sectors: {sectors}")
                elif 'f1' in res and 'f2' in res:  # Relationship result
                    print(f"\n{i+1}. 🔗 Relationship:")
                    print(f"   A: {res['f1'].get('claim', '')}")
                    print(f"   B: {res['f2'].get('claim', '')}")
        else:
            print("❌ No results found")
        
        retriever.close()
        
    except Exception as e:
        print(f"❌ Natural language search failed: {e}")

def interactive_mode():
    """Interactive mode for testing"""
    print("🎮 Interactive GraphRAG Mode")
    print("=" * 40)
    print("Commands:")
    print("  v <query>     - Vector search")
    print("  c <query>     - Cypher/Natural language search")
    print("  both <query>  - Both searches")
    print("  examples      - Show example queries")
    print("  quit          - Exit")
    print()
    
    while True:
        try:
            user_input = input("GraphRAG> ").strip()
            
            if not user_input or user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'examples':
                show_examples()
                continue
            
            parts = user_input.split(' ', 1)
            if len(parts) < 2:
                print("❌ Please provide a command and query. Type 'examples' for help.")
                continue
            
            command, query = parts[0].lower(), parts[1]
            
            print()  # Add spacing
            
            if command == 'v':
                vector_search(query)
            elif command == 'c':
                cypher_search(query)
            elif command == 'both':
                vector_search(query)
                print("\n" + "-" * 60 + "\n")
                cypher_search(query)
            else:
                print("❌ Unknown command. Use 'v', 'c', 'both', 'examples', or 'quit'")
            
            print()  # Add spacing
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def show_examples():
    """Show example queries"""
    print("\n📚 Example Queries:")
    print("-" * 30)
    
    examples = [
        ("OPEC oil production impact", "Energy sector analysis"),
        ("inflation monetary policy", "Macroeconomic policy"),
        ("banking sector lending", "Financial sector"),
        ("transportation fuel costs", "Supply chain impact"),
        ("semiconductor shortage", "Technology sector"),
        ("Find FactBlocks about OPEC", "Natural language entity search"),
        ("Get high impact FactBlocks", "Natural language impact search"),
        ("Show me claims in energy sector", "Natural language sector search")
    ]
    
    for query, description in examples:
        print(f"  '{query}'")
        print(f"    → {description}")
        print()

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="GraphRAG CLI for Investment Knowledge Graph")
    parser.add_argument("--mode", choices=['vector', 'cypher', 'both', 'interactive'], 
                       default='interactive', help="Search mode")
    parser.add_argument("--query", "-q", help="Query string")
    parser.add_argument("--limit", "-k", type=int, default=5, help="Number of results")
    parser.add_argument("--no-graph", action='store_true', help="Disable graph expansion for vector search")
    
    args = parser.parse_args()
    
    print("🚀 GraphRAG CLI - Investment Knowledge Graph")
    print("=" * 50)
    
    if args.mode == 'interactive' or not args.query:
        interactive_mode()
    else:
        if args.mode in ['vector', 'both']:
            vector_search(args.query, k=args.limit, graph_expansion=not args.no_graph)
        
        if args.mode in ['cypher', 'both']:
            if args.mode == 'both':
                print("\n" + "-" * 60 + "\n")
            cypher_search(args.query)

if __name__ == "__main__":
    main()