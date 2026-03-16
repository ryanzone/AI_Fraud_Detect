from backend.db.chroma_utils import ChromaCloudDB
import os
from dotenv import load_dotenv

def verify():
    load_dotenv()
    print("Connecting to ChromaDB...")
    db = ChromaCloudDB()
    
    test_queries = [
        "win a cash prize reward",
        "free entry to competition",
        "urgent bank account update required",
        "hey darling how are you"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        results = db.query_similar(query, n_results=3)
        
        if not results:
            print("No results found.")
            continue
            
        for i, res in enumerate(results):
            text = res['text'][:100] + "..." if len(res['text']) > 100 else res['text']
            risk = res['metadata'].get('risk_level', 'unknown')
            score = res.get('score', 'N/A')
            print(f"{i+1}. [Risk: {risk}] [Score: {score}] {text}")

if __name__ == "__main__":
    verify()
