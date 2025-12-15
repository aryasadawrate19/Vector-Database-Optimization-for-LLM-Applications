"""Quick test to check if search works"""
import weaviate
from weaviate_client import WeaviateVectorDB
import random

# Create simple test
db = WeaviateVectorDB(collection_name="TestSearch")
db.create_schema(vector_dimension=768, ef_construction=64, ef=32)

# Insert one test object
test_embedding = [random.random() for _ in range(768)]
db.insert_data(["Test document"], [test_embedding], show_progress=False)

# Try to search
print("\nAttempting search...")
try:
    results = db.search(test_embedding, limit=1)
    print(f"Search successful! Found {len(results)} results")
    print(f"Result: {results[0]['text']}")
except Exception as e:
    print(f" Search failed: {e}")

db.close()
