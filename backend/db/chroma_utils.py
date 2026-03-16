import os
import chromadb
from chromadb import Search
from chromadb.api import Schema, VectorIndexConfig, SparseVectorIndexConfig, K, Rrf, Knn, GroupBy, MinK
from chromadb.utils.embedding_functions import (
    ChromaCloudQwenEmbeddingFunction,
    ChromaCloudSpladeEmbeddingFunction
)
from chromadb.utils.embedding_functions.chroma_cloud_qwen_embedding_function import ChromaCloudQwenEmbeddingModel
from chromadb.utils.embedding_functions.chroma_cloud_splade_embedding_function import ChromaCloudSpladeEmbeddingModel
from dotenv import load_dotenv

# Step 1: Tell Python to load the secrets from the .env file
load_dotenv()

class ChromaCloudDB:
    def __init__(self):
        # Step 2: Grab the secret values from the loaded environment
        self.tenant = os.getenv("CHROMA_TENANT")
        self.database = os.getenv("CHROMA_DATABASE")
        self.api_key = os.getenv("CHROMA_API_KEY")

        print("Connecting to Chroma Cloud...")
        
        # Step 3: Create the 'Client'. The client is the connection cable between your app and the cloud database.
        self.client = chromadb.CloudClient(
            tenant=self.tenant,
            database=self.database,
            api_key=self.api_key
        )
        print("Connected Successfully!")

        # -------------------------------------------------------------------
        # NEW CODE: Defining our AI "Brains" (Models)
        # -------------------------------------------------------------------
        
        class SafeListEmbeddingFunction:
            """A wrapper to fix a known ChromaDB 1.5.5 bug returning numpy arrays instead of lists"""
            def __init__(self, base_func):
                self.base_func = base_func
            def __call__(self, input):
                res = self.base_func(input)
                # Convert numpy arrays to lists if necessary
                return [r.tolist() if hasattr(r, 'tolist') else r for r in res]

        # Brain 1 (Dense): Qwen. This model reads text and understands the "meaning".
        self.dense_model = SafeListEmbeddingFunction(
            ChromaCloudQwenEmbeddingFunction(
                model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B,
                task="text_retrieval_query"
            )
        )
        
        # Brain 2 (Sparse): Splade. This model reads text and finds exact "Keywords".
        self.sparse_model = ChromaCloudSpladeEmbeddingFunction(
            model=ChromaCloudSpladeEmbeddingModel.SPLADE_PP_EN_V1
        )

        # Step 4: Define the shape of our "drawer" (Collection).
        schema = Schema() 
        
        # Tell the drawer to use Qwen for finding meaning (This is the default search)
        schema.create_index(config=VectorIndexConfig(
            space="cosine",
            embedding_function=self.dense_model
        ))
        
        # Tell the drawer to also keep an index of Keywords using Splade
        schema.create_index(
            config=SparseVectorIndexConfig(
                source_key=K.DOCUMENT,
                embedding_function=self.sparse_model
            ),
            key="sparse" # We give this index a name so we can call it later
        )

        # Step 5: Open the drawer (or build it if it's the first time).
        print("Opening the 'ai_fraud_learning' collection...")
        self.collection = self.client.get_or_create_collection(
            name="ai_fraud_learning",
            schema=schema
        )
        print("Collection is ready!")

    # -------------------------------------------------------------------
    # NEW CODE: Chunking and Adding Data
    # -------------------------------------------------------------------

    def chunk_text(self, text, max_bytes=16000):
        """
        Slices a long document into smaller 'chunks' so Chroma Cloud doesn't reject it for being too large.
        We split by line so we don't cut words in half.
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        if not text:
            return [""]

        # Go through the text line by line
        for line in text.splitlines(keepends=True):
            line_size = len(line.encode('utf-8'))
            
            # If adding this line pushes us over 16,000 bytes, save the current chunk and start a new one
            if current_size + line_size > max_bytes:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Don't forget to save the last chunk!
        if current_chunk:
            chunks.append("".join(current_chunk))
        return chunks

    def add_pattern(self, text, metadata, doc_id):
        """
        Takes an entire fraud document, chunks it if necessary, 
        and adds every chunk into the Cloud Database.
        """
        print(f"Adding document: {doc_id}")
        chunks = self.chunk_text(text)
        
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # We give each piece a unique name like: doc123_chunk_0
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # We copy the original metadata (like risk_level="high")
            chunk_metadata = metadata.copy()
            
            # CRITICAL: We record EXACTLY where this chunk came from. 
            # This is required for "GroupBy" later!
            chunk_metadata.update({
                "source_doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(chunk_metadata)
            
        # Finally, upload the chunks to the cloud drawer
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def add_batch(self, entries):
        """
        Takes a list of dictionaries, where each dict has 'text', 'metadata', and 'doc_id'.
        Chunks everything and uploads in a single cloud request.
        'entries' format: [{'text': '...', 'metadata': {...}, 'doc_id': '...'}, ...]
        """
        all_ids = []
        all_documents = []
        all_metadatas = []

        for entry in entries:
            text = entry['text']
            metadata = entry['metadata']
            doc_id = entry['doc_id']
            
            chunks = self.chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "source_doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                
                all_ids.append(chunk_id)
                all_documents.append(chunk)
                all_metadatas.append(chunk_metadata)

        if all_ids:
            self.collection.add(
                ids=all_ids,
                documents=all_documents,
                metadatas=all_metadatas
            )

    def query_similar(self, query_text, n_results=5):
        """
        Searches the database using both Dense (meaning) and Sparse (keyword) models.
        Combines them using RRF, then deduplicates using GroupBy so we don't get 
        5 chunks from the exact same original document.
        """
        print(f"Searching for: '{query_text}'...")
        
        # Step 0: Translate the text into AI language (Vectors)
        dense_vector = self.dense_model([query_text])[0]
        sparse_vector = self.sparse_model([query_text])[0]
        
        # Step 1: Define what we want to search for
        search_request = (
            Search()
            # Rank uses Rrf (Reciprocal Rank Fusion) to combine our two "Brains"
            .rank(Rrf([
                # Brain 1: Pass the raw dense vector
                Knn(query=dense_vector, return_rank=True),
                # Brain 2: Pass the raw sparse vector
                Knn(query=sparse_vector, key="sparse", return_rank=True)
            ]))
            # Deduplicate the results
            .group_by(GroupBy(
                # Group by the source document ID
                keys=K("source_doc_id"), 
                # For each group, only keep the 1 best match
                aggregate=MinK(keys=K.SCORE, k=1) 
            ))
            # Limit to the top 'n_results' unique documents
            .limit(n_results)
            # Tell it what data to give us back
            .select(K.DOCUMENT, K.SCORE, "source_doc_id", "chunk_index", "risk_level")
        )
        
        # Step 2: Actually run the search
        search_response = self.collection.search(search_request)
        
        # Step 3: Parse the results (Format: {'ids': [[...]], 'documents': [[...]], ...})
        final_results = []
        
        # Since we only did one query, we look at index 0 of the lists
        ids = search_response.get('ids', [[]])[0]
        docs = search_response.get('documents', [[]])[0]
        metadatas = search_response.get('metadatas', [[]])[0]
        scores = search_response.get('scores', [[]])[0]
        
        for i in range(len(ids)):
            final_results.append({
                "text": docs[i] if i < len(docs) else "",
                "score": scores[i] if i < len(scores) else 0,
                "metadata": metadatas[i] if i < len(metadatas) else {}
            })
            
        return final_results

# Let's test the connection right now!
if __name__ == "__main__":
    db = ChromaCloudDB()
    
    # 1. Let's add a fake long document
    long_email = (
        "Subject: URGENT ACCOUNT UPDATE\n"
        "Dear customer,\n"
        "Your account has been locked due to suspicious activity.\n"
        "Please click the link below to verify your identity.\n"
        "If you do not click within 24 hours, your money will be seized.\n"
    )
    # Give it a fake ID
    db.add_pattern(long_email, {"risk_level": "high"}, "email_54321")
    
    # 2. Let's search for a similar fraud pattern!
    search_results = db.query_similar("account locked click link to verify")
    print("\nSearch Results:")
    print(search_results)
