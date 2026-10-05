from sentence_transformers import SentenceTransformer
import chromadb


# ---------------------------------------------------------
# AI EMBEDDING MODEL
# ---------------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# CHROMADB DATABASE
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path="data/chroma_db"
)


# ---------------------------------------------------------
# MEMORY COLLECTION
# ---------------------------------------------------------

collection = client.get_or_create_collection(
    name="memorybridge"
)


# ---------------------------------------------------------
# CREATE EMBEDDING
# ---------------------------------------------------------

def create_embedding(text):
    """
    Convert text into a numerical vector.
    """

    embedding = model.encode(text)

    return embedding.tolist()


# ---------------------------------------------------------
# STORE MEMORY
# ---------------------------------------------------------

def store_memory(memory_id, text):
    """
    Store a memory and its AI embedding in ChromaDB.
    """

    embedding = create_embedding(text)

    collection.add(
        ids=[str(memory_id)],
        documents=[text],
        embeddings=[embedding]
    )


# ---------------------------------------------------------
# SEARCH MEMORIES
# ---------------------------------------------------------

def search_memories(query, number_of_results=3):
    """
    Find memories that are semantically related
    to the user's question.
    """

    query_embedding = create_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=number_of_results
    )

    return results