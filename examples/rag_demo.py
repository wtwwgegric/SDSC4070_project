"""Small demo showing how to index text chunks into Chroma and query them."""
from career_copilot.pdf_loader import chunk_text, load_pdf_from_bytes
from career_copilot.rag import create_collection, query_collection


def demo_index_from_pdf_bytes(pdf_bytes: bytes, collection_name: str = "cv"):
    text = load_pdf_from_bytes(pdf_bytes)
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]
    create_collection(collection_name, chunks, metadatas=metadatas, persist_directory="./chromadb")


def demo_query(collection_name: str = "cv"):
    q = input("Query: ")
    res = query_collection(collection_name, q, k=5, persist_directory="./chromadb")
    print("Results:")
    for i, doc in enumerate(res.get("documents", [[]])[0]):
        meta = res.get("metadatas", [[{}]])[0][i]
        dist = res.get("distances", [[None]])[0][i]
        print(f"--- result {i+1} (dist={dist}) ---")
        print(doc[:400].strip())
        print("meta:", meta)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "rb") as f:
            demo_index_from_pdf_bytes(f.read(), collection_name="cv_demo")
        print("Indexed. Now run without args to query.")
    else:
        demo_query("cv_demo")
