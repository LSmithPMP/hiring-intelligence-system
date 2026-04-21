import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "../knowledge_base/hiring_benchmarks.md")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../knowledge_base/chroma_db")


def build_vector_store():
    print("Building RAG knowledge base...")

    # Load document
    loader = TextLoader(KNOWLEDGE_BASE_PATH)
    documents = loader.load()

    # Split by markdown headers for clean chunks
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "section"),
            ("##", "subsection"),
            ("###", "topic"),
        ]
    )
    splits = splitter.split_text(documents[0].page_content)

    # Add source metadata
    for split in splits:
        split.metadata["source"] = "hiring_benchmarks"

    # Create embeddings and store in Chroma
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Knowledge base built: {len(splits)} chunks indexed")
    return vectorstore


def get_vector_store():
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


def retrieve_context(query: str, k: int = 3) -> str:
    vectorstore = get_vector_store()
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


if __name__ == "__main__":
    build_vector_store()
    
    # Test retrieval
    test_queries = [
        "What is the healthy offer acceptance rate for engineers?",
        "What are SLA targets for time to hire?",
        "Which sourcing channels produce the best candidates?"
    ]
    
    print("\nTesting retrieval...")
    for query in test_queries:
        context = retrieve_context(query)
        print(f"\nQuery: {query}")
        print(f"Retrieved: {context[:200]}...")
