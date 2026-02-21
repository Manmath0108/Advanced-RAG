# Two Requirted Tools
# 1. retrieve_docs
# 2. web_search

### 1.
from langchain_core.tools import tool
from dotenv import load_dotenv

import os
from scripts import utils

load_dotenv()

@tool
def retriever_docs(query: str, k: int = 5):
    """
    Retrieve relevant financial documents from ChromaDB.
    Extracts filters and retrieves matching documents.

    Args:
        query: Search the query (e.g "what was amazon's revenue in q3 2025?")
        k: Number of documents to retrieve. Generaly prefer 5 docs
    
    Returns:
        Retrieved documents with with metadata as formatted string
    """
    filters = utils.extract_filters(query)
    ranking_keywords = utils.generate_ranking_keywords(query)
    results = utils.search_docs(query, filters, ranking_keywords, k=k)

    docs = utils.rank_documents_by_keywords(results, ranking_keywords, k=k)
    
    print(f"[RETRIEVED] {len(docs)} documents")

    if len(docs) == 0:
        return f"No documents found for the query: '{query}'. Try rephrasing the query or change filters."
    
    retrieved_text = []
    for i, doc in enumerate(docs, 1):
        doc_text = [f"--- Document {i} ---"]

        for key, value in doc.metadata.items():
            doc_text.append(f"{key}: {value}")
        
        doc_text.append(f"\nContent:\n{doc.page_content}")

        text = "\n".join(doc_text)
        retrieved_text.append(text)
    
    retrieved_text = "\n".join(retrieved_text)

    os.makedirs("debug_logs", exist_ok=True)
    with open("debug_logs/retrieved_reranked_docs.md", "w", encoding='utf-8') as f:
        f.write(retrieved_text)
    
    return retrieved_text


from ddgs import DDGS

@tool
def web_search(query:str, num_results: int = 10) -> str:
    """Use this tool whenever you need to access realtime or latest information.
        Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 5)
    
    Returns:
        Formatted search results with titles, descriptions, and URLs
    """

    results = DDGS().text(query=query, max_results=num_results, region='us-en')

    if not results:
        return f"No results found for '{query}'"
    
    formatted_results = [f"Search results for search query: '{query}'"]
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        href = result.get('href', '')
        body = result.get('body', 'No description available')
        
        text = f"{i}. **{title}**\n   {body}\n   {href}"
        
        formatted_results.append(text)

    return "\n\n".join(formatted_results)