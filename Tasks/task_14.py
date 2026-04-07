import os
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

def format_docs(docs):
    """Joins the content of retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def basic_rag_pipeline(documents: list, question: str) -> str:
    """Indexes documents and answers the question using RAG."""
    
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    docs = [Document(page_content=text) for text in documents]
    
    connection = "postgresql://postgres:Pass%40123@localhost:5432/langchain_assesment"
    collection_name = "task_14_rag"
    
    vectorstore = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(question)

result = basic_rag_pipeline(RAG_DOCUMENTS, "What dimensions do OpenAI small embeddings produce?")
print(result)