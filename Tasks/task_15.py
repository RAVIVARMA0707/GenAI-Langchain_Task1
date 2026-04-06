from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
import os

RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

def format_docs(docs):
    """Formats retrieved documents for the LLM context window."""
    return "\n\n".join(doc.page_content for doc in docs)

def rag_with_sources(documents: list, question: str) -> dict:
    """Returns the answer AND the source documents used with their scores."""
    
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    docs = [Document(page_content=text) for text in documents]
    connection = "postgresql://postgres:Pass%40123@localhost:5432/langchain_assesment"
    
    vectorstore = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name="task_15_sources",
        connection=connection,
    )

    def retrieve_with_scores(query: str):
        return vectorstore.similarity_search_with_score(query, k=3)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the following context:\n{context}\n\nQuestion: {question}"
    )

    
    retrieval_step = RunnableParallel({
        "docs_with_scores": lambda x: retrieve_with_scores(x),
        "question": RunnablePassthrough()
    })

    generation_chain = (
        {
            "context": lambda x: format_docs([d[0] for d in x["docs_with_scores"]]),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    inputs = retrieval_step.invoke(question)
    answer = generation_chain.invoke(inputs)

    sources = [
        {"content": doc.page_content, "score": round(float(score), 4)}
        for doc, score in inputs["docs_with_scores"]
    ]

    return {
        "answer": answer,
        "sources": sources
    }

result = rag_with_sources(RAG_DOCUMENTS, "How does pgvector support similarity search?")
print(result["answer"])
print(result["sources"])