from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

def conversational_rag(documents: list) -> list:
    """Returns [answer_turn1, answer_turn2] using history-aware retrieval."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings()
    connection = "postgresql://postgres:Pass%40123@localhost:5432/langchain_assesment"
    
    vectorstore = PGVector.from_texts(
        texts=documents,
        embedding=embeddings,
        collection_name="task_16_history",
        connection=connection,
    )
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chat_history = []
    
    input1 = "What is LangChain?"
    result1 = rag_chain.invoke({"input": input1, "chat_history": chat_history})
    answer1 = result1["answer"]
    
    chat_history.extend([
        HumanMessage(content=input1),
        AIMessage(content=answer1)
    ])
    
    input2 = "What version introduced LCEL?"
    result2 = rag_chain.invoke({"input": input2, "chat_history": chat_history})
    answer2 = result2["answer"]

    return [answer1, answer2]

answers = conversational_rag(RAG_DOCUMENTS)
for i, ans in enumerate(answers, 1):
    print(f"Turn {i}: {ans}\n")