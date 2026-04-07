from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent

# Use the documents from the previous task context
RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

def rag_agent(question: str) -> str:
    """Uses a ReAct agent with a retriever tool to answer the question."""
    
    # 1. Setup Vector Store & Retriever
    embeddings = OpenAIEmbeddings()
    connection = "postgresql://postgres:Pass%40123@localhost:5432/langchain_assesment"
    
    vectorstore = PGVector.from_texts(
        texts=RAG_DOCUMENTS,
        embedding=embeddings,
        collection_name="task_17_agent",
        connection=connection,
    )
    retriever = vectorstore.as_retriever()

    # 2. Wrap Retriever in a Tool
    retriever_tool = create_retriever_tool(
        retriever,
        name="knowledge_base",
        description="Search this tool for technical info about LangChain, pgvector, and RAG."
    )
    tools = [retriever_tool]

    # 3. Initialize LLM and Prompt
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Pull a standard ReAct prompt from the LangChain Hub
    # This prompt contains the instructions for the Thought/Action/Observation loop
    prompt = hub.pull("hwchase17/react")

    # 4. Create the ReAct Agent
    agent = create_react_agent(llm, tools, prompt)
    
    # 5. Create the Agent Executor (The runtime)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

    # 6. Execute and Return
    response = agent_executor.invoke({"input": question})
    return response["output"]

# --- Execution ---
answer = rag_agent("What distance metrics does pgvector support?")
print(f"\nFinal Answer: {answer}")
