import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.context import collect_runs

def traced_chain(topic: str) -> dict:
    """Runs a chain with LangSmith tracing. Returns answer and run_id."""
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Task_18_Instrumentation"

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("Tell me a short fact about {topic}.")
    parser = StrOutputParser()
    
    chain = prompt | llm | parser
    
    with collect_runs() as cb:
        answer = chain.invoke(
            {"topic": topic},
            config={
                "run_name": "task18_trace_execution", 
                "tags": ["challenge", "unit-test"]
            }
        )
        
        run_id = str(cb.traced_runs[0].id)
    
    return {
        "answer": answer,
        "run_id": run_id
    }


result = traced_chain("Artificial Intelligence")
print(f"Run ID: {result['run_id']}")
print(f"Answer: {result['answer']}")
