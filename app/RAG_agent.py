from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from config import (
    LANGSMITH_PROJECT,
    OPENAI_LLM_MODEL,
    OPENAI_EMBEDDING_MODEL,
    KNOWLEDGE_BASE_PATH,
    PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    SYSTEM_PROMPT,
)
from jinja2 import Environment, FileSystemLoader

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

llm = ChatOpenAI(model=OPENAI_LLM_MODEL, temperature=0)

embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
)

env = Environment(loader=FileSystemLoader("prompts"))

template = env.get_template(SYSTEM_PROMPT)
system_prompt = template.render()

if not os.path.exists(KNOWLEDGE_BASE_PATH):
    raise FileNotFoundError(f"knowledge_base not found: {KNOWLEDGE_BASE_PATH}")

try:
    documents = []
    for file in os.listdir(KNOWLEDGE_BASE_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(KNOWLEDGE_BASE_PATH, file))
            docs = loader.load()
            documents.extend(docs)

    print(f"Loaded {len(documents)} pages from PDFs")

except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)


pages_split = text_splitter.split_documents(documents)

if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)


try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    print("Created ChromaDB vector store!")

except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the knowledge base documents.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the knowledge base."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


tools_dict = {our_tool.name: our_tool for our_tool in tools}


def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}


def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(
            f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}"
        )

        if t["name"] not in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            print(f"Result length: {len(str(result))}")

        results.append(
            ToolMessage(
                tool_call_id=t["id"], name=t["name"], content=str(result)
            )
        )

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm", should_continue, {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def rag_agent_reply(user_input: str) -> str:
    """Takes user input and returns agent response as string."""
    messages = [HumanMessage(content=user_input)]
    result = rag_agent.invoke({"messages": messages})
    return result["messages"][-1].content
