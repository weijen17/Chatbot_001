"""
LangGraph Chatbot with Intent Routing and Memory

This chatbot uses LangGraph to route user queries through different nodes:
- Intent classification
- Decision routing (Doc Retrieval, Small Talk, Other Tools)
- Response generation with conversation memory
"""

from typing import TypedDict, Literal, Annotated
import operator
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
import faiss
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import logging

from src.assets.prompts import system_prompt__intention_recog, system_prompt__context_extraction,system_prompt__entity_extraction,system_prompt__name_search,system_prompt__search_content
from src.tools.tools_module import search_tool,retrieval_tool
from src.agents.agents_module import *
from src.config.settings import settings
from src.data import faiss_index_loading

################################################################################
# -----------------------------
# Configuration
# -----------------------------

TOP_K=settings.TOP_K
TOP_K2=settings.TOP_K2

EMBED_MODEL = settings.EMBED_MODEL
embedder = SentenceTransformer(EMBED_MODEL)
faiss_index, data = faiss_index_loading(embedder)
# df_data = pd.DataFrame(data, columns=['index', '_id', 'author', 'user_followers_count', 'engagement', 'title_body'])

llm = init_chat_model(model=settings.MODEL_NAME, model_provider=settings.MODEL_PROVIDER, temperature=settings.TEMPERATURE,top_p=settings.TOP_P)

LOG_FILE = os.path.join(settings.LOG_DIR, "app.log")
# Configure logging
logging.basicConfig(
    level=logging.INFO,                           # logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),           # save to file
        logging.StreamHandler()                  # also print to console
    ]
)

################################################################################
# -----------------------------
# Flow
# -----------------------------

# Define the state structure
class ChatState(TypedDict):
    messages: Annotated[list, operator.add]
    intent: str
    route: str
    context: str


# Intent Classification Node
def intent_node(state: ChatState) -> ChatState:
    """Classify the user's intent from their message"""
    last_message = state["messages"][-1].content

    human_input = f"""User message: {last_message}"""

    intent = intent_recognition_agent(llm,human_input)

    return {"intent": intent}

# Decision Node
def decision_node(state: ChatState) -> Literal["doc_retrieval", "small_talk", "search_tool"]:
    """Route based on classified intent"""
    intent = state["intent"]

    if "document" in intent:
        return "doc_retrieval"
    elif "small_talk" in intent:
        return "small_talk"
    else:
        return "search_tool"


# Document Retrieval Node
def doc_retrieval_node(state: ChatState) -> ChatState:
    """Handle document retrieval requests"""
    last_message = state["messages"][-1].content
    l_entities = entities_extraction_agent(llm,last_message)
    print(l_entities)
    if l_entities:
        l_entities = similar_name_expansion_agent(llm,l_entities)
    print(l_entities)
    query = context_extraction_agent(llm,last_message)

    print(query)
    print(l_entities)
    print('####################')
    context = retrieval_tool(data,l_entities,embedder, faiss_index, query, TOP_K=TOP_K,TOP_K2=TOP_K2)
    # Simulate document retrieval (replace with actual vector DB in production)
    retrieved_context = (f"Retrieved documents keys or fields are ['index','_id','author','user_followers_count','engagement','title_body']"
                         f"\n\nRetrieved documents related to: \n{context}")
    for _ in context:
        print(_)

    return {"context": retrieved_context, "route": "doc_retrieval"}


# Small Talk Node
def small_talk_node(state: ChatState) -> ChatState:
    """Handle casual conversation"""
    return {"context": "Engaging in friendly conversation", "route": "small_talk"}


# Search Tool Node
def search_tool_node(state: ChatState) -> ChatState:
    """Handle other tool requests"""
    last_message = state["messages"][-1].content
    context = context_extraction_agent(llm,last_message)
    searched_context = search_tool(context)
    return {"context": searched_context, "route": "search_tool"}


# Response Generation Node
def response_node(state: ChatState) -> ChatState:
    """Generate final response using conversation history and context"""
    # Build conversation history
    conversation_history = state["messages"][:-1]  # All but last message
    user_message = state["messages"][-1].content
    context = state.get("context", "")
    route = state.get("route", "")
    user_message = user_message + '\n\n If user question is in mandarin, then reply in mandarin.'

    # Create system message based on route
    system_prompts = {
        "doc_retrieval": "You are a helpful assistant that provides information based on retrieved documents. Use the context provided to answer questions accurately.",
        "small_talk": "You are a friendly conversational assistant. Engage warmly and naturally with the user.",
        "search_tool": "You are a helpful assistant that can help users accomplish tasks using search_tool."
    }

    system_msg = SystemMessage(content=system_prompts.get(route, "You are a helpful assistant."))

    # Build prompt with context
    if context and route == "doc_retrieval":
        enhanced_message = f"Context: {context}\n\nUser question: {user_message}"
        print('#### enhanced message')
        print(enhanced_message)
    elif context and route =='search_tool':
        enhanced_message = f"Searched Result:\n{context}\n\nUser question: {user_message}"
    else:
        enhanced_message = user_message

    # Generate response with full conversation history
    messages_for_llm = [system_msg] + conversation_history + [HumanMessage(content=enhanced_message)]
    response = llm.invoke(messages_for_llm)

    return {"messages": [AIMessage(content=response.content)]}


# Build the graph
def create_chatbot_graph():
    """Create and compile the LangGraph chatbot"""
    workflow = StateGraph(ChatState)

    # Add nodes
    workflow.add_node("intent", intent_node)
    workflow.add_node("doc_retrieval", doc_retrieval_node)
    workflow.add_node("small_talk", small_talk_node)
    workflow.add_node("search_tool", search_tool_node)
    workflow.add_node("response", response_node)

    # Set entry point
    workflow.set_entry_point("intent")

    # Add conditional edges from intent to routing
    workflow.add_conditional_edges(
        "intent",
        decision_node,
        {
            "doc_retrieval": "doc_retrieval",
            "small_talk": "small_talk",
            "search_tool": "search_tool"
        }
    )

    # Add edges from routing nodes to response
    workflow.add_edge("doc_retrieval", "response")
    workflow.add_edge("small_talk", "response")
    workflow.add_edge("search_tool", "response")

    # Add edge from response to end
    workflow.add_edge("response", END)

    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Main chatbot class
class Chatbot:
    def __init__(self):
        self.graph = create_chatbot_graph()
        self.thread_id = "conversation_1"

    def chat(self, user_message: str) -> str:
        """Send a message and get a response"""
        config = {"configurable": {"thread_id": self.thread_id}}

        state = {
            "messages": [HumanMessage(content=user_message)],
            "intent": "",
            "route": "",
            "context": ""
        }

        result = self.graph.invoke(state, config)
        return result["messages"][-1].content

    def reset_memory(self):
        """Start a new conversation"""
        import uuid
        self.thread_id = str(uuid.uuid4())



