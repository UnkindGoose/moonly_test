from typing import TypedDict, List
import os

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from datetime import datetime

__all__=['graph']

class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]

@tool
def get_current_time():
    """Return the current UTC time in ISO‑8601 format.
    Example → {"utc": "2025‑05‑21T06:42:00Z"}"""
    return datetime.now()

tools = [get_current_time]


llm = init_chat_model(os.environ['MODEL_NAME'], model_provider='ollama').bind_tools(tools)

def llm_node(state: AgentState):
    response = llm.invoke(state["messages"])
    state['messages'] = state['messages'] + [response]
    return state


def router(state: AgentState):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return "end"


tool_node = ToolNode([get_current_time])

def tools_node(state: AgentState):
    result = tool_node.invoke(state)
    return {
        'messages': state['messages'] + result['messages']
    }
    

builder = StateGraph(AgentState)

builder.add_node('llm', llm_node)
builder.add_node('tools', tools_node)

builder.add_edge(START, 'llm')
builder.add_edge('tools', 'llm')
builder.add_conditional_edges('llm', router, {'tools':'tools', 'end': END})

graph = builder.compile()
