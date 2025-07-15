from backend.langraph_agent.rag_node import build_rag_chain, rag_node_factory
from backend.langraph_agent.finance_node import finance_node
from backend.langraph_agent.plot_node import plot_node
from backend.langraph_agent.router_node import router_node_factory
from backend.langraph_agent.utils import GraphState
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage


def response_node(state: GraphState) -> GraphState:
    if state.get("graph_base64"):
        final_response = f"Here is the genarated graph"
    elif state.get("stock_data"):
        final_response = f"Données boursières extraites :\n{state['stock_data']}"
    elif state.get("rag_result"):
        final_response = state["rag_result"]
    else:
        final_response = "Je n'ai pas pu générer de réponse avec les éléments disponibles."
    
    return {"final_response": final_response}


def build_langgraph_agent(retriever, llm):
    retriever_obj, stuff_chain = build_rag_chain(retriever, llm)
    rag_node = rag_node_factory(retriever_obj, stuff_chain)
    router_node = router_node_factory(llm)

    graph = StateGraph(GraphState)

    graph.set_entry_point("router")
    graph.add_edge("router", "rag_node")

    graph.add_node("router", router_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("finance_node", finance_node)
    graph.add_node("graph_node", plot_node)
    graph.add_node("response_node", response_node)

    graph.add_conditional_edges(
        "rag_node",
        lambda state: state["route"],
        {
            "rag_only": "response_node",
            "graph_flow_internal": "graph_node",
            "graph_flow_stock": "finance_node",
            "finance_only": "finance_node"
        }
    )

    graph.add_conditional_edges(
        "finance_node",
        lambda state: state["route"],
        {
            "finance_only": "response_node",
            "graph_flow_stock": "graph_node"
        }
    )
   
    graph.add_edge("graph_node", "response_node")
    graph.add_edge("finance_node", "response_node")

    
    graph.set_finish_point("response_node")

    return graph.compile()