from rag_node import build_rag_chain, rag_node_factory
from utils import GraphState
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from finance_node import finance_node
from plot_node import plot_node


def router_node_factory(llm):
    def router_node(state: GraphState) -> GraphState:
        question = state["question"]
        system_prompt = (
            "Tu es un routeur intelligent. Ta tâche est de classer une question dans l'une des catégories suivantes :\n"
            "- 'rag_only' : si la question porte uniquement sur des informations à extraire des documents (comme ventes, résumé, plan).\n"
            "- 'finance_only' : si la question porte uniquement sur la bourse, les cours d'action, CAC40, ou les données financières.\n"
            "- 'graph_flow_sales' : si la question demande une visualisation ou une comparaison de données internes (ex: nombre de véhicules vendus par année).\n"
            "- 'graph_flow_finance' : si la question demande une corrélation ou une comparaison entre des données internes et financières (ex: ventes vs stock).\n"
            "Répond uniquement par : rag_only, finance_only, graph_flow_sales, ou graph_flow_finance.\n"
            "Exemples :\n"
            "- 'Compare les ventes de 2023 à celles de 2022' → graph_flow_sales\n"
            "- 'Compare l'évolution des ventes Renault et de son cours en bourse' → graph_flow_finance\n"
            "- 'Quels sont les indicateurs DPEF ?' → rag_only\n"
            "- 'Quel était le prix de l’action Renault lors des résultats 2023 ?' → finance_only\n"
        )

        # Support mémoire si dispo
        history = state.get("chat_history", [])
        messages = history + [
            HumanMessage(content=system_prompt),
            HumanMessage(content=f"Question : {question}")
        ]

        response = llm.invoke(messages)
        decision = response.content.strip().lower()
        if decision not in {"rag_only", "finance_only", "graph_flow_sales", "graph_flow_finance"}:
            decision = "rag_only"

        print(f"Router decision: {decision}")
        return {"route": decision}

    return router_node




def response_node(state: GraphState) -> GraphState:
    if state.get("graph_base64"):
        final_response = f"{state['graph_base64']}"
    elif state.get("stock_data"):
        final_response = f"Données boursières extraites :\n{state['stock_data']}"
    elif state.get("rag_result"):
        final_response = state["rag_result"]
    else:
        final_response = "Je n'ai pas pu générer de réponse avec les éléments disponibles."
    
    return {"final_response": final_response}


def build_langgraph_rag(retriever, llm):
    retriever_obj, stuff_chain = build_rag_chain(retriever, llm)
    rag_node = rag_node_factory(retriever_obj, stuff_chain)
    router_node = router_node_factory(llm)

    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("rag_only", rag_node)
    graph.add_node("finance_only", finance_node)
    graph.add_node("graph_flow_sales", rag_node)
    graph.add_node("graph_flow_finance", rag_node)
    graph.add_node("plot_node", plot_node)
    graph.add_node("response", response_node)

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "rag_only": "rag_only",
            "finance_only": "finance_only",
            "graph_flow_sales": "graph_flow_sales",
            "graph_flow_finance": "graph_flow_finance"
        }
    )
    graph.add_edge("plot_node", "response")
    graph.add_edge("rag_only", "response")
    graph.add_edge("graph_flow_sales", "plot_node")
    graph.add_edge("graph_flow_finance", "plot_node")
    graph.add_edge("rag_only", "response")
    graph.add_edge("finance_only", "response")

    graph.set_entry_point("router")
    graph.set_finish_point("response")

    return graph.compile()