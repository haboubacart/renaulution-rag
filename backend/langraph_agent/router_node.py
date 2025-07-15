from backend.langraph_agent.utils import GraphState
from langchain_core.messages import HumanMessage


def router_node_factory(llm):
    def router_node(state: GraphState) -> GraphState:
        question = state["question"]
        system_prompt = (
            "Tu es un routeur intelligent. Ta tâche est de classer une question dans l'une des catégories suivantes :\n"
            "- 'rag_only' : si la question porte uniquement sur des informations à extraire des documents (comme ventes, résumé, plan).\n"
            "- 'finance_only' : si la question porte uniquement sur la bourse, les cours d'action, CAC40, ou les données financières.\n"
            "- 'graph_flow_internal' : si la question demande une visualisation ou une comparaison de données internes (ex: nombre de véhicules vendus par année).\n"
            "- 'graph_flow_stock' : si la question demande une corrélation ou une comparaison entre des données internes et financières (ex: ventes vs stock).\n"
            "Répond uniquement par : rag_only, finance_only, graph_flow_internal, ou graph_flow_stock.\n"
            "Exemples :\n"
            "- 'Compare les ventes de 2023 à celles de 2022' → graph_flow_internal\n"
            "- 'Compare l'évolution des ventes Renault et de son cours en bourse' → graph_flow_stock\n"
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
        if decision not in {"rag_only", "finance_only", "graph_flow_internal", "graph_flow_stock"}:
            decision = "rag_only"

        print(f"Router decision: {decision}")
        return {"route": decision}

    return router_node