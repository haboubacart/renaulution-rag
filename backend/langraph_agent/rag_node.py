from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.chains import StuffDocumentsChain
from utils import GraphState


def build_rag_chain(retriever, llm):
    prompt = PromptTemplate(
        template="""
        Tu es un expert sur l'entreprise Renault.
        Tu dois repondre à la question de l'utilisateur à partir du contexte fournir ci-dessous.
        Utilise uniquement les éléments de contexte suivants pour répondre à la question.
        Si tu ne retrouve pas la réponse, reponds simplement que tu ne connais pas la réponse. Soit simple, professionnel et chalereux.
        \n
        Instructions à respecter : \n
        {instruction}
        \n
        ---
        Contexte: {context}
        Question: {question}
        Réponse :
        ---
        """,
        input_variables=["context", "question", "instruction"]
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    return retriever, stuff_chain


# Noeud RAG avec instruction dynamique selon la route
def rag_node_factory(retriever, stuff_chain):
    def rag_node(state: GraphState) -> GraphState:
        query = state["question"]
        route = state.get("route", "rag_only")
        print(f"Current route: {route}")

        if route == "graph_flow_sales":
            instruction = (
                "Extrait uniquement les chiffres de vente par année (ou par période) depuis 2020. "
                "Formate les résultats en json, par exemple : {'yyyy' : 'vente1', 'yyyy2' : 'vente2'} etc..."
                "repond juste avec les données sans rien ajouter, ni de balise de code, juste le json"
            )
        elif route == "graph_flow_finance":
            instruction = (
                "Prépare une réponse qui extrait à la fois les chiffres de vente et les dates clés "
                "comme les annonces de résultats. Formate sous la forme : {'yyyy1-01-01' : 'vente1', 'yyyy2-01-01' : 'vente2'} etc..."
                "repond juste avec les données sans rien ajouter, ni de balise de code, juste le json"
            )
        elif route == "finance_only":
            instruction = (
                "Fournis uniquement les dates d'annonce des résultats ou les périodes précises nécessaires à l'analyse boursière."
            )
        else:
            instruction = (
                "Fournis une réponse directe et claire à la question en utilisant uniquement le contexte fourni."
            )

        docs = retriever.get_relevant_documents(query)
        result = stuff_chain.run(
            input_documents=docs,
            question=query,
            instruction=instruction
        )

        state.update({"rag_result": result, "route": route})
        return state

    return rag_node