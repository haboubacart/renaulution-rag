from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.chains import StuffDocumentsChain
from backend.langraph_agent.utils import GraphState


def build_rag_chain(retriever, llm):
    prompt = PromptTemplate(
    template="""
    You are an expert on the Renault company.
    You must answer the user's question using only the context provided below.
    Use exclusively the information from the context to formulate your answer.
    If the answer cannot be found in the context, say so clearly.
    Be simple, clear, and professional.
    Answer in the same language as the question.
    ---- IMPORTANT ----
    If the question is in English, answer in English.
    If the question is in French, answer in French.
    ----

    Follow these instructions carefully:
    {instruction}

    ---
    Context: {context}
    Question: {question}
    Answer:
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

        if route == "graph_flow_internal":
            instruction = (
                "Extract only the sales figures by year (or by period) since the year mentionned. "
                "Format the results as JSON, for example: {'yyyy1': 'sales1', 'yyyy2': 'sales2'}, etc. "
                "Respond with the Json data only — no explanations, no extra text, no code block — just raw JSON."
            )
        elif route == "graph_flow_stock":
            instruction = (
                "Prepare a response that extracts figures which can later be used to retrieve stock market data, "
                "such as earnings announcement dates and similar information. "
                "Respond with the data only — no explanations, no extra text, no code block."
            )
        elif route == "finance_only":
            instruction = (
                "Only provide the requested dates or specific time periods relevant for stock analysis."
            )
        else:
            instruction = (
                "Provide a direct and clear answer to the user's question using only the given context. "
                "Dig into the context to extract as much relevant content as possible and build a concise answer."
            )

        docs = retriever.get_relevant_documents(query)
        result = stuff_chain.run(
            input_documents=docs,
            question=query,
            instruction=instruction
        )
        #print(docs)
        state.update({"rag_result": result, "route": route})
        return state

    return rag_node