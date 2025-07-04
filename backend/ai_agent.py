import os
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
warnings.filterwarnings("ignore")


# Initialisation du LLM
"""llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    temperature=0.7
)"""

#Initialiser le LLM (gpt-4o par défaut)
def load_llm():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
        temperature=0.1
    )

# 🧠 Prompt personnalisé pour le RAG
def build_rag_prompt():
    prompt_template = """
    Utilise uniquement les éléments de contexte suivants pour répondre à la question.
    Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.

    Contexte: {context}

    Question: {question}

    Réponse utile:
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 🔍 Crée une chaîne RAG basée sur un retriever
def create_rag_chain(llm, retriever):
    prompt = build_rag_prompt()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# 🛠️ Fonction d'accès pour le RAG tool
def retriever_relevant_docs_factory(rag_chain):
    def _retriever(query: str):
        result = rag_chain.invoke({"query": query})
        return result["result"]
    return _retriever

# 🚀 Construire l'agent LangChain
def build_agent(retriever):
    llm = load_llm()
    rag_chain = create_rag_chain(llm, retriever)
    search_func = retriever_relevant_docs_factory(rag_chain)

    rag_tool = Tool(
        name="rag",
        func=search_func,
        description="Recherche dans les documents de Renault fournis en contexte pour répondre à des questions."
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    agent = initialize_agent(
        tools=[rag_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True
    )

    return agent

# 🎤 Fonction d'appel de l'agent
def ask_agent(agent, question: str) -> str:
    return agent.run(question)

