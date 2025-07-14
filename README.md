# Renaulution-rag
 Le projet a pour objection d'implémenter un chatbot sur un pattern multiagentique, capable de répondre aux question des utilisateurs sur Renault. Le RAG est basé sur un ensemble de doc : pdf, talk youtube, données financières provenant de l'API yfinace.

### Choix de solution technique
J'ai fait le choix d'opter pour un RAG agentique implémenté avec le framework langgraph pour répondre au cas d'usage. Le système se décline comme le montre la ficgure ci-dessous : 
<img src="doc_images/graph_agent.png" alt="Graph Agent" width="450"/>.

Le système est composé de cinq noeuds : 
- Router : le noeud d'atrée qui agent en "plannigicateur". Il détecte l'intention usager et défini la route (enchainement des étapes) à suivre. Les différentes routes possibles sont au nombre de quatre :
    - rag_only : la doc indexée suffira pour répondre à la question
    - finance_only : l'utilisateur veut des données financières exclusives mais sans graphe
    - graph_flow_stock : l'utilisateur demande des données boursières sous un rendu graphiquement
    - graph_flow_internal : l'utilisateur demande des données d'exploitation renudes graphiquement
    
- rag_node : noeud RAG qui permet de récuperer le contexte permettant de répondre à la question
- graph_node : noeud d'affichage. Il construit tout ce qui est graphe, demandé par l'utilisateur
- finance_node : Ce noeud sert à interroger l'API yfinace pour extraire des données boursières
