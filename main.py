from aiagents.manager import RouteManager
from aiagents.retriver import build_vector_store, answer_query as rag_answer
from aiagents.web_agent import answer_query as web_answer

document_path = "H:\ChatBots\Multi-Agentic-RAG\YOLO.pdf"
vector_store, document_topics = build_vector_store(document_path=document_path)
route_manager = RouteManager(document_topics)

print(f"Document topics detected: {document_topics}")

def process_query(query):
        route = route_manager.decide(query)
        print(f"Manager decided to route to: {route.upper()}")

        if route == "rag":
            answer = rag_answer(query, vector_store)
            source = "RAG (Document)"
        else:
            answer = web_answer(query)
            source = "Web (Tavily)"

        return f"Answer from {source}:\n{answer}"


query = input("Enter the query you have relevant to AI? ")
print(process_query(query=query))