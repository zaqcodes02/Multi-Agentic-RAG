from aiagents.manager import decide
from aiagents.retriver import build_vector_store, answer_query
from aiagents.web_agent import answer_query

document_path = ""
vector_store = build_vector_store(document_path=document_path)


def process_query(query):
        route = decide(query)
        print(f"Manager decided to route to: {route.upper()}")

        if route == "rag":
            answer = answer_query(query, vector_store)
            source = "RAG (Document)"
        else:
            answer = answer_query(query)
            source = "Web (Tavily)"

        return f"Answer from {source}:\n{answer}"



query = input("Enter the querry you have relevant to AI?")

print(process_query(query=query))