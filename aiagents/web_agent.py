import groq
from tavily import TavilyClient
from dotenv import load_dotenv
import os


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

client = groq.Groq(GROQ_API_KEY)
model = "llama3-8b-8192"
tavily_client = TavilyClient(TAVILY_API_KEY)

def answer_query( query):
        search_results = tavily_client.search(query, max_results=5)
        context = "\n".join([result["content"] for result in search_results["results"]])
        prompt = f"Based on the following web content:\n{context}\n\nAnswer the query: {query}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content