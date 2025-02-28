import groq
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = groq.Groq(GROQ_API_KEY)
model = "llama3-8b-8192"

def decide(query):
    prompt = f"""
        Given the user query: "{query}"
        Should this be answered using a local document (RAG) or a web search?
        Respond with 'RAG' or 'WEB'.
        """
    response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
    decision = response.choices[0].message.content.strip().lower()
    return "rag" if "rag" in decision else "web"