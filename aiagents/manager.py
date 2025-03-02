from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RouteManager:
    def __init__(self, document_topics):
        self.topics = document_topics
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "llama3-8b-8192"
    
    def decide(self, query):
        topics_str = "\n".join([f"       - {topic}" for topic in self.topics])
        prompt = f"""
        Given the user query: "{query}"
        You are a routing agent that must STRICTLY decide whether to use:
        1. RAG (local document) - ONLY for queries specifically about:
{topics_str}
        
        2. WEB - for ALL other queries including any topic not directly related to the above topics.
        
        You must respond with ONLY one word: 'RAG' or 'WEB'.
        Think step by step:
        1. Is this query specifically about any of the listed topics?
        2. If not 100% certain it's about the document topics, choose WEB.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        decision = response.choices[0].message.content.strip().lower()
        return "rag" if "rag" in decision else "web"