import groq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = groq.Groq(api_key=GROQ_API_KEY)
model = "llama3-8b-8192"
embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
)


def analyze_document_topic(documents):
    prompt = f"""
    Analyze the following document content and provide 3-5 main topics it covers.
    Return ONLY the topics, one per line, no numbering or bullets.
    {documents[0].page_content[:2000]}
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().split('\n')


def build_vector_store(document_path):
    # Load and split the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    # Analyze document topics
    topics = analyze_document_topic(documents)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Create Chroma vector store with Hugging Face embeddings
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_store, topics


def answer_query(query, vector_store):
        # Retrieve relevant chunks
        
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Generate answer with Groq
        prompt = f"Based on the following context:\n{context}\n\nAnswer the query: {query}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content