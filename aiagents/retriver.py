import groq
from langchain.vectorstores import chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = groq.Groq(GROQ_API_KEY)
model = "llama3-8b-8192"
embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
)


def build_vector_store(document_path):
        # Load and split the document
        loader = PyPDFLoader(document_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create FAISS vector store with Hugging Face embeddings
        vector_store = chroma.from_documents(chunks,embeddings)
        return vector_store





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