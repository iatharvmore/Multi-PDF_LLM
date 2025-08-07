import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import uuid
import time
import requests
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.docstore.document import Document
from dotenv import load_dotenv

# Import Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Load environment variables and configure API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key not found. Please check your .env file.")
    st.stop()
genai.configure(api_key=api_key)

# Qdrant collection settings
COLLECTION_NAME = "pdf_documents"
VECTOR_SIZE = 768  # Size for Google's embedding-001 model
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

@st.cache_resource
def initialize_qdrant():
    """Initialize Qdrant client with robust error handling and health check"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Try to connect to Qdrant
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10.0)
            
            # Test connection with collections endpoint
            collections = client.get_collections()
            st.success("✅ Connected to Qdrant server")
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Retrying Qdrant connection ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                st.warning(f"Could not connect to Qdrant server: {str(e)}")
                st.info("Falling back to in-memory database. Data will be lost when app stops.")
                try:
                    memory_client = QdrantClient(":memory:")
                    return memory_client
                except Exception as mem_e:
                    st.error(f"Failed to initialize in-memory database: {str(mem_e)}")
                    return None
    
    return None

def probe_qdrant_api():
    """Probe the Qdrant API to check if it's available"""
    endpoints = [
        "/collections",
        "/readyz",
        "/livez",
        "/healthz",
    ]

    for path in endpoints:
        try:
            url = f"http://{QDRANT_HOST}:{QDRANT_PORT}{path}"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
    return False

def ensure_collection_exists(client):
    """Ensure the collection exists, creating it if necessary"""
    try:
        collections = client.get_collections().collections
        collection_exists = any(col.name == COLLECTION_NAME for col in collections)
        
        if not collection_exists:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            st.info(f"Created new collection: {COLLECTION_NAME}")
        return True
    except Exception as e:
        st.error(f"Error ensuring collection exists: {str(e)}")
        return False

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    if not pdf_docs:
        return ""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error processing PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into manageable chunks"""
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector embeddings using Qdrant"""
    if not text_chunks:
        return False
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        client = initialize_qdrant()
        if not client:
            return False
            
        # Clear existing collection
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)
        except Exception:
            pass  # Collection might not exist yet
            
        # Create collection
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        
        # Create embeddings and store them in batches
        with st.spinner("Creating embeddings and storing in Qdrant..."):
            points = []
            for i, chunk in enumerate(text_chunks):
                embedding = embeddings.embed_query(chunk)
                
                # Create a unique ID for each chunk
                point_id = str(uuid.uuid4())
                
                # Create the point with vector and payload
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "chunk_id": i,
                    }
                )
                points.append(point)
                
                # Upload in batches of 100 to avoid memory issues
                if len(points) >= 100:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points
                    )
                    points = []
            
            # Upload any remaining points
            if points:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                
        # Store collection info in session state
        st.session_state.collection_name = COLLECTION_NAME
        st.session_state.pdfs_processed = True
        return True
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return False

def get_conversational_chain():
    """Set up the QA chain with the LLM"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error setting up the model: {str(e)}")
        return None

def user_input(user_question):
    """Process user questions and generate responses"""
    try:
        # Initialize Qdrant client
        client = initialize_qdrant()
        if not client:
            st.error("Failed to connect to vector database.")
            return
            
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(col.name == COLLECTION_NAME for col in collections)
        if not collection_exists:
            st.error("No document database found. Please upload and process PDFs first.")
            return
            
        # Create embeddings for the query
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        query_vector = embeddings.embed_query(user_question)
        
        # Search for similar documents
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5  # Retrieve top 5 most similar chunks
        )
        
        if not search_results:
            st.write("No relevant information found in the documents.")
            return
            
        # Convert search results to Document objects for LangChain
        docs = []
        for result in search_results:
            doc = Document(
                page_content=result.payload.get("text", ""),
                metadata={"score": result.score, "id": result.id}
            )
            docs.append(doc)
            
        # Get the conversational chain
        chain = get_conversational_chain()
        if not chain:
            return
            
        # Generate response
        with st.spinner("Generating answer..."):
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            
            st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")

def check_qdrant_status():
    """Check Qdrant status and return appropriate message"""
    qdrant_available = probe_qdrant_api()
    client = initialize_qdrant()
    
    if qdrant_available:
        return "✅ Connected to Qdrant server"
    elif client and str(client._client.host) == ":memory:":
        return "⚠️ Using in-memory database (data will be lost on restart)"
    else:
        return "❌ Failed to connect to Qdrant"

def main():
    """Main application function"""
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖")
    
    # Initialize session state
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = COLLECTION_NAME
    if "pdfs_processed" not in st.session_state:
        # Check if collection exists
        client = initialize_qdrant()
        if client:
            collections = client.get_collections().collections
            st.session_state.pdfs_processed = any(col.name == COLLECTION_NAME for col in collections)
        else:
            st.session_state.pdfs_processed = False

    # Main area for questions
    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")
    if user_question:
        if not st.session_state.pdfs_processed:
            st.warning("Please upload and process PDF files first.")
        else:
            user_input(user_question)

    # Sidebar for file upload
    with st.sidebar:
        st.title("Multi-PDF's Chat bot 🤖")
        st.write("---")
        
        # Add storage status indicator
        status_msg = check_qdrant_status()
        if "✅" in status_msg:
            st.success(status_msg)
        elif "⚠️" in status_msg:
            st.warning(status_msg)
        else:
            st.error(status_msg)
        
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            success = get_vector_store(text_chunks)
                            if success:
                                st.success("Done! You can now ask questions about your documents.")
        
        # Clear data button
        if st.session_state.pdfs_processed and st.button("Clear PDF Data"):
            try:
                client = initialize_qdrant()
                if client:
                    client.delete_collection(collection_name=COLLECTION_NAME)
                    st.session_state.pdfs_processed = False
                    st.success("PDF data cleared successfully!")
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error clearing data: {str(e)}")
        
        # Check status and show feedback
        if st.session_state.pdfs_processed:
            st.info("PDF data has been processed and is ready for questions.")
        
        st.write("---")
        st.write("AI App created by @ Atharv More")

if __name__ == "__main__":
    main()
