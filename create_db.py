import os
# 1. Setup - Paste your API Key here
os.environ["GOOGLE_API_KEY"] = "AIzaSyAGLCJ_VZRiepn6iUv8MxCKu5WXn4u9whc"

# Import necessary libraries
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def create_database():
    # --- THE FIX: USE ABSOLUTE PATHS ---
    # This finds the exact folder where this python file is living
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # This builds the full path to the 'data' folder
    data_path = os.path.join(current_dir, "data")
    
    print(f"Looking for data in: {data_path}")
    
    # Check if folder exists
    if not os.path.exists(data_path):
        print("❌ ERROR: The folder 'data' does not exist at that path.")
        print("   Make sure you created a folder named 'data' inside 'Hackathon Project'.")
        return

    print("Loading data...")
    # Load all .txt files from the correct path
    loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    
    if len(docs) == 0:
        print("❌ ERROR: The 'data' folder is empty! Put 'law_data.txt' inside it.")
        return

    print(f"Found {len(docs)} documents. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print("Creating database (this might take a minute)...")
    
    # Create the database folder in the same project directory
    db_path = os.path.join(current_dir, "chroma_db")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        persist_directory=db_path
    )
    
    print("✅ SUCCESS: Database created successfully!")

if __name__ == "__main__":
    create_database()