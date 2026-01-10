import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# --- 1. CONFIGURATION ---
# PASTE YOUR KEY HERE (Keep the quotes "")
os.environ["GOOGLE_API_KEY"] = "PASTE_YOUR_GOOGLE_API_KEY_HERE"

# Define Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "chroma_db")

# Initialize Embedding Function (Must match the one used to create the DB)
embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Load Database (with Error Check)
if os.path.exists(db_path):
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_function)
    print("✅ Backend: Database loaded successfully.")
else:
    print("⚠️ WARNING: chroma_db folder not found. Please run create_database.py first!")
    vectorstore = None

# Initialize AI Model (Gemini 2.5 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# --- 2. THE CORE LOGIC FUNCTION ---
def get_chat_response(user_input, chat_history):
    """
    The Brain of the operation.
    1. Receives text (typed or voice-converted) from app.py.
    2. Searches the vector database for laws (Eviction AND Civil).
    3. Generates a legal response using the Balanced Prompt.
    """
    
    # Safety Check
    if not vectorstore:
        return "System Error: The legal database is missing. Please ask the admin to run 'create_database.py'."

    # A. INTELLIGENT RETRIEVAL 
    # We use k=6 to ensure we fetch enough documents to find 'General Law' 
    # even if there are many 'Eviction' documents.
    try:
        results = vectorstore.similarity_search(user_input, k=6)
        context_text = "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Database Search Error: {e}"

    # B. FORMAT HISTORY
    # Convert list of chat turns into a single string block for the AI
    history_text = "\n".join(chat_history)

    # C. THE BALANCED PROMPT (The "Judge" Logic)
    prompt = f"""
    You are an expert Legal Assistant for the Bodoland Territorial Region (BTR).
    
    INSTRUCTIONS FOR ANALYSIS:
    1. **CLASSIFY THE ISSUE:**
       - Is it a **Government/Forest Issue**? (Eviction notice, Tribal Belt, Forest Dept).
       - Is it a **Private Civil Dispute**? (Neighbor boundary, Family inheritance, Money, Fraud).
    
    2. **SELECT THE RIGHT LAWS:**
       - **If Private Dispute:** Prioritize "General Law" (Hindu Succession Act, Registration Act) and "Civil Remedies" (Simana Nirdharan, Partition Suit).
       - **If Government Eviction:** Prioritize "Prodyut Bora Case", "Section 165 ALRR", and "Due Process" (15-day notice).
    
    3. **BE HELPFUL:**
       - Do not just say "It is illegal." Give the user a Step-by-Step Action Plan (e.g., "File an FIR," "Apply for Mutation").
    
    --- LEGAL KNOWLEDGE BASE (Source of Truth) ---
    {context_text}
    
    --- CONVERSATION HISTORY (Context) ---
    {history_text}
    
    --- CURRENT USER QUESTION ---
    User: {user_input}
    
    --- YOUR LEGAL ADVICE ---
    """

    # D. GENERATION
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"AI Generation Error: {e}"

# --- 3. BACKEND TESTING BLOCK (Terminal Only) ---
if __name__ == "__main__":
    print("\n--- ⚖️  LEGAL AI BACKEND TEST (Type 'quit' to exit) ---")
    print("This mode tests logic only. Voice input happens in app.py.\n")
    
    test_history = []
    
    while True:
        user_in = input("\nYou: ")
        if user_in.lower() in ["quit", "exit"]:
            break
        
        # Simulate the function call
        reply = get_chat_response(user_in, test_history)
        
        print(f"AI: {reply}")
        
        # Update fake history for the next turn
        test_history.append(f"User: {user_in}")
        test_history.append(f"AI: {reply}")