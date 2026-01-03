import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# --- SETUP ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyAGLCJ_VZRiepn6iUv8MxCKu5WXn4u9whc"

def start_legal_chat():
    # 1. INITIALIZE DATABASE CONNECTION
    # We do this OUTSIDE the loop so we don't reload it every time (Performance optimization)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
    
    # 2. INITIALIZE THE MODEL
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # 3. INITIALIZE MEMORY (BACKEND JOB)
    # This list will store the conversation so the AI remembers context.
    chat_history = []

    print("\n⚖️  KOKRAJHAR LEGAL ASSISTANT (Type 'quit' to exit) ⚖️")
    print("AI: Hello. I am your legal assistant. Please describe your situation.\n")

    # 4. THE INFINITE LOOP (The Engine)
    while True:
        # A. Get User Input
        user_input = input("You: ")
        
        # B. Check for Exit Condition
        if user_input.lower() in ["quit", "exit", "stop"]:
            print("AI: Goodbye.")
            break

        # C. RETRIEVAL (RAG)
        # We search the DB for laws relevant to the *current* input
        results = db.similarity_search(user_input, k=3)
        context_text = "\n\n".join([doc.page_content for doc in results])

        # D. CONSTRUCT THE PROMPT (The "Packet" sent to the API)
        # We inject the Chat History so the AI knows what happened before.
        
        # Convert list format to string format for the prompt
        history_text = "\n".join(chat_history)

        prompt = f"""
        You are a strict Legal Assistant for Kokrajhar.
        
        GOAL: Determine if the user's situation is legal or illegal based on the Context.
        
        RULES:
        1. If you do not know if the user is Tribal or Non-Tribal, ASK THEM.
        2. If you do not know if they received a 'Notice', ASK THEM.
        3. Do not give a final verdict until you have these details.
        
        --- LEGAL KNOWLEDGE (DATABASE) ---
        {context_text}
        
        --- CONVERSATION HISTORY (MEMORY) ---
        {history_text}
        
        --- CURRENT USER INPUT ---
        User: {user_input}
        
        --- YOUR RESPONSE ---
        """

        # E. GENERATION (Call the AI)
        response = llm.invoke(prompt)
        ai_reply = response.content

        # F. UPDATE MEMORY (BACKEND JOB)
        # Save this turn into our history list
        chat_history.append(f"User: {user_input}")
        chat_history.append(f"AI: {ai_reply}")

        # G. Display Output
        print(f"AI: {ai_reply}\n")

if __name__ == "__main__":
    start_legal_chat()