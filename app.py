import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Please set it in a .env file.")

chat_history_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create a chat history instance for a given session ID."""
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]


def load_pdf(pdf_path):
    """Load and extract text from a PDF file."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_text(documents):
    """Split the extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def create_vectorstore(chunks, persist_directory="./chroma_db"):
    """Create and persist the vector store using embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    return vectorstore

def load_vectorstore(persist_directory="./chroma_db"):
    """Load the existing vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def setup_rag_pipeline(vectorstore):
    """Set up the conversational RAG pipeline."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

    # System prompt for answering questions based on retrieved context
    system_prompt = """
    Answer the following question based on the context provided. 
    Think step by step before providing the answer.
    <context>
    {context}
    </context>
    """

    # Create LLM chain for answering questions
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Document retrieval chain
    retriever = vectorstore.as_retriever()
    
    # Contextualized question reformulation prompt
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question, 
    which might reference context in the chat history, 
    formulate a standalone question which can be understood 
    without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    # Wrap RAG pipeline with session-based chat history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

def interactive_chat(rag_chain, session_id):
    """Run an interactive chat session with stateful session history."""
    print(f"\n🔹 Session ID: {session_id} (Type 'exit' or 'end' to quit) 🔹")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "end"]:
            print("🔹 Session ended. Chat history saved. 🔹")
            break

        response = rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        
        print("\nAI:", response["answer"])

def main():
    parser = argparse.ArgumentParser(description="Conversational RAG Chatbot with Stateful Chat History.")
    parser.add_argument("pdf_path", nargs="?", type=str, help="Path to the PDF file for processing.")

    args = parser.parse_args()

    # If no argument is provided, ask the user for input
    if args.pdf_path is None:
        args.pdf_path = input("\nEnter the path to your PDF file: ").strip()

    if not os.path.exists(args.pdf_path):
        print("\nError: The provided file path does not exist. Please check and try again.")
        return

    # Ask for a Session ID
    session_id = input("\nEnter a unique Session ID (e.g., 'user1', 'chat123'): ").strip()

    print("\n📜 Processing PDF and setting up the RAG pipeline...")

    # Load, split, and embed the document
    documents = load_pdf(args.pdf_path)
    chunks = split_text(documents)
    vectorstore = create_vectorstore(chunks)

    # Load the vectorstore and set up the RAG pipeline
    vectorstore_disk = load_vectorstore()
    rag_chain = setup_rag_pipeline(vectorstore_disk)

    # Start interactive chat with session tracking
    interactive_chat(rag_chain, session_id)

if __name__ == "__main__":
    main()
