# Conversational RAG Chatbot

## Project Overview

This project is a Conversational Retrieval-Augmented Generation (RAG) chatbot that processes PDFs, extracts relevant text, and allows interactive querying based on the extracted content. It leverages Google's Generative AI for embeddings and chat responses, along with Chroma for vector storage. The chatbot maintains stateful session-based chat history, improving contextual understanding in conversations.

## Installation

### Prerequisites
- Python 3.8+
- Required Python libraries (install via `requirements.txt`)
- Google API key (set in a `.env` file)

### Steps to Install
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/your-repository.git
   ```
2. Navigate to the project directory:
   ```sh
   cd your-repository
   ```
3. Create and activate a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
5. Set up your `.env` file with your Google API key:
   ```sh
   echo "GOOGLE_API_KEY=your_google_api_key" > .env
   ```

## Usage

### Running the Chatbot
To start the chatbot, run the following command:
```sh
python main.py path/to/your.pdf
```
If no PDF path is provided, the script will prompt you to enter one.

### Example Interaction
1. The chatbot loads and processes the provided PDF.
2. It extracts and stores relevant text as embeddings.
3. Users enter a session ID to track chat history.
4. The chatbot allows interactive conversations using the extracted document context.

Example:
```sh
Enter a unique Session ID: user123
📜 Processing PDF and setting up the RAG pipeline...
You: What is the main topic of this document?
AI: The document discusses...
```

### Exiting
Type `exit` or `end` to quit the interactive session.

## Contact

For inquiries, feel free to reach out:
- LinkedIn: [Tayyab Tanveer](https://www.linkedin.com/in/tayyab-tanveer-b000282b3/)
- Email: adamjosaph@gmail.com

