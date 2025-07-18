
📚 Multi-PDF ChatApp AI Agent 🤖
Chat with multiple PDFs using Google Gemini, LangChain & FAISS.
Built with Streamlit, this app lets users upload and interact with PDFs in a conversational interface.

🧠 How It Works
PDF Upload – Load multiple PDFs.

Text Chunking – Split content for better context handling.

Embedding – Convert text to vectors using LLMs.

Semantic Search – Match user queries with relevant chunks.

AI Response – Generate responses using selected content.

🚀 Features
Adaptive Chunking – Dynamic window sizing for smarter retrieval.

Multi-hop QA – Query across multiple documents.

Supports PDF/TXT

LLM Options – Gemini Pro, GPT-3, Claude, LLaMA2, and more.

🧰 Tech Stack
Streamlit – UI

google-generativeai – Gemini model

LangChain – Embedding + RAG

FAISS – Vector DB

PyPDF2 – PDF handling

dotenv – Secure API config

🛠️ Setup
bash
Copy
Edit
git clone 
cd Multi-PDFs_ChatApp_AI-Agent
pip install -r requirements.txt
Create a .env file:

env
Copy
Edit
GOOGLE_API_KEY=your-key-here
Run the app:

bash
Copy
Edit
streamlit run app.py


🧪 Usage Steps
Upload one or more PDFs.

Click Submit & Process.

Ask natural language questions via chat.

Get answers pulled from all uploaded documents.

📄 License
MIT License

