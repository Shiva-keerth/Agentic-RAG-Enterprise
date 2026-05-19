# Dual-Domain Agentic RAG Platform — Complete Project Guide

> [!IMPORTANT]
> This document covers EVERYTHING: architecture, line-by-line code, ML/AI concepts, how to use it, and 30+ interview Q&A. After reading this, you will be able to explain, defend, and reuse every component.

---

## 1. What Is This Project?

An **Agentic RAG (Retrieval-Augmented Generation)** platform that answers questions using **3 sources**:

| Source | How It Works |
|--------|-------------|
| **Medical Vector DB** | Searches embedded healthcare PDFs stored in ChromaDB |
| **Finance Vector DB** | Searches embedded financial documents stored in ChromaDB |
| **Live Web (Tavily AI)** | Searches the real-time internet for current data |

The **"Agentic"** part means the AI **autonomously decides** which source to use — you don't tell it. Ask about a patient → it picks Medical DB. Ask about stock prices → it picks Tavily Web Search.

---

## 2. Architecture Diagram

```
User Question
     │
     ▼
┌─────────────────────────────┐
│  Streamlit UI (app.py)      │
│  - Chat interface           │
│  - PDF uploader             │
│  - Metric dashboard         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  AgenticRAG (rag_engine.py) │
│  - LangGraph ReAct Agent    │
│  - Groq Llama-3.1 LLM       │
│  - System Prompt (Rules)    │
└──────────┬──────────────────┘
           │
     Agent THINKS → Picks a Tool
           │
     ┌─────┼──────────┐
     ▼     ▼          ▼
┌────────┐ ┌────────┐ ┌──────────┐
│Medical │ │Finance │ │ Tavily   │
│Retriever│ │Retriever│ │Web Search│
│(ChromaDB)│(ChromaDB)│ │(Internet)│
└────────┘ └────────┘ └──────────┘
     │         │           │
     └─────────┴───────────┘
               │
               ▼
        Agent SYNTHESIZES
        Final Answer + Sources
               │
               ▼
        Displayed in Chat UI
        with Tool Badge + Citations
```

---

## 3. Project Files Overview

| File | Purpose | Lines |
|------|---------|-------|
| `rag_engine.py` | The AI brain — Agent + 3 Tools | 204 |
| `ingestion.py` | PDF/Image → Text → Chunks → ChromaDB | 153 |
| `app.py` | Streamlit UI with Midnight FinTech theme | 538 |
| `run_ingestion.py` | Script to load initial healthcare data | 29 |
| `.env` | API keys (Groq + Tavily) | 2 |
| `requirements.txt` | Python dependencies | 13 |

---

## 4. File 1: `ingestion.py` — The Data Pipeline

### What It Does
Takes raw documents (PDFs, images) → extracts text → splits into chunks → converts to vectors → stores in ChromaDB.

### Line-by-Line Breakdown

#### Imports (Lines 7-13)
```python
import fitz          # PyMuPDF — reads PDF files digitally (no OCR needed)
import cv2           # OpenCV — image processing (grayscale conversion)
import pytesseract   # Tesseract OCR — extracts text from images
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Chunks text
from langchain_huggingface import HuggingFaceEmbeddings              # BERT embeddings
from langchain_chroma import Chroma                                  # Vector database
```

#### Class Init (Lines 21-38)
```python
class DocumentIngestor:
    def __init__(self, persist_directory="./chroma_db"):
        # 1. Load the embedding model (BERT variant)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Each chunk = max 1000 characters
            chunk_overlap=200,  # 200 chars overlap between chunks
        )
        
        # 3. Connect to ChromaDB (vector database)
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
```

**Why chunk_size=1000?** LLMs have token limits. If you feed them a 50-page PDF at once, it won't fit. Chunking splits the document into digestible pieces.

**Why chunk_overlap=200?** Without overlap, a sentence like "The patient was diagnosed with diabetes" could get split into "The patient was diagnosed" and "with diabetes" — losing meaning. Overlap ensures sentences aren't cut.

#### PDF Text Extraction (Lines 40-54)
```python
def extract_text_from_pdf(self, pdf_path):
    doc = fitz.open(pdf_path)          # Open PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load each page
        text = page.get_text("text")    # Extract text digitally
        pages.append((page_num + 1, text))
    return pages  # Returns [(1, "page 1 text"), (2, "page 2 text"), ...]
```

**PyMuPDF vs OCR:** PyMuPDF reads text directly from the PDF file structure (fast, accurate). OCR (Tesseract) is only needed for scanned images/photos of documents.

#### Upload Support (Lines 56-70)
```python
def extract_text_from_pdf_bytes(self, file_bytes, filename):
    doc = fitz.open(stream=file_bytes, filetype="pdf")  # Open from bytes, not file path
```
This method accepts raw bytes from `st.file_uploader()` — enabling drag-and-drop uploads.

#### OCR Image Extraction (Lines 72-82)
```python
def extract_text_from_image(self, image_path):
    img = cv2.imread(image_path)                    # Read image with OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Convert to grayscale
    text = pytesseract.image_to_string(gray)         # OCR: image → text
```

**Why grayscale?** OCR works better on black-and-white images. Color adds noise.

#### Chunking + Embedding + Storage (Lines 84-104)
```python
def process_and_store(self, text, source_name, domain, page_num=None):
    chunks = self.text_splitter.split_text(text)   # Split into 1000-char chunks
    
    metadatas = []
    for i, _ in enumerate(chunks):
        meta = {
            "source": source_name,   # e.g., "john_doe_bloodwork.pdf"
            "domain": domain,        # "healthcare" or "finance"
            "chunk_index": i
        }
        if page_num: meta["page"] = page_num
        metadatas.append(meta)
    
    self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
```

**What happens inside `add_texts()`:**
1. Each chunk gets passed through the BERT model → produces a 384-dimensional vector
2. The vector + text + metadata get saved to ChromaDB on disk
3. The `domain` metadata field is what allows us to filter later ("only search healthcare docs")

### Key ML Concepts in This File

| Concept | What It Means |
|---------|--------------|
| **Embedding** | Converting text → numbers (vectors). "Diabetes" → [0.23, -0.14, 0.87, ...] |
| **all-MiniLM-L6-v2** | A lightweight BERT model (22M params) that creates 384-dim vectors |
| **Vector Database** | A database optimized for similarity search on vectors, not SQL queries |
| **Chunking** | Splitting large documents into smaller pieces for processing |
| **Metadata Filtering** | Tagging chunks with labels so you can search within a specific domain |

---

## 5. File 2: `rag_engine.py` — The AI Brain

### What It Does
Creates an autonomous AI Agent that receives a question, decides which tool to use, retrieves information, and synthesizes a professional answer.

### Line-by-Line Breakdown

#### Imports (Lines 7-16)
```python
from langchain_groq import ChatGroq              # Groq's ultra-fast LLM API
from langchain_huggingface import HuggingFaceEmbeddings  # Same BERT embeddings
from langchain_chroma import Chroma               # Same vector store
from langchain_core.tools.retriever import create_retriever_tool  # Wraps retriever as a tool
from langchain_tavily import TavilySearch          # Real-time web search
from langchain_core.messages import HumanMessage   # Message format for agents
from langgraph.prebuilt import create_react_agent  # Agent factory
```

#### LLM Setup (Lines 34-39)
```python
self.llm = ChatGroq(
    temperature=0,                              # 0 = deterministic (same input = same output)
    groq_api_key=os.getenv("GROQ_API_KEY"),    # API key from .env
    model_name="llama-3.1-8b-instant"           # Meta's Llama 3.1 model
)
```

**Why temperature=0?** For a RAG system, you want factual, consistent answers. Temperature=1 would make it creative/random — bad for medical/financial data.

**Why Groq?** Groq uses custom LPU chips that run inference 10x faster than GPU. Free tier gives ~30 requests/min.

#### Domain-Specific Retrievers (Lines 48-54)
```python
self.medical_retriever = self.vectorstore.as_retriever(
    search_kwargs={"k": 4, "filter": {"domain": "healthcare"}}
)
self.finance_retriever = self.vectorstore.as_retriever(
    search_kwargs={"k": 4, "filter": {"domain": "finance"}}
)
```

**`k=4`** → Return top 4 most similar chunks.
**`filter={"domain": "healthcare"}`** → Only search chunks tagged as healthcare. This is why metadata matters!

**How similarity search works:**
1. User asks "What is Bob's blood pressure?"
2. The question gets embedded → [0.45, -0.12, 0.78, ...]
3. ChromaDB compares this vector against ALL healthcare chunk vectors
4. Returns the 4 chunks with highest cosine similarity

#### Tool Creation (Lines 83-106)
```python
# Tool 1: Medical DB
medical_tool = create_retriever_tool(
    self.medical_retriever,
    "search_medical_docs",
    "Search the internal Medical/Healthcare vector database..."
)

# Tool 2: Finance DB
finance_tool = create_retriever_tool(
    self.finance_retriever,
    "search_finance_docs",
    "Search the internal Finance vector database..."
)

# Tool 3: Live Web
web_tool = TavilySearch(
    max_results=3,       # Return top 3 web results
    topic="general",
    include_answer=True  # Tavily also generates a quick answer
)
```

**Why tool descriptions matter:** The Agent reads these descriptions to decide which tool to use. If the description says "patient records, prescriptions," the Agent will use it for medical questions.

#### The ReAct Agent (Lines 59-79)
```python
system_prompt = """You are an elite Enterprise AI Research Agent...
RULES:
- ALWAYS use a tool before answering
- Medical questions → search_medical_docs FIRST
- Finance questions → search_finance_docs FIRST
- General/current events → search_web directly
..."""

self.agent = create_react_agent(
    model=self.llm,
    tools=self.tools,
    prompt=system_prompt
)
```

**ReAct Pattern = Reason + Act:**
1. **Reason:** "The user asked about stock prices. This is finance. Let me search the finance DB."
2. **Act:** Calls `search_finance_docs`
3. **Observe:** Gets results back
4. **Reason:** "The results don't have current prices. Let me try the web."
5. **Act:** Calls `search_web`
6. **Synthesize:** Combines all retrieved data into a final answer

#### The Ask Method (Lines 108-164)
```python
def ask(self, question):
    response = self.agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )
    
    # Extract final answer (last message in the chain)
    final_message = response["messages"][-1]
    answer = final_message.content
    
    # Detect which tools the agent used
    tools_used = []
    sources = []
    for msg in response["messages"]:
        if hasattr(msg, "name") and msg.name:  # Tool messages have a "name" field
            tool_name = msg.name
            tools_used.append(tool_name)
            sources.append({"tool": tool_name, "content": msg.content[:300]})
```

**The message chain looks like:**
1. `HumanMessage` — "What is Nvidia stock price?"
2. `AIMessage` — (Agent thinks: "I need to search the web")
3. `ToolMessage(name="tavily_search")` — Results from Tavily
4. `AIMessage` — Final synthesized answer

### Key ML/AI Concepts in This File

| Concept | What It Means |
|---------|--------------|
| **RAG** | Retrieval-Augmented Generation — LLM answers using retrieved documents, not just its training data |
| **Agentic RAG** | The AI decides WHICH source to search autonomously |
| **ReAct Agent** | Reason-Act loop — the agent thinks, acts, observes, and repeats |
| **LangGraph** | Modern framework for building stateful AI agents |
| **Cosine Similarity** | Measures how similar two vectors are (1.0 = identical, 0 = unrelated) |
| **Retriever** | A component that fetches relevant documents from a vector store |
| **Tool Calling** | LLM can call external functions/APIs as "tools" |
| **System Prompt** | Instructions that control how the Agent behaves |

---

## 6. File 3: `app.py` — The Premium UI

### What It Does
Streamlit web application with the Midnight FinTech glassmorphism theme. Contains the chat interface, PDF uploader, and metrics dashboard.

### Key Sections

#### Caching (Lines 381-390)
```python
@st.cache_resource
def load_rag_engine():
    return AgenticRAG()
```
**Why cache?** Loading the BERT model + ChromaDB takes ~5 seconds. Without caching, this would run on EVERY page refresh. `@st.cache_resource` loads it ONCE and reuses it.

#### Chat State Management (Lines 470-472)
```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```
**`st.session_state`** persists data across Streamlit reruns. Without it, chat history would disappear every time you send a message.

#### The Chat Loop (Lines 499-537)
```python
if prompt := st.chat_input("Ask anything..."):
    # 1. Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Call the Agent
    result = rag.ask(prompt)
    
    # 3. Display answer + tool badges + source citations
    st.markdown(f'<div class="chat-ai">{result["answer"]}</div>')
    
    # 4. Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tools_used": result["tools_used"],
        "sources": result["sources"]
    })
```

---

## 7. How to Use This Project

### First-Time Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add API keys to .env file
GROQ_API_KEY=gsk_xxxxx
TAVILY_API_KEY=tvly-xxxxx

# 3. Load initial data into ChromaDB
python run_ingestion.py

# 4. Launch the app
streamlit run app.py
```

### How to Query
| Question Type | What Happens |
|--------------|-------------|
| "What is Bob's diagnosis?" | Agent uses Medical DB tool |
| "What is Tesla's revenue?" | Agent uses Finance DB tool |
| "What is Bitcoin price today?" | Agent uses Tavily Web Search |
| "Compare diabetes treatments" | Agent may use Medical DB + Web Search |

### How to Add New Documents
1. Click **Browse Files** in the sidebar
2. Select a PDF file
3. Choose domain: **healthcare** or **finance**
4. Click **Ingest Document**
5. The PDF is now searchable in the chat!

---

## 8. How to Reuse These Concepts

### Using Embeddings in Other Projects
```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector = embeddings.embed_query("any text here")
# vector is now a list of 384 numbers
```
**Use cases:** Semantic search, recommendation systems, duplicate detection, clustering.

### Using RAG in Other Projects
```python
# Any project that needs an LLM to answer from YOUR data:
# 1. Embed your documents into a vector store
# 2. Create a retriever
# 3. Chain: retriever → LLM → answer
```
**Use cases:** Customer support bots, legal document Q&A, code documentation search.

### Using Agents in Other Projects
```python
from langgraph.prebuilt import create_react_agent
# Give the agent tools = functions it can call
# The agent decides WHEN and WHICH tool to use
```
**Use cases:** Research assistants, data analysis bots, workflow automation.

---

## 9. Interview Questions & Answers (30+)

### Basics

**Q1: What is RAG?**
> RAG (Retrieval-Augmented Generation) is a technique where an LLM first retrieves relevant documents from a knowledge base, then uses those documents as context to generate an accurate answer. This prevents hallucination because the LLM answers from real data, not just its training.

**Q2: Why RAG instead of fine-tuning?**
> Fine-tuning is expensive ($$$), requires GPU training, and the model becomes outdated. RAG is cheaper — you just update the documents in the database. Also, RAG provides source citations, which fine-tuning cannot.

**Q3: What makes your RAG "Agentic"?**
> Normal RAG always searches the same database. My system has an autonomous Agent that DECIDES which of 3 tools to use based on the question. It can even chain multiple tools — search the DB first, then search the web if results are insufficient.

### Embeddings & Vectors

**Q4: What is a vector embedding?**
> A numerical representation of text in high-dimensional space. Similar meanings → similar vectors. "Doctor" and "Physician" would have vectors close together, while "Doctor" and "Pizza" would be far apart.

**Q5: Why all-MiniLM-L6-v2 specifically?**
> It's a lightweight BERT model (22M params, 384 dimensions) optimized for semantic similarity. It's fast enough for real-time search and accurate enough for production use. Full BERT (110M params) would be slower without significant accuracy gain for this use case.

**Q6: What is cosine similarity?**
> It measures the angle between two vectors. cosine(A, B) = (A·B) / (|A|×|B|). Value ranges from -1 to 1. In our system, when a user asks a question, the question gets embedded and compared against all stored chunk vectors using cosine similarity. Top-k most similar chunks are returned.

**Q7: How does ChromaDB store vectors?**
> ChromaDB stores each chunk as: (ID, vector_embedding, text_content, metadata). When you search, it computes cosine similarity between the query vector and all stored vectors, and returns the top-k matches. It persists data to disk in `./chroma_db/`.

### Chunking

**Q8: Why chunk documents? Why not feed the whole PDF?**
> LLMs have context window limits (8K-128K tokens). A 50-page PDF could be 25K+ tokens. Also, retrieving a small, relevant chunk gives better answers than feeding the entire document where most content is irrelevant.

**Q9: Explain your chunking strategy.**
> RecursiveCharacterTextSplitter with chunk_size=1000 and overlap=200. "Recursive" means it tries to split at paragraph breaks first, then sentences, then words — preserving semantic meaning. The 200-char overlap ensures no sentence is cut in half.

**Q10: What would happen with chunk_overlap=0?**
> Sentences at chunk boundaries would get split. "The patient was prescribed Metformin for diabetes" could become two chunks: "...prescribed Metformin" and "for diabetes..." — losing the connection.

### Agent Architecture

**Q11: What is the ReAct pattern?**
> ReAct = Reason + Act. The agent follows a loop: (1) Reason about what to do, (2) Act by calling a tool, (3) Observe the results, (4) Reason again. This continues until it has enough information to answer.

**Q12: How does the agent decide which tool to use?**
> Through the system prompt + tool descriptions. The LLM reads the tool descriptions and the user's question, then reasons: "This question is about medical data, so I should use search_medical_docs." The tool descriptions act as a routing mechanism.

**Q13: What is LangGraph and why use it over AgentExecutor?**
> LangGraph is the modern replacement for LangChain's deprecated AgentExecutor. It provides stateful graph-based agent orchestration with better error handling, streaming support, and the ability to build complex multi-step workflows.

**Q14: What is tool calling?**
> Tool calling is when an LLM outputs a structured function call instead of text. The LLM says "I want to call search_medical_docs with query='blood pressure'" and the framework actually executes that function and feeds the results back.

### APIs & Infrastructure

**Q15: Why Groq instead of OpenAI?**
> Groq runs on custom LPU (Language Processing Unit) chips that are 10x faster than GPU inference. Their free tier gives ~30 req/min with Llama 3.1. OpenAI would cost money and be slower for this use case.

**Q16: What is Tavily Search?**
> Tavily is an AI-optimized search API designed specifically for LLM agents. Unlike Google Search, Tavily returns clean, structured content that's ready for LLM consumption. It also provides a pre-generated answer alongside raw results.

**Q17: What is the role of the .env file?**
> It stores sensitive API keys (GROQ_API_KEY, TAVILY_API_KEY) outside the codebase. `load_dotenv()` loads them as environment variables. This prevents accidental exposure of keys in version control.

### Data Pipeline

**Q18: Explain the full data flow from PDF upload to answer.**
> 1. User uploads PDF → `st.file_uploader()` returns bytes
> 2. PyMuPDF extracts text page-by-page
> 3. RecursiveCharacterTextSplitter chunks the text (1000 chars, 200 overlap)
> 4. all-MiniLM-L6-v2 embeds each chunk → 384-dim vector
> 5. Vectors + text + metadata stored in ChromaDB
> 6. User asks question → question gets embedded
> 7. ChromaDB finds top-4 similar chunks via cosine similarity
> 8. Chunks fed to Llama-3.1 as context → generates answer

**Q19: What is metadata filtering and why is it important?**
> Each chunk is tagged with `{"domain": "healthcare"}` or `{"domain": "finance"}`. When the medical retriever searches, it adds `filter={"domain": "healthcare"}` so it ONLY searches healthcare chunks. Without filtering, a medical question could return irrelevant finance data.

**Q20: How does OCR work in your system?**
> For scanned documents (images, not digital PDFs): OpenCV converts the image to grayscale (removes color noise), then Tesseract OCR uses pattern recognition to identify characters in the image and convert them to text. This extracted text then follows the same chunking → embedding → storage pipeline.

### UI & Architecture

**Q21: Why Streamlit?**
> Streamlit allows building full web apps in pure Python — no HTML/CSS/JS expertise needed. It has built-in components for file uploads, chat interfaces, and state management. Ideal for ML demos and data apps.

**Q22: What is `@st.cache_resource`?**
> A Streamlit decorator that caches expensive operations. Loading the BERT model takes ~5 seconds. Without caching, it would reload on every page interaction. `cache_resource` loads it once and reuses the same object across all sessions.

**Q23: What is `st.session_state`?**
> Streamlit reruns the entire script on every interaction. `session_state` is a persistent dictionary that survives reruns. We use it to store chat history — without it, messages would vanish after each new message.

### Advanced / Production

**Q24: How would you scale this for 1000 users?**
> 1. Replace ChromaDB with Pinecone/Weaviate (cloud-hosted, horizontally scalable)
> 2. Add Redis caching for frequent queries
> 3. Deploy on Kubernetes with load balancing
> 4. Use a message queue (RabbitMQ) for async ingestion
> 5. Add user authentication and multi-tenant isolation

**Q25: How do you handle hallucination?**
> Three layers: (1) The system prompt says "ALWAYS use a tool, never answer from your own knowledge." (2) RAG itself grounds the LLM in real retrieved documents. (3) Source citations let users verify the answer against the original document.

**Q26: What would you improve?**
> 1. Add hybrid search (keyword + vector) for better retrieval
> 2. Implement re-ranking with a cross-encoder model
> 3. Add conversation memory so the agent remembers previous questions
> 4. Support more file types (DOCX, CSV, Excel)
> 5. Add user authentication and document access control

**Q27: What is the difference between dense and sparse retrieval?**
> Dense retrieval uses vector embeddings (our approach) — good for semantic meaning. Sparse retrieval uses keyword matching (like BM25/TF-IDF) — good for exact terms. Hybrid search combines both for best results.

**Q28: What is a cross-encoder re-ranker?**
> After retrieving top-k chunks, a cross-encoder model re-scores each (query, chunk) pair together. It's more accurate than cosine similarity because it sees both texts simultaneously, but too slow for initial retrieval. Used as a second-stage filter.

**Q29: Can the agent use multiple tools in one query?**
> Yes! The ReAct loop allows chaining. If the agent searches the Medical DB and gets insufficient results, it can THEN call Tavily Web Search in the same turn. The system prompt explicitly instructs this behavior.

**Q30: How does `create_retriever_tool` work internally?**
> It wraps a LangChain Retriever into a Tool object with a name and description. When the agent calls this tool with a query string, it: (1) embeds the query, (2) searches ChromaDB with the configured filters, (3) returns the top-k document contents as a string back to the agent.

**Q31: What is the difference between RAG and a regular chatbot?**
> A regular chatbot answers purely from its training data (can be outdated, can hallucinate). RAG adds a retrieval step — it first fetches relevant documents from YOUR database, then uses those as context. This means: (1) answers are grounded in real data, (2) you can update knowledge without retraining, (3) you get source citations.

**Q32: Explain the embedding dimension (384). Why not 768 or 1536?**
> 384 dimensions is a trade-off. More dimensions = more information captured, but slower search and more storage. all-MiniLM-L6-v2 uses 384 dims and achieves 95%+ of the accuracy of full BERT (768 dims) at 3x the speed. For our use case, 384 is the sweet spot.

---

## 10. Technology Stack Summary

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| LLM | Groq Llama-3.1-8b | Free, fast (LPU), open-source |
| Embeddings | all-MiniLM-L6-v2 | Lightweight, fast, 384-dim |
| Vector DB | ChromaDB | Local, persistent, metadata filtering |
| Web Search | Tavily AI | Built for LLM agents, structured output |
| Agent Framework | LangGraph | Modern, stateful, replaces AgentExecutor |
| PDF Parsing | PyMuPDF (fitz) | Fast digital PDF text extraction |
| OCR | Tesseract + OpenCV | Handles scanned/image documents |
| UI | Streamlit | Python-only web apps, rapid prototyping |
| Orchestration | LangChain | Chains, tools, retrievers, prompts |
