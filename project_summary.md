# Project Summary: Dual-Domain Agentic RAG Platform

## Overview
The Dual-Domain Agentic RAG Platform is an advanced, autonomous AI agent built to intelligently retrieve information across multiple data sources. Unlike traditional RAG (Retrieval-Augmented Generation) systems that blindly query a single vector database, this system uses an Agentic routing architecture. It autonomously decides whether to search an internal Healthcare database, an internal Corporate Finance database, or pivot to a live web search (via Tavily AI) when proprietary data is insufficient. 

## Technology Stack
- **LLM Engine:** Llama-3.1-8B-Instant (via Groq API)
- **Agent Orchestration:** LangGraph (ReAct Agent architecture)
- **Vector Database:** ChromaDB 
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Web Search Tool:** Tavily AI Search API
- **Document Processing:** PyMuPDF, OpenCV, PyTesseract (OCR)
- **Frontend UI:** Streamlit (with a custom "Midnight FinTech Glass" CSS theme)

## Core Features
1. **Multi-Domain Routing:** The agent determines the context of the user's question and dynamically selects the correct database (Healthcare vs. Finance).
2. **Live Web Fallback:** If a user asks for real-time data (e.g., current stock prices, recent news), the agent recognizes the limitation of its internal database and autonomously executes a live web scrape to find the answer.
3. **Citation & Transparency:** The system prevents hallucinations by explicitly citing its sources, displaying UI badges to show exactly which tool it used, and providing dropdowns with raw source URLs and document excerpts.

---

# Video Demonstrations Summary

## Video 1: Healthcare Domain & Web Pivot
**What the video shows:**
1. **Ingestion:** The user uploads a medical PDF (`john_doe_bloodwork.pdf`) and classifies it as "healthcare". The system successfully chunks and embeds the document.
2. **Internal Retrieval:** The user asks for the patient's LDL cholesterol level. The Agent autonomously routes the query to the **Internal Medical Database** tool and accurately extracts the data (160 mg/dL).
3. **Live Web Pivot:** The user asks a follow-up question about the newest FDA-approved treatments for high LDL. Recognizing this is not in the patient's file, the Agent autonomously triggers the **Tavily Web Search** tool. It scrapes the live internet, provides the latest treatments, and displays the source URLs used to generate the answer.

## Video 2: Corporate Finance Domain & Web Pivot
**What the video shows:**
1. **Ingestion:** The user uploads a financial PDF (e.g., `tesla_q3_earnings.pdf`) and classifies it as "finance".
2. **Internal Retrieval:** The user asks for the total automotive revenue for Q3. The Agent autonomously routes the query to the **Internal Finance Database** tool and extracts the precise figure ($22.4 billion).
3. **Live Web Pivot:** The user asks for the current, real-time stock price of the company. The Agent bypasses the static vector database, triggers the **Tavily Web Search** tool, retrieves the live market price, and displays the citation URLs. 

## Key Takeaway for Recruiters
These videos demonstrate a deep understanding of modern Generative AI architecture. They prove the developer can build systems that overcome the biggest limitations of standard LLMs: static knowledge cutoffs and hallucinations. By implementing tool-calling and autonomous routing, the developer has built a resilient, production-ready Enterprise AI application.
