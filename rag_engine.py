"""
Agentic RAG Engine — Phase 6
Autonomous AI Agent that decides whether to search Medical DB, Finance DB, or the Live Web.
Uses LangGraph ReAct Agent + Tavily AI Search + ChromaDB Vector Retrievers.
"""

import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()


class AgenticRAG:
    """
    An autonomous RAG Agent that can:
    1. Search the Medical Vector Database
    2. Search the Finance Vector Database
    3. Search the Live Web via Tavily AI
    The Agent autonomously decides which tool to use based on the user's question.
    """

    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        print("[AgenticRAG] Initializing Agentic RAG Engine...")

        # 1. Setup the LLM (Groq Llama-3)
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )

        # 2. Setup Embeddings & Vector Store
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

        # 3. Build Domain-Specific Retrievers
        self.medical_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4, "filter": {"domain": "healthcare"}}
        )
        self.finance_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4, "filter": {"domain": "finance"}}
        )

        # 4. Create Agent Tools
        self.tools = self._build_tools()

        # 5. Create the ReAct Agent
        system_prompt = """You are an elite Enterprise AI Research Agent with access to three specialized tools:

1. **search_medical_docs**: Search the internal Medical/Healthcare vector database for patient records, prescriptions, diagnoses, lab results, and clinical data.
2. **search_finance_docs**: Search the internal Finance vector database for financial reports, revenue data, stock analysis, and corporate financial information.
3. **search_web**: Search the live internet via Tavily AI for real-time data, current events, market prices, or any information NOT available in the internal databases.

RULES:
- ALWAYS use a tool before answering. Never answer from your own knowledge alone.
- If the question is about medical/health topics, use search_medical_docs FIRST. If results are insufficient, THEN use search_web.
- If the question is about finance/business topics, use search_finance_docs FIRST. If results are insufficient, THEN use search_web.
- If the question is general or about current events, use search_web directly.
- After retrieving information, synthesize a clear, professional answer.
- DO NOT apologize or mention any previous mistakes. 
- DO NOT output your internal thought process. Provide only the final, confident answer.
- ALWAYS mention which source you used (Internal Medical DB, Internal Finance DB, or Live Web Search) at the end of your answer.
- Format your answer clearly with proper paragraphs."""

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=system_prompt
        )

        print("[AgenticRAG] Agent initialized with 3 tools: Medical DB, Finance DB, Web Search")

    def _build_tools(self):
        """Create the three tools the Agent can use."""
        # Tool 1: Medical Vector DB Search
        medical_tool = create_retriever_tool(
            self.medical_retriever,
            "search_medical_docs",
            "Search the internal Medical/Healthcare vector database. Use this tool for questions about patient records, prescriptions, diagnoses, lab results, medical conditions, and clinical data stored in our system."
        )

        # Tool 2: Finance Vector DB Search
        finance_tool = create_retriever_tool(
            self.finance_retriever,
            "search_finance_docs",
            "Search the internal Finance vector database. Use this tool for questions about financial reports, revenue data, stock analysis, corporate earnings, and financial documents stored in our system."
        )

        # Tool 3: Live Web Search via Tavily (Custom Wrapper to fix Llama 3.1 JSON errors)
        from langchain_core.tools import tool
        from tavily import TavilyClient

        @tool
        def search_web(query: str) -> str:
            """Search the live internet via Tavily AI for real-time data, current events, market prices, or any information NOT available in the internal databases."""
            try:
                client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
                response = client.search(query=query, max_results=3)
                results = response.get("results", [])
                output = ""
                for res in results:
                    output += f"Source URL: {res.get('url')}\nContent: {res.get('content')}\n\n"
                return output if output else "No results found on the web."
            except Exception as e:
                return f"Web search failed: {str(e)}"

        web_tool = search_web

        return [medical_tool, finance_tool, web_tool]

    def ask(self, question):
        """
        Send a question to the Agent and get back a structured response.
        Returns: dict with keys 'answer', 'sources', 'tool_used', 'response_time'
        """
        start_time = time.time()

        try:
            response = self.agent.invoke(
                {"messages": [HumanMessage(content=question)]}
            )

            # Extract the final answer
            final_message = response["messages"][-1]
            answer = final_message.content

            # Detect which tools were used
            tools_used = []
            sources = []
            for msg in response["messages"]:
                if hasattr(msg, "name") and msg.name:
                    tool_name = msg.name
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
                    # Extract source content for citations
                    if msg.content and len(msg.content) > 10:
                        source_preview = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                        sources.append({
                            "tool": tool_name,
                            "content": source_preview
                        })

            elapsed = round(time.time() - start_time, 2)

            # Map tool names to human-readable labels
            tool_labels = {
                "search_medical_docs": "Internal Medical Database",
                "search_finance_docs": "Internal Finance Database",
                "tavily_search": "Live Web Search (Tavily AI)"
            }
            readable_tools = [tool_labels.get(t, t) for t in tools_used]

            return {
                "answer": answer,
                "sources": sources,
                "tools_used": readable_tools,
                "response_time": elapsed
            }

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "sources": [],
                "tools_used": ["Error"],
                "response_time": elapsed
            }

    def get_db_stats(self):
        """Return statistics about the Vector Database."""
        try:
            collection = self.vectorstore._collection
            total_docs = collection.count()
            return {
                "total_chunks": total_docs,
                "embedding_model": "all-MiniLM-L6-v2",
                "llm_model": "llama-3.1-8b-instant",
                "tools_available": 3
            }
        except Exception:
            return {
                "total_chunks": 0,
                "embedding_model": "all-MiniLM-L6-v2",
                "llm_model": "llama-3.1-8b-instant",
                "tools_available": 3
            }


if __name__ == "__main__":
    rag = AgenticRAG()

    print("\n" + "=" * 50)
    print("TEST 1: Medical Question (Should use Medical DB)")
    print("=" * 50)
    result = rag.ask("What did Dr. Adams prescribe for Bob Williams?")
    print(f"Answer: {result['answer']}")
    print(f"Tools Used: {result['tools_used']}")
    print(f"Time: {result['response_time']}s")

    print("\n" + "=" * 50)
    print("TEST 2: Web Question (Should use Tavily)")
    print("=" * 50)
    result = rag.ask("What is the current price of Nvidia stock?")
    print(f"Answer: {result['answer']}")
    print(f"Tools Used: {result['tools_used']}")
    print(f"Time: {result['response_time']}s")
