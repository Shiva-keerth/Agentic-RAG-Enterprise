from rag_engine import AgenticRAG
import warnings
warnings.filterwarnings('ignore')

rag = AgenticRAG()

print("\n--- TEST 1: Medical ---")
r1 = rag.ask("Based on the uploaded healthcare document, what is John Doe's LDL cholesterol level?")
print(r1["answer"])

print("\n--- TEST 2: Finance ---")
r2 = rag.ask("What was Acme Corp's total revenue in Q3 according to the finance report?")
print(r2["answer"])
