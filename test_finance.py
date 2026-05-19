from rag_engine import DualRAG

print("\n" + "="*40)
print("TESTING FINANCE RAG ENGINE")
print("="*40)

rag = DualRAG()

q1 = "What was the gross margin for Apple in fiscal 2025 and why did it improve?"
print(f"\nUser: {q1}")
print(f"AI: {rag.ask_finance_question(q1)}")

q2 = "How many vehicles did Tesla deliver in Q3 2026?"
print(f"\nUser: {q2}")
print(f"AI: {rag.ask_finance_question(q2)}")

q3 = "What were the results of John Doe's blood test?"
print(f"\nUser: {q3}")
print(f"AI: {rag.ask_finance_question(q3)}")
