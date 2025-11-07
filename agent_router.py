from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from helper import extract_all_pdf_contexts

@traceable(name="route_agent", tags=["router"])
def route_to_agent(question: str, pdf_available: bool) -> str:
    """Classify user question into weather, rag, or unknown."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ✅ Quick check for weather
    weather_keywords = ["weather", "temperature", "rain", "climate", "humidity", "forecast"]
    if any(word in question.lower() for word in weather_keywords):
        return "weather"

    # ✅ If PDF is available, use its summarized context
    pdf_context = ""
    if pdf_available:
        pdf_context = extract_all_pdf_contexts("uploaded_pdfs")

    # ✅ If no PDF or empty context, fallback
    if not pdf_context:
        return "unknown"

    # Build intelligent routing prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
Classify the user's question based on the PDF content.

Categories:
- "rag": Question is answerable or related to the uploaded PDF.
- "weather": Question is about temperature, humidity, or climate.
- "unknown": If neither applies.

Logic:
If question is NOT about weather:
→ Check if it can be answered using the PDF context.
→ If yes, output "rag".
→ Otherwise, output "unknown".

PDF Context:
{pdf_context[:3000]}
"""),
        ("human", "{question}")
    ])

    try:
        result = (prompt | llm).invoke({"question": question})
        category = result.content.strip().lower()
        return category if category in ["rag", "weather", "unknown"] else "unknown"
    except Exception as e:
        print(f"Routing error: {e}")
        return "unknown"
