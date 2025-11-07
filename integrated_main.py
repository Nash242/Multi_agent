import sys
from pathlib import Path
from integrated_app import IntegratedApp

def main():
    print("🤖 Integrated RAG + Weather Assistant")
    print("=" * 60)
    
    pdf_path = None
    if len(sys.argv) > 1:
        pdf_file = Path(sys.argv[1])
        if pdf_file.exists():
            pdf_path = str(pdf_file)
            print(f"📄 Loaded PDF: {pdf_file.name}")
    
    print("\nI can help with:")
    print("  🌤️  Weather - 'What's the weather in Mumbai?'")
    if pdf_path:
        print("  📄 Document Q&A - 'Summarize chapter 1'")
    print("\nType 'exit' to quit")
    print("-" * 60)
    
    app = IntegratedApp()
    
    try:
        while True:
            question = input("\n💬 You: ").strip()
            
            if not question or question.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            result = app.query(question=question, pdf_path=pdf_path)
            
            print(f"\n🤖 Assistant: {result['answer']}")
            print(f"\n📊 Agent: {result.get('agent_type', 'unknown').upper()}")
            
            steps = result.get('steps', [])
            if steps:
                print("\n📋 Steps:")
                for step in steps:
                    print(f"  {step}")
            print("-" * 60)
    
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()
