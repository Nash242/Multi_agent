from typing import Dict
from langsmith import traceable
from weather_agent import extract_location_from_query, fetch_weather_data, format_weather_answer

@traceable(name="weather_node", tags=["weather"])
def weather_node(state: Dict) -> Dict:
    """Handle weather queries."""
    location = extract_location_from_query(state["question"])
    city = location.get("city")
    state_name = location.get("state")
    
    if not city:
        return {
            "answer": "I couldn't identify a city from your question. Please specify a city name.\n\n**Example:** 'What's the weather in Mumbai?' or 'Temperature in Pune, Maharashtra?'",
            "metadata": {"agent": "weather", "success": False},
            "steps": ["❌ No city found in query"]
        }
    
    weather_data = fetch_weather_data(city, state_name)
    
    if not weather_data:
        location_str = f"{city}, {state_name}" if state_name else city
        return {
            "answer": f"Sorry, couldn't fetch weather data for {location_str}. Please check the city/state name and try again.",
            "metadata": {"agent": "weather", "success": False, "city": city, "state": state_name},
            "steps": [f"❌ Weather API failed for {location_str}"]
        }
    
    answer = format_weather_answer(weather_data, city)
    
    return {
        "city": city,
        "state": state_name,
        "weather_data": weather_data,
        "answer": answer,
        "metadata": {
            "agent": "weather",
            "success": True,
            "city": city,
            "state": state_name,
            "temperature": weather_data["main"]["temp"]
        },
        "steps": [f"✓ Fetched weather for {city}" + (f", {state_name}" if state_name else "")]
    }

@traceable(name="unknown_node", tags=["fallback"])
def unknown_node(state: Dict) -> Dict:
    """Handle unknown queries."""
    pdf_status = "✓ PDF loaded" if state.get("pdf_path") else "✗ No PDF"
    
    return {
        "answer": f"""I can help you with:

1. 📄 **Document Questions** - Ask about your uploaded PDF
   Status: {pdf_status}
   
2. 🌤️ **Weather Information** - Get weather for any city
   Example: "What's the weather in Mumbai?"

Please ask a question in one of these categories!""",
        "metadata": {"agent": "unknown"},
        "steps": ["❓ Unknown query type"]
    }
