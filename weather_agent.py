import requests
import json
from typing import Optional
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import re
import requests
import pgeocode
from config import OPENWEATHER_API_KEY, LLM_MODEL

@traceable(name="extract_location", tags=["weather"])
def extract_location_from_query(question: str) -> dict:
    """Extract city and state from query."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    
#     prompt = ChatPromptTemplate.from_messages([
#     ("system", """Extract city and state from the user's weather question. If only State is mentioned in the question then consider city as the capital of that state. Eg What is weather in Uttar Pradesh? then cosider city as Lucknow.
# Return ONLY JSON: {{"city": "...", "state": "..."}}
# If not found, use null.

# Examples:
# "Weather in Mumbai?" -> {{"city": "Mumbai", "state": "Maharashtra"}}
# "Temperature in jaipur" -> {{"city": "Jaipur", "state": "Rajasthan"}}
# "Humidity in goa" -> {{"city": "Goa", "state": "Goa"}}
# "Temperature in kolkata, West Bengal" -> {{"city": "Kolkata", "state": "West Bengal"}}
# "What is the Weather in Rajasthan?" -> {{"city": "Jaipur", "state": "Rajasthan"}}
# """),
#     ("human", "{question}")
# ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a location extraction expert for weather queries.

    RULES:
    1. Extract city and state from the user's weather question
    2. If ONLY a state is mentioned (no city), use the state's CAPITAL as the city
    3. If both city and state are mentioned, use them as-is
    4. If only a city is mentioned, identify its state if possible
    5. Return ONLY valid JSON: {{"city": "...", "state": "..."}}
    6. Use null if information cannot be determined

    IMPORTANT: When only state is mentioned, automatically use its capital city.

    Indian State Capitals Reference:
    - Maharashtra → Mumbai
    - Rajasthan → Jaipur
    - Uttar Pradesh → Lucknow
    - West Bengal → Kolkata
    - Karnataka → Bengaluru
    - Tamil Nadu → Chennai
    - Gujarat → Gandhinagar
    - Kerala → Thiruvananthapuram
    - Punjab → Chandigarh
    - Haryana → Chandigarh
    - Delhi → New Delhi
    - Goa → Panaji
    - Madhya Pradesh → Bhopal
    - Bihar → Patna
    - Odisha → Bhubaneswar
    - Telangana → Hyderabad
    - Andhra Pradesh → Amaravati
    - Assam → Dispur
    - Jharkhand → Ranchi
    - Chhattisgarh → Raipur

    Examples:
    "Weather in Mumbai?" → {{"city": "Mumbai", "state": "Maharashtra"}}
    "Temperature in Jaipur" → {{"city": "Jaipur", "state": "Rajasthan"}}
    "What is the weather in Rajasthan?" → {{"city": "Jaipur", "state": "Rajasthan"}}
    "Weather in Uttar Pradesh?" → {{"city": "Lucknow", "state": "Uttar Pradesh"}}
    "How's the climate in Maharashtra?" → {{"city": "Mumbai", "state": "Maharashtra"}}
    "Temperature in Kolkata, West Bengal" → {{"city": "Kolkata", "state": "West Bengal"}}
    "Weather in Karnataka" → {{"city": "Bengaluru", "state": "Karnataka"}}
    "Humidity in Goa" → {{"city": "Panaji", "state": "Goa"}}

    Return ONLY the JSON object, nothing else."""),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    
    try:
        return json.loads(result)
    except:
        return {"city": None, "state": None}

@traceable(name="fetch_weather", tags=["weather"])
def fetch_weather_data(city: str, state: Optional[str] = None) -> Optional[dict]:
    """Fetch weather from OpenWeather API."""
    if not city:
        return None
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},{state}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

@traceable(name="format_weather", tags=["weather"])
def format_weather_answer(weather_data: dict, city: str) -> str:
    """Format weather data."""
    if not weather_data:
        return f"Sorry, couldn't fetch weather for {city}."
    
    try:
        temp = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data["wind"]["speed"]
        
        return f"""🌤️ **Weather in {city.title()}**

**Temperature:** {temp}°C (Feels like {feels_like}°C)
**Condition:** {description.title()}
**Humidity:** {humidity}%
**Wind Speed:** {wind_speed} m/s

Have a great day! 🌈"""
    except:
        return "Weather data received but couldn't parse it."


# Optional: lightweight city→state map for India
INDIAN_CITY_TO_STATE = {
    "mumbai": "Maharashtra",
    "pune": "Maharashtra",
    "nagpur": "Maharashtra",
    "delhi": "Delhi",
    "bengaluru": "Karnataka",
    "bangalore": "Karnataka",
    "chennai": "Tamil Nadu",
    "kolkata": "West Bengal",
    "hyderabad": "Telangana",
    "ahmedabad": "Gujarat",
    "jaipur": "Rajasthan",
    "lucknow": "Uttar Pradesh",
    "surat": "Gujarat",
    "indore": "Madhya Pradesh",
    "patna": "Bihar",
    "bhopal": "Madhya Pradesh",
    "thane": "Maharashtra",
    "nashik": "Maharashtra",
    "chandigarh": "Chandigarh",
    "coimbatore": "Tamil Nadu",
    "vadodara": "Gujarat",
    "visakhapatnam": "Andhra Pradesh",
    "noida": "Uttar Pradesh",
    "gurugram": "Haryana",
    "goa": "Goa"
}


# def extract_city_state(question: str):
#     """Extract city/state from a natural language question."""
#     # Normalize
#     text = question.lower()
    
#     # Basic regex match for 'in <city>'
#     match = re.search(r"in\s+([a-zA-Z\s]+)", text)
#     if not match:
#         return None, None
    
#     possible_location = match.group(1).strip().rstrip("?.,")
    
#     # Split if state and city both mentioned (e.g. "in Mumbai, Maharashtra")
#     if "," in possible_location:
#         parts = [p.strip().title() for p in possible_location.split(",")]
#         city = parts[0]
#         state = parts[1] if len(parts) > 1 else None
#         return city, state
    
#     city = possible_location.title()
#     state = None

#     # If the city is in India mapping, fetch state
#     if city.lower() in INDIAN_CITY_TO_STATE:
#         state = INDIAN_CITY_TO_STATE[city.lower()]
    
#     return city, state


# def resolve_city_state(city: str, state: str = None, country: str = "India"):
#     """
#     If only city is provided, try to find the state (for India).
#     Otherwise return both as title case strings.
#     """
#     if not city:
#         return None, None

#     city = city.title()
#     if state:
#         return city, state.title()

#     # Try local map first
#     if city.lower() in INDIAN_CITY_TO_STATE:
#         return city, INDIAN_CITY_TO_STATE[city.lower()]

#     # Try pgeocode fallback
#     try:
#         nomi = pgeocode.Nominatim(country_code="IN")
#         loc = nomi.query_postal_code(city)
#         if loc is not None and hasattr(loc, "state_name") and isinstance(loc.state_name, str):
#             return city, loc.state_name
#     except Exception:
#         pass

#     return city, "India"  # Fallback


# def get_weather(city: str, state: str, api_key: str):
#     """Fetch weather data using OpenWeather API."""
#     try:
#         url = f"https://api.openweathermap.org/data/2.5/weather?q={city},{state},IN&appid={api_key}&units=metric"
#         response = requests.get(url)
#         data = response.json()

#         if data.get("cod") != 200:
#             return {"success": False, "error": data.get("message", "Unknown error")}

#         return {"success": True, "data": data}

#     except Exception as e:
#         return {"success": False, "error": str(e)}
