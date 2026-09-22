import requests
import streamlit as st
from openai import OpenAI

if "openai_api_key" not in st.secrets:
    st.error("Please add your openai_api_key to secrets.toml")
    st.stop()

client = OpenAI(api_key=st.secrets["openai_api_key"])
MODEL = "gpt-4o-mini"

#weather function tool 
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather and today's forecast for a given city, to help suggest what to wear and what outdoor activities are appropriate.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. 'Syracuse, NY' or 'Paris'. Defaults to 'Syracuse, NY' if not specified by the user.",
                }
            },
            "required": ["location"],
        },
    },
}
def get_current_weather(location="Syracuse, NY"):
    url = f'https://wttr.in/{location}?format=j1'
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f'wttr.in error: status {response.status_code}')
    try:
        data = response.json()
    except ValueError:
        raise Exception(f'Could not find a location named {location}')

    current = data['current_condition'][0]
    today = data['weather'][0]
    astronomy = today['astronomy'][0]
    midday = next((h for h in today['hourly'] if h['time'] == '1200'), today['hourly'][4])

    return {
        'location': location,
        'temp_F': float(current['temp_F']),
        'feels_like_F': float(current['FeelsLikeF']),
        'description': current['weatherDesc'][0]['value'],
        'humidity': current['humidity'],
        'wind_speed_mph': current['windspeedMiles'],
        'uv_index': current['uvIndex'],
        'max_temp_F': today['maxtempF'],
        'min_temp_F': today['mintempF'],
        'chance_of_rain_midday': midday['chanceofrain'],
        'chance_of_snow_midday': midday['chanceofsnow'],
        'wind_gust_mph_midday': midday['WindGustMiles'],
        'sunrise': astronomy['sunrise'],
        'sunset': astronomy['sunset'],
    }

st.title("What to Wear Bot")

location = st.text_input("Enter a city:", value="Syracuse, NY")

if st.button("Get advice"):
    # 1. Ask the LLM what to wear, giving it access to the weather tool
    messages = [
        {
            "role": "user",
            "content": f"What should I wear today in {location}, and what outdoor activities would be good?"
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=[weather_tool],
        tool_choice="auto",
    )

    response_message = response.choices[0].message

    # 2. Check if the LLM wants to call the weather tool
    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        import json
        args = json.loads(tool_call.function.arguments)
        call_location = args.get("location", "Syracuse, NY")

        # 3. Actually call the weather function
        weather_data = get_current_weather(call_location)

        # 4. Send the weather data back to the LLM for its final answer
        messages.append(response_message.to_dict())
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(weather_data),
        })

        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        st.write(second_response.choices[0].message.content)
    else:
        # LLM didn't need the tool (shouldn't normally happen here, but just in case)
        st.write(response_message.content)

