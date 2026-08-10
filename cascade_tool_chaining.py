# Three function tools called in sequence, each fed by the previous result. The
# model does the chaining from the instructions and the tool schemas alone;
# every tool body runs in this process.


import os

import zrt
from zrt import Agent, Pipeline, Room, function_tool
from zrt.inference import TurnDetector
from zrt.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "tool-chaining-agent")


@function_tool
async def get_weather(city: str) -> dict:
    """Get the current temperature for a city. Call this first.

    Args:
        city: The city to look up, e.g. "Paris" or "Mumbai".
    """
    import aiohttp

    async with aiohttp.ClientSession() as http:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        async with http.get(geo_url) as response:
            geo = await response.json()
        if not geo.get("results"):
            return {"error": f"I could not find {city}."}

        place = geo["results"][0]
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={place['latitude']}&longitude={place['longitude']}"
            "&current=temperature_2m"
        )
        async with http.get(url) as response:
            data = await response.json()

    temperature = data["current"]["temperature_2m"]
    logger.info("%s is %s C", city, temperature)
    return {"city": place["name"], "temperature": temperature, "unit": "Celsius"}


@function_tool
async def get_clothing_advice(temperature: float) -> dict:
    """Suggest what to wear for a temperature. Call this after get_weather.

    Args:
        temperature: The temperature in Celsius, from get_weather.
    """
    if temperature < 5:
        clothing = "a heavy coat, hat and gloves"
    elif temperature < 15:
        clothing = "a warm jacket"
    elif temperature < 25:
        clothing = "a light jacket or long sleeves"
    else:
        clothing = "light clothing and sunscreen"

    logger.info("%s C -> %s", temperature, clothing)
    return {"clothing": clothing}


@function_tool
async def get_activity_suggestion(temperature: float, clothing: str) -> dict:
    """Suggest an activity. Call this last, using both earlier results.

    Args:
        temperature: The temperature in Celsius, from get_weather.
        clothing: The advice from get_clothing_advice.
    """
    if temperature < 5:
        activity = "a museum or a long lunch indoors"
    elif temperature < 15:
        activity = "a walk through the old town"
    elif temperature < 25:
        activity = "a park, a market, or anything outdoors"
    else:
        activity = "somewhere shaded, or a swim"

    logger.info("suggesting %s", activity)
    return {"activity": activity, "wearing": clothing}


class ToolChainingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful travel assistant. When a user asks what to do "
                "in a city:\n"
                "1. FIRST call get_weather to get the temperature\n"
                "2. THEN call get_clothing_advice with that temperature\n"
                "3. THEN call get_activity_suggestion with the temperature AND "
                "clothing advice\n"
                "4. Finally, combine all three results into a natural spoken "
                "response.\n\n"
                "You MUST call all three tools in sequence -- do NOT skip any "
                "step. Keep your final response concise and conversational "
                "(2-3 sentences max)."
            ),
            agent_id=AGENT_ID,
            tools=[get_weather, get_clothing_advice, get_activity_suggestion],
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=GoogleLLM(),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi! I'm your travel assistant. Ask me what to do in any city and "
            "I'll check the weather, suggest what to wear, and recommend an "
            "activity."
        )

    async def on_exit(self) -> None:
        logger.info("call finished")


def on_ready() -> None:
    result = zrt.invoke(AGENT_ID, room=Room(
        name="Tool Chaining", playground=True))


if __name__ == "__main__":
    zrt.serve(ToolChainingAgent, on_ready=on_ready)
