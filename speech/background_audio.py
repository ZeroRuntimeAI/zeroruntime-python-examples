# Ambience under the call: set on the agent to play from the start, or started
# and stopped mid-call from the session. The file is resolved in the runtime,
# and mixing mode keeps the track off the speech path.

import os

import zrt
from zrt import Agent, BackgroundAudio, Pipeline, Room, function_tool
from zrt.inference import TurnDetector
from zrt.plugins import DeepgramSTT, OpenAILLM, OpenAITTS, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)



AGENT_ID = os.getenv("AGENT_ID", "background-audio-agent")

MUSIC = os.getenv("BACKGROUND_MUSIC", "")


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant that can answer questions and "
                "help with tasks. If the user asks to play music, use the "
                "control_background_music tool with action 'play'. To stop, use "
                "the action 'stop'."
            ),
            agent_id=AGENT_ID,
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=OpenAILLM(),
                tts=OpenAITTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
            background_audio=BackgroundAudio(
                file_path="https://cdn.zeroruntime.ai/zrt/bg-audio/bg-noise-1.ogg",
                volume=0.9,
                looping=True,
                mode="mixing",
            ),
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello, how can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")

    @function_tool
    async def control_background_music(self, action: str) -> str:
        """Control the background music.

        Args:
            action: 'play' to start the music, 'stop' to end it.
        """
        if action.lower() == "play":
            await self.session.play_background_audio(
                MUSIC or None,
                volume=0.8,
                looping=True,
                override_thinking=False,
            )
            return "Background music started."

        if action.lower() == "stop":
            await self.session.stop_background_audio()
            return "Background music stopped."

        return "Invalid action. Please use 'play' or 'stop'."


def on_ready() -> None:
    zrt.invoke(AGENT_ID, room=Room(
        name="Background Audio", playground=True))


if __name__ == "__main__":
    zrt.serve(VoiceAgent, on_ready=on_ready)
