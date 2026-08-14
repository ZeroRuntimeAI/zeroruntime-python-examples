# Two sounds under the call, both on the room's mixing track.
#
#   thinking audio    plays while the LLM is generating, automatically, every turn
#   background audio  plays when you ask for it, and keeps playing until you stop
#
# Both need Room(background_audio=True). That flag is the track itself, not the
# sound: without it the runtime has nowhere to put audio that is not speech, and
# both calls below are declined. Neither plays by itself -- the Room opens the
# track, these two put something on it.

import os

import zeroruntime
from zeroruntime import Agent, Pipeline, Room, function_tool
from zeroruntime.inference import TurnDetector,GoogleLLM,SarvamAITTS
from zeroruntime.plugins import DeepgramSTT, SileroVAD

from dotenv import load_dotenv
load_dotenv(override=True)


AGENT_ID = os.getenv("AGENT_ID", "background-audio-agent")

#: Any file libav can decode -- wav, mp3, ogg, flac, m4a -- fetched by the
#: runtime, so a URL it can reach rather than a path on this machine. Leave
#: either unset to take the runtime's own default sound.
MUSIC = os.getenv("BACKGROUND_MUSIC", "")
THINKING = os.getenv("THINKING_AUDIO", "")


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
                llm=GoogleLLM(),
                tts=SarvamAITTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector(),
            ),
        )

    async def on_enter(self) -> None:
        await self.session.set_thinking_audio(THINKING or None, volume=0.3)
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
                # False makes the music exclusive: the thinking sound is held
                # back while it plays, rather than the two layering. True lets
                # them overlap.
                override_thinking=False,
            )
            return "Background music started."

        if action.lower() == "stop":
            await self.session.stop_background_audio()
            return "Background music stopped."

        return "Invalid action. Please use 'play' or 'stop'."


def on_ready() -> None:
    zeroruntime.invoke(
        AGENT_ID,
        room=Room(name="Background Audio", playground=True),
    )


if __name__ == "__main__":
    zeroruntime.serve(VoiceAgent, on_ready=on_ready, room=Room(background_audio=True))
