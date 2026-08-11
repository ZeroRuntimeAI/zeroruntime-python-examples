# zeroruntime examples

Runnable examples for the [zeroruntime](https://github.com/ZeroRuntimeAI/zeroruntime-py)
Python SDK.

Every file here is standalone. One file is one idea — a pipeline shape, a tool
pattern, a piece of call control — and each opens with a comment explaining
what it demonstrates and which detail is the point. Read the top of a file
before running it; that comment is the documentation.

## Setup

```bash
git clone https://github.com/ZeroRuntimeAI/zeroruntime-python-examples
cd zeroruntime-python-examples
```

### With uv

```bash
uv sync
```

That creates `.venv` and installs everything. Run an example without
activating anything:

```bash
uv run cascade_basic.py
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Either way you get one environment that runs every example in this repo.

## Credentials

Every example calls `load_dotenv()`, so put your keys in a `.env` file next to
the example you are running:

```bash
ZERORUNTIME_AUTH_TOKEN=...
```

Then add a key per vendor the example's pipeline names. The full set across
this repo, though no single example needs all of them:

```bash
DEEPGRAM_API_KEY=...      # DeepgramSTT
CARTESIA_API_KEY=...      # CartesiaTTS
ELEVENLABS_API_KEY=...    # ElevenLabsTTS
GOOGLE_API_KEY=...        # GoogleLLM, GoogleTTS, GeminiRealtime
OPENAI_API_KEY=...        # OpenAILLM, OpenAITTS
ANTHROPIC_API_KEY=...     # AnthropicLLM
SARVAMAI_API_KEY=...      # SarvamAISTT, SarvamAITTS
SIMLI_API_KEY=...         # SimliAvatar
ANAM_API_KEY=...          # AnamAvatar
```

Providers imported from `zrt.inference` reach the gateway instead of the vendor,
so they need `ZERORUNTIME_AUTH_TOKEN` and no vendor key at all — see
`videosdk_cascade_inference_agent.py`. `SileroVAD` runs locally and needs
nothing.

## Running

```bash
uv run cascade_basic.py       # uv
python cascade_basic.py       # pip, with .venv activated
```

The agent serves, joins a room, and prints a playground URL once on stdout.
Open it and talk to the agent. Ctrl-C to stop.

## The examples

### Start here

| File | What it shows |
| --- | --- |
| `cascade_basic.py` | The smallest complete agent: STT, LLM, TTS, VAD, turn detector |
| `realtime_basic.py` | The same call with one speech-to-speech model doing all of it |
| `cascade_advanced.py` | Tuning end-of-utterance and barge-in on a cascade pipeline |

### Pipeline shapes

| File | What it shows |
| --- | --- |
| `composable_pipelines/agent_multimodal.py` | Voice in, voice out — the map for the other three |
| `composable_pipelines/agent_llm.py` | Text in, text out |
| `composable_pipelines/agent_text_to_voice.py` | Text in, voice out |
| `composable_pipelines/agent_voice_to_text.py` | Voice in, text out |
| `hybrid_mode/hybrid_custom_stt_realtime.py` | Your transcriber in front of a realtime model |
| `hybrid_mode/hybrid_realtime_custom_tts.py` | A realtime model with your voice on the output |
| `fallback_recovery.py` | A pipeline slot as a list — head serves, tail stands by |
| `videosdk_cascade_inference_agent.py` | The same cascade through the gateway, one credential |
| `videosdk_realtime_inference_agent.py` | A realtime model through the gateway, no vendor key |

### Tools

| File | What it shows |
| --- | --- |
| `cascade_tool_chaining.py` | Three tools called in sequence, each fed by the last |
| `mcp_example.py` | Tools from an MCP server, connected in your process |
| `mcp_servers/current_time.py` | A small stdio MCP server for the above to talk to |
| `n8n_workflow/appointment_telephony.py` | An n8n workflow as the toolset, over HTTP |
| `human_in_the_loop/customer_agent.py` | A tool that waits on a person before answering |
| `human_in_the_loop/discord_mcp_server.py` | The Discord server that blocks behind it |

### Conversation and context

| File | What it shows |
| --- | --- |
| `agent_context_window.py` | Bounding a long call — summarise or truncate older turns |
| `agent_memory.py` | Long-term memory across calls, searched and written per turn |
| `context_management/agent_sequential_handoff.py` | A tool that returns an Agent is the handoff |
| `context_management/cascade_to_realtime_handoff.py` | Swapping a live call onto a realtime model |
| `context_management/realtime_to_cascade_handoff.py` | And back again |
| `multi_agent_switch.py` | One caller, three agents, handoff in either direction |
| `persona_switch.py` | Five personas rebuilt live from a chat message |
| `translator_agent.py` | Detect the caller's language mid-call and follow it |
| `demo_multilang.py` | One agent in four languages, picked at startup |

### Speech control

| File | What it shows |
| --- | --- |
| `utterance_handle_agent.py` | Awaiting an utterance, and tools that notice interruption |
| `reply_interrupt_agent.py` | say, reply and process_text — three different things |
| `enhanced_pronounciation.py` | Substitution rules applied between LLM and TTS |
| `cached_tts.py` | Fixed phrases synthesised once and replayed as PCM |
| `background_audio.py` | Ambience under the call, from the start or mid-call |
| `wakeup_call.py` | Nudging a caller who has gone quiet |

### Telephony

| File | What it shows |
| --- | --- |
| `call_transfer.py` | Moving the caller to another number |
| `dtmf_voicemail.py` | Keypad input and answering-machine detection |
| `agent_hangup.py` | The agent ending the call itself |

### Room and observability

| File | What it shows |
| --- | --- |
| `pubsub_example.py` | Publishing from a tool, reacting to a subscription |
| `pipeline_events.py` | Component errors, recording state, latency metrics |
| `voice_pipeline_hooks.py` | The turn lifecycle of a cascade pipeline |
| `realtime_pipeline_hooks.py` | The same lifecycle on a realtime call |
| `observability_hooks.py` | OpenTelemetry, recording, and history on exit |

### Video

| File | What it shows |
| --- | --- |
| `vision/vision_cascade.py` | Showing the model what the camera sees |
| `vision/vision_realtime.py` | The same, on a speech-to-speech pipeline |
| `avatar_agent.py` | Giving a cascade agent a face |
| `avatar_anam_realtime.py` | An avatar on a realtime pipeline |
