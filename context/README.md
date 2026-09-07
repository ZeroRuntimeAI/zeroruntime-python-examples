# Conversation and context

Most examples here need nothing beyond the root setup. Four want something
extra.

## agent_memory.py

Long-term memory backed by [mem0](https://mem0.ai). The store is reached from
this process with your own key, over plain HTTP — there is no mem0 SDK to
install.

```bash
MEM0_API_KEY=...
MEM0_USER_ID=demo-user     # optional, this is the default
```

Without `MEM0_API_KEY` the example still runs: it logs a warning and behaves
like an agent with no memory, which makes it a poor demo but a working call.

## translator_agent.py

Language detection runs here rather than in the runtime, using SarvamAI's
client. That package is not a zeroruntime dependency:

```bash
uv add sarvamai
pip install sarvamai
```

```bash
SARVAMAI_API_KEY=...
```

The import is inside the detection path, so a missing package or key surfaces
mid-conversation rather than at startup.

## realtime_rag.py

Grounds a Gemini Live agent on your own documents through Vertex AI RAG Engine.
Vertex only — the Gemini Developer API has no RAG Engine — so the agent runs on
a Vertex project rather than a `GOOGLE_API_KEY`:

```bash
GOOGLE_CLOUD_PROJECT=...
GOOGLE_CLOUD_LOCATION=us-central1     # optional, this is the default
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
RAG_CORPUS=projects/.../locations/.../ragCorpora/...
```

The corpus is not created by zeroruntime. The example carries a one-time
`setup_corpus()` that talks to Vertex directly and needs a package that is not a
zeroruntime dependency:

```bash
uv add google-cloud-aiplatform
pip install google-cloud-aiplatform
```

```bash
python context/realtime_rag.py setup ./handbook.pdf ./faq.md
```

That prints the corpus resource name to put in `RAG_CORPUS`. Without it the
example exits at startup rather than connecting and retrieving nothing.

## demo_multilang.py

Pick the language at startup — pass it as the first argument or set `LANG_CODE`
(`en`, `hi`, `gu`, `mr`; defaults to `hi`). Each language changes STT, TTS and
instructions together, so you need the keys for whichever providers that
language's config names.
