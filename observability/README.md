# Room and observability

`agent_events.py`, `pipeline_events.py`, `voice_pipeline_hooks.py` and
`realtime_pipeline_hooks.py` need nothing beyond the root setup — they log to
your terminal.

## observability_hooks.py

This one exports to OpenTelemetry, and wants a collector to export to:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

Any OTLP/HTTP receiver works — a local collector, Jaeger, Grafana, your
existing vendor.

Leaving it unset is fine and is what the example does by default: traces and
metrics stay enabled and go to the platform's own backend, and logs switch off.
Logs are the noisy one, so they turn on only when you have named somewhere to
put them.

The same example also turns on platform recording and fetches the conversation
history in `on_exit`. Neither needs configuration here.

## agent_events.py

The other direction from the hooks. Everything else in this folder carries data
back to *your* Python process; `Room(agent_events=...)` publishes into the
*room*, for whoever has joined it — a browser client, the playground, a mobile
SDK. Nothing it sends reaches your process.

Three switches:

| | default | what it publishes |
|---|---|---|
| `turn_metrics` | on | one payload per turn, once every stage has finished |
| `component_metrics` | off | each stage's metrics as that stage finishes |
| `send_interim` | off | interim transcripts as well as final ones |

`component_metrics` is additive — the per-turn payload still arrives, so a
client that only reads turn metrics is unaffected. It is off by default because
it multiplies the messages sent per turn.

`send_interim` is a gate rather than a command: it lets interim transcripts
through, but each provider still decides whether it emits any —
`DeepgramSTT(forward_interim_transcripts=...)` for the caller's speech,
`CartesiaTTS(word_timestamps=True)` for the agent's. Both layers have to agree,
which is what lets you stream the caller's words live while the agent's own
speech stays settled, or the reverse. Final transcripts are always sent.

None of it reaches a participant who joined without `enableAgentEvents`.
