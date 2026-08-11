# Room and observability

`pipeline_events.py`, `voice_pipeline_hooks.py` and
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
