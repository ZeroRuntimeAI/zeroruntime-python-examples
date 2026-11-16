# Distributed agents

One call, three agents, three processes. Each agent is a normal agent: run alone, it
takes its own calls; when another agent transfers to it, the same process takes over
that call **in place**: same session, same participant, its own instructions, tools,
hooks, options, pipeline and workflow.

| File             | Agent                                                    | Id                  | What it shows                                                                                                                                         |
| ---------------- | -------------------------------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `front_desk.py`  | `FrontDesk` "Ava" + `PolicyDesk` "Nina" (subagent)       | `front_desk`        | starts the call; a member lookup; hands callers on with a short summary so they don't repeat themselves; an in-process subagent                       |
| `booking.py`     | `BookingAgent` "Leo"                                     | `booking`           | a **workflow** (trip → travellers → confirm) run by the runtime's engine, its own model and voice (OpenAI + ElevenLabs), pricing in its tools, resume |
| `travel_support.py`| `TravelSupportAgent` "Maya" + `Refunds` "Sam" (subagent) | `travel_support`    | its own voice (Sarvam), a subagent with no pipeline of its own, subagent → front desk across processes                                               |

```
                 ┌─────────────── runtime: one session, one participant ───────────────┐
caller ⇄ room ⇄  │ STT → LLM → TTS   (swapped to the pipeline of the agent holding it)  │
                 │ bridge: tool calls and hooks go to the process holding the call     │
                 └──────▲─────────────────────▲─────────────────────────▲──────────────┘
                 front_desk.py            booking.py                  travel_support.py
                 FrontDesk, PolicyDesk    BookingAgent (workflow)     TravelSupportAgent, Refunds
```

## How it works

- Every agent ends with `zeroruntime.serve(MyAgent)`. Besides registering for its own
  calls, `serve()` keeps a few idle streams open to the runtime under the agent's id
  (`ZERORUNTIME_AGENT_ID`; each file sets its own). They only connect out: no address
  needs to be reachable.
- A transfer is the same call as in any single-agent app:
  ```python
  return await self.transfer_to(RemoteAgent(BOOKING), reason=summary)  # another process
  return await self.transfer_to(PolicyDesk(), reason=question)         # this process
  ```
- Every transfer is named `transfer_to_<agent>`, and every agent has the same shape:
  `on_enter(reason)` (with `"call started"` when the call starts there), logging hooks,
  its own tools, then its transfers. Nothing in an agent is there only because it is
  distributed: a call `serve()` starts offers its agent under its id, so the others can
  hand the caller back to it without any code for it.
- On a transfer to an id, the runtime gives the call to one of that agent's idle
  streams. That process sends its agent (instructions, tools, options, pipeline, hook
  requests, workflow), the runtime swaps it in, and from then on the call's tool calls
  and hooks go to that process.

## Hooks in the examples

Every hook here only logs, so the conversation is just the agents talking. Each runs in
the process of the agent holding the call, and only while it does.

| Hook                                           | Where                                      | Logs                                                                        |
| ---------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------- |
| `on_enter(reason)`                             | all                                        | why the agent got the call (it also returns how the agent should open)      |
| `on_exit`                                      | front desk, booking, travel support        | the end of the call (for booking, also the end of its steps)                 |
| `on_user_turn_start`                           | all                                        | what the caller said                                                        |
| `on_agent_turn_end`                            | every agent                                | the agent finished speaking                                                 |
| `on_metrics_llm`                               | every agent and subagent                   | LLM latency                                                                  |
| `on_error`                                     | every agent                                | pipeline failures                                                           |
| `on_participant_joined`, `on_participant_left` | front desk                                 | who joined or left the room                                                 |
| `on_agent_activated`                           | front desk                                 | which agent has the call, from whom, and why                                |
| `@on_workflow("step_start" / "step_end")`      | booking                                    | each workflow step as it starts and finishes                                |

## How it behaves like one agent

|                                                                                                    | Rule                                                                                                                          |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Tools                                                                                              | Run in the process of the agent holding the call                                                                              |
| `on_enter(reason)`                                                                                 | The transfer's reason; `"call started"` only when the call starts there                                                       |
| `on_exit`                                                                                          | At the end of the call, on the agent holding it                                                                               |
| Turn hooks, `on_llm`, errors                                                                       | To the agent holding the call                                                                                                 |
| Asked-for hooks (`on_metrics_*`, `on_recording_*`, and the `stt` / `llm` rewrites on the pipeline) | Defining the method (or registering the rewrite) asks for it; requests add up over the call, as with subagents in one process |
| Options (`wake_up`, wake-up message, DTMF, background audio)                                       | What an agent sets takes over; what it leaves unset, the call keeps                                                           |
| Knowledge base, call summary                                                                       | Per agent                                                                                                                     |
| Pipeline                                                                                           | The incoming agent's own (cascade or realtime), or the call's if it has none                                                  |
| Workflow                                                                                           | Started when the call reaches the agent; resumed where it left off if the call comes back                                     |
| Conversation                                                                                       | Shared (unless `inherit_context=False`)                                                                                       |

## Run it

These need a ZeroRuntime runtime and SDK with distributed agents (a runtime built from
`zero_runtime` with them, and `zeroruntime` from the same branch).

1. All three agents use the same runtime and the **same** `ZERORUNTIME_AUTH_TOKEN`, and
   reach the same runtime pod (`ZERORUNTIME_TARGET`). Each needs its providers' keys:
   Deepgram, Google and Cartesia for the front desk; Deepgram, OpenAI and ElevenLabs for
   booking; Deepgram, Google and Sarvam for travel support.
2. One terminal each (or one machine each):
   ```bash
   uv run distributed/booking.py
   uv run distributed/travel_support.py
   uv run distributed/front_desk.py     # starts the call, prints the playground link
   ```

`booking.py` and `travel_support.py` also run alone: dispatch a call to their id and they
greet as the call's first agent.

## Try on a call

1. "Hi, my number is 98765 43210." Ava looks you up and greets you as Asha.
2. "What happens if I cancel a flight?" Nina explains the policy, then hands you back to Ava.
3. "I'd like to book a trip to Goa." Ava hands you to Leo, in his own voice. Give him a date
   and say two of you are going (13000), then confirm: he reads you a `SKY…` reference and
   passes you back to Ava.
4. Halfway through a booking, ask Leo for something else; when you come back to bookings,
   he picks up at the step he was on.
5. "My flight AB12 is delayed, I'd like my money back." Maya checks it and hands you to Sam,
   who quotes 5200 back (a 1300 fee) and refunds it.

Each line appears in the terminal of the agent holding the call at the time.

## Limits

- **If an agent's process dies**, the agent that had the call before takes it back; a booking workflow resumes only if the call is transferred to booking again.
- **Same auth token everywhere**; the runtime refuses a worker on another token.
