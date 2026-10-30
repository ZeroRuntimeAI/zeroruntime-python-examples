# Sunrise Clinic's appointment line: a workflow of two subagents.
#
#   Intake       name -> reason -> referral (specialists only) -> insurance (optional)
#   Scheduling   slot -> confirm
#
# Each step's answer is a dataclass. A tool checks what the caller said and calls
# self.complete(Answer(...)); the runtime then moves to the next step, and on to
# the next subagent once on_exit says so.
#
#   python clinic_showcase.py

import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv

import zeroruntime
from zeroruntime import Agent, Pipeline, Room, Step, SubAgent, function_tool
from zeroruntime.inference import TurnDetector
from zeroruntime.plugins import CartesiaTTS, DeepgramSTT, GoogleLLM, SileroVAD
from zeroruntime.workflow import on_workflow, workflow_status

load_dotenv(override=True)

VISITS = ("checkup", "illness", "specialist")
SPECIALTIES = ("cardiology", "dermatology", "orthopaedics")
SLOTS = {"S1": "Monday 10:00", "S2": "Tuesday 14:30", "S3": "Thursday 09:15"}


# Answers: one dataclass per step

@dataclass
class Patient:
    full_name: str


@dataclass
class Visit:
    kind: str
    details: str


@dataclass
class Referral:
    specialty: str
    doctor: str


@dataclass
class Insurance:
    provider: str
    member_id: str


@dataclass
class Booking:
    slot_id: str
    when: str
    doctor: str


@dataclass
class Confirmation:
    reference: str


# Intake: who's calling and why

class Intake(SubAgent):
    name = "intake"
    instructions = "You're the clinic's front desk. Ask one thing at a time, and keep it short and kind."
    workflow = [
        Step(
            id="name",
            description="who is calling",
            instructions="Get the patient's full name, then call record_name.",
            opening="Ask for the patient's full name.",
            result=Patient,
        ),
        Step(
            id="reason",
            description="why they're coming in",
            instructions="Find out if it's a checkup, an illness or a specialist visit, and a few words "
                         "about it, then call record_reason.",
            opening="Ask what they'd like to come in for.",
            result=Visit,
            max_attempts=2,  # two wrong answers, and on_step_failed runs
        ),
        Step(
            id="referral",
            description="referral",
            instructions="Get the specialty and the referring doctor's name, then call record_referral.",
            opening="Ask which specialist they need, and who referred them.",
            result=Referral,
            condition={"field": "reason.kind", "in": ["specialist"]},  # only for a specialist visit
        ),
        Step(
            id="insurance",
            description="insurance",
            instructions="Ask for their insurer and member id, then call record_insurance.",
            opening="Ask if they'll be using insurance.",
            result=Insurance,
            optional=True,  # if they decline, the model calls the built-in skip_step
        ),
    ]

    async def on_enter(self, reason: str):
        if reason == "call started":
            name = self.userdata.get("caller_name", "the caller")  # from invoke(metadata=...)
            return f"Greet {name} as Sunrise Clinic, and say you'll get them booked in."
        if reason == "caller changed the visit":  # sent back from scheduling by change_visit
            return "Say of course. If they've already said what they'd like instead, record it; otherwise ask."
        return None

    # A parameter with no default is asked for until the caller gives it.

    @function_tool
    async def record_name(self, full_name: str) -> dict:
        """Record the patient's name.

        Args:
            full_name: The patient's full name.
        """
        return await self.complete(Patient(full_name.strip()))

    @function_tool
    async def record_reason(self, kind: str, details: str) -> dict:
        """Record why they're coming in.

        Args:
            kind: checkup, illness or specialist.
            details: A few words about it.
        """
        if kind.lower() not in VISITS:
            return {"error": f"kind is one of {', '.join(VISITS)}."}  # the model asks again
        return await self.complete(Visit(kind.lower(), details))

    @function_tool
    async def record_referral(self, specialty: str, doctor: str) -> dict:
        """Record the specialist they need and the referring doctor.

        Args:
            specialty: cardiology, dermatology or orthopaedics.
            doctor: The referring doctor's name.
        """
        if specialty.lower() not in SPECIALTIES:
            return {"error": f"We see {', '.join(SPECIALTIES)}."}
        return await self.complete(Referral(specialty.lower(), doctor))

    @function_tool
    async def record_insurance(self, provider: str, member_id: str) -> dict:
        """Record their insurance.

        Args:
            provider: The insurer.
            member_id: Their member id.
        """
        return await self.complete(Insurance(provider, member_id))

    @function_tool
    async def fix_answer(self, step_id: str) -> dict:
        """Go back to change one earlier answer; the others are kept.

        Args:
            step_id: name, reason, referral or insurance.
        """
        return await self.go_to_step(step_id, reason="caller correction")

    async def on_exit(self):
        # All steps done: on to scheduling.
        return await self.transfer_to("scheduling", reason="intake complete")

    async def on_step_failed(self, step_id: str, reason: str):
        # max_attempts reached: book a general checkup instead.
        return await self.transfer_to("scheduling", reason="general checkup")


# Scheduling: pick a time and confirm it

class Scheduling(SubAgent):
    name = "scheduling"
    instructions = "You book the appointment. Offer the slots find_slots returns, never others."
    workflow = [
        Step(
            id="slot",
            description="pick a time",
            instructions="Find slots, offer them, then call book_slot with their choice.",
            opening="Say you'll look at the diary, then call find_slots.",
            result=Booking,
        ),
        Step(
            id="confirm",
            description="confirm",
            instructions="Read the booking back and get a clear yes, then call confirm_booking.",
            opening="Read the booking back and ask if that's right.",
            result=Confirmation,
        ),
    ]

    def doctor(self) -> str:
        referral = self.workflow.intake.referral
        return referral.specialty if referral else "GP"

    async def on_enter(self, reason: str):
        if reason == "general checkup":
            return "Say you'll book a general checkup, since you couldn't take all the details."
        patient = self.workflow.intake.name
        return f"Thank {patient.full_name}, and say you'll find a time."

    @function_tool
    async def find_slots(self) -> dict:
        """Look up open slots for their visit. Takes a moment."""
        await self.session.say("One moment while I check the diary.", add_to_chat_context=False)
        return {"doctor": self.doctor(), "slots": SLOTS}

    @function_tool
    async def book_slot(self, slot_id: str) -> dict:
        """Hold the slot they chose.

        Args:
            slot_id: S1, S2 or S3.
        """
        if slot_id not in SLOTS:
            return {"error": f"slot_id is one of {', '.join(SLOTS)}."}
        return await self.complete(Booking(slot_id, SLOTS[slot_id], self.doctor()))

    @function_tool
    async def confirm_booking(self, confirmed: bool) -> dict:
        """The caller's answer to the booking read back.

        Args:
            confirmed: True if they said yes.
        """
        if not confirmed:
            return await self.go_to_step("slot", reason="caller wants another time")
        booking = self.workflow.slot
        return await self.complete(Confirmation(f"SUN-{booking.slot_id}"))

    @function_tool
    async def change_visit(self) -> dict:
        """The caller wants to change why they're coming in."""
        # A step in another subagent: the caller is handed back to intake for it.
        return await self.go_to_step("intake.reason", reason="caller changed the visit")

    @on_workflow("step_end")
    def track(self, data: dict) -> None:
        self.log.info("%s.%s done; to do: %s", data["agent"], data["step"], data["progress"]["to_do"])

    async def on_exit(self):
        return "Thanks for calling Sunrise Clinic. See you soon."


EVENTS = ("agent_handoff", "step_start", "step_end", "step_back", "step_failed")


class Clinic(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You answer the phone at Sunrise Clinic. This is a phone call: a sentence or two at a time. "
                         "Once the caller has nothing else, call end_call.",
            pipeline=Pipeline(
                stt=DeepgramSTT(),
                llm=GoogleLLM(),
                tts=CartesiaTTS(),
                vad=SileroVAD(),
                turn_detector=TurnDetector()),
            tools=[workflow_status],  # with Clinic's own methods, every subagent has these
            subagents=[Intake(), Scheduling()],  # the first one opens the call
        )
        for event in EVENTS:
            self.on(event, print)

    @function_tool
    async def end_call(self) -> dict:
        """End the call when the user asks to hang up or says goodbye."""
        asyncio.create_task(self._announce_and_hangup())
        return {"status": "ending_call"}

    async def _announce_and_hangup(self) -> None:
        await self.session.interrupt()
        await asyncio.sleep(1)
        handle = await self.session.say("Have a nice day!", interruptible=False)
        await handle
        await self.hangup(farewell="Goodbye!")


def on_ready() -> None:
    zeroruntime.invoke(room=Room(name="Sunrise Clinic", playground=True), metadata={"caller_name": "Ravi"})


if __name__ == "__main__":
    zeroruntime.serve(Clinic, on_ready=on_ready)
