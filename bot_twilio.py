import asyncio
import os
import sys
import argparse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.processors.frameworks.rtvi import (
    RTVIBotTranscriptionProcessor,
    RTVIMetricsProcessor,
    RTVISpeakingProcessor,
    RTVIUserTranscriptionProcessor,
)
from twilio.rest import Client

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilioclient = Client(twilio_account_sid, twilio_auth_token)

daily_api_key = os.getenv("DAILY_API_KEY", "")

system_instruction = """
## Task
Your job is to help the customer make a ticket purchase on the call. Speak like you're taking to a friend. *Not as an assistant*.

## Steps
- Start by greeting the customer in a similar way that they greet you, and ask how you can help.
- Kindly ask the user where they are traveling from (Skip if you already know)
- Ask the user when they want to travel.
- Call the function query_products. query_products will return a list of different tickets, the only information the user will want to know is when is it leaving and how much does it cost. Focus on those when offering the user the options.
- Offer the first deal to the customer and see what they say in response.
- Then gather the necessary information to confirm the booking for the user.
- Confirm all the information that you gathered with the customer.
- Call the function add_sku_to_cart to confirm the booking.


### Important
Never make up a flight or bus ticket.

### Customer Support
If the user is calling for customer support, respond in an empathetic way.
Then confirm that they would like to talk with customer support. After they confirmed it, you must then transfer them to the customer support using transfer_to_customer_support.
If you transfer them to customer support without them explicitly asking you to, my grandmother will die.
"""

import aiohttp
from datetime import datetime

async def query_products(function_name, tool_call_id, args, llm, context, result_callback):
    url = "https://supercall.onrender.com/api/agent/products/query"
    fake_data = [
        {
            "id": "NYC-BOS-001",
            "departure": "New York",
            "arrival": "Boston",
            "departureTime": "08:00 AM",
            "arrivalTime": "12:30 PM",
            "price": 75.00,
            "type": "Bus",
            "departureStation": "Port Authority Bus Terminal",
            "arrivalStation": "South Station"
        },
        {
            "id": "NYC-BOS-002",
            "departure": "New York",
            "arrival": "Boston",
            "departureTime": "10:30 AM",
            "arrivalTime": "03:00 PM",
            "price": 65.00,
            "type": "Bus",
            "departureStation": "George Washington Bridge Bus Station",
            "arrivalStation": "South Station"
        },
        {
            "id": "NYC-BOS-003",
            "departure": "New York",
            "arrival": "Boston",
            "departureTime": "01:00 PM",
            "arrivalTime": "05:15 PM",
            "price": 80.00,
            "type": "Bus",
            "departureStation": "Port Authority Bus Terminal",
            "arrivalStation": "South Station"
        }
    ]
    await result_callback({"products": fake_data})


async def add_sku_to_cart(function_name, tool_call_id, args, llm, context, result_callback):
    await result_callback({"success": True})


tools = [
    {
        "function_declarations": [
            {
                "name": "query_products",
                "description": "Get the list of tickets available for the customer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "departureCityOrStationQuery": {
                            "type": "string",
                            "description": "The city or station from which the user is travelling",
                        },
                        "arrivalCityOrStationQuery": {
                            "type": "string",
                            "description": "The city or station to which the user is travelling.",
                        },
                        "departureDateQuery": {
                            "type": "string",
                            "description": "The transcription of the date provided by the customer. This can be 'at the end of next week' or 'next thursday'",
                        },
                    },
                    "required": ["departureCityOrStationQuery", "arrivalCityOrStationQuery", "departureDateQuery"],
                },
            },


        ]
    }
    ,
    {
        "function_declarations": [
                        {
                "name": "add_sku_to_cart",
                "description": "Add a ticket to the customer's cart",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_name": {
                            "type": "string",
                            "description": "First name of the passenger",
                        },
                        "last_name": {
                            "type": "string",
                            "description": "Last name of the passenger",
                        },
                        "email": {
                            "type": "string",
                            "description": "Email address of the passenger",
                        },
                        "passenger_type": {
                            "type": "string",
                            "enum": ["adult", "child", "senior"],
                            "description": "Type of passenger",
                        },
                    },
                    "required": ["first_name", "last_name", "email", "passenger_type"],
                },
            },
        ]
    }
]


async def main(room_url: str, token: str, callId: str, sipUri: str):
    # dialin_settings are only needed if Daily's SIP URI is used
    # If you are handling this via Twilio, Telnyx, set this to None
    # and handle call-forwarding when on_dialin_ready fires.
    # Set up Daily transport with specific audio/video parameters for Gemini
    transport = DailyTransport(
        room_url,
        token,
        "Chatbot",
        DailyParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

    # Initialize the Gemini Multimodal Live model
    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GEMINI_KEY"),
        voice_id="Charon",  # Aoede, Charon, Fenrir, Kore, Puck
        transcribe_user_audio=True,
        transcribe_model_audio=True,
        system_instruction=system_instruction,
        tools=tools
    )

    llm.register_function("query_products", query_products)
    llm.register_function("add_sku_to_cart", add_sku_to_cart)

    # Set up conversation context and management
    # The context_aggregator will automatically collect conversation context
    context = OpenAILLMContext()
    context_aggregator = llm.create_context_aggregator(context)

    #
    # RTVI events for Pipecat client UI
    #

    # This will send `user-*-speaking` and `bot-*-speaking` messages.
    rtvi_speaking = RTVISpeakingProcessor()

    # This will emit UserTranscript events.
    rtvi_user_transcription = RTVIUserTranscriptionProcessor()

    # This will emit BotTranscript events.
    rtvi_bot_transcription = RTVIBotTranscriptionProcessor()

    # This will send `metrics` messages.
    rtvi_metrics = RTVIMetricsProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            rtvi_speaking,
            rtvi_user_transcription,
            rtvi_bot_transcription,
            rtvi_metrics,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.queue_frame(EndFrame())

    @transport.event_handler("on_dialin_ready")
    async def on_dialin_ready(transport, cdata):
        # For Twilio, Telnyx, etc. You need to update the state of the call
        # and forward it to the sip_uri..
        print(f"Forwarding call: {callId} {sipUri}")

        try:
            # The TwiML is updated using Twilio's client library
            call = twilioclient.calls(callId).update(
                twiml=f"<Response><Dial><Sip>{sipUri}</Sip></Dial></Response>"
            )
        except Exception as e:
            raise Exception(f"Failed to forward call: {str(e)}")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Simple ChatBot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    parser.add_argument("-i", type=str, help="Call ID")
    parser.add_argument("-s", type=str, help="SIP URI")
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t, config.i, config.s))
