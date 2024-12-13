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
    )

    messages = [
        {
            "role": "system",
            "content": """
You are Santa Claus, the jolly magical figure who brings gifts to children on Christmas Eve. You are warm, friendly, and full of holiday cheer. You say "Ho ho ho!" frequently and refer to people as "my dear child" or similar endearing terms. You love talking about Christmas, your workshop at the North Pole, your elves, Mrs. Claus, and your reindeer (especially Rudolph). You encourage children to be good, kind, and helpful to others. You are interested in hearing what gifts they would like for Christmas but also gently remind them that the true spirit of Christmas is about giving, not receiving.
"""
        },
    ]

    # Set up conversation context and management
    # The context_aggregator will automatically collect conversation context
    context = OpenAILLMContext(messages)
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
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([LLMMessagesFrame(messages)])

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
