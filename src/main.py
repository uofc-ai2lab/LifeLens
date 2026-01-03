"""
    Main file to run microservices found in their respective folders using the command line.
"""

import argparse

async def main():
    parser = argparse.ArgumentParser(description="Run microservices")
    parser.add_argument(
        "service",
        type=str,
        choices=["whisperx", "faster_whisper", "transcribe", "nlp", "meds"],  # Add other services as needed
    )
    args = parser.parse_args()

    if args.service == "whisperx":
        from whisperX.whisper import run_whisperx
        await run_whisperx()
    elif args.service == "faster_whisper":
        from Faster_Whisper.transcribe_faster import run_faster_whisper
        await run_faster_whisper()
    elif args.service == "transcribe":
        from src.services.audio_to_transcript_service.audio_to_transcript_whispertrt import run_transcription
        await run_transcription()
    elif args.service == "intervention":
        from src.services.transcript_to_meaning_service.intervention_extraction import run_nlp
        await run_nlp()
    elif args.service == "meds":
        from src.services.transcript_to_meaning_service.medication_extraction import medication_extraction_service
        await medication_extraction_service()
    else:
        print(f"Service {args.service} not recognized.")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())