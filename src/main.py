"""
    Main file to run microservices found in their respective folders using the command line.
"""

import argparse

async def main():
    parser = argparse.ArgumentParser(description="Run microservices")
    parser.add_argument(
        "service",
        nargs="?",
        type=str,
        choices=["transcribe", "meds", "sem"],
        default=None
    )
    args = parser.parse_args()

    if args.service == "transcribe":
        from src.services.transcription_service.transcription_whispertrt import run_transcription
        await run_transcription()
    elif args.service == "meds":
        from src.services.medication_extraction_service.medication_extraction import medication_extraction_service
        await medication_extraction_service()
    elif args.service == "sem":
        from src.services.semantic_filtering_service.semantic_filtering import run_semantic_filtering
        await run_semantic_filtering()
    else:
        from src.services.transcription_service.transcription_whispertrt import run_transcription
        await run_transcription()
        from src.services.medication_extraction_service.medication_extraction import medication_extraction_service
        await medication_extraction_service()
        from src.services.semantic_filtering_service.semantic_filtering import run_semantic_filtering
        await run_semantic_filtering()
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())