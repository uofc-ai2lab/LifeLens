import argparse
from src.services.transcription_service.transcription_whispertrt import run_transcription
from src.services.medication_extraction_service.medication_extraction import run_medication_extraction_service
# from src.services.intervention_extraction_service.intervention_extraction import run_intervention_extraction
from src.services.semantic_filtering_service.semantic_filtering import run_semantic_filtering

async def main():
    """
    Main function to run microservices found in their respective folders using the command line.
    """
    parser = argparse.ArgumentParser(description="Run microservices")
    parser.add_argument(
        "service",
        nargs="?",
        type=str,
        choices=["transcribe", "meds", "inter", "sem"],
        default=None
    )
    args = parser.parse_args()

    if args.service == "transcribe":
        await run_transcription()
    elif args.service == "meds":
        await run_medication_extraction_service()
    # elif args.service == "inter":
    #     await run_intervention_extraction()
    elif args.service == "sem":
        await run_semantic_filtering()
    else:
        await run_transcription()
        await run_medication_extraction_service()
        await run_semantic_filtering()
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())