import argparse
from src.services.transcription_service.transcription_whispertrt import run_transcription
from src.services.medication_extraction_service.medication_extraction import run_medication_extraction
from src.services.intervention_extraction_service.intervention_extraction import run_intervention_extraction
from src.services.semantic_filtering_service.semantic_filtering import run_semantic_filtering
from src.entities import AUDIO_PIPELINE_METADATA
from src.utils.metadata import setup_metadata, finalize_metadata

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
    
    setup_metadata()

    try:
        if args.service == "transcribe":
            await run_transcription()
        elif args.service == "meds":
            await run_medication_extraction()
        elif args.service == "inter":
            await run_intervention_extraction()
        elif args.service == "sem":
            await run_semantic_filtering()
        else:  
            try:
                print("Starting transcription...\n")
                await run_transcription()
                print("Transcription finished.\n")
            except Exception as e:
                print("Transcription failed:", e)

            try:
                print("Starting medication extraction...\n")
                await run_medication_extraction()
                print("Medication extraction finished\n")
            except Exception as e:
                print("Medication extraction failed:", e)

            try:
                print("Starting intervention extraction...\n")
                await run_intervention_extraction()
                print("All services finished!\n")
            except Exception as e:
                print("Intervention extraction failed:", e)
                
            # try:
            #     print("Starting semantic filtering...\n")
            #     await run_semantic_filtering()
            #     print("All services finished!\n")
            # except Exception as e:
            #     print("Semantic filtering failed:", e)
                
    finally:
        finalize_metadata()
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())