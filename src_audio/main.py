import argparse
from src_audio.services.transcription_service.transcription_whispertrt import run_transcription
from src_audio.services.medication_extraction_service.medication_extraction import run_medication_extraction
from src_audio.services.intervention_extraction_service.intervention_extraction import run_intervention_extraction
from src_audio.services.semantic_filtering_service.semantic_filtering import run_semantic_filtering
from src_audio.services.anonymization_service.transcript_anonymization import run_anonymization_service
from scripts.trim_audio import run_audio_trimming
from src_audio.utils.metadata import setup_metadata, finalize_metadata

async def main():
    """
    Main function to run microservices found in their respective folders using the command line.
    """
    parser = argparse.ArgumentParser(description="Run microservices")
    parser.add_argument(
        "service",
        nargs="?",
        type=str,
        choices=["transcribe", "meds", "inter", "sem", "anonymize", "trim"],
        default=None
    )
    args = parser.parse_args()
    
    await run_audio_trimming()
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
        elif args.service == "anonymize":
            await run_anonymization_service()
        elif args.service == "trim":
            await run_audio_trimming()
        else:  
            try:
                print("Starting transcription...\n")
                await run_transcription()
                print("Transcription finished.\n")
            except Exception as e:
                print("Transcription failed:", e)

            try:
                print("Starting anonymization...\n")
                await run_anonymization_service()
                print("Anonymization finished.\n")
            except Exception as e:
                print("Anonymization failed:", e)

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