import argparse
from src_audio.services.transcription_service.transcription_whispertrt import run_transcription
from src_audio.services.medication_extraction_service.medication_extraction import run_medication_extraction
from src_audio.services.intervention_extraction_service.intervention_extraction import run_intervention_extraction
from src_audio.services.semantic_filtering_service.semantic_filtering import run_semantic_filtering
from src_audio.services.anonymization_service.transcript_anonymization import run_anonymization_service
from src_audio.services.recording_audio_service.record_functions import run_recording_service
from src_audio.services.audio_chunking_service.trim_audio import run_audio_trimming
from src_audio.services.denoising_service.denoise import run_denoise_service
from src_audio.utils.metadata import setup_metadata, finalize_metadata
from datetime import datetime

async def main():
    """
    Main function to run microservices found in their respective folders using the command line.
    """
    parser = argparse.ArgumentParser(description="Run microservices")
    parser.add_argument(
        "service",
        nargs="?",
        type=str,
        choices=["transcribe", "meds", "inter", "sem", "anonymize", "trim", "record", "denoise"],
        default=None
    )
    args = parser.parse_args()
    
    start_time = datetime.now()
    end_time = start_time
    
    try:
        if args.service == "transcribe":
            setup_metadata()
            await run_transcription()
        elif args.service == "meds":
            setup_metadata()
            await run_medication_extraction()
        elif args.service == "inter":
            setup_metadata()
            await run_intervention_extraction()
        elif args.service == "sem":
            setup_metadata()
            await run_semantic_filtering()
        elif args.service == "anonymize":
            setup_metadata()
            await run_anonymization_service()
        elif args.service == "record":
            await run_recording_service()
            setup_metadata()
        elif args.service == "trim":
            await run_audio_trimming()
            setup_metadata()
        elif args.service == "denoise":
            await run_denoise_service()
            setup_metadata()
        
        else:  
            
            try:
                print("Starting recording...\n")
                await run_recording_service()
                print("Recording finished.\n")
            except Exception as e:
                print("Recording failed:", e)

            try:
                print("Starting de-noising service...\n")
                await run_denoise_service()
                print("De-noising service finished.\n")
            except Exception as e:
                print("De-noising service failed:", e)
            
            try:
                print("Starting audio trimming...\n")
                await run_audio_trimming()
                print("Audio trimming finished.\n")
            except Exception as e:
                print("Audio trimming failed:", e)
            
            setup_metadata()
            
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
    
    end_time = datetime.now()
    
    total_time = end_time - start_time
    
    total_seconds = int(total_time.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    print(f"Complete AUDIO pipeline time: {hours} hours, {minutes} minutes, and {seconds} seconds")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
