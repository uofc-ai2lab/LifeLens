"""
    Main file to run microservices found in their respective folders using the command line.
"""

import argparse

async def main():
    parser = argparse.ArgumentParser(description="Run microservices")
    parser.add_argument(
        "service",
        type=str,
        choices=["whisperx"],  # Add other services as needed
        help="The microservice to run",
    )
    args = parser.parse_args()

    if args.service == "whisperx":
        from whisperX.whisper import run_whisperx
        await run_whisperx()
    elif args.service == "faster-whisper":
        from Faster_Whisper.transcribe_faster import run_faster_whisper
        await run_faster_whisper()
    else:
        print(f"Service {args.service} not recognized.")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())