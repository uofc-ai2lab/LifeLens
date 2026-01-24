#!/bin/bash

# Define output folder
OUTPUT_DIR="/home/capstone/recordings"

# Ensure the directory exists
mkdir -p "$OUTPUT_DIR"

# Define output file name with timestamp
OUTPUT_FILE="$OUTPUT_DIR/holding_$(date +%Y%m%d_%H%M%S).wav"

# Signal file to indicate completion
SIGNAL_FILE="$OUTPUT_DIR/recording_done.flag"

# Remove any previous signal file, just in case
rm -f "$SIGNAL_FILE"

echo "Starting recording for 10 seconds..."

# Start the recording in the background
timeout 10 arecord -D hw:CARD=ArrayUAC10,DEV=0 -f S16_LE -r 16000 -c 6 "$OUTPUT_FILE" &

# Get the process ID (PID) of the recording
RECORD_PID=$!

# Wait for the recording to finish
wait $RECORD_PID

# Create a signal file to indicate the recording is complete
touch "$SIGNAL_FILE"

# Notify user after recording ends
echo "Recording saved to $OUTPUT_FILE and recording is complete."
