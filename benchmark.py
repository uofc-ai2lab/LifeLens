import time
from faster_whisper import WhisperModel
from whisper_trt import load_trt_model

audio_file = "test_speech.wav"

print("="*60)
print("BENCHMARK: faster-whisper vs WhisperTRT")
print("="*60)

# Test faster-whisper
print("\n[1] Testing faster-whisper (base)...")
fw_model = WhisperModel("base", device="cuda", compute_type="float16")
start = time.time()
segments, info = fw_model.transcribe(audio_file)
list(segments)  # Consume generator
fw_time = time.time() - start
print(f"Time: {fw_time:.2f}s")

# Test WhisperTRT
print("\n[2] Testing WhisperTRT (base.en)...")
trt_model = load_trt_model("base.en")
start = time.time()
result = trt_model.transcribe(audio_file)
trt_time = time.time() - start
print(f"Time: {trt_time:.2f}s")

# Summary
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"faster-whisper: {fw_time:.2f}s")
print(f"WhisperTRT:     {trt_time:.2f}s")
print(f"Speedup:        {fw_time/trt_time:.2f}x")