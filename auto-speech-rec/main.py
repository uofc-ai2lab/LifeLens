from transformers import pipeline

cls = pipeline("automatic-speech-recognition", device=-1)  # device=-1 forces CPU

# audio used: https://www.youtube.com/watch?v=VJQq1ROW5G4
res = cls("trauma_patient_pov.mp3")

print(res)