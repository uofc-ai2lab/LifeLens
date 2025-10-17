# EMS Synthetic Audio Generator

Generate realistic, noisy EMS audio for fine-tuning speech-to-text models by combining medical transcripts with TTS and domain-specific background sounds (e.g., helicopter blades, sirens, traffic).

## Setup & Usage

1. **Install dependencies**:
   ```bash
   pip install requests pydub audiomentations librosa numpy scipy pyttsx3
   ```

2. **Get a Freesound API key** from [https://freesound.org/apiv2/apply/](https://freesound.org/apiv2/apply/) and add it to your environment variables. MacOS:
   ```bash
    export FREESOUND_API_KEY="your_api_key_here"
   ```

or Windows:
```bash
setx FREESOUND_API_KEY your_api_key_here
```

**NOTE**: It may take some time for your api key to work and you can run the python script.

3. **Download EMS background noises** (helicopter, sirens, etc.):
   ```bash
   python generate_noise_library.py
   ```

4. **Generate synthetic EMS audio**:
   ```bash
   python generate_synthetic_audio.py
   ```

> Output will be saved in `synthetic_medical_audio/` as noisy `.wav` files.