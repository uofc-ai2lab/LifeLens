# capstone-audio-to-text

Research phase for learning audio-to-text diarization.

---

## 🔧 Environment Setup

1. **Check current Python**

   ```bash
   which python
   ```

   * If the output looks like:

     ```
     .../anaconda3/bin/python
     ```

     then run:

     ```bash
     conda deactivate
     python3.11 -m venv ./venv311
     source ./venv311/bin/activate
     ```

2. **Verify Python location**

   ```bash
   which python
   ```

   Expected output:

   ```
   .../capstone-audio-to-text/nvidia-nemo/venv311/bin/python
   ```

---

## 🔑 Hugging Face Authentication

1. Register on [Hugging Face](https://huggingface.co).
2. Copy your **access token**.
3. Log in from CLI:

   ```bash
   huggingface-cli login
   ```

   Paste your token when prompted.

---

## 📦 Install Dependencies

Upgrade pip:

```bash
pip install --upgrade pip
```

Core packages:

```bash
pip install torch torchvision torchaudio
```

System libraries (macOS):

```bash
brew install swig
```

Extra dependencies:

```bash
pip install Cython packaging
pip install faiss-cpu==1.7.4
pip install "nemo_toolkit[all]"
```

---

## ▶️ Run Diarization

```bash
python3 diarize_audio.py
```

---
