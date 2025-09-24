# WhisperX Environment Setup & Fine-Tuning Guide

## 3. Create and Activate Python Virtual Environment

```sh
python3.11 -m venv venv311
# On Windows:
venv311\Scripts\activate
# On Mac/Linux:
source venv311/bin/activate
```

## 4. Upgrade pip

```sh
pip install --upgrade pip
```

## 5. Install Dependencies

```sh
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install .
```

## 6. (Optional) Install CUDA for GPU Acceleration

- Make sure you have a compatible NVIDIA GPU and CUDA drivers.
- See [PyTorch CUDA Installation Guide](https://pytorch.org/get-started/locally/).

## 7. Hugging Face Authentication

1. Register at [Hugging Face](https://huggingface.co).
2. Get your access token.
3. Login via CLI:

    ```sh
    huggingface-cli login
    ```

## 8. Fine-Tuning WhisperX

- Prepare your dataset (see [WhisperX docs](https://github.com/m-bain/whisperx#fine-tuning)).
- Run fine-tuning:

    ```sh
    python whisper.py --train --data_dir /path/to/your/data --output_dir /path/to/save/model
    ```

- For more options:

    ```sh
    python whisper.py --help
    ```

## 9. Running Inference

```sh
python whisper.py --audio_file /path/to/audio.wav --model_dir /path/to/save/model
```

---

## Troubleshooting

- If you have dependency issues, reinstall them or check [WhisperX Issues](https://github.com/m-bain/whisperx/issues).
- For advanced usage, see the [WhisperX documentation](https://github.com/m-bain/whisperx).

---

## References

- [WhisperX GitHub](https://github.com/m-bain/whisperx)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face](https://huggingface.co)
- [PyTorch](https://pytorch.org)

---

**Note:** Fine-tuning requires a suitable dataset and a GPU is recommended for best performance.