import io
import os

import torch
from fastapi import FastAPI, UploadFile, Form
from faster_whisper import WhisperModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SAVE_DIR = "models/"
MODEL_PATH = os.path.join(SAVE_DIR, "whisper-large-v3")

app = FastAPI()


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

        model.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)

        print("Model downloaded and saved at", MODEL_PATH)
    else:
        print("Model already exists at", MODEL_PATH)


def initialize_model():
    download_model()

    model_path = "/models/whisper-large-v3"
    if torch.cuda.is_available():
        print("CUDA is available")
        return WhisperModel("large-v2", device="cuda", compute_type="float16", download_root=model_path)
    else:
        print("CUDA is not available or not enabled")
        cpu_threads = os.cpu_count()
        return WhisperModel("large-v2", device="cpu", compute_type="int8", cpu_threads=cpu_threads,
                            download_root=model_path)


model = initialize_model()


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = Form(...)):
    try:
        # ファイルの内容をバイナリデータとして読み込む
        file_content = await file.read()

        # バイナリデータをBinaryIOオブジェクトに変換
        file_stream = io.BytesIO(file_content)

        result_text = ""

        # 音声ファイルの文字起こし
        segments, info = model.transcribe(
            audio=file_stream,  # BinaryIOオブジェクトを渡す
            beam_size=5,
            language="ja",
            vad_filter=True,
            without_timestamps=True,
        )

        total_time = 0.0
        for segment in segments:
            segment_duration = segment.end - segment.start
            total_time += segment_duration
            result_text += segment.text
            print(f"{segment.start} - {segment.end}: {segment.text}")

    except Exception as e:
        return {"error": str(e)}
