"""Pre-download all models used in the project (priority order), retrying on dropped connections."""
import time

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B", "intfloat/multilingual-e5-large", "microsoft/mdeberta-v3-base",
    "Qwen/Qwen3-4B-Instruct-2507", "yandex/YandexGPT-5-Lite-8B-instruct",
]
IGNORE = ["onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.onnx", "*.ot", "*.gguf", "*.pth"]

for m in MODELS:
    for attempt in range(50):
        try:
            snapshot_download(m, ignore_patterns=IGNORE, max_workers=2)
            print("done", m, flush=True)
            break
        except Exception as e:  # noqa: BLE001 -- flaky proxy; resume from .incomplete
            print("retry", m, attempt, type(e).__name__, flush=True)
            time.sleep(5)
