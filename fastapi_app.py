"""FastAPI-сервис: классификация вида спорта по фото (100 классов, PyTorch EfficientNet-B0).

Запуск:  uvicorn fastapi_app:app --reload
Файлы модели (получаются в Colab): model/pytorch_sports_model.pth и model/classes.json
"""
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

log = logging.getLogger("sports")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "pytorch_sports_model.pth"))
CLASSES_PATH = Path(os.getenv("CLASSES_PATH", BASE_DIR / "model" / "classes.json"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 МБ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SportsClassifier(nn.Module):
    """Та же архитектура, что в Colab-ноутбуке (иначе веса не загрузятся)."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

state: dict = {"model": None, "classes": []}


def load_model() -> None:
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Не найдены файлы модели: {MODEL_PATH.name} и {CLASSES_PATH.name} "
            f"(ожидаются в папке {MODEL_PATH.parent})"
        )
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    model = SportsClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    state["model"], state["classes"] = model, classes
    log.info("Модель загружена: %d классов, устройство: %s", len(classes), device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as exc:  # сервер запускается и объясняет, чего не хватает
        log.error("Модель не загружена: %s", exc)
    yield


app = FastAPI(title="Sports 100 Classifier API", lifespan=lifespan)


@app.get("/")
def home():
    return {"message": "FastAPI Sports Classifier is running!", "model_loaded": state["model"] is not None}


@app.get("/health")
def health():
    return {"status": "ok" if state["model"] is not None else "model_not_loaded", "classes": len(state["classes"])}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(
            503,
            "Модель не загружена. Положите pytorch_sports_model.pth и classes.json в папку model и перезапустите сервер.",
        )
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Файл слишком большой (максимум 10 МБ)")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Не удалось прочитать изображение (нужен jpg или png)")

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(state["model"](tensor), dim=1)[0]
    top_p, top_i = torch.topk(probs, k=min(5, len(state["classes"])))
    top5 = [
        {"class_id": int(i), "class_name": state["classes"][int(i)], "confidence": round(float(p), 4)}
        for p, i in zip(top_p, top_i)
    ]
    return {
        "class_id": top5[0]["class_id"],
        "class_name": top5[0]["class_name"],
        "confidence": top5[0]["confidence"],
        "top5": top5,
    }
