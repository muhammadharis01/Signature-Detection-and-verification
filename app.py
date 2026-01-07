import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import json
import io
import base64

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Config
DETECTOR = "detector_yolo_4cls.pt"
VERIFIER = "verifier_scripted.pt"
BASE_DIR = Path(__file__).parent

app = FastAPI(title="Signature Verification")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class Pipeline:
    def __init__(self):
        self.detector = YOLO(str(BASE_DIR / DETECTOR))
        self.verifier = torch.jit.load(str(BASE_DIR / VERIFIER))
        self.verifier.eval()
        self.transform = T.Compose([
            T.Grayscale(), T.Resize((105, 105)), T.ToTensor(), T.Normalize([0.5], [0.5])
        ])
        print(f"Pipeline ready | Classes: {self.detector.names}")

    def _prep(self, img):
        if isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))
        return self.transform(img.convert("RGB")).unsqueeze(0)

    def verify(self, doc_bytes, ref_bytes, client, threshold):
        ref = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
        doc = Image.open(io.BytesIO(doc_bytes)).convert("RGB")

        # Detect signatures
        results = self.detector(doc, conf=0.25, verbose=False)
        detections = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': round(box.conf[0].item(), 3),
                        'class': self.detector.names[int(box.cls[0])],
                        'crop': doc.crop((x1, y1, x2, y2))
                    })

        if not detections:
            return {
                'status_code': 'MISSING',
                'status': 'No signature found',
                'signature_found': False,
                'signature_count': 0,
                'is_match': None,
                'details': []
            }, doc

        # Verify each detection
        details, any_match = [], False
        for i, det in enumerate(detections):
            t1, t2 = self._prep(det['crop']), self._prep(ref)
            with torch.no_grad():
                dist = F.pairwise_distance(*self.verifier(t1, t2)).item()
            match = dist < threshold
            details.append({
                'id': i + 1, 'class': det['class'], 'bbox': det['bbox'],
                'confidence': det['confidence'], 'distance': round(dist, 4),
                'similarity': round(1/(1+dist), 4), 'is_match': match
            })
            if match: any_match = True

        # Annotate
        annotated = doc.copy()
        draw = ImageDraw.Draw(annotated)
        for d in details:
            draw.rectangle(d['bbox'], outline='green' if d['is_match'] else 'red', width=3)

        return {
            'status_code': 'MATCH' if any_match else 'MISMATCH',
            'status': f"Signature {'matches' if any_match else 'does NOT match'} {client}",
            'signature_found': True,
            'signature_count': len(details),
            'is_match': any_match,
            'details': details
        }, annotated


pipe = None

def get_pipe():
    global pipe
    if not pipe:
        pipe = Pipeline()
    return pipe


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/verify")
async def verify(
    request: Request,
    reference: UploadFile = File(...),
    document: UploadFile = File(...),
    client_name: str = Form(default="Client"),
    threshold: float = Form(default=0.5)
):
    ref_bytes = await reference.read()
    doc_bytes = await document.read()
    result, annotated = get_pipe().verify(doc_bytes, ref_bytes, client_name, threshold)

    # Browser request → HTML, API request → JSON
    if "text/html" in request.headers.get("accept", ""):
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": result['status'],
            "status_class": result['status_code'].lower(),
            "signature_count": result.get('signature_count', 0),
            "image_base64": base64.b64encode(buf.getvalue()).decode(),
            "json_result": json.dumps(result, indent=2)
        })
    return JSONResponse(content=result)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
