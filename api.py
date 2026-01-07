from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
import io
import numpy as np
from ultralytics import YOLO

DETECTOR_MODEL = "detector_yolo_4cls.pt"
VERIFIER_MODEL = "verifier_scripted.pt"

app = FastAPI(title="Signature Verification API")

# Pipeline
class Pipeline:
    def __init__(self):
        base = Path(__file__).parent
        self.detector = YOLO(str(base / DETECTOR_MODEL))
        self.verifier = torch.jit.load(str(base / VERIFIER_MODEL))
        self.verifier.eval()
        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((105, 105)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        print(f"Pipeline ready | Classes: {self.detector.names}")
    
    def verify(self, doc_bytes: bytes, ref_bytes: bytes, client: str, threshold: float):
        ref_img = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
        doc_img = Image.open(io.BytesIO(doc_bytes)).convert("RGB")
        
        # Detect
        results = self.detector(doc_img, conf=0.25, verbose=False)
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': round(box.conf[0].item(), 3),
                        'class': self.detector.names[int(box.cls[0])],
                        'crop': doc_img.crop((x1, y1, x2, y2))
                    })
        
        if not detections:
            return {
                'status_code': 'MISSING',
                'status': 'No signature found on document',
                'signature_found': False,
                'signature_count': 0,
                'is_match': None,
                'details': []
            }
        
        # Verify each
        details = []
        any_match = False
        for i, det in enumerate(detections):
            t1 = self.transform(det['crop']).unsqueeze(0)
            t2 = self.transform(ref_img).unsqueeze(0)
            with torch.no_grad():
                e1, e2 = self.verifier(t1, t2)
                dist = F.pairwise_distance(e1, e2).item()
            is_match = dist < threshold
            details.append({
                'id': i + 1,
                'class': det['class'],
                'bbox': det['bbox'],
                'distance': round(dist, 4),
                'similarity': round(1/(1+dist), 4),
                'is_match': is_match
            })
            if is_match:
                any_match = True
        
        return {
            'status_code': 'MATCH' if any_match else 'MISMATCH',
            'status': f'Signature {"matches" if any_match else "does NOT match"} {client}',
            'signature_found': True,
            'signature_count': len(details),
            'is_match': any_match,
            'details': details
        }


pipeline = None

@app.on_event("startup")
def startup():
    global pipeline
    pipeline = Pipeline()


@app.post("/verify")
async def verify(
    reference: UploadFile = File(...),
    document: UploadFile = File(...),
    client_name: str = Form(default="Client"),
    threshold: float = Form(default=0.5)
):
    ref_bytes = await reference.read()
    doc_bytes = await document.read()
    
    result = pipeline.verify(doc_bytes, ref_bytes, client_name, threshold)
    return JSONResponse(content=result)


@app.get("/")
def root():
    return {"message": "Signature Verification API", "endpoint": "/verify", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
