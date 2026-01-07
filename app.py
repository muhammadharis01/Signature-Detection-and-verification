import torch
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from pathlib import Path
import json
import io

from ultralytics import YOLO

DETECTOR_MODEL = "detector_yolo_4cls.pt"
VERIFIER_MODEL = "verifier_scripted.pt"
DEFAULT_THRESHOLD = 0.5


class SignatureVerificationPipeline:
    
    def __init__(self, detector_path: str, verifier_path: str):
        self.detector = YOLO(detector_path)
        self.verifier = torch.jit.load(verifier_path)
        self.verifier.eval()
        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((105, 105)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        self.threshold = DEFAULT_THRESHOLD
        print(f"Pipeline ready | Detector: {self.detector.names}")
    
    def detect(self, image):
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        else:
            img = Image.open(image).convert("RGB")
        
        results = self.detector(img, conf=0.25, verbose=False)
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': round(box.conf[0].item(), 3),
                        'class': self.detector.names[int(box.cls[0])],
                        'crop': img.crop((x1, y1, x2, y2))
                    })
        return detections, img
    
    def compare(self, img1, img2, threshold=None):
        if threshold is None:
            threshold = self.threshold
        
        def prep(img):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
            return self.transform(img).unsqueeze(0)
        
        t1, t2 = prep(img1), prep(img2)
        with torch.no_grad():
            e1, e2 = self.verifier(t1, t2)
            dist = F.pairwise_distance(e1, e2).item()
        
        return {
            'distance': round(dist, 4),
            'similarity': round(1/(1+dist), 4),
            'is_match': dist < threshold
        }
    
    def verify(self, document, reference, client="Client"):
        if isinstance(reference, np.ndarray):
            ref_img = Image.fromarray(reference).convert("RGB")
        else:
            ref_img = Image.open(reference).convert("RGB")
        
        detections, doc_img = self.detect(document)
        
        if not detections:
            return {'status_code': 'MISSING', 'status': 'No signature found', 'is_match': None, 'details': []}, doc_img
        
        details = []
        any_match = False
        for i, det in enumerate(detections):
            result = self.compare(det['crop'], ref_img)
            details.append({'id': i + 1, 'class': det['class'], 'bbox': det['bbox'], 'confidence': det['confidence'], **result})
            if result['is_match']:
                any_match = True
        
        status_code = 'MATCH' if any_match else 'MISMATCH'
        status = f'Signature {"matches" if any_match else "does NOT match"} {client}'
        return {'status_code': status_code, 'status': status, 'is_match': any_match, 'details': details}, doc_img


pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        base = Path(__file__).parent
        pipeline = SignatureVerificationPipeline(str(base / DETECTOR_MODEL), str(base / VERIFIER_MODEL))
    return pipeline


def verify(reference, document, client_name, threshold):
    if reference is None or document is None:
        return "Upload both images", None, "{}"
    
    pipe = get_pipeline()
    pipe.threshold = threshold
    result, doc_img = pipe.verify(document, reference, client_name or "Client")
    
    annotated = doc_img.copy()
    draw = ImageDraw.Draw(annotated)
    for d in result.get('details', []):
        color = 'green' if d['is_match'] else 'red'
        draw.rectangle(d['bbox'], outline=color, width=3)
        draw.text((d['bbox'][0], d['bbox'][1] - 15), f"{d['similarity']:.0%}", fill=color)
    
    return result['status'], annotated, json.dumps(result, indent=2)


demo = gr.Interface(
    fn=verify,
    inputs=[
        gr.Image(label="Reference Signature", type="numpy"),
        gr.Image(label="Document", type="numpy"),
        gr.Textbox(label="Client Name", value="Haris"),
        gr.Slider(0.1, 2.0, 0.5, label="Threshold")
    ],
    outputs=[
        gr.Textbox(label="Result"),
        gr.Image(label="Annotated"),
        gr.Code(label="JSON", language="json")
    ],
    title="Signature Verification",
    description="Upload reference signature + document to verify.",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
