# Signature Verification

Detect and verify signatures in documents using YOLO + Siamese CNN.

## Models
- `detector_yolo_4cls.pt` - YOLO detector (signature, initials, redaction, date)
- `verifier_scripted.pt` - Siamese CNN for verification (TorchScript)

## Usage

### Local
```bash
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:7860
```

### API (n8n/Postman)
```
POST /api/predict
Body: {"data": [ref_base64, doc_base64, "ClientName", 0.5]}
```

## Response Codes
- `MISSING` - No signature found
- `MATCH` - Signature verified
- `MISMATCH` - Different person
