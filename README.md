---
title: Signature Verification
emoji: ✍️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Signature Verification

Detect and verify signatures in documents.

## API
- **POST /verify** - Upload reference + document images
- **GET /health** - Health check

## Response
```json
{
  "status_code": "MATCH|MISMATCH|MISSING",
  "status": "Signature matches/does NOT match Client",
  "signature_found": true,
  "signature_count": 1,
  "is_match": true
}
```
