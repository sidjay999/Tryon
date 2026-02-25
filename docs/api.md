# AI Virtual Try-On – API Documentation

## Base URL
```
http://your-server/api
```

All endpoints accept and return JSON unless noted.

---

## POST /api/tryon

Submit a virtual try-on job.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `person_image` | `file` | ✅ | Full-body photo of the person (JPG/PNG/WEBP, max 20MB) |
| `clothing_image` | `file` | ✅ | Photo of the clothing item (JPG/PNG/WEBP, max 20MB) |

**Response 202**
```json
{
  "job_id": "c3e8a4b1-...",
  "status": "queued"
}
```

**Error 400** — invalid file type or size
```json
{ "detail": "Unsupported image type: image/gif" }
```

---

## GET /api/tryon/{job_id}

Poll job status and retrieve result.

**Path params**

| Param | Description |
|---|---|
| `job_id` | Job ID returned by POST /api/tryon |

**Response – pending**
```json
{ "job_id": "...", "status": "pending", "progress": 0 }
```

**Response – processing**
```json
{
  "job_id": "...",
  "status": "processing",
  "step": "inpainting",
  "progress": 55
}
```

Pipeline steps and progress mapping:

| Step | Progress |
|---|---|
| `segmentation` | 10% |
| `pose` | 25% |
| `warp` | 40% |
| `inpainting` | 55% |
| `blend` | 85% |
| `done` | 100% |

**Response – completed (S3 configured)**
```json
{
  "job_id": "...",
  "status": "completed",
  "progress": 100,
  "result_url": "https://s3.amazonaws.com/tryon-results/results/xxx.png?..."
}
```

**Response – completed (no S3, base64 fallback)**
```json
{
  "job_id": "...",
  "status": "completed",
  "progress": 100,
  "result_b64": "iVBORw0KGgoAAAANSUh..."
}
```

**Error 500** — pipeline failure
```json
{
  "detail": {
    "error": "CUDA out of memory",
    "job_id": "..."
  }
}
```

---

## GET /health

Returns service health and GPU memory status.

**Response 200**
```json
{
  "status": "ok",
  "models_loaded": true,
  "cuda_available": true,
  "gpu": {
    "gpu_name": "NVIDIA A10G",
    "memory_allocated_gb": 8.24,
    "memory_reserved_gb": 10.1,
    "total_memory_gb": 23.69
  }
}
```

---

## Interactive Docs

- **Swagger UI** → `http://your-server/docs`
- **ReDoc** → `http://your-server/redoc`

---

## Example: cURL

```bash
# Submit job
curl -X POST http://localhost:8000/api/tryon \
  -F person_image=@person.jpg \
  -F clothing_image=@shirt.jpg

# Poll status
curl http://localhost:8000/api/tryon/c3e8a4b1-...
```

## Example: Python

```python
import requests, time

# Submit
res = requests.post("http://localhost:8000/api/tryon",
    files={
        "person_image": open("person.jpg", "rb"),
        "clothing_image": open("shirt.jpg", "rb"),
    }
)
job_id = res.json()["job_id"]

# Poll
while True:
    r = requests.get(f"http://localhost:8000/api/tryon/{job_id}").json()
    print(r["status"], r.get("progress", 0))
    if r["status"] == "completed":
        print("Result:", r.get("result_url") or "base64 available")
        break
    if r["status"] in ("failed", "failure"):
        raise RuntimeError(r)
    time.sleep(2)
```
