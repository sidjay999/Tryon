# Model Optimization Notes

## 1. FP16 Precision

All models are loaded with `torch_dtype=torch.float16`. This halves VRAM usage compared to FP32 with minimal quality loss on SDXL.

```python
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
```

**Effect:** Reduces VRAM from ~20GB → ~10GB. Required for 24GB GPU cards.

---

## 2. xFormers Memory-Efficient Attention

```python
pipe.enable_xformers_memory_efficient_attention()
```

Replaces PyTorch's native attention with xFormers Flash Attention. Reduces memory by ~15-30% and speeds up inference by ~20%.

**Disable on CPU-only:** Comment out `xformers` in `requirements.txt` and set `USE_XFORMERS=false`.

---

## 3. Attention Slicing

```python
pipe.enable_attention_slicing()
```

Processes attention heads one slice at a time. Trades a small speed penalty for significantly lower peak VRAM. Enabled by default since it has minimal quality impact.

---

## 4. VAE Tiling

```python
pipe.enable_vae_tiling()
```

Processes the image in tiles during VAE encode/decode. Critical for 1024px generation on ≤16GB VRAM cards.

---

## 5. Celery Worker Restart Per Task

```python
worker_max_tasks_per_child = 1
```

The Celery worker process is recycled after each task. Ensures GPU memory is fully freed between jobs, preventing memory fragmentation buildup over long uptime.

---

## 6. Model Caching

All Hugging Face models are cached at `/app/model_cache` (mounted as a Docker volume). After first download, subsequent container restarts load from disk rather than re-downloading.

```yaml
volumes:
  - model_cache:/app/model_cache
```

**First-run time:** ~20-40 minutes (downloads ~15GB of model weights)
**Subsequent starts:** ~60-90 seconds (load from disk to GPU)

---

## 7. Inference Steps Tuning

| Steps | Quality | Inference Time (A10G) |
|---|---|---|
| 20 | Good | ~8s |
| 30 | High (default) | ~12s |
| 50 | Best | ~20s |

Default is 30 steps for the best quality/speed tradeoff.

---

## 8. Batching

Currently configured for single-image inference (`concurrency=1`). For throughput-optimized deployments:

```python
# In tasks.py, modify run_inpainting to accept batch:
pipe(prompt=[prompt] * batch_size, image=[img]*n, ...)
```

Recommended batch size: 2–4 images on an A100 80GB.

---

## 9. TorchScript / Torch Compile (Future)

```python
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

PyTorch 2.x compile can provide a further 20-40% speedup after warmup. Not enabled by default due to longer first-inference compile time (~3 minutes).

---

## 10. Target GPU Requirements

| GPU | VRAM | Status |
|---|---|---|
| RTX 3090 / 4090 | 24GB | ✅ Supported (FP16) |
| A10G | 24GB | ✅ Recommended |
| A100 40/80GB | 40-80GB | ✅ Best performance |
| RTX 3080 | 10GB | ⚠️ Tight — enable VAE tiling + attention slicing |
| CPU | - | ⚠️ Supported but ~5 minutes per inference |
