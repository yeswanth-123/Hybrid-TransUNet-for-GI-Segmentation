# Hybrid TransUNet for GI Segmentation

A hybrid CNN + Transformer segmentation model for gastrointestinal lesion
detection on the Kvasir-SEG dataset, with an optimized FastAPI + ONNX
deployment pipeline.

## Architecture

- **Encoder**: 4-stage convolutional encoder (`ConvBlock`), each stage
  doubling channels (3 → 64 → 128 → 256 → 512) with max-pooling downsampling.
- **Bottleneck**: a Transformer bottleneck (`TransformerBottleneck`) applies
  multi-head self-attention (8 heads, 4 layers) over the flattened spatial
  feature map, with learnable 1D positional embeddings added to each patch
  token to preserve spatial/geometric information lost during flattening.
- **Decoder**: transposed-convolution upsampling with skip connections from
  every encoder stage, standard U-Net style.
- **Loss**: a custom combined BCE + Dice loss (`BCEDiceLoss`).

## Dataset

[Kvasir-SEG](https://www.kaggle.com/datasets/debeshjha1/kvasirseg) — 1000
images split 80/10/10 (train/val/test), loaded via `kagglehub`.

## Results

Trained for 40 epochs (AdamW, cosine LR schedule, mixed precision). Final
test-set results with 4-way test-time augmentation (original, horizontal
flip, vertical flip, 180° rotation):

| Metric          | Score  |
|-----------------|--------|
| Pixel Accuracy  | ~94.7% |
| Mean Dice       | ~0.82  |
| Mean IoU        | ~0.735 |

Reproduced independently across two separate training runs (Kaggle and
Google Colab) with consistent results.

## Deployment

- `app.py` — FastAPI server serving the FP32 ONNX-exported model.
- `app2.py` — FastAPI server serving an INT8 dynamically-quantized version
  of the model (via `quantize.py`), reducing model size by ~3.7x and CPU
  inference latency to roughly 250ms per request on commodity hardware.
- `test.py` — sends a sample image to a running server and saves the
  predicted mask.

### Running locally

```bash
# 1. Train (see the notebook) or supply your own best_transunet.pth,
#    then export to ONNX and quantize:
python quantize.py

# 2. Start the server (INT8 quantized model)
uvicorn app2:app --host 127.0.0.1 --port 8000

# 3. Send a test image
python test.py
```

Note: the exported `.onnx` model weight files are not committed to this
repo (see `.gitignore`) — generate them locally via the training notebook
and `quantize.py`.

## Files

- `hybrid-transunet-for-gastrointestinal-lesion.ipynb` — full training,
  evaluation, and ONNX export pipeline.
- `app.py`, `app2.py`, `quantize.py`, `test.py` — deployment scripts.
- `test_image.jpg`, `result_mask.png` — sample input/output pair.
