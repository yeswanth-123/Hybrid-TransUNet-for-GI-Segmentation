import time
import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image

app = FastAPI(title="TransUNet INT8 Optimized API")

# Load INT8 Quantized ONNX session with Threading Optimization
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4  # Maximize CPU cores
sess_options.inter_op_num_threads = 4

# Load the NEW quantized model
session = ort.InferenceSession("transunet_quantized.onnx", sess_options)

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((256, 256))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_arr = (img_arr - mean) / std
    return np.expand_dims(np.transpose(img_arr, (2, 0, 1)), axis=0).astype(np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    input_tensor = preprocess(await file.read())
    
    # Run inference and time it
    start_time = time.perf_counter()
    logits = session.run(["output"], {"input": input_tensor})[0]
    latency = (time.perf_counter() - start_time) * 1000
    
    # Post-process
    mask = ((1 / (1 + np.exp(-logits)))[0, 0] > 0.5).astype(np.uint8) * 255
    
    img_io = io.BytesIO()
    Image.fromarray(mask, mode='L').save(img_io, format='PNG')
    img_io.seek(0)
    
    print(f"Optimized INT8 Latency: {latency:.2f} ms")
    return StreamingResponse(img_io, media_type="image/png")