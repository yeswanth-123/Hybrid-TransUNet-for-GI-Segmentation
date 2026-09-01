from onnxruntime.quantization import quantize_dynamic, QuantType
import os

model_fp32 = "transunet_optimized.onnx"
model_int8 = "transunet_quantized.onnx"

print(f"Original size: {os.path.getsize(model_fp32) / (1024 * 1024):.2f} MB")
print("Quantizing model from FP32 to INT8... (This takes about 10 seconds)")

# Compress the weights to 8-bit integers
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_int8,
    weight_type=QuantType.QUInt8
)

print(f"Quantized size: {os.path.getsize(model_int8) / (1024 * 1024):.2f} MB")
print(f"Success! Saved as {model_int8}")