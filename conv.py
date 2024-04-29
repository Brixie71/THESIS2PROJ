import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'YuNet_model.onnx'
model_quant = 'YuNet_model.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant)