import warnings
import whisper

# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

model = whisper.load_model("small")
result = model.transcribe("spk1_00014.wav")
print(result["text"])