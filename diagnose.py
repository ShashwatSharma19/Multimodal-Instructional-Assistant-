import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

def diagnose():
    model_id = "microsoft/Florence-2-large"
    print(f"Diagnostics for {model_id}...")

    # Hardware Check
    has_cuda = torch.cuda.is_available()
    print(f"\n[Hardware Check]")
    print(f"CUDA Available: {has_cuda}")
    if has_cuda:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch Version: {torch.__version__}")
        
    print(f"\n[Model Loading Check]")
    print(f"Attempting to load processor for {model_id}...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Processor loaded successfully!")
    except Exception as e:
        print(f"FAILED to load processor. Error:\n{e}")
        return

    print(f"\nAttempting to load model for {model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"FAILED to load model. Error:\n{e}")

if __name__ == "__main__":
    diagnose()
