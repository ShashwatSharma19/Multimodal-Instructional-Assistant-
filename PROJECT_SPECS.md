# Multimodal Educational Assistant - Project Documentation

## 📋 Project Overview

This project is a **Multimodal Educational Assistant** that uses the **Florence-2** vision-language model to:
1. **Generate captions** for educational images (diagrams, charts, etc.)
2. **Generate questions** from image content using OCR detection

---

## 🤖 Model Used

### Florence-2-large (Microsoft)
- **Model ID**: `microsoft/Florence-2-large`
- **Type**: Vision-Language Model (VLM)
- **Size**: ~1.5GB
- **Capabilities**:
  - Image captioning (basic, detailed, more detailed)
  - Optical Character Recognition (OCR)
  - Object Detection
  - Dense Region Captioning
  - Region-to-Description

### Why Florence-2?
- **Multimodal**: Understands both images and text
- **Open-source**: Free to use and modify
- **Task-flexible**: Supports multiple vision tasks with simple prompt tokens
- **Efficient**: Can run on consumer GPUs (like RTX 3050)

---

## 🧠 Key Concepts

### 1. Multimodal Learning
Combining multiple types of data (images + text) to understand content. The model "sees" the image and "reads" text within it simultaneously.

### 2. Vision-Language Models (VLMs)
Neural networks trained on image-text pairs. They learn to:
- Describe what's in an image
- Answer questions about images
- Detect and recognize objects/text

### 3. Task Prompts
Florence-2 uses special tokens to specify what task to perform:
- `<CAPTION>` - Basic caption
- `<MORE_DETAILED_CAPTION>` - Rich, detailed description
- `<OCR>` - Extract text from image
- `<OD>` - Object detection

### 4. Optical Character Recognition (OCR)
Detecting and reading text within images. Used to identify labels in diagrams (like "Aorta", "Left Ventricle" in heart diagrams).

### 5. Float16 (Half Precision)
Model weights stored in 16-bit floats instead of 32-bit. Benefits:
- 50% less memory usage
- Faster inference on GPU
- Minimal quality loss

---

## 📁 Project Structure

```
multimodal captioner/
├── app.py              # Gradio web interface
├── model_handler.py    # Florence-2 model wrapper
├── train.py            # LoRA fine-tuning script
├── diagnose.py         # Environment diagnostics
├── create_dummy_data.py# Test data generator
├── requirements.txt    # Python dependencies
├── .venv/              # Virtual environment
├── data/               # Training/test data
│   └── dummy_dataset/
└── output/             # Generated outputs
    └── lora_adapter/   # Fine-tuned adapter
```

---

## 📄 File Descriptions

### `model_handler.py`
Core model logic. Contains the `Florence2Handler` class.

| Method | Description |
|--------|-------------|
| `__init__(model_id, device)` | Initialize handler with model ID and device (cuda/cpu) |
| `load_model()` | Load processor and model from HuggingFace |
| `generate(image, task_prompt, text_input)` | Core generation method - processes image with given task |
| `caption(image)` | Generate detailed caption using `<MORE_DETAILED_CAPTION>` |
| `generate_questions(image, num_questions)` | Use OCR to detect labels and generate educational questions |

**Key Implementation Details:**
```python
# Dtype matching (GPU fix)
model_dtype = next(self.model.parameters()).dtype
inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

# Cache disabled (compatibility fix)
use_cache=False  # Required for Florence-2 with current transformers
```

---

### `app.py`
Gradio web interface with two tabs.

| Function | Description |
|----------|-------------|
| `init_handler(mock, token)` | Initialize model (real or mock mode) |
| `process_single(image, mode)` | Handle single image - caption or generate questions |
| `process_batch(files, mode)` | Process multiple images, export to CSV |
| `create_ui()` | Build Gradio interface with tabs |

**UI Modes:**
1. **Caption** - Generate detailed image description
2. **Generate Questions** - Create educational questions from image labels

---

### `train.py`
Fine-tuning script using LoRA (Low-Rank Adaptation).

| Component | Purpose |
|-----------|---------|
| `PaliGemmaProcessor` | Tokenizes text and processes images |
| `LoraConfig` | Configures adapter parameters (rank, alpha, dropout) |
| `Trainer` | HuggingFace training loop |

**LoRA Parameters:**
- `r=8` - Low rank dimension
- `lora_alpha=16` - Scaling factor
- `lora_dropout=0.1` - Regularization

---

### `diagnose.py`
Environment verification script.

Checks:
- `HF_TOKEN` environment variable
- Processor loading
- Model loading capability

---

### `create_dummy_data.py`
Generates test images and metadata for development.

---

## 🔧 Dependencies

| Package | Purpose |
|---------|---------|
| `transformers` | HuggingFace model loading |
| `torch` | PyTorch deep learning framework |
| `gradio` | Web UI framework |
| `timm` | Vision backbone (required by Florence-2) |
| `peft` | Parameter-efficient fine-tuning (LoRA) |
| `accelerate` | Training acceleration |
| `pillow` | Image processing |
| `pandas` | Data handling (batch exports) |

---

## 🚀 How It Works

### Caption Generation Flow
```
Image → Processor → Pixel Values (float16)
                  ↓
        Florence-2 Model
                  ↓
        <MORE_DETAILED_CAPTION>
                  ↓
        "The image shows a detailed anatomical
         diagram of a human heart with labeled
         structures including the aorta..."
```

### Question Generation Flow
```
Image → OCR Task → Extract Labels
                  ↓
        ["Aorta", "Left Ventricle", "Pulmonary Valve"]
                  ↓
        Question Templates
                  ↓
        1. What is the function of the Aorta?
        2. Describe the role of Left Ventricle...
        3. Where is the Pulmonary Valve located?
```

---

## 🛠️ Technical Fixes Applied

### 1. `use_cache=False`
**Problem**: Florence-2's beam search crashed with `'NoneType' object has no attribute 'shape'`
**Fix**: Disabled KV-caching in generation

### 2. Dtype Matching
**Problem**: `Input type (float) and bias type (Half) should be the same`
**Fix**: Convert `pixel_values` to model's dtype (float16) before inference

### 3. `timm` Installation
**Problem**: Model failed to load - missing vision backbone dependency
**Fix**: Installed `timm` package (PyTorch Image Models)

### 4. CUDA PyTorch
**Problem**: Running on CPU (slow inference)
**Fix**: Installed `torch+cu121` for NVIDIA GPU support

---

## 🎮 Running the Project

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run the web app
python app.py

# Access at: http://127.0.0.1:7860
```

---

## 📊 Hardware Used

- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
- **CUDA Version**: 12.1
- **PyTorch**: 2.5.1+cu121
- **Python**: 3.12

---

## 📚 Further Learning

1. **Florence-2 Paper**: [arxiv.org/abs/2311.06242](https://arxiv.org/abs/2311.06242)
2. **HuggingFace Model Card**: [huggingface.co/microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)
3. **LoRA Paper**: [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
4. **Gradio Docs**: [gradio.app/docs](https://gradio.app/docs)
