# Multimodal Instructional Assistant with Florence-2

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **Multimodal Instructional Assistant** that uses Microsoft's **Florence-2** vision-language model to generate captions and educational questions from images. Upload a diagram, chart, or any educational image and get instant AI-generated descriptions and study questions.

## ✨ Features

1. **Image Captioning**: Generate detailed descriptions of educational images (diagrams, charts, etc.)
2. **Question Generation**: Create educational questions from image labels using OCR detection
3. **Batch Processing**: Process multiple images at once with CSV export
4. **Fine-tuning Support**: Script to fine-tune the model using PEFT/LoRA

## 🤖 Model

- **Model**: [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) by Microsoft
- **Size**: ~1.5GB
- **Capabilities**: Image captioning, OCR, object detection, dense region captioning

## 📦 Setup

1. **Create virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or: source .venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For GPU support** (NVIDIA CUDA 12.1):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

4. **(Optional) Set HuggingFace token** for higher API rate limits:
   ```bash
   # Create a .env file or set the env variable directly
   set HF_TOKEN=your_token_here   # Windows
   export HF_TOKEN=your_token_here # Linux/Mac
   ```

## 🚀 Usage

### Run the Web App
```bash
python app.py
```
Access at: http://127.0.0.1:7860

### Command Line Options
```bash
python app.py --mock     # Use mock model for testing
python app.py --share    # Create public Gradio link
```

### Fine-tuning (Optional)
```bash
python train.py --dataset_path ./data/custom_dataset --output_dir ./output
```
*Requires GPU. Uses QLoRA for memory efficiency.*

## 📁 Project Structure

```
multimodal captioner/
├── app.py              # Gradio web interface
├── model_handler.py    # Florence-2 model wrapper
├── train.py            # LoRA fine-tuning script
├── diagnose.py         # Environment diagnostics
├── create_dummy_data.py# Test data generator
├── requirements.txt    # Python dependencies
├── PROJECT_SPECS.md    # Detailed documentation
├── .venv/              # Virtual environment
├── data/               # Training/test data
└── output/             # Generated outputs & adapters
```

## 🔧 Dependencies

| Package | Purpose |
|---------|---------|
| `transformers>=4.42.0` | HuggingFace model loading |
| `torch` | PyTorch deep learning framework |
| `gradio` | Web UI framework |
| `timm` | Vision backbone (required by Florence-2) |
| `peft` | Parameter-efficient fine-tuning (LoRA) |
| `pillow` | Image processing |
| `pandas` | Data handling (batch exports) |

## 📊 Hardware Requirements

- **CPU**: Works, but slow inference
- **GPU (Recommended)**: NVIDIA GPU with CUDA support
- **RAM**: 8GB+ recommended

## 📚 Documentation

For detailed technical documentation, architecture, and troubleshooting, see [PROJECT_SPECS.md](PROJECT_SPECS.md).

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🔗 References

- [Florence-2 Paper](https://arxiv.org/abs/2311.06242)
- [Florence-2 on HuggingFace](https://huggingface.co/microsoft/Florence-2-large)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Gradio Documentation](https://gradio.app/docs)
