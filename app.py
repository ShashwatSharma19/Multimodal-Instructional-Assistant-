import gradio as gr
import argparse
from model_handler import Florence2Handler, MockFlorenceHandler
import os
from PIL import Image
import pandas as pd
import time

# Global handler
handler = None

def init_handler(mock=False, token=None):
    global handler
    if mock:
        print("Initializing in MOCK mode.")
        handler = MockFlorenceHandler()
    else:
        print("Initializing in REAL mode (Florence-2).")
        handler = Florence2Handler()

    if not handler.load_model():
        print("Failed to load real model. Falling back to Mock for UI demo.")
        handler = MockFlorenceHandler()
        handler.load_model()
        return

    # Eagerly load Phi-3-mini at startup — never downloads mid-request
    if hasattr(handler, 'enhancer'):
        try:
            handler.enhancer.load()
        except Exception as e:
            print(f"Warning: Phi-3-mini failed to load: {e}")
            print("Tip: run 'python download_models.py' first to pre-cache all models.")

def process_single(image, mode):
    if image is None:
        return "⚠️ Please upload an image first."

    start = time.time()
    if mode == "Caption":
        result = handler.caption(image)
    else:
        result = handler.generate_questions(image)

    elapsed = time.time() - start

    # Clean up result
    if isinstance(result, dict):
        result = next(iter(result.values()), str(result))
    result = str(result).strip()

    return f"{result}\n\n─────────────────────\n⏱ Generated in {elapsed:.1f}s"

def process_batch(files, mode, progress=gr.Progress()):
    if not files:
        return None, None

    images = []
    filenames = []

    progress(0, desc="Loading images...")
    for i, file_path in enumerate(files):
        try:
            path = file_path if isinstance(file_path, str) else file_path.name
            image = Image.open(path)
            images.append(image)
            filenames.append(os.path.basename(path))
        except Exception as e:
            filenames.append(os.path.basename(str(file_path)))
            images.append(None)

    valid_indices = [i for i, img in enumerate(images) if img is not None]
    valid_images = [images[i] for i in valid_indices]

    results_map = {}
    for idx, (vi, img) in enumerate(zip(valid_indices, valid_images)):
        progress((idx + 1) / len(valid_images),
                 desc=f"Processing {filenames[vi]} ({idx+1}/{len(valid_images)})...")
        if mode == "Caption":
            out = handler.caption(img)
        else:
            out = handler.generate_questions(img)
        if isinstance(out, dict):
            out = next(iter(out.values()), str(out))
        results_map[vi] = str(out).strip()

    results = []
    for i, filename in enumerate(filenames):
        if i in valid_indices:
            results.append({"Filename": filename, "Output": results_map[i]})
        else:
            results.append({"Filename": filename, "Output": "❌ Failed to load image"})

    df = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    csv_path = "output/batch_results.csv"
    df.to_csv(csv_path, index=False)
    return df, csv_path


def create_ui():
    with gr.Blocks(
        title="Multimodal Instructional Assistant",
    ) as demo:

        gr.HTML("""
        <div class="title-block">
            <h1 style="font-size:2rem; font-weight:700; margin:0;">
                🎓 Multimodal Instructional Assistant
            </h1>
            <p class="subtitle">
                Powered by <strong>Microsoft Florence-2</strong> · 
                Upload an educational image to generate captions or study questions
            </p>
        </div>
        """)

        with gr.Tabs():
            # ── Tab 1: Interactive Demo ───────────────────────────────────
            with gr.Tab("🖼️ Interactive Demo"):
                gr.Markdown("Upload any diagram, chart, or educational image and click **Generate**.")
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            type="pil",
                            label="Upload Image",
                            height=320
                        )
                        mode_input = gr.Radio(
                            ["Caption", "Generate Questions"],
                            label="Output Mode",
                            value="Caption",
                            info="Caption → rich description · Generate Questions → study Q&A"
                        )
                        submit_btn = gr.Button(
                            "✨ Generate",
                            variant="primary",
                            elem_classes=["generate-btn"]
                        )

                    with gr.Column(scale=1):
                        output_text = gr.Textbox(
                            label="Output",
                            lines=14,
                            placeholder="Output will appear here after clicking Generate...",
                        )

                submit_btn.click(
                    fn=process_single,
                    inputs=[img_input, mode_input],
                    outputs=output_text,
                    api_name="generate"
                )

            # ── Tab 2: Batch Processing ───────────────────────────────────
            with gr.Tab("📦 Batch Processing"):
                gr.Markdown(
                    "Upload **multiple images** at once. Results are exported as a downloadable CSV."
                )
                with gr.Row():
                    file_input = gr.File(
                        file_count="multiple",
                        label="Upload Images",
                        file_types=["image"]
                    )
                    batch_mode = gr.Radio(
                        ["Caption", "Generate Questions"],
                        label="Output Mode",
                        value="Caption"
                    )

                batch_btn = gr.Button("⚡ Process All", variant="primary")

                with gr.Row():
                    batch_df = gr.Dataframe(
                        label="Results",
                        wrap=True
                    )
                    batch_file = gr.File(label="📥 Download CSV")

                batch_btn.click(
                    fn=process_batch,
                    inputs=[file_input, batch_mode],
                    outputs=[batch_df, batch_file]
                )

        gr.HTML("""
        <div style="text-align:center; margin-top:16px; color:#999; font-size:0.85rem;">
            Florence-2-large · Microsoft Research · 
            <a href="https://huggingface.co/microsoft/Florence-2-large" target="_blank">Model Card</a>
        </div>
        """)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock model for testing")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--token", type=str, help="Hugging Face Token (optional)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    init_handler(mock=args.mock, token=token)

    demo = create_ui()
    demo.launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
