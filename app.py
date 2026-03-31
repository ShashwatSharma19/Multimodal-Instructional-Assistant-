import gradio as gr
import argparse
from model_handler import Florence2Handler, MockFlorenceHandler
import os
from PIL import Image
import pandas as pd

# Global handler
handler = None

def init_handler(mock=False, token=None):
    global handler
    if mock:
        print("Initializing in MOCK mode.")
        handler = MockFlorenceHandler()
    else:
        print("Initializing in REAL mode (Florence-2).")
        # Florence-2 doesn"t need a token strictly, but we pass device if needed
        handler = Florence2Handler()
    
    if not handler.load_model():
        print("Failed to load real model. Falling back to Mock for UI demo.")
        handler = MockFlorenceHandler()
        handler.load_model()

def process_single(image, mode):
    if image is None:
        return "Please upload an image."
    
    if mode == "Caption":
        return handler.caption(image)
    else:
        return handler.generate_questions(image)

def process_batch(files, mode):
    if not files:
        return None, None
    
    # Load all images first for batching
    images = []
    filenames = []
    
    for file_path in files:
        try:
            path = file_path if isinstance(file_path, str) else file_path.name
            image = Image.open(path)
            images.append(image)
            filenames.append(os.path.basename(path))
        except Exception as e:
            filenames.append(os.path.basename(str(file_path)))
            images.append(None)  # Placeholder for failed loads
    
    # Filter out failed loads
    valid_indices = [i for i, img in enumerate(images) if img is not None]
    valid_images = [images[i] for i in valid_indices]
    
    # Process batch using true tensor batching
    if mode == "Caption":
        batch_results = handler.caption_batch(valid_images)
    else:
        batch_results = handler.generate_questions_batch(valid_images)
    
    # Build results list with proper ordering
    results = []
    result_idx = 0
    for i, filename in enumerate(filenames):
        if i in valid_indices:
            results.append({"filename": filename, "output": batch_results[result_idx]})
            result_idx += 1
        else:
            results.append({"filename": filename, "output": "Error: Failed to load image"})
    
    df = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    csv_path = "output/batch_results.csv"
    df.to_csv(csv_path, index=False)
    return df, csv_path

def create_ui():
    with gr.Blocks(title="Multimodal Educational Assistant") as demo:
        gr.Markdown("# Multimodal Educational Assistant (florence)")
        gr.Markdown("Upload images to generate captions or ask questions.")
        
        with gr.Tabs():
            # Tab 1: Interactive
            with gr.Tab("Interactive Demo"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(type="pil", label="Upload Image")
                        mode_input = gr.Radio(["Caption", "Generate Questions"], label="Mode", value="Caption")
                        
                        submit_btn = gr.Button("Generate")
                    
                    with gr.Column():
                        output_text = gr.Textbox(label="Output", lines=8)
                
                submit_btn.click(process_single, inputs=[img_input, mode_input], outputs=output_text)

            # Tab 2: Batch Processing
            with gr.Tab("Batch Processing"):
                gr.Markdown("Upload multiple images to process them in bulk.")
                file_input = gr.File(file_count="multiple", label="Upload Images")
                batch_mode = gr.Radio(["Caption", "Generate Questions"], label="Mode", value="Caption")
                
                batch_btn = gr.Button("Process Batch")
                
                with gr.Row():
                    batch_df = gr.Dataframe(label="Results")
                    batch_file = gr.File(label="Download CSV")
                
                batch_btn.click(process_batch, inputs=[file_input, batch_mode], outputs=[batch_df, batch_file])

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock model for testing")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app")
    parser.add_argument("--token", type=str, help="Hugging Face Token")
    args = parser.parse_args()
    
    # Check for HF Token in args or env
    token = args.token or os.environ.get("HF_TOKEN")
    
    init_handler(mock=args.mock, token=token)
    
    demo = create_ui()
    demo.launch(share=args.share, server_name="0.0.0.0")
