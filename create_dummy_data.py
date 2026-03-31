import os
import json
from PIL import Image, ImageDraw
import random

def create_dummy_data(output_dir="data/dummy_dataset", num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    
    colors = ["red", "blue", "green", "yellow", "purple"]
    shapes = ["rectangle", "ellipse"]
    
    for i in range(num_samples):
        color = random.choice(colors)
        shape = random.choice(shapes)
        filename = f"image_{i}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Create image
        img = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(img)
        
        if shape == "rectangle":
            draw.rectangle([50, 50, 174, 174], fill=color)
        else:
            draw.ellipse([50, 50, 174, 174], fill=color)
            
        img.save(filepath)
        
        # Create different task types for PaliGemma
        # 1. Captioning
        data.append({
            "file_name": filename,
            "prompt": "caption en",
            "completion": f"A {color} {shape} on a white background."
        })
        
        # 2. Q&A
        data.append({
            "file_name": filename,
            "prompt": f"question en What color is the {shape}?",
            "completion": color
        })
        
    # Save metadata.jsonl
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Created {num_samples} images and {len(data)} metadata entries in {output_dir}")

if __name__ == "__main__":
    create_dummy_data()
