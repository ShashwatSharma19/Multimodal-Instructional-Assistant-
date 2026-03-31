import os
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
import argparse

def train(args):
    # 1. Setup Model ID
    # Using the mix-224 model which contains ScienceQA tasks as requested
    model_id = args.model_id
    
    print(f"Loading processor for {model_id}...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading processor: {e}")
        return

    # 3. Load Dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset("imagefolder", data_dir=args.dataset_path, split="train")
    
    # 4. Preprocessing
    def collate_fn(examples):
        # PaliGemma uses prompts like "caption en" or "question en ..."
        # Our dummy data has "prompt" and "completion".
        
        inputs_text = [example["prompt"] for example in examples]
        targets_text = [example["completion"] for example in examples]
        images = [example["image"].convert("RGB") for example in examples]
        
        # Use the 'suffix' argument to handle input (prompt) and label (completion) automatically.
        # The processor creates 'labels' where the prompt tokens are masked with -100.
        model_inputs = processor(
            text=inputs_text,
            images=images,
            suffix=targets_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        
        return model_inputs

    # 5. Load Model
    device_map = "auto"
    quantization_config = None
    
    if torch.cuda.is_available():
        print("CUDA detected. Using 4-bit quantization.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        print("CUDA not detected. Using CPU mode (slow, mostly for verification).")
        device_map = {"": "cpu"}

    print(f"Loading model {model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=device_map
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 6. Apply LoRA (PEFT)
    print("Applying PEFT/LoRA...")
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=args.max_steps, 
        learning_rate=2e-4,
        logging_steps=1,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        save_strategy="steps",
        save_steps=10,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none" 
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn
    )

    # 9. Train
    print("Starting training...")
    trainer.train()
    
    # 10. Save
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/Florence-2-large")
    parser.add_argument("--dataset_path", type=str, default="data/dummy_dataset")
    parser.add_argument("--output_dir", type=str, default="output/lora_adapter")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    
    train(args)
