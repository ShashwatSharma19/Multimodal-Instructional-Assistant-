from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from PIL import Image
import os
import time
import warnings


class TechnicalCaptionEnhancer:
    """
    CPU-based Phi-3-mini text LLM.
    Takes OCR labels extracted by Florence-2 and produces coherent
    educational explanations — handles any diagram with readable text.
    Loaded lazily on first use so app startup stays fast.
    """

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    _SYS = "<|system|>\nYou are a concise educational assistant.\n</s>\n"
    _USR = "<|user|>\n{prompt}\n</s>\n<|assistant|>\n"

    def __init__(self):
        self._pipe = None

    def load(self):
        """Eagerly load TinyLlama at app startup. Must be called before serving requests."""
        if self._pipe is not None:
            return
        print("Loading TinyLlama on GPU (~10s)...")
        self._pipe = pipeline(
            "text-generation",
            model=self.MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        print("TinyLlama ready.")

    def _load(self):
        """Internal: ensures model is loaded (fallback safety call)."""
        self.load()

    def _run(self, prompt: str, max_new_tokens: int = 120) -> str:
        self._load()
        full_prompt = self._SYS + self._USR.format(prompt=prompt)
        out = self._pipe(
            full_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False,
        )
        return out[0]["generated_text"].strip()

    def enhance_caption(self, ocr_text: str, visual_caption: str = "") -> str:
        prompt = (
            f"A diagram has these visible labels: \"{ocr_text}\".\n"
            f"Visual description: \"{visual_caption}\".\n"
            "In 2-3 sentences, explain what type of diagram this is and what "
            "educational concept it illustrates. Be specific."
        )
        return self._run(prompt, max_new_tokens=120)

    def generate_questions(self, ocr_text: str, visual_caption: str = "") -> str:
        prompt = (
            f"A diagram has these labels: \"{ocr_text}\".\n"
            f"Visual description: \"{visual_caption}\".\n"
            "Write exactly 5 numbered study questions a student should answer "
            "after studying this diagram."
        )
        return self._run(prompt, max_new_tokens=150)


class Florence2Handler:
    def __init__(self, model_id="microsoft/Florence-2-large", device=None, 
                 use_quantization=True, use_flash_attention=True, use_compile=True):
        """
        Optimized Florence-2 Handler with performance enhancements.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to run on (auto-detected if None)
            use_quantization: Enable 4-bit quantization for faster inference
            use_flash_attention: Enable Flash Attention 2 / SDPA
            use_compile: Enable torch.compile for JIT optimization
        """
        self.model_id = model_id
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None
        self.processor = None

        # Technical caption enhancer — lazy-loaded on first diagram with OCR text
        self.enhancer = TechnicalCaptionEnhancer()

        # Optimization flags
        # Quantization disabled - causes hallucinations and slowdowns with Florence-2
        self.use_quantization = False  # Forced off for Florence-2
        self.use_flash_attention = use_flash_attention and self.device == "cuda"
        # torch.compile disabled - incompatible with Florence-2's custom model
        self.use_compile = False  # Forced off for Florence-2
    
    def _get_attention_implementation(self):
        """Determine best available attention implementation."""
        # Florence-2 custom model lacks SDPA support flag in current transformers
        return 'eager'

    def _pad_to_square(self, image: Image.Image) -> Image.Image:
        """
        Pad image to square with white background.
        Florence-2's DaViT vision backbone requires square feature maps;
        non-square inputs cause 'only support square feature maps' AssertionError.
        """
        w, h = image.size
        if w == h:
            return image
        max_dim = max(w, h)
        square = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        square.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
        return square
    
    def _get_quantization_config(self):
        """Create 4-bit quantization config."""
        if not self.use_quantization:
            return None
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
        )
    
    def load_model(self):
        try:
            print(f"Loading processor for {self.model_id}...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            
            # Bypass HuggingFace's AST import checker that strictly requires flash_attn
            # even when the code gracefully falls back to eager attention.
            import transformers.dynamic_module_utils
            transformers.dynamic_module_utils.check_imports = lambda filename: []

            
            print(f"Loading model {self.model_id} on {self.device}...")
            print(f"  - Quantization: {'4-bit' if self.use_quantization else 'disabled'}")
            print(f"  - Flash Attention: {'enabled' if self.use_flash_attention else 'disabled'}")
            print(f"  - torch.compile: {'enabled' if self.use_compile else 'disabled'}")
            
            # Prepare model loading kwargs
            model_kwargs = {
                "trust_remote_code": True,
                "attn_implementation": self._get_attention_implementation(),
            }
            
            # Add quantization if enabled
            quant_config = self._get_quantization_config()
            if quant_config:
                model_kwargs["quantization_config"] = quant_config
                model_kwargs["device_map"] = "auto"  # Required for quantization
            else:
                # Manual dtype and device handling for non-quantized
                model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            # Move to device if not using quantization (quantization uses device_map)
            if not quant_config:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Apply torch.compile for JIT optimization
            if self.use_compile:
                try:
                    print("Compiling model with torch.compile (this may take a moment on first run)...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("Model compiled successfully.")
                except Exception as e:
                    warnings.warn(f"torch.compile failed, falling back to eager mode: {e}")
            
            print("Florence-2 Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate(self, image, task_prompt, text_input=None):
        """Generate output for a single image."""
        if not self.model or not self.processor:
            return "Model not loaded."
            
        try:
            if image is None:
                return "Error: No image provided"
                
            image = image.convert("RGB")
            image = self._pad_to_square(image)  # Florence-2 requires square feature maps
            
            if text_input is None:
                prompt = task_prompt
            else:
                prompt = task_prompt + text_input
                
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            if "pixel_values" not in inputs or inputs["pixel_values"] is None:
                return "Error: Failed to process image"
            
            # Move to device and convert to model dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False,  # Disable KV cache — EncoderDecoderCache incompatibility with Florence-2 custom generate()
                )
                
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            
            return parsed_answer
            
        except Exception as e:
            import traceback
            return f"Error during generation: {e}\nTraceback: {traceback.format_exc()}"

    def generate_batch(self, images, task_prompt, text_inputs=None):
        """
        TRUE TENSOR BATCHING: Process multiple images in parallel on GPU.
        
        Args:
            images: List of PIL Images
            task_prompt: Task prompt to use for all images
            text_inputs: Optional list of text inputs (one per image)
        
        Returns:
            List of parsed results
        """
        if not self.model or not self.processor:
            return ["Model not loaded."] * len(images)
        
        if not images:
            return []
        
        try:
            start_time = time.time()
            
            # Convert all images to RGB and pad to square
            images = [self._pad_to_square(img.convert("RGB")) for img in images]
            image_sizes = [(img.width, img.height) for img in images]
            
            # Build prompts
            if text_inputs is None:
                prompts = [task_prompt] * len(images)
            else:
                prompts = [task_prompt + (ti or "") for ti in text_inputs]
            
            # Batch process with processor - creates batched tensors
            inputs = self.processor(
                text=prompts, 
                images=images, 
                return_tensors="pt",
                padding=True,  # Pad to same length for batching
            )
            
            # Move to device and convert dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
            
            print(f"Processing batch of {len(images)} images...")
            
            with torch.no_grad():
                # Single batched forward pass for all images
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False,  # Disable KV cache — EncoderDecoderCache incompatibility with Florence-2 custom generate()
                )
            
            # Decode all at once
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
            
            # Post-process each result
            results = []
            for i, text in enumerate(generated_texts):
                parsed = self.processor.post_process_generation(
                    text,
                    task=task_prompt,
                    image_size=image_sizes[i]
                )
                if isinstance(parsed, dict):
                    results.append(parsed.get(task_prompt, str(parsed)))
                else:
                    results.append(str(parsed))
            
            elapsed = time.time() - start_time
            print(f"Batch processing completed in {elapsed:.2f}s ({elapsed/len(images):.2f}s per image)")
            
            return results
            
        except Exception as e:
            import traceback
            error_msg = f"Error during batch generation: {e}\nTraceback: {traceback.format_exc()}"
            return [error_msg] * len(images)

    def caption(self, image):
        """
        Generate caption for a single image.
        Pipeline:
          1. Run Florence-2 OCR to extract visible text labels.
          2. Run Florence-2 visual caption for scene context.
          3. If meaningful OCR text found → send both to Phi-3-mini for an
             accurate, context-aware educational explanation.
          4. If no OCR text (purely visual image) → return visual caption directly.
        """
        # Step 1: OCR
        ocr_result = self.generate(image, "<OCR>")
        ocr_text = ""
        if isinstance(ocr_result, dict):
            ocr_text = ocr_result.get("<OCR>", "").strip()
        else:
            ocr_text = str(ocr_result).strip()

        # Step 2: Visual caption
        visual_task = "<MORE_DETAILED_CAPTION>"
        visual_result = self.generate(image, visual_task)
        visual_caption = ""
        if isinstance(visual_result, dict):
            visual_caption = visual_result.get(visual_task, str(visual_result)).strip()
        else:
            visual_caption = str(visual_result).strip()

        # Step 3: Enhance if OCR has meaningful content (>= 3 words)
        if ocr_text and len(ocr_text.split()) >= 3:
            try:
                return self.enhancer.enhance_caption(ocr_text, visual_caption)
            except Exception as e:
                warnings.warn(f"Phi-3-mini enhancement failed, falling back: {e}")

        # Step 4: Fallback — return Florence-2 visual caption
        return visual_caption
    
    def caption_batch(self, images):
        """Generate captions for multiple images using true batching."""
        task = "<MORE_DETAILED_CAPTION>"
        return self.generate_batch(images, task)

    def generate_questions(self, image, num_questions=5):
        """
        Generate educational questions about the image.
        Pipeline:
          1. Run Florence-2 OCR to extract visible text labels.
          2. Run Florence-2 visual caption for scene context.
          3. If meaningful OCR text found → send both to Phi-3-mini to generate
             specific, contextual study questions.
          4. If no OCR text → generate generic visual questions from caption.
        """
        try:
            # Step 1: OCR
            ocr_result = self.generate(image, "<OCR>")
            ocr_text = ""
            if isinstance(ocr_result, dict):
                ocr_text = ocr_result.get("<OCR>", "").strip()
            else:
                ocr_text = str(ocr_result).strip()

            # Step 2: Visual caption
            caption_result = self.generate(image, "<MORE_DETAILED_CAPTION>")
            visual_caption = ""
            if isinstance(caption_result, dict):
                visual_caption = caption_result.get("<MORE_DETAILED_CAPTION>", str(caption_result)).strip()
            else:
                visual_caption = str(caption_result).strip()

            # Step 3: Phi-3-mini generates specific questions if OCR has content
            if ocr_text and len(ocr_text.split()) >= 3:
                try:
                    return self.enhancer.generate_questions(ocr_text, visual_caption)
                except Exception as e:
                    warnings.warn(f"Phi-3-mini question generation failed, falling back: {e}")

            # Step 4: Fallback — generic questions from visual caption
            if visual_caption:
                return (
                    "1. What is the main subject of this image?\n"
                    "2. Describe the key components shown in this image.\n"
                    "3. What educational concept does this image illustrate?\n"
                    "4. What relationships can you identify in this diagram?\n"
                    "5. How would you explain this image to a student?"
                )

            return "Could not generate questions. Please try a different image."

        except Exception as e:
            import traceback
            return f"Error generating questions: {e}\nTraceback: {traceback.format_exc()}"
    
    def generate_questions_batch(self, images, num_questions=5):
        """Generate questions for multiple images."""
        # For question generation, we need OCR + caption per image
        # This is inherently sequential due to the multi-step logic
        results = []
        for img in images:
            results.append(self.generate_questions(img, num_questions))
        return results

    def qa(self, image, question):
        """Answer a question about an image."""
        image = image.convert("RGB")
        prompt = question
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        model_dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text


# Mock class for CPU/Sandbox verification if real model fails to load
class MockFlorenceHandler:
    def __init__(self):
        self.loaded = False
        
    def load_model(self):
        print("MOCK MODEL LOADED (Sandbox mode)")
        self.loaded = True
        return True
        
    def caption(self, image):
        return "This is a mock caption for the image."
    
    def caption_batch(self, images):
        return [f"Mock caption for image {i+1}" for i in range(len(images))]
    
    def generate_questions(self, image, num_questions=5):
        return "1. Mock question 1?\n2. Mock question 2?\n3. Mock question 3?"
    
    def generate_questions_batch(self, images, num_questions=5):
        return [self.generate_questions(img, num_questions) for img in images]
        
    def qa(self, image, question):
        return f"This is a mock answer to '{question}'."
