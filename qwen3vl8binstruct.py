"""
Run Qwen3-VL-8B-Instruct locally on each page image of a PDF and save combined markdown.
Uses aggressive CPU offloading for GPUs with limited VRAM (e.g., 8GB).
"""

import os
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import gc

PDF_PATH = "qp/physics2025unwaterremove.pdf"
OUT_MD = "md/physics2025qwen3vl8b.md"
TMP_IMG_DIR = "tmp_pdf_pages"

# Prompts (exactly as you provided)
SYSTEM_PROMPT = """You are Qwen3 VL 8B Instruct, a large language model from qwen.

Formatting Rules:
- Use Markdown **only when semantically appropriate**. Examples: inline code, code fences, tables, and lists.
- In assistant responses, format file names, directory paths, function names, and class names with backticks (`).
- For math: use \\( and \\) for inline expressions, and \\[ and \\] for display (block) math.
"""

USER_PROMPT = """convert the image(questions, options) to markdown format, if there is any image, describe it.
"""

# Set environment variable BEFORE any CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

os.makedirs(TMP_IMG_DIR, exist_ok=True)
os.makedirs("offload", exist_ok=True)

print("Converting PDF pages to images (this can take a while)...")
# Use lower DPI to reduce memory usage
pages = convert_from_path(PDF_PATH, dpi=150)  # Further reduced to 150
print(f"Converted {len(pages)} pages.")

# Save page images
page_image_paths = []
for i, pil_img in enumerate(pages, start=1):
    path = os.path.join(TMP_IMG_DIR, f"page_{i:03d}.png")
    # Resize if image is too large (more aggressive resizing)
    max_size = 896  # Reduced from 1024
    if max(pil_img.size) > max_size:
        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    pil_img.save(path, "PNG", optimize=True)
    page_image_paths.append(path)

# Clear the pages list to free memory
del pages
gc.collect()

print("Loading Qwen3-VL model and processor (may take a while)...")

# Create explicit device map with more aggressive CPU offloading
# This leaves only essential layers on GPU
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import get_balanced_memory

# Check available GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    # Reserve only 4GB for GPU, rest for CPU offload
    max_memory = {0: "4GiB", "cpu": "16GiB"}
else:
    max_memory = {"cpu": "16GiB"}

print(f"Loading model with memory map: {max_memory}")

# Load with aggressive CPU offloading
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory=max_memory,  # Limit GPU memory usage
    low_cpu_mem_usage=True,
    offload_folder="offload",
    offload_state_dict=True,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

print("Model loaded. Device map:")
if hasattr(model, 'hf_device_map'):
    for name, device in model.hf_device_map.items():
        print(f"  {name}: {device}")

def run_on_image(pil_image, system_prompt, user_prompt, max_new_tokens=1024):
    # Clear cache before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # Resize image more aggressively if needed
    max_dim = 768  # Further reduced
    if max(pil_image.size) > max_dim:
        pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    
    # build chat messages
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move inputs to the first device (could be CPU if heavily offloaded)
    input_device = next(model.parameters()).device
    inputs = {k: v.to(input_device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    # generation with memory-efficient settings
    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                num_beams=1,  # Greedy decoding (most memory efficient)
            )
        
        # trim input prefix to get only generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        result = output_texts[0] if len(output_texts) else ""
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM Error: {e}")
        result = None
    finally:
        # Aggressive cleanup
        del inputs
        if 'generated_ids' in locals():
            del generated_ids
        if 'generated_ids_trimmed' in locals():
            del generated_ids_trimmed
        if 'output_texts' in locals():
            del output_texts
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    return result

print("Processing pages with the model and writing combined markdown...")

with open(OUT_MD, "w", encoding="utf-8") as f_out:
    f_out.write("# Combined output from Qwen3-VL-8B-Instruct\n\n")
    
    for idx, img_path in enumerate(page_image_paths, start=1):
        print(f"Page {idx}/{len(page_image_paths)} -> sending to model...")
        pil_img = Image.open(img_path).convert("RGB")
        
        try:
            out_text = run_on_image(pil_img, SYSTEM_PROMPT, USER_PROMPT, max_new_tokens=1536)
            
            if out_text is None:
                # Retry with even smaller image
                print(f"  Retrying page {idx} with smaller image...")
                pil_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                out_text = run_on_image(pil_img, SYSTEM_PROMPT, USER_PROMPT, max_new_tokens=1024)
            
            if out_text is None:
                out_text = f"*Error: Could not process page {idx} due to memory constraints*"
                
        except Exception as e:
            out_text = f"*Error processing page {idx}: {e}*"
            print(f"  Error: {e}")
        finally:
            pil_img.close()
            del pil_img
        
        # Write heading and model output
        f_out.write(f"## Page {idx}\n\n")
        f_out.write(out_text.strip() + "\n\n---\n\n")
        f_out.flush()
        
        # Progress update
        if idx % 5 == 0:
            print(f"  Completed {idx}/{len(page_image_paths)} pages")

print(f"Done. Combined Markdown written to {OUT_MD}")