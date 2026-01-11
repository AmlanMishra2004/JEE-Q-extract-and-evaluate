"""
Run Qwen3-VL-8B-Instruct locally on each page image of a PDF and save combined markdown.
Optimized for RTX 4060 8GB laptop GPU.
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
# Use moderate DPI - balance between quality and memory
pages = convert_from_path(PDF_PATH, dpi=200)
print(f"Converted {len(pages)} pages.")

# Save page images
page_image_paths = []
for i, pil_img in enumerate(pages, start=1):
    path = os.path.join(TMP_IMG_DIR, f"page_{i:03d}.png")
    # Moderate resizing
    max_size = 1024
    if max(pil_img.size) > max_size:
        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    pil_img.save(path, "PNG", optimize=True)
    page_image_paths.append(path)

# Clear the pages list to free memory
del pages
gc.collect()

print("Loading Qwen3-VL model and processor (may take a while)...")

# Optimized memory map for RTX 4060 8GB
# Leave ~1.5GB buffer for activations during generation
max_memory = {0: "6.5GiB", "cpu": "16GiB"}

print(f"Loading model with memory map: {max_memory}")

# Load with optimized CPU offloading
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    offload_folder="offload",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

print("Model loaded. Device map summary:")
gpu_layers = sum(1 for device in model.hf_device_map.values() if device == 0)
cpu_layers = sum(1 for device in model.hf_device_map.values() if device == 'cpu')
print(f"  GPU layers: {gpu_layers}")
print(f"  CPU layers: {cpu_layers}")

def run_on_image(pil_image, system_prompt, user_prompt, max_new_tokens=1536):
    # Clear cache before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
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

    # Move inputs to the first device
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
                num_beams=1,
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
        print(f"  Retrying with smaller image and tokens...")
        
        # Cleanup
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Retry with smaller parameters
        pil_image_small = pil_image.copy()
        pil_image_small.thumbnail((768, 768), Image.Resampling.LANCZOS)
        
        messages_retry = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image_small},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        
        inputs_retry = processor.apply_chat_template(
            messages_retry,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs_retry = {k: v.to(input_device) if isinstance(v, torch.Tensor) else v 
                       for k, v in inputs_retry.items()}
        
        try:
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs_retry,
                    max_new_tokens=1024,  # Reduced
                    do_sample=False,
                    use_cache=True,
                    num_beams=1,
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs_retry['input_ids'], generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            result = output_texts[0] if len(output_texts) else ""
        except:
            result = None
        finally:
            del inputs_retry, pil_image_small
            
    finally:
        # Aggressive cleanup
        if 'inputs' in locals():
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
print("Note: First page may take longer as model warms up.\n")

with open(OUT_MD, "w", encoding="utf-8") as f_out:
    f_out.write("# Combined output from Qwen3-VL-8B-Instruct\n\n")
    
    import time
    
    for idx, img_path in enumerate(page_image_paths, start=1):
        start_time = time.time()
        print(f"Page {idx}/{len(page_image_paths)} -> processing...", end=" ", flush=True)
        
        pil_img = Image.open(img_path).convert("RGB")
        
        try:
            out_text = run_on_image(pil_img, SYSTEM_PROMPT, USER_PROMPT, max_new_tokens=1536)
            
            if out_text is None:
                out_text = f"*Error: Could not process page {idx} due to memory constraints*"
                
        except Exception as e:
            out_text = f"*Error processing page {idx}: {e}*"
            print(f"\n  Error: {e}")
        finally:
            pil_img.close()
            del pil_img
        
        elapsed = time.time() - start_time
        print(f"done in {elapsed:.1f}s")
        
        # Write heading and model output
        f_out.write(f"## Page {idx}\n\n")
        f_out.write(out_text.strip() + "\n\n---\n\n")
        f_out.flush()
        
        # Progress update every 10 pages
        if idx % 10 == 0:
            avg_time = elapsed  # Simplified - you could track moving average
            est_remaining = (len(page_image_paths) - idx) * avg_time / 60
            print(f"  Progress: {idx}/{len(page_image_paths)} pages ({idx*100//len(page_image_paths)}%) - Est. {est_remaining:.1f} min remaining\n")

print(f"\nDone! Combined Markdown written to {OUT_MD}")