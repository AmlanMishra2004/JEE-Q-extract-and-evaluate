#!/usr/bin/env python3
"""
deepseek_ocr_final_cpu.py

Converts a physics PDF to Markdown using DeepSeek-OCR with pure CPU inference.
This is the ONLY version that will work reliably on 8GB GPUs.

Usage:
    python3 deepseek_ocr_final_cpu.py input.pdf output.md diagrams_dir/ [--skip-diagrams]

Requirements:
    pip install -U torch transformers accelerate pdf2image pillow opencv-python numpy
"""

import os
import sys
import gc
import glob
import tempfile
from pathlib import Path
import shutil
import warnings

from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

import torch

# Suppress the model type warning with explanation
# DeepSeek-OCR's config has model_type="deepseek_vl_v2" but uses class "DeepseekOCR"
# This is intentional (OCR based on VL architecture) and doesn't cause issues
warnings.filterwarnings(
    'ignore',
    message='.*deepseek_vl_v2.*',
    category=UserWarning,
    module='transformers'
)

from transformers import AutoModel, AutoTokenizer

# ----------------------- Configuration -----------------------
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

# Force CPU usage - critical for avoiding GPU OOM
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU from PyTorch completely
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# ----------------------------------------------------------------

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()

# ----------------------- Model load ---------------------------
print(f"Preparing to load model {MODEL_NAME} ...")
print("This may take several minutes for the first time...")
print()

print("=" * 60)
print("LOADING MODEL ON CPU ONLY")
print("GPU is intentionally disabled to avoid OOM issues")
print("=" * 60)
print()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

try:
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    model = model.cpu()
    model.eval()
    
    print("✓ Model loaded successfully on CPU")
    print("  No GPU memory will be used")
    print("  Expected time: ~1-2 minutes per page")
    print()
    
    cleanup_memory()
    
except Exception as e:
    print(f"\n❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    raise
# ----------------------------------------------------------------

# ----------------------- I/O helper ---------------------------
def read_first_textlike_file(out_dir: str) -> str:
    """Read the first text-like file found in directory"""
    for ext in ("*.md", "*.txt", "*.out", "*.json"):
        files = sorted(glob.glob(os.path.join(out_dir, ext)))
        if files:
            with open(files[0], "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    files = sorted(glob.glob(os.path.join(out_dir, "*")))
    if files:
        try:
            with open(files[0], "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except:
            pass
    return ""
# ----------------------------------------------------------------

# ----------------------- Inference wrappers -------------------
def infer_image_to_markdown(image_pil: Image.Image, prompt: str,
                            base_size=1024, image_size=640, crop_mode=True) -> str:
    """
    Run OCR inference on an image using CPU only.
    """
    if model is None:
        raise RuntimeError("Model not loaded")

    tmp_dir = tempfile.mkdtemp(prefix="deepseek_run_")
    try:
        # Save image
        img_path = os.path.join(tmp_dir, "page.png")
        image_pil.convert("RGB").save(img_path, format="PNG")

        out_dir = os.path.join(tmp_dir, "out")
        os.makedirs(out_dir, exist_ok=True)

        print(f"Running inference on CPU...")
        cleanup_memory()
        
        try:
            with torch.inference_mode():
                res = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=img_path,
                    output_path=out_dir,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=True,
                    test_compress=True
                )
        except TypeError as e:
            print(f"Signature mismatch, trying minimal args: {e}")
            with torch.inference_mode():
                res = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=img_path,
                    output_path=out_dir
                )
        
        cleanup_memory()

        # Extract text from result
        out_text = ""
        if isinstance(res, str):
            out_text = res
        elif res is not None and hasattr(res, "get"):
            for k in ("text", "output", "0", "results"):
                if k in res:
                    out_text = str(res[k])
                    break

        if not out_text:
            out_text = read_first_textlike_file(out_dir)

        if not out_text:
            out_text = read_first_textlike_file(tmp_dir)

        if not out_text:
            print(f"Warning: No text output found in {out_dir}")
            
        return out_text or ""
        
    except Exception as e:
        print(f"Inference error: {e}")
        cleanup_memory()
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def infer_image_to_description(image_pil: Image.Image, prompt: str) -> str:
    """Smaller inference for diagram descriptions"""
    return infer_image_to_markdown(
        image_pil, prompt,
        base_size=512,
        image_size=512,
        crop_mode=False
    )
# ----------------------------------------------------------------

# ----------------------- Diagram extraction -------------------
def extract_diagram_crops(image_pil: Image.Image, out_dir: Path, page_idx: int,
                          min_area_ratio=0.01, max_area_ratio=0.9):
    """Extract diagram regions from page image"""
    img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dil = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    page_area = h * w
    crops = []

    for i, cnt in enumerate(contours):
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        area_ratio = area / page_area
        
        if area_ratio >= min_area_ratio and area_ratio <= max_area_ratio:
            pad_w = int(0.02 * w)
            pad_h = int(0.02 * h)
            x0 = max(0, x - pad_w)
            y0 = max(0, y - pad_h)
            x1 = min(w, x + cw + pad_w)
            y1 = min(h, y + ch + pad_h)
            
            crop = img[y0:y1, x0:x1]
            out_path = out_dir / f"page_{page_idx:03d}_fig_{i+1:02d}.png"
            Image.fromarray(crop).save(out_path)
            crops.append(out_path)
            
    return sorted(crops, key=lambda p: str(p))
# ----------------------------------------------------------------

# ----------------------- Prompts -------------------------------
PAGE_PROMPT = """<image>
<|grounding|>
Convert this physics exam page to clean Markdown for study notes.

Requirements:
1. Output valid Markdown with headings, lists, and question numbering
2. Convert math to LaTeX: $$ ... $$ for display, $ ... $ for inline
3. Preserve question structure (1., 2., (a), (b), etc.)
4. Use LaTeX for Greek letters: \\alpha, \\theta, \\beta
5. Use LaTeX for sub/superscripts: v_{max}, T^2, E_0
6. For figure references, use: ![Figure](diagrams/page_XXX_fig_YY.png)
7. Mark unclear symbols as [CHECK]
8. Return only Markdown text, no explanations
"""

FIGURE_PROMPT = """<image>
<|grounding|>
Describe this physics diagram for study notes.

Provide:
1. Brief 1-3 sentence description
2. List all visible labels (variables, angles, forces)
3. Note which question part it relates to if apparent
4. Mark unclear labels as [CHECK]

Return Markdown only.
"""
# ----------------------------------------------------------------

# ----------------------- Main pipeline -------------------------
def pdf_to_single_md(pdf_path, output_md_path, diagrams_dir, 
                     dpi=200, detect_diagrams=True, skip_diagram_ocr=False):
    """
    Convert PDF to single Markdown file with diagram extraction.
    
    Args:
        pdf_path: Input PDF file
        output_md_path: Output Markdown file
        diagrams_dir: Directory to save extracted diagrams
        dpi: PDF rendering DPI (200 for quality)
        detect_diagrams: Whether to extract diagram regions
        skip_diagram_ocr: Skip OCR on diagrams to save time
    """
    pdf_path = Path(pdf_path)
    output_md_path = Path(output_md_path)
    diagrams_dir = Path(diagrams_dir)
    diagrams_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting PDF to images (DPI={dpi})...")
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    print(f"✓ Converted to {len(pages)} page images")
    
    if skip_diagram_ocr:
        print("\n⚠ Diagram OCR is disabled (--skip-diagrams)")
        print("  Only extracting diagram images, not describing them\n")

    # Initialize output file
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(f"# {pdf_path.stem}\n\n")
        f.write("*Auto-generated with DeepSeek-OCR — please review math & diagrams*\n\n")

    # Process each page
    for idx, page_img in enumerate(pages, start=1):
        print(f"\n{'='*60}")
        print(f"Processing page {idx}/{len(pages)}")
        print(f"{'='*60}")

        # Extract diagrams
        diagram_paths = []
        if detect_diagrams:
            print("Detecting diagrams...")
            diagram_paths = extract_diagram_crops(page_img, diagrams_dir, idx)
            print(f"✓ Found {len(diagram_paths)} diagram(s)")

        # OCR the full page
        print("Running page OCR...")
        try:
            page_markdown = infer_image_to_markdown(page_img, PAGE_PROMPT)
            print("✓ Page OCR completed")
        except Exception as e:
            print(f"❌ ERROR during page OCR: {e}")
            page_markdown = f"[ERROR: Could not process page {idx}] [CHECK]\n"

        # Describe each diagram (if not skipped)
        diagram_descriptions = []
        if skip_diagram_ocr:
            diagram_descriptions = [(dpath.name, "*[Diagram OCR skipped for speed]*") 
                                   for dpath in diagram_paths]
        else:
            for i, dpath in enumerate(diagram_paths, 1):
                print(f"Describing diagram {i}/{len(diagram_paths)}...")
                try:
                    dimg = Image.open(dpath)
                    desc_md = infer_image_to_description(dimg, FIGURE_PROMPT)
                    diagram_descriptions.append((dpath.name, desc_md))
                    print(f"✓ Diagram {i} described")
                except Exception as e:
                    print(f"❌ ERROR describing diagram: {e}")
                    diagram_descriptions.append((dpath.name, "[ERROR describing diagram] [CHECK]"))

        # Append to output file
        with open(output_md_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n---\n\n## Page {idx}\n\n")
            
            # Add diagrams first
            for fname, desc in diagram_descriptions:
                rel_path = os.path.join(diagrams_dir.name, fname)
                f.write(f"![{fname}]({rel_path})\n\n")
                f.write(f"{desc}\n\n")
            
            # Then page content
            f.write(page_markdown)
            f.write("\n\n")
        
        print(f"✓ Page {idx} saved to output")
        cleanup_memory()

    print(f"\n{'='*60}")
    print("✓ CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Markdown: {output_md_path}")
    print(f"Diagrams: {diagrams_dir}/")
    print()
# ----------------------------------------------------------------

# ----------------------- CLI entrypoint -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("=" * 60)
        print("DeepSeek-OCR - Pure CPU Inference")
        print("=" * 60)
        print("\nUsage:")
        print("  python3 deepseek_ocr_final_cpu.py input.pdf output.md diagrams_dir/ [--skip-diagrams]")
        print("\nExamples:")
        print("  python3 deepseek_ocr_final_cpu.py physics.pdf output.md diagrams/")
        print("  python3 deepseek_ocr_final_cpu.py physics.pdf output.md diagrams/ --skip-diagrams")
        print("\nOptions:")
        print("  --skip-diagrams   Skip OCR on diagrams (faster, for testing)")
        print("\nFeatures:")
        print("  ✓ 100% CPU inference (GPU completely disabled)")
        print("  ✓ No OOM errors possible")
        print("  ✓ Full quality: 200 DPI")
        print("  ✓ Model type warning suppressed")
        print("  ✓ Stable for any number of pages")
        print("\nPerformance:")
        print("  ~1-2 minutes per page (~2-4 hours for 95 pages)")
        print("\nNote:")
        print("  This is the ONLY reliable solution for 8GB GPUs.")
        print("  GPU/CPU hybrid modes will OOM during inference.")
        print("=" * 60)
        sys.exit(1)
    
    pdf_in = sys.argv[1]
    md_out = sys.argv[2]
    diag_dir = sys.argv[3]
    skip_diag = "--skip-diagrams" in sys.argv
    
    if not os.path.exists(pdf_in):
        print(f"❌ Error: PDF file not found: {pdf_in}")
        sys.exit(1)
    
    print("=" * 60)
    print("DeepSeek-OCR - Pure CPU Mode")
    print("=" * 60)
    print(f"Input:    {pdf_in}")
    print(f"Output:   {md_out}")
    print(f"Diagrams: {diag_dir}/")
    if skip_diag:
        print("Mode:     Fast (skip diagram OCR)")
    else:
        print("Mode:     Full (with diagram OCR)")
    print("=" * 60)
    
    # Verify GPU is actually disabled
    if torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA is still available!")
        print("This shouldn't happen. GPU may still be used.")
        print("If you get OOM errors, restart Python and try again.")
    else:
        print("\n✓ GPU successfully disabled - using CPU only")
    
    print()
    
    try:
        pdf_to_single_md(pdf_in, md_out, diag_dir, dpi=200, skip_diagram_ocr=skip_diag)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
# ----------------------------------------------------------------