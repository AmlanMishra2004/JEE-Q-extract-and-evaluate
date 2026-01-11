from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
import torch

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model with quantization
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "parani01/Fine-tuned-physics-VLM-on-LoRA-and-QLoRA")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def solve_physics_problem(image_path='circle.png', question='describe the diagram. Mention all points, angles, arrows, shapes.'):
    from PIL import Image
    
    # Load the image
    image = Image.open(image_path)
    
    # Prepare the conversation format for Qwen2-VL
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": question
                },
            ],
        }
    ]
    
    # Apply chat template and prepare inputs
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Process inputs (image + text)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    # Move inputs to the same device as model
    inputs = inputs.to(model.device)
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    
    # Decode the generated tokens
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return response

# Actually CALL the function (not just print it)
result = solve_physics_problem()
print(result)