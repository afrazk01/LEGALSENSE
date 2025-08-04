from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

def check_bitsandbytes_gpu_support():
    """Check if bitsandbytes has GPU support"""
    try:
        import bitsandbytes as bnb
        # Try to create a simple 4-bit tensor to test GPU support
        if torch.cuda.is_available():
            test_tensor = torch.randn(10, 10).cuda()
            quantized = bnb.nn.Linear4bit(10, 10, compute_dtype=torch.float16)
            return True
        return False
    except Exception:
        return False

def load_falcon_model():
    """Load optimized Falcon model with faster loading and memory efficiency"""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'falcon-rw-1b')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check if we can use quantization
    use_quantization = check_bitsandbytes_gpu_support()
    
    if use_quantization:
        # Configure quantization for faster loading and less memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Use 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None
    
    # Determine optimal device mapping
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        offload_folder=None  # Disable offloading for faster loading
    )
    
    # Optimize for inference
    model.eval()
    
    return tokenizer, model

def generate_answer(prompt, tokenizer, model):
    """Generate answer with optimized inference"""
    # Tokenize with optimized settings
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=2048, 
        truncation=True,
        padding=True
    )
    
    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with optimized settings
    with torch.no_grad():  # Disable gradient computation
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache for faster generation
            repetition_penalty=1.1  # Reduce repetition
        )
    
    # Decode and clean output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|assistant|>" in generated_text:
        return generated_text.split("<|assistant|>")[-1].strip()
    else:
        return generated_text.strip()