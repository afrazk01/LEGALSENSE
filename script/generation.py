from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_falcon_model():
    tokenizer = AutoTokenizer.from_pretrained("U:\\LEGALSENSE\\models\\falcon-rw-1b")
    model = AutoModelForCausalLM.from_pretrained(
        "U:\\LEGALSENSE\\models\\falcon-rw-1b",
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    return tokenizer, model

def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.5,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()