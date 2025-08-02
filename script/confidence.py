import torch
import numpy as np

def calculate_confidence(logits):
    probs = torch.softmax(logits, dim=-1)
    top_probs, _ = torch.topk(probs, 5)
    return torch.mean(top_probs).item()