import transformers
import torch
import sys
import os

model_id = sys.argv[1]
save_path = sys.argv[2]
hf_token = sys.argv[3]
device_type = sys.argv[4]

# Determine the best available device
if device_type == 'cuda':
    device = torch.device('cuda')
    print('Using CUDA device')
    quantization_config = {"load_in_4bit": True}
elif device_type == 'mps':
    device = torch.device('mps')
    print('Using MPS device')
    quantization_config = None
else:
    device = torch.device('cpu')
    print('Using CPU device')
    quantization_config = None

# Check if model is already downloaded
if not os.path.exists(save_path):
    model_kwargs = {
        "torch_dtype": torch.float16 if device.type in ['cuda', 'mps'] else torch.float32,
        "low_cpu_mem_usage": True,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        **model_kwargs
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model {model_id} saved to {save_path}")
else:
    print(f"Model {model_id} already exists at {save_path}")