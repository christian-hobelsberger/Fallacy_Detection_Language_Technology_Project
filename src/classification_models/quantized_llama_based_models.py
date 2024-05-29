import logging
import os
# from urllib import request
import re
from huggingface_hub import hf_hub_download

def extract_hf_url_details(url):
    # Regex pattern to extract details from the URL
    pattern = re.compile(
        r"https://huggingface\.co/(?P<repo_owner>[^/]+)/(?P<repo_name>[^/]+)/resolve/(?P<revision>[^/]+)/(?P<file_path>.+)"
    )
    match = pattern.match(url)
    if match:
        return match.group("repo_owner"), match.group("repo_name"), match.group("revision"), match.group("file_path")
    else:
        raise ValueError("URL does not match the expected Hugging Face URL pattern")

def download_from_hf(url, local_filename):
    try:
        repo_owner, repo_name, revision, file_path = extract_hf_url_details(url)
        repo_id = f"{repo_owner}/{repo_name}"
        downloaded_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            revision=revision,
            local_dir=".",
            local_dir_use_symlinks=False
        )
        # Rename the downloaded file to the desired local filename
        os.rename(downloaded_file_path, local_filename)
        print(f"Downloaded {local_filename} successfully.")
    except Exception as e:
        print(f"Error downloading file: {e}")

import torch
from llama_cpp import Llama
from transformers import pipeline

logger = logging.getLogger("MafaldaLogger")

LLAMA_BASED_MODELS = {
    "7B": {
        "4-bit": {
            "LLaMA2": {
                "repo": "TheBloke/Llama-2-7B-GGUF",
                "model_file_name": "llama-2-7b.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "LLaMA2-Chat": {
                "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
                "model_file_name": "llama-2-7b-chat.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "LLaMA2-Instruct": {
                "repo": "TheBloke/Llama-2-7B-32K-Instruct-GGUF",
                "model_file_name": "llama-2-7b-32k-instruct.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GGUF/resolve/main/llama-2-7b-32k-instruct.Q4_K_M.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "Falcon": {
                "repo": "NikolayKozloff/falcon-7b-GGUF",
                "model_file_name": "falcon-7b-Q4_0-GGUF.gguf",
                "model_url": "https://huggingface.co/NikolayKozloff/falcon-7b-GGUF/resolve/main/falcon-7b-Q4_0-GGUF.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "Mistral": {
                "repo": "TheBloke/Mistral-7B-v0.1-GGUF",
                "model_file_name": "mistral-7b-v0.1.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf",
                "instruction_begin": "<s> Instruction:",
                "instruction_end": "",
            },
            "Mistral-Instruct": {
                "repo": "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
                "model_file_name": "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
                "instruction_begin": "<s>[INST]",
                "instruction_end": "[/INST]",
            },
            "Vicuna": {
                "repo": "TheBloke/vicuna-7B-v1.5-16K-GGUF",
                "model_file_name": "vicuna-7b-v1.5-16k.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/TheBloke/vicuna-7B-v1.5-16K-GGUF/resolve/main/vicuna-7b-v1.5-16k.Q4_K_M.gguf",
                "instruction_begin": "USER:",
                "instruction_end": "ASSISTANT:",
            },
            "WizardLM": {
                "repo": "TheBloke/WizardLM-7B-uncensored-GGUF",
                "model_file_name": "WizardLM-7B-uncensored.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGUF/resolve/main/WizardLM-7B-uncensored.Q4_K_M.gguf",
                "instruction_begin": "USER:",
                "instruction_end": "ASSISTANT:",
            },
            "Zephyr": {
                "repo": "TheBloke/zephyr-7B-alpha-GGUF",
                "model_file_name": "zephyr-7b-alpha.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/resolve/main/zephyr-7b-alpha.Q4_K_M.gguf",
                "instruction_begin": "<|system|> </s> <|user|>",
                "instruction_end": "</s> <|assistant|>",
            },
        },
        "8-bit": {
            "LLaMA2": {
                "repo": "TheBloke/Llama-2-7B-GGUF",
                "model_file_name": "llama-2-7b.Q8_0.gguf",
                "model_url": "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q8_0.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "LLaMA2-Chat": {
                "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
                "model_file_name": "llama-2-7b-chat.Q8_0.gguf",
                "model_url": "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q8_0.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "LLaMA2-Instruct": {
                "repo": "TheBloke/Llama-2-7B-32K-Instruct-GGUF",
                "model_file_name": "llama-2-7b-32k-instruct.Q8_0.gguf",
                "model_url": "https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GGUF/resolve/main/llama-2-7b-32k-instruct.Q8_0.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "Falcon": {
                "repo": "NikolayKozloff/falcon-7b-GGUF",
                "model_file_name": "falcon-7b-Q8_0-GGUF.gguf",
                "model_url": "https://huggingface.co/NikolayKozloff/falcon-7b-GGUF/resolve/main/falcon-7b-Q8_0-GGUF.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "Mistral": {
                "repo": "TheBloke/Mistral-7B-v0.1-GGUF",
                "model_file_name": "mistral-7b-v0.1.Q8_0.gguf",
                "model_url": "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q8_0.gguf",
                "instruction_begin": "<s> Instruction:",
                "instruction_end": "",
            },
            "Mistral-Instruct": {
                "repo": "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
                "model_file_name": "Mistral-7B-Instruct-v0.3.Q8_0.gguf",
                "model_url": "https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q8_0.gguf",
                "instruction_begin": "<s>[INST]",
                "instruction_end": "[/INST]",
            },
            "Vicuna": {
                "repo": "TheBloke/vicuna-7B-v1.5-16K-GGUF",
                "model_file_name": "vicuna-7b-v1.5-16k.Q8_0.gguf",
                "model_url": "https://huggingface.co/TheBloke/vicuna-7B-v1.5-16K-GGUF/resolve/main/vicuna-7b-v1.5-16k.Q8_0.gguf",
                "instruction_begin": "USER:",
                "instruction_end": "ASSISTANT:",
            },
            "WizardLM": {
                "repo": "TheBloke/WizardLM-7B-uncensored-GGUF",
                "model_file_name": "WizardLM-7B-uncensored.Q8_0.gguf",
                "model_url": "https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGUF/resolve/main/WizardLM-7B-uncensored.Q8_0.gguf",
                "instruction_begin": "USER:",
                "instruction_end": "ASSISTANT:",
            },
            "Zephyr": {
                "repo": "TheBloke/zephyr-7B-alpha-GGUF",
                "model_file_name": "zephyr-7b-alpha.Q8_0.gguf",
                "model_url": "https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/resolve/main/zephyr-7b-alpha.Q8_0.gguf",
                "instruction_begin": "<|system|> </s> <|user|>",
                "instruction_end": "</s> <|assistant|>",
            },
        },
    },
    "8B": {
        "4-bit": {
            "LLaMA3-Instruct": {
                "repo": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
                "model_file_name": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
        },
        "8-bit": {
            "LLaMA3-Instruct": {
                "repo": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
                "model_file_name": "Meta-Llama-3-8B-Instruct.Q8_0.gguf",
                "model_url": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            }
        }
    },
    "13B": {
        "4-bit": {
            "LLaMA2": {
                "repo": "TheBloke/Llama-2-13B-GGUF",
                "model_file_name": "llama-2-13b.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_K_M.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "LLaMA2-Chat": {
                "repo": "TheBloke/Llama-2-13B-Chat-GGUF",
                "model_file_name": "llama-2-13b-chat.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "Vicuna": {
                "repo": "TheBloke/vicuna-13B-v1.5-16K-GGUF",
                "model_file_name": "vicuna-13b-v1.5-16k.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/TheBloke/vicuna-13B-v1.5-16K-GGUF/resolve/main/vicuna-13b-v1.5-16k.Q4_K_M.gguf",
                "instruction_begin": "USER:",
                "instruction_end": "ASSISTANT:",
            },
            "WizardLM": {
                "repo": "TheBloke/WizardLM-13B-V1.2-GGUF",
                "model_file_name": "wizardlm-13b-v1.2.Q4_K_M.gguf",
                "model_url": "https://huggingface.co/TheBloke/WizardLM-13B-V1.2-GGUF/resolve/main/wizardlm-13b-v1.2.Q4_K_M.gguf",
                "instruction_begin": "USER:",
                "instruction_end": "ASSISTANT:",
            },
        },
        "8-bit": {
            "LLaMA2": {
                "repo": "TheBloke/Llama-2-13B-GGUF",
                "model_file_name": "llama-2-13b.Q8_0.gguf",
                "model_url": "https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q8_0.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "LLaMA2-Chat": {
                "repo": "TheBloke/Llama-2-13B-Chat-GGUF",
                "model_file_name": "llama-2-13b-chat.Q8_0.gguf",
                "model_url": "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q8_0.gguf",
                "instruction_begin": "[INST]",
                "instruction_end": "[/INST]",
            },
            "Vicuna": {
                "repo": "TheBloke/vicuna-13B-v1.5-16K-GGUF",
                "model_file_name": "vicuna-13b-v1.5-16k.Q8_0.gguf",
                "model_url": "https://huggingface.co/TheBloke/vicuna-13B-v1.5-16K-GGUF/resolve/main/vicuna-13b-v1.5-16k.Q8_0.gguf",
                "instruction_begin": "USER:",
                "instruction_end": "ASSISTANT:",
            },
            "WizardLM": {
                "repo": "TheBloke/WizardLM-13B-V1.2-GGUF",
                "model_file_name": "wizardlm-13b-v1.2.Q8_0.gguf",
                "model_url": "https://huggingface.co/TheBloke/WizardLM-13B-V1.2-GGUF/resolve/main/wizardlm-13b-v1.2.Q8_0.gguf",
                "instruction_begin": "USER:",
                "instruction_end": "ASSISTANT:",
            },
        },
    },
}

def initialize_model(tmp_model_name: str, url: str, n_gpu_layers: int = 0):
    model_path = f"models/{tmp_model_name}"
    if not os.path.exists(model_path):
        logger.info("Downloading model...")
        # request.urlretrieve(url, model_path)
        download_from_hf(url, model_path)
    model = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=n_gpu_layers, seed=42)
    return model


class LLaMABasedQuantizedModel:
    def __init__(
        self,
        model_size: str,
        model_name: str,
        quantization: int,
        n_gpu_layers: int = 0,
    ) -> None:
        try:
            self.model = None
            self.model_name = model_name
            self.model_size = model_size
            self.quantization = quantization
            self.repo = LLAMA_BASED_MODELS[model_size][quantization][model_name]["repo"]
            self.model_url = LLAMA_BASED_MODELS[model_size][quantization][model_name][
                "model_url"
            ]
            self.model_file_name = LLAMA_BASED_MODELS[model_size][quantization][
                model_name
            ]["model_file_name"]
            self.instruction_begin = LLAMA_BASED_MODELS[model_size][quantization][
                model_name
            ]["instruction_begin"]
            self.instruction_end = LLAMA_BASED_MODELS[model_size][quantization][
                model_name
            ]["instruction_end"]

            self.model = initialize_model(
                self.model_file_name, self.model_url, n_gpu_layers=n_gpu_layers
            )
        except:
            logger.error(
                f"Error: Model not found for {model_size} {quantization} {model_name}"
            )

    def predict(self, text: str, max_tokens: int = 10, echo: bool = False):
        out = self.model(text, max_tokens=max_tokens, echo=echo)
        return out["choices"][0]["text"]
