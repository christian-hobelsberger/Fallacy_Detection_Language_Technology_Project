import argparse
from types import MethodType

import torch

from transformers import AutoTokenizer, AutoModel

from src.classification_models.baseline_models import RandomModel, SilentModel
from src.classification_models.openai_based_models import ChatGPTModel
from src.classification_models.quantized_llama_based_models import (
    LLaMABasedQuantizedModel,
)
from src.evaluate import eval_dataset
from src.experiments_pipelines.pipelines import zero_or_few_shots_pipeline
from src.users_study_evaluation import users_study_evaluation
from src.utils import setup_logger

HF_TOKEN = "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"

# Determine the best available device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def dynamic_predict(self, text):
    inputs = self.tokenizer(text, return_tensors='pt')
    outputs = self.generate(**inputs)
    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_original_experiment(
    model: str, size: str, quantization: str, level: int, n_gpu_layers: int = 0,
):
    model = LLaMABasedQuantizedModel(
        model_size=size,
        model_name=model,
        quantization=quantization,
        n_gpu_layers=n_gpu_layers,
    )

    zero_or_few_shots_pipeline(
        model=model,
        dataset_path="datasets/gold_standard_dataset.jsonl",
        prediction_path=f"results/{model.model_name}_{size}_{quantization}_level_{level}_results.jsonl",
        level=level,
    )

def run_new_experiment(
        model: str, size: str, quantization: str, level: int, n_gpu_layers: int = 0,
        ):
    # model = AutoModel.from_pretrained(model_id,
    #                                   torch_dtype = torch.float16 if device.type in ['cuda', 'mps'] else torch.float32,
    #                                   low_cpu_mem_usage=True,
    #                                   token=HF_TOKEN).to(device)
    # model.tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model.predict = MethodType(dynamic_predict, model)

    model = LLaMABasedQuantizedModel(
        model_size=size,
        model_name=model,
        quantization=quantization,
        n_gpu_layers=n_gpu_layers,
    )

    zero_or_few_shots_pipeline(
        model=model,
        dataset_path="datasets/gold_standard_dataset.jsonl",
        prediction_path=f"results/{model.split('/')[-1]}_level_{level}_results.jsonl",
        level=level,
    )

def run_chatgpt_experiment(
    model_name: str, level: int, api_key: str,
):
    model = ChatGPTModel(model_name=model_name, api_key=api_key)
    zero_or_few_shots_pipeline(
        model=model,
        dataset_path="datasets/gold_standard_dataset.jsonl",
        prediction_path=f"results/{model.model_name}_level_{level}_results.jsonl",
        level=level,
    )

def run_baseline_experiment(
    model_name: str, level: int,
):
    if model_name == "base-silent":
        model = SilentModel(model_name=model_name)
    elif model_name == "base-random":
        model = RandomModel(model_name=model_name)
    else:
        raise Exception("Unknown baseline model")

    zero_or_few_shots_pipeline(
        model=model,
        dataset_path="datasets/gold_standard_dataset.jsonl",
        prediction_path=f"results/{model.model_name}_level_{level}_results.jsonl",
        level=level,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--size", type=str, help="Model size")
    parser.add_argument("--quantization", type=str, help="Quantization")
    parser.add_argument("--level", type=int, help="Level")
    parser.add_argument(
        "--n_gpu_layers", type=int, help="Number of GPUs for layer", default=0
    )
    parser.add_argument("--models_eval", help="Evaluate", action="store_true")
    parser.add_argument("--humans_eval", help="Evaluate", action="store_true")
    parser.add_argument("--api_key", type=str, help="OpenAI API Key")

    args = parser.parse_args()

    logger_filename = (
        f"logs/{args.model}_{args.size}_{args.quantization}_level_{args.level}.log"
    )
    logger = setup_logger(logger_filename)
    try:
        if args.models_eval:
            # models_evaluation()
            print('lol')
        elif args.humans_eval:
            # humans_evaluation()
            print('lol')
        else:
            if args.model[:3] == "gpt":
                # run_chatgpt_experiment(
                #     model_name=args.model, level=args.level, api_key=args.api_key,
                # )
                print('lol')
            elif args.model[:4] == "base":
                # run_baseline_experiment(
                #     model_name=args.model, level=args.level,
                # )
                print('lol')
            else:
                run_new_experiment(
                    model=args.model,
                    size=args.size,
                    quantization=args.quantization,
                    level=args.level,
                    n_gpu_layers=args.n_gpu_layers,
                )

    except Exception as e:
        logger.error(e, exc_info=True)