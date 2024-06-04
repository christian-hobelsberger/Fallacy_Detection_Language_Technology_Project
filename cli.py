import argparse

import os
import time

from src.classification_models.baseline_models import RandomModel, SilentModel
from src.classification_models.openai_based_models import ChatGPTModel
from src.classification_models.quantized_llama_based_models import (
    LLaMABasedQuantizedModel,
)
from src.evaluate import eval_dataset
from src.experiments_pipelines.pipelines import zero_or_few_shots_pipeline
from src.users_study_evaluation import users_study_evaluation
from src.utils import setup_logger

def run_experiment(
        model: str, size: str, quantization: str, prompt: str, level: int, n_gpu_layers: int = 0,
        ):
    
    # # For performance/load scheduling, locally.
    # time.sleep(20000)

    model = LLaMABasedQuantizedModel(
        model_size=size,
        model_name=model,
        quantization=quantization,
            n_gpu_layers=n_gpu_layers,
        )

    zero_or_few_shots_pipeline(
        model=model,
        dataset_path="datasets/gold_standard_dataset.jsonl",
        prompt=prompt,
        prediction_path=f"results/{prompt}/{model.model_name.split('/')[-1]}_level_{level}_results.jsonl",
        level=level,
    )

def models_evaluation():
    results_dir = "results"
    dataset_path = "datasets/gold_standard_dataset.jsonl"
    
    # List all immediate child directories of the results directory
    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        
        # Check if it is a directory
        if os.path.isdir(subdir_path):
            eval_dataset(dataset_path, subdir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--size", type=str, help="Model size")
    parser.add_argument("--quantization", type=str, help="Quantization")
    parser.add_argument("--prompt", type=str, help="Prompting technique")
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
            models_evaluation()
        elif args.humans_eval:
            # humans_evaluation()
            print('NOT IMPLEMENTED')
        else:
            if args.model[:3] == "gpt":
                # run_chatgpt_experiment(
                #     model_name=args.model, level=args.level, api_key=args.api_key,
                # )
                print('NOT IMPLEMENTED')
            elif args.model[:4] == "base":
                # run_baseline_experiment(
                #     model_name=args.model, level=args.level,
                # )
                print('NOT IMPLEMENTED')
            else:
                run_experiment(
                    model=args.model,
                    size=args.size,
                    quantization=args.quantization,
                    prompt=args.prompt,
                    level=args.level,
                    n_gpu_layers=args.n_gpu_layers,
                )

    except Exception as e:
        logger.error(e, exc_info=True)