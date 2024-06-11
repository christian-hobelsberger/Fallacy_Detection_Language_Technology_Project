# A Comparative Analysis of Prompting Techniques for Fallacy Detection

**This branch is a fork of the [MAFALDA repository](https://github.com/ChadiHelwe/MAFALDA).** It contains the code and resources for evaluating a baseline, Chain-of-Thought (CoT), Thread-of-Thought (ThoT), and simplified Knowledge Generation prompting techniques using the Level 2 annotation scheme based on the taxonomy proposed in the [MAFALDA paper](https://arxiv.org/pdf/2311.09761), using same sentence-level fallacy attribution method.

## Abstract
This study explores the effectiveness of different prompting techniques for fallacy detection, a task requiring com- plex reasoning, using three State-of-the- Art (SotA) LLMs: Llama3, Gemma 7B, and Mistral 7B. We evaluate five prompting techniques: a handcrafted baseline, Chain-of-Thought (CoT), Thread-of-Thought (ThoT), Knowledge Generation, and Self-Consistency, against the Multi-level Annotated Fallacy Dataset (MAFALDA). The study aims to deter- mine the impact of these techniques on LLM performance in fallacy detection, identifying the best combination of LLM and prompting technique according to the MAFALDA taxonomy. The findings highlight the individual strengths and limitations of each prompting technique and discuss future research directions.

The group report is available via the Brightspace submissions portal (restricted access).

## Installation
```bash
git clone -b MAFALDA https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project.git
cd Fallacy_Detection_Language_Technology_Project
pip install -r requirements.txt
```

## Run Experiments with Local Models

The following script will download a quantized version of the selected model in [GGUF format](https://huggingface.co/docs/hub/gguf) from [HugggingFace](huggingface.co). Some models may require an access token due to the original developer's policy, while others are sourced from third party repositories.

### with GPU
```bash
./run_with_gpu.sh
```

## Run Evaluation
```bash
./run_eval.sh
```

## Evaluation Summary

| **Model**  | **Prompting technique**  | **Precision (%)**  | **Recall (%)**  | **F1-Score (%)** |
|------------|---------------------------|--------------------|-----------------|------------------|
| **Llama3** |                           |                    |                 |                  |
|            | Baseline                  | **15.51**          | 44.63           | **21.25**        |
|            | Thread-of-Thought         | 10.79              | 53.33           | 15.87            |
|            | Chain-of-Thought          | 8.45               | **60.75**       | 12.92            |
|            | Generated Knowledge       | 6.24               | 58.08           | 10.95            |
| **Gemma**  |                           |                    |                 |                  |
|            | Baseline                  | 15.81              | 31.83           | 19.53            |
|            | Thread-of-Thought         | **23.44**          | 40.66           | **27.75**        |
|            | Chain-of-Thought          | 18.37              | 40.54           | 23.00            |
|            | Generated Knowledge       | 5.54               | **64.58**       | 9.99             |
| **Mistral**|                           |                    |                 |                  |
|            | Baseline                  | 19.54              | 39.79           | 24.39            |
|            | Thread-of-Thought         | **25.65**          | 43.54           | **30.11**        |
|            | Chain-of-Thought          | 19.20              | **54.13**       | 24.02            |
|            | Generated Knowledge       | 14.50              | 52.79           | 20.74            |


# Additional notes
- All models evaluated were the instruct-tuned variants.
- Due to logistical constraints the Self-Consistency prompting technique was not evaluated. While the Thread-of-Thought prompting technique was evaluated for both one-shot and few-shot versions, only the few-shot evaluation was reported.
- GitHub CoPilot and GPT-4o were used to suggest changes to the codebase.
- The final results were obtained with Python 3.11.9 using Apple Silicon M2 (Pro) chips either running locally or on a cloud-based bare-metal Scaleway instance.
