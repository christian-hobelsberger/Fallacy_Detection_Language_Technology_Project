# Fallacy Detection Using Different Prompting Techniques

## Abstract

Insert Abstract here

## Results
### Multi-label with Spans

| Model  | Prompting            | Precision (%) | Recall (%) | F1-Score (%) |
|--------|----------------------|---------------|------------|--------------|
| Llama3 | Thread-of-Thought    | 30.61         | 28.09      | 28.19        |
|        | Chain-of-Thought     | 20.50         | 17.24      | 17.94        |
|        | Generation Knowledge | **31.25**         | **30.35**      | **30.58**        |
|        | Self-Consistency     | 19.25         | 18.29      | 18.06        |
| Gemma  | Thread-of-Thought    | 10.16         | 10.09      | 10.12        |
|        | Chain-of-Thought     | **25.09**         | **27.85**      | **25.45**        |
|        | Generation Knowledge | 23.44         | 22.40      | 22.32        |
|        | Self-Consistency     | 24.88         | 25.62      | 24.51        |
| Mistral| Thread-of-Thought    | 29.12         | 26.27      | 26.87        |
|        | Chain-of-Thought     | 31.55         | 30.58      | 30.01        |
|        | Generation Knowledge | **34.32**         | 29.68      | 30.06        |
|        | Self-Consistency     | 32.75         | **32.85**      | **31.71**        |

### Multi-label without Spans

| Model  | Prompting            | Precision (%) | Recall (%) | F1-Score (%) |
|--------|----------------------|---------------|------------|--------------|
| Llama3 | Thread-of-Thought    | **40.04**     | **37.87**  | **37.92**    |
|        | Chain-of-Thought     | 24.61         | 22.08      | 22.57        |
|        | Generation Knowledge | 31.83         | 30.92      | 31.20        |
|        | Self-Consistency     | 23.42         | 19.43      | 20.18        |
| Gemma  | Thread-of-Thought    | 16.90         | 19.91      | 17.53        |
|        | Chain-of-Thought     | 25.85         | **31.25**  | **26.72**    |
|        | Generation Knowledge | **27.25**     | 25.17      | 25.47        |
|        | Self-Consistency     | 25.10         | 25.43      | 19.57        |
| Mistral| Thread-of-Thought    | 37.05         | **38.01**  | **36.57**    |
|        | Chain-of-Thought     | **38.32**     | 35.79      | 35.89        |
|        | Generation Knowledge | 37.08         | 34.50      | 35.02        |
|        | Self-Consistency     | 31.54         | 29.94      | 30.05        |

### Multi-class

| Model  | Prompting            | Precision (%) | Recall (%) | F1-Score (%) |
|--------|----------------------|---------------|------------|--------------|
| Llama3 | Thread-of-Thought    | **47.50**     | **34.80**      | **40.17**    |
|        | Chain-of-Thought     | 43.00         | 33.08      | 37.39        |
|        | Generation Knowledge | 42.00         | 30.22      | 35.15        |
|        | Self-Consistency     | 27.00         | 18.56      | 22.00        |
| Gemma  | Thread-of-Thought    | **52.00**     | **38.52**  | **44.26**    |
|        | Chain-of-Thought     | 26.00         | 17.99      | 21.27        |
|        | Generation Knowledge | 50.00         | 38.02      | 43.20        |
|        | Self-Consistency     | 34.50         | 24.82      | 28.87        |
| Mistral| Thread-of-Thought    | 57.50     | 44.06  | 49.89    |
|        | Chain-of-Thought     | **65.50**     | **52.61**  | **58.35**    |
|        | Generation Knowledge | 48.00         | 35.42      | 40.76        |
|        | Self-Consistency     | 37.50         | 28.85      | 32.61        |

### Usage instructions

Clone the repository

```
git clone https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project.git
```
The [requirements.txt](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/blob/main/requirements.txt) file lists the dependencies necessary to run the project, ensuring that all required libraries and packages are installed.

### Repository Structure

This repository is organized into several key directories, each serving a distinct purpose in the project workflow:

- [Data](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/tree/main/Data): Contains raw and processed data used throughout the project.

#### Data Pre/post-processing
- [Postprocessing](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/tree/main/Postprocessing): Includes scripts for refining and transforming model outputs.
- [Preprocessing](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/tree/main/Preprocessing): Contains scripts for preparing raw data for model evaluation.

#### Models
- [Models](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/tree/main/Models): Stores the code nessecary for deploying the models.

#### Promptings
- [Prompting](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/tree/main/Prompting): Contains files with different prompting techniques used on the models.

#### Evaluation
- [Evaluation](https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project/tree/main/Evaluation): Houses scripts and tools for assessing model performance.

