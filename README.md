# Fallacy Detection

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


## Usage instructions

Clone the repository

```
git clone https://github.com/christian-hobelsberger/Fallacy_Detection_Language_Technology_Project.git
```
### Promptings

### Data Postprocessing

### Evaluation
