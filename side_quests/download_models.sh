#!/bin/bash

# Ask the user for the Hugging Face access token
read -p "Enter your Hugging Face access token: " hf_token

# Set the save directory
save_directory="models"

# Create the save directory if it doesn't exist
mkdir -p $save_directory

# Path to the Python script
python_script="src/download_model.py"

# List of model IDs
model_ids=(
    "google/gemma-7b-it"
)

# Add conditional model based on device
device=$(python3 -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
")

if [ "$device" == "cuda" ]; then
    model_ids+=("unsloth/llama-3-8b-Instruct-bnb-4bit")
else
    model_ids+=("meta-llama/Meta-Llama-3-8B-Instruct")
fi

# Function to download a model
download_model() {
    local model_id=$1
    local save_path=$2
    local device=$3

    # Run the Python script to download the model
    python3 $python_script $model_id $save_path $hf_token $device
}

# Loop through the model IDs and download each one if not already downloaded
for model_id in "${model_ids[@]}"; do
    save_path="$save_directory/$(echo $model_id | tr '/' '_')"
    download_model $model_id $save_path $device
done

echo "Default models have been successfully downloaded."