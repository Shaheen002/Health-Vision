from transformers.utils import cached_file, TRANSFORMERS_CACHE
import shutil
import os

# Get the cache directory
cache_dir = TRANSFORMERS_CACHE if TRANSFORMERS_CACHE else os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")

# Define the model directory
model_dir = os.path.join(cache_dir, "models--EleutherAI--gpt-neo-2.7B")

# Check if the model directory exists and delete it
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
    print(f"Deleted {model_dir}")
else:
    print(f"{model_dir} does not exist")