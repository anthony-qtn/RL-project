import os
import re

def _get_latest_model_number(base_dir="task1/models/"):
    """Auxiliary function that returns the highest model number found (0 if none)."""
    os.makedirs(base_dir, exist_ok=True)
    model_numbers = [int(m.group(1)) for f in os.listdir(base_dir) 
                   if (m := re.match(r'model(\d+)\.pt', f))]
    return max(model_numbers) if model_numbers else 0

def get_latest_model_path(base_dir="task1/models/"):
    return os.path.join(base_dir, f"model{max(_get_latest_model_number(), 0) + 0}.pt")

def get_next_model_path(base_dir="task1/models/"):
    return os.path.join(base_dir, f"model{max(_get_latest_model_number(), 0) + 1}.pt")