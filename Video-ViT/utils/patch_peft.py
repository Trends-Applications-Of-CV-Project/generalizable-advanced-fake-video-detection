import os

file_path = ".venv/lib/python3.12/site-packages/peft/tuners/lora/model.py"
#file_path = "venv_Lollo/lib/python3.12/site-packages/peft/tuners/lora/model.py"
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, "r") as f:
    content = f.read()

target_import = "from transformers.modeling_layers import GradientCheckpointingLayer"
replacement_import = "class GradientCheckpointingLayer: pass # Patched by Agent"

if target_import in content:
    content = content.replace(target_import, replacement_import)
    print("Patched import.")
else:
    print("Import already patched or missing.")

target_check = "if isinstance(layer, GradientCheckpointingLayer) and layer.gradient_checkpointing:"
replacement_check = "if getattr(layer, 'gradient_checkpointing', False): # Patched by Agent"

if target_check in content:
    content = content.replace(target_check, replacement_check)
    print("Patched isinstance check.")
else:
    print("Check already patched or missing.")

with open(file_path, "w") as f:
    f.write(content)

print("Patching complete.")
