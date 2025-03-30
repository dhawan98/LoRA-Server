from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
from diffusers import FluxPipeline
import gc
import os
import subprocess
from typing import List
from huggingface_hub import login, list_repo_files, hf_hub_download
from lora_loading_patch import load_lora_into_transformer
from Crypto.Cipher import AES
import base64
import hashlib
import random
import string
from io import BytesIO
from datetime import datetime

app = FastAPI()

# === Config ===
MODEL_ID = "black-forest-labs/FLUX.1-dev"
CACHE_DIR = "./huggingface_cache"
GDRIVE_FOLDER = "gdrive1:output"
HF_TOKEN = ""
KEY = hashlib.sha256(b"same key as prompt").digest()

login(token=HF_TOKEN)
os.makedirs(CACHE_DIR, exist_ok=True)

# === Load base model ===
print("🔧 Loading FLUX...")
txt2img_pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
).to("cuda")
txt2img_pipe.__class__.load_lora_into_transformer = classmethod(load_lora_into_transformer)

loaded_adapters = {}  # adapter_name → weight

# === Utils ===
def decrypt_prompt(payload: str) -> str:
    data = base64.b64decode(payload)
    iv, tag, ciphertext = data[:12], data[12:28], data[28:]
    cipher = AES.new(KEY, AES.MODE_GCM, nonce=iv)
    return cipher.decrypt_and_verify(ciphertext, tag).decode()

def random_filename(extension="png"):
    return "img_" + ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + f".{extension}"

def stream_upload(image, remote_folder):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    remote_path = f"{remote_folder}/{random_filename()}"
    start = datetime.now()
    result = subprocess.run(
        ["rclone", "rcat", remote_path, "-P"],
        input=buffer.read(),
        capture_output=True,
        check=True
    )
    print(f"📤 Uploaded to {remote_path} in {(datetime.now() - start).total_seconds():.2f}s")
    return remote_path

def download_lora(repo_id: str):
    files = list_repo_files(repo_id)
    safetensors = next((f for f in files if f.endswith(".safetensors")), None)
    return hf_hub_download(repo_id=repo_id, filename=safetensors, cache_dir=CACHE_DIR)

def clear_loras():
    global loaded_adapters
    if not loaded_adapters:
        print("ℹ️ No adapters to clear.")
        return

    print("🧹 Deep-clearing all LoRAs from model...")
    try:
        # Walk every module in the model
        for name, module in txt2img_pipe.transformer.named_modules():
            # Clear PEFT config if present
            if hasattr(module, "peft_config"):
                module.peft_config.clear()
            # Disable adapters
            if hasattr(module, "disable_adapters"):
                module.disable_adapters()
            if hasattr(module, "active_adapter"):
                module.active_adapter = None
            # Remove adapter layers
            if hasattr(module, "base_model") and hasattr(module.base_model, "model"):
                for sub in module.base_model.model.modules():
                    if hasattr(sub, "disable_adapters"):
                        sub.disable_adapters()
                    if hasattr(sub, "peft_config"):
                        sub.peft_config.clear()

        # Finally, if set_adapters() has internal stack, clear that
        if hasattr(txt2img_pipe, "set_adapters"):
            txt2img_pipe.set_adapters([], [])

    except Exception as e:
        print(f"❌ Error during forced LoRA unload: {e}")

    loaded_adapters.clear()
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ LoRAs completely removed.")



def update_weights_only(repos, weights):
    for repo, weight in zip(repos, weights):
        if repo in loaded_adapters:
            loaded_adapters[repo] = weight
            print(f"🔁 Updated weight for {repo} → {weight}")
        else:
            print(f"⚠️ LoRA {repo} not loaded; skipping.")
    if loaded_adapters:
        txt2img_pipe.set_adapters(list(loaded_adapters.keys()), list(loaded_adapters.values()))
        torch.cuda.empty_cache()
        gc.collect()

# === API Models ===
class PromptItem(BaseModel):
    payload: str
    seed: int
    num_images: int = 1

class BatchRequest(BaseModel):
    batch: List[PromptItem]

class LoadLoRARequest(BaseModel):
    repos: List[str]
    weights: List[float]

# === Endpoints ===

@app.post("/generate_batch")
def generate_batch(req: BatchRequest):
    results = []
    for item in req.batch:
        try:
            prompt = decrypt_prompt(item.payload)
            generator = torch.Generator("cuda").manual_seed(item.seed)
            images = txt2img_pipe(
                prompt=prompt,
                width=1024,
                height=1024,
                guidance_scale=7.0,
                num_inference_steps=32,
                generator=generator,
                num_images_per_prompt=item.num_images
            ).images
            for i, image in enumerate(images):
                remote_path = stream_upload(image, GDRIVE_FOLDER)
                results.append({
                    "seed": item.seed,
                    "image_index": i,
                    "remote_path": remote_path
                })
        except Exception as e:
            results.append({
                "seed": item.seed,
                "error": str(e)
            })
    return {"message": "✅ Batch complete", "results": results}

@app.post("/load_loras")
def load_loras(req: LoadLoRARequest):
    for repo, weight in zip(req.repos, req.weights):
        if repo in loaded_adapters:
            print(f"ℹ️ Updating weight only: {repo}")
            loaded_adapters[repo] = weight
        else:
            try:
                path = download_lora(repo)
                txt2img_pipe.load_lora_weights(path, adapter_name=repo, low_cpu_mem_usage=True)
                loaded_adapters[repo] = weight
                print(f"✅ Loaded adapter: {repo}")
            except Exception as e:
                print(f"❌ Failed to load {repo}: {str(e)}")

    if loaded_adapters:
        txt2img_pipe.set_adapters(list(loaded_adapters.keys()), list(loaded_adapters.values()))
    torch.cuda.empty_cache()
    gc.collect()
    return {"message": "✅ LoRAs loaded/updated", "active": loaded_adapters}

@app.post("/update_weights")
def update_weights(req: LoadLoRARequest):
    try:
        update_weights_only(req.repos, req.weights)
        return {"message": "✅ Weights updated", "active": loaded_adapters}
    except Exception as e:
        return {"error": str(e)}

@app.post("/clear_loras")
def clear_loras_endpoint():
    try:
        clear_loras()
        return {"message": "🧼 All LoRAs cleared"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
def get_status():
    return {"active_loras": loaded_adapters}

@app.on_event("shutdown")
def shutdown_cleanup():
    clear_loras()
    txt2img_pipe.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
