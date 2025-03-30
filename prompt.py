import requests
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64
import hashlib
import random

KEY = hashlib.sha256(b"My Super Key is None of Your Super Key").digest()

PROMPTS = [
    "Divyanshu, as Aquaman",
    "Divyanshu, as god of water, holding a trident in a holy pose",
    "Divyanshu, underwater hero with glowing aqua armor and a futuristic trident, riding a giant manta ray through the deep ocean, commanding sea creatures, epic cinematic shot"
]

def encrypt_prompt(prompt: str, key: bytes) -> str:
    iv = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    ciphertext, tag = cipher.encrypt_and_digest(prompt.encode())
    return base64.b64encode(iv + tag + ciphertext).decode()

batch_payloads = []
for prompt in PROMPTS:
    seed = random.randint(0, 2**32 - 1)
    encrypted = encrypt_prompt(prompt, KEY)
    batch_payloads.append({
        "payload": encrypted,
        "seed": seed,
        "num_images": 3
    })

response = requests.post(
    "http://localhost:8501/generate_batch",
    json={"batch": batch_payloads}
)

print(response.json())
