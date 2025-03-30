# LoRA Server 

Dynamic LoRA loading, unloading, and weight adjustment through an HTTP API – all without restarting your model server.

## Features

-  **Dynamically load/unload** LoRA adapters at runtime
-  **Adjust LoRA scaling weights** on the fly
-  Simple **FastAPI**-based HTTP interface
-  Built on top of Hugging Face `diffusers` and `peft`
-  Fully tested with minimal test case included
