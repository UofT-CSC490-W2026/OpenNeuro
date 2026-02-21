This project contains python scripts for serving large language models on modal.

To run the serving scripts:
```bash
cd modal-infra
uv run python3 -m modal setup
uv run modal deploy src/llm.py
uv run modal deploy src/vlm.py
```