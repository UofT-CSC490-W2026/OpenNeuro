"""
Synthetic SFT dataset generator for delta-captioning task.

Loads videos from HF dataset, samples 2 consecutive 1-sec windows @ 4fps,
uses Gemini 2.5 Flash Lite (via OpenRouter) to produce:
  - Window 1: full caption
  - Window 2: delta caption (only new observations)

Outputs an HF Dataset with a defined Features schema, pushed to Hub or saved to disk.

Usage:
    export OPENROUTER_API_KEY=<your-key>
    python synth_sft_gen.py \
        --dataset_name "Wild-Heart/Disney-VideoGeneration-Dataset" \
        --output_path ./synth_delta_caption_dataset \
        --push_to_hub my-org/delta-caption-disney \
        --max_examples 0        # 0 = process all
"""

import os
import io
import base64
import random
import argparse
import time
import json
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
import datasets
from datasets import Dataset, Features, Value, Sequence, Image as HFImage
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VLM_FPS = 4
WINDOW_DURATION = 2.0  # seconds per window
FRAMES_PER_WINDOW = int(WINDOW_DURATION * VLM_FPS)  # 4
TOTAL_DURATION = 2.0 * WINDOW_DURATION  # 2 seconds total
TOTAL_FRAMES = int(TOTAL_DURATION * VLM_FPS)  # 8

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "google/gemini-2.5-flash-lite"

# Prompts
PROMPT_WINDOW1 = (
    "You are an advanced real-time vision module for blind people. "
    "Given the current observation, use short phrases to caption what you see, include movement and composition if needed. "
    "Keep it short, efficient, real-time."
)

PROMPT_WINDOW2_TEMPLATE = (
    "You are an advanced real-time vision module for blind people. "
    "Given the current observation history, generate an efficient delta caption "
    "only introducing new observations that were not mentioned in the history context. "
    "Keep it efficient, no redundant information. DO NOT REPEAT anything that is already mentioned.\n\n"
    "History context:\n{context}"
)


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
DATASET_FEATURES = Features({
    # Video identification
    "video_id": Value("string"),

    # Sampling metadata
    "start_time": Value("float64"),
    "fps": Value("float64"),
    "source_fps": Value("float64"),
    "window_duration": Value("float64"),

    # Frames stored as HF Image (PNG bytes internally)
    "frames_window1": Sequence(HFImage()),  # 4 PIL images
    "frames_window2": Sequence(HFImage()),  # 4 PIL images

    # Captions
    "caption_window1": Value("string"),     # full caption
    "caption_window2_delta": Value("string"),  # delta caption

    # Conversation format (for SFT training)
    "conversations": [
        {
            "role": Value("string"),
            "content": Value("string"),
        }
    ],
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_pil_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """Encode a PIL image to a base64 data-URI string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def decode_video_frames(video_input, total_duration: float, vlm_fps: int) -> tuple:
    """
    Decode `total_duration` seconds of video at `vlm_fps` from a random offset.
    Returns (frames_np [T,H,W,3], source_fps, start_time).
    Handles both torchvision VideoReader objects and raw bytes.
    """
    import torch
    # If it's already a VideoReader-like object
    if hasattr(video_input, 'get_metadata') and hasattr(video_input, 'seek'):
        metadata = video_input.get_metadata()
        v_meta = metadata.get('video', {})
        fps = v_meta.get('fps', [30.0])
        if isinstance(fps, (list, tuple, torch.Tensor)):
            fps = float(fps[0])
        duration = v_meta.get('duration', [0.0])
        if isinstance(duration, (list, tuple, torch.Tensor)):
            duration = float(duration[0])

        total_frames = int(total_duration * vlm_fps)
        start_time = random.uniform(0, max(0, duration - total_duration))
        end_time = start_time + total_duration

        target_ts = np.linspace(start_time, max(start_time, end_time - 0.01), total_frames)
        frames = []
        video_input.seek(start_time)
        target_idx = 0

        for frame in video_input:
            pts = float(frame.get('pts', -1.0))
            if pts >= target_ts[target_idx]:
                img = frame['data'].permute(1, 2, 0).cpu().numpy()
                frames.append(img)
                target_idx += 1
                if target_idx >= len(target_ts):
                    break

        if len(frames) < total_frames:
            # Pad by repeating last frame
            while len(frames) < total_frames:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        return np.stack(frames[:total_frames]), fps, start_time

    # If it's bytes / file-like — use PyAV
    import av
    if isinstance(video_input, bytes):
        container = av.open(io.BytesIO(video_input))
    else:
        container = av.open(video_input)

    stream = container.streams.video[0]
    source_fps = float(stream.average_rate) if stream.average_rate else 30.0
    duration = float(stream.duration * stream.time_base) if stream.duration else 0.0

    total_frames = int(total_duration * vlm_fps)
    start_time = random.uniform(0, max(0, duration - total_duration))
    end_time = start_time + total_duration

    target_ts = np.linspace(start_time, max(start_time, end_time - 0.01), total_frames)
    frames = []
    target_idx = 0

    # Seek
    container.seek(int(start_time * av.time_base), any_frame=False, backward=True)

    for frame in container.decode(video=0):
        pts_sec = float(frame.pts * stream.time_base)
        if pts_sec >= target_ts[target_idx]:
            arr = frame.to_ndarray(format='rgb24')
            frames.append(arr)
            target_idx += 1
            if target_idx >= len(target_ts):
                break

    container.close()

    if len(frames) < total_frames:
        while len(frames) < total_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    return np.stack(frames[:total_frames]), source_fps, start_time


def numpy_to_pil_list(frames_np: np.ndarray) -> List[Image.Image]:
    """Convert [T,H,W,3] uint8 numpy array to list of PIL Images."""
    out = []
    for i in range(frames_np.shape[0]):
        arr = frames_np[i]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out.append(Image.fromarray(arr))
    return out


def call_openrouter(
    api_key: str,
    frames: List[Image.Image],
    text_prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
    retries: int = 5,
    backoff: float = 2.0,
) -> str:
    """
    Call OpenRouter chat completions with images + text.
    Returns the assistant reply text.
    """
    # Build content array: images first, then text
    content = []
    for img in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_pil_to_base64(img)},
        })
    content.append({"type": "text", "text": text_prompt})

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def build_conversations(
    caption_w1: str,
    caption_w2_delta: str,
) -> List[Dict[str, str]]:
    """
    Build the SFT conversation turns.
    Turn 1: user asks for caption of window 1 → assistant responds with full caption.
    Turn 2: user asks for delta caption of window 2 → assistant responds with delta.
    """
    return [
        {
            "role": "user",
            "content": PROMPT_WINDOW1,
        },
        {
            "role": "assistant",
            "content": caption_w1,
        },
        {
            "role": "user",
            "content": PROMPT_WINDOW2_TEMPLATE.format(context=caption_w1),
        },
        {
            "role": "assistant",
            "content": caption_w2_delta,
        },
    ]


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(
    dataset_name: str,
    api_key: str,
    max_examples: int = 0,
    save_path: Optional[str] = None,
    push_to_hub: Optional[str] = None,
    resume_path: Optional[str] = None,
):
    """
    Stream through the source dataset, call Gemini for each video,
    and accumulate rows into a new HF Dataset.
    """
    print(f"Loading source dataset: {dataset_name} (streaming)")
    src = datasets.load_dataset(dataset_name, split="train", streaming=True)
    src_iter = iter(src)

    # Resume support: load partial results if they exist
    rows: List[Dict[str, Any]] = []
    done_ids: set = set()
    if resume_path and os.path.isdir(resume_path):
        print(f"Resuming from {resume_path}")
        partial = Dataset.load_from_disk(resume_path)
        for row in partial:
            rows.append(row)
            done_ids.add(row["video_id"])
        print(f"  Loaded {len(rows)} existing examples")

    idx = 0
    pbar = tqdm(desc="Generating", unit="ex")

    while True:
        if 0 < max_examples <= len(rows):
            break

        try:
            sample = next(src_iter)
        except StopIteration:
            break

        video_id = str(sample.get("id", sample.get("video_id", f"row_{idx}")))
        idx += 1

        if video_id in done_ids:
            continue

        # ------ Decode frames ------
        video_data = sample["video"]  # could be VideoReader, bytes, path, etc.
        frames_np, source_fps, start_time = decode_video_frames(
            video_data, TOTAL_DURATION, VLM_FPS
        )

        if frames_np.shape[0] < TOTAL_FRAMES:
            print(f"[skip] video_id={video_id}: only {frames_np.shape[0]} frames")
            continue

        mid = FRAMES_PER_WINDOW
        frames_w1_np = frames_np[:mid]
        frames_w2_np = frames_np[mid:]

        frames_w1_pil = numpy_to_pil_list(frames_w1_np)
        frames_w2_pil = numpy_to_pil_list(frames_w2_np)

        # ------ Window 1: full caption ------
        caption_w1 = call_openrouter(api_key, frames_w1_pil, PROMPT_WINDOW1)
        
        print(f"[{video_id}] Window 1 caption: {caption_w1}")

        # ------ Window 2: delta caption ------
        prompt_w2 = PROMPT_WINDOW2_TEMPLATE.format(context=caption_w1)
        caption_w2 = call_openrouter(api_key, frames_w2_pil, prompt_w2)

        # ------ Build row ------
        conversations = build_conversations(caption_w1, caption_w2)

        row = {
            "video_id": video_id,
            "start_time": start_time,
            "fps": float(VLM_FPS),
            "source_fps": float(source_fps),
            "window_duration": WINDOW_DURATION,
            "frames_window1": frames_w1_pil,
            "frames_window2": frames_w2_pil,
            "caption_window1": caption_w1,
            "caption_window2_delta": caption_w2,
            "conversations": conversations,
        }
        rows.append(row)
        done_ids.add(video_id)
        pbar.update(1)

        # Periodic checkpoint every 50 examples
        if save_path and len(rows) % 50 == 0:
            _save_checkpoint(rows, save_path)

    pbar.close()

    if not rows:
        print("No examples generated!")
        return

    # ------ Finalize dataset ------
    ds = Dataset.from_list(rows, features=DATASET_FEATURES)
    print(f"Dataset created: {ds}")

    if save_path:
        ds.save_to_disk(save_path)
        print(f"Saved to {save_path}")

    if push_to_hub:
        ds.push_to_hub(push_to_hub, private=False)
        print(f"Pushed to hub: {push_to_hub}")

    return ds


def _save_checkpoint(rows: List[Dict], path: str):
    """Quick checkpoint to disk."""
    try:
        ds = Dataset.from_list(rows, features=DATASET_FEATURES)
        ds.save_to_disk(path)
        print(f"[checkpoint] {len(rows)} examples saved to {path}")
    except Exception as e:
        print(f"[checkpoint-error] {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic delta-captioning SFT dataset")
    parser.add_argument("--dataset_name", type=str, default="Wild-Heart/Disney-VideoGeneration-Dataset",
                        help="Source HuggingFace dataset name")
    parser.add_argument("--output_path", type=str, default="./synth_delta_caption_dataset",
                        help="Local path to save the dataset")
    parser.add_argument("--push_to_hub", type=str, default="BarryFutureman/synth-delta-caption",
                        help="HuggingFace Hub repo to push to (e.g. my-org/my-dataset)")
    parser.add_argument("--max_examples", type=int, default=1,
                        help="Max examples to generate (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint at output_path")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("Provide --api_key or set OPENROUTER_API_KEY env var")

    generate_dataset(
        dataset_name=args.dataset_name,
        api_key=api_key,
        max_examples=args.max_examples,
        save_path=args.output_path,
        push_to_hub=args.push_to_hub,
        resume_path=args.output_path if args.resume else None,
    )


if __name__ == "__main__":
    main()
