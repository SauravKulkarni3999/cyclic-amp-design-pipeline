import os
import sys

# MUST be before torch import — MPS reads these at import time
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'

import pandas as pd
import json
import torch
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
from trl import DPOTrainer, DPOConfig

import psutil

def log_memory(label):
    mem = psutil.virtual_memory()
    print(f"[MEM] {label}: {mem.used/1e9:.1f}GB used / {mem.total/1e9:.1f}GB total "
          f"({mem.available/1e9:.1f}GB available)")


def load_dpo_dataset(pref_path, seq_csv_path):
    df_seq = pd.read_csv(seq_csv_path)
    sequences = df_seq['sequence'].tolist()

    dpo_pairs = []
    with open(pref_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            chosen_idx = int(pair['chosen'].split('_')[1])
            rejected_idx = int(pair['rejected'].split('_')[1])
            dpo_pairs.append({
                "prompt": "<|endoftext|>",
                "chosen": sequences[chosen_idx],
                "rejected": sequences[rejected_idx]
            })

    return dpo_pairs


def main(seq_csv_path, pref_path, output_dir):
    log_memory("Node 04 START")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "aligned_model"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "optimized_sequences"), exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else \
            "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Mapping IDs to sequences via CSV index...")
    dpo_pairs = load_dpo_dataset(pref_path, seq_csv_path)
    print(f"Loaded {len(dpo_pairs)} DPO pairs")
    train_dataset = Dataset.from_list(dpo_pairs)

    print("Loading model and tokenizer...")
    model_name = os.environ.get("PROTGPT2_DIR", "nferruz/ProtGPT2")
    local = os.path.isdir(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    # Policy model on MPS/GPU
    model = GPT2LMHeadModel.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=local
    ).to(device)
    model.gradient_checkpointing_enable()
    log_memory("After policy model load")

    # Reference model on CPU — only does forward passes, doesn't need GPU
    # This saves ~1.5GB of MPS memory
    ref_model = GPT2LMHeadModel.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=local
    )  # intentionally no .to(device) — stays on CPU
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    log_memory("After reference model load")

    print(f"Policy model memory: "
          f"{sum(p.nbytes for p in model.parameters()) / 1024**3:.2f}GB")
    print(f"Reference model memory: "
          f"{sum(p.nbytes for p in ref_model.parameters()) / 1024**3:.2f}GB")
    print(f"Policy dtype: {next(model.parameters()).dtype}")
    print(f"Reference dtype: {next(ref_model.parameters()).dtype}")

    training_args = DPOConfig(
        output_dir=os.path.join(output_dir, "checkpoints"),
        per_device_train_batch_size=1,
        bf16=False,
        fp16=False,
        num_train_epochs=2,
        learning_rate=1e-5,
        save_strategy="no",
        logging_steps=1,
        remove_unused_columns=False,
        beta=0.1,
        max_length=64,
        gradient_accumulation_steps=4,
        use_cpu=(device == "cpu")  # effective batch = 8, low memory
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        ref_model=ref_model,
    )

    print("Aligning model to physical constraints...")
    log_memory("Before DPO train()")
    dpo_trainer.train()
    log_memory("After DPO train()")

    model.save_pretrained(os.path.join(output_dir, "aligned_model"))
    print("Training complete. Generating samples...")

    model.eval()
    new_sequences = []
    for _ in range(50):
        input_ids = tokenizer(
            "<|endoftext|>", add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        attention_mask = tokenizer(
            "<|endoftext|>", add_special_tokens=False, return_tensors="pt"
        ).attention_mask.to(device)
        output = model.generate(
            input_ids,
            do_sample=True,
            top_k=50,
            repetition_penalty=1.7,
            temperature=1.0,
            attention_mask=attention_mask,
            max_new_tokens=11,
            top_p=0.95,
        )
        seq = tokenizer.decode(output[0], skip_special_tokens=True).replace("\n", "")
        if seq.startswith('M') and len(seq) > 1:
            seq = seq[1:]
        new_sequences.append(seq)

    df = pd.DataFrame({"sequence": new_sequences})
    output_path = os.path.join(output_dir, "optimized_sequences")
    df.to_csv(os.path.join(output_path, "optimized_sequences.csv"), index=False)

    log_memory("Node 04 COMPLETE")
    print("Node 04 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_csv_path", type=str, required=True)
    parser.add_argument("--pref_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.seq_csv_path, args.pref_path, args.output_dir)