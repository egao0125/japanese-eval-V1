"""LoRA fine-tuning for Qwen3-ASR-1.7B on Japanese telephony data.

Key insight: Apply LoRA to model.thinker (the Qwen3ASRThinkerForConditionalGeneration)
which has standard input_ids/labels forward, NOT the outer wrapper.
"""
import argparse, json, os, time, random
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import get_cosine_schedule_with_warmup


class ASRDataset(Dataset):
    def __init__(self, manifest_path, max_audio_sec=30.0):
        with open(manifest_path) as f:
            entries = json.load(f)
        self.max_samples = int(max_audio_sec * 16000)
        self.entries = [e for e in entries if Path(e['audio_path']).exists() and e.get('transcript', '').strip()]
        print(f'Dataset: {len(self.entries)}/{len(entries)} valid')

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        try:
            audio, sr = sf.read(e['audio_path'])
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            if len(audio) < 160:
                audio = np.zeros(1600, dtype=np.float32)
            return audio, e['transcript']
        except Exception:
            return np.zeros(1600, dtype=np.float32), ''


def train(args):
    from qwen_asr import Qwen3ASRModel

    print('=== Qwen3-ASR-1.7B LoRA Fine-Tuning ===')
    qwen = Qwen3ASRModel.from_pretrained('Qwen/Qwen3-ASR-1.7B')
    model = qwen.model.to('cuda', dtype=torch.float16)
    processor = qwen.processor

    # Apply LoRA to THINKER (has standard forward with input_ids/labels)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        task_type=TaskType.CAUSAL_LM, use_rslora=True,
    )
    thinker = get_peft_model(model.thinker, lora_cfg)
    thinker.print_trainable_parameters()
    model.thinker = thinker
    model.thinker.train()

    # Freeze encoder
    for p in model.thinker.base_model.model.audio_tower.parameters():
        p.requires_grad = False

    # Prompt template
    chat = [{'role': 'user', 'content': '<|audio|>Transcribe this audio to Japanese text.'}]
    prompt_text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Data
    train_ds = ASRDataset(os.path.join(args.data_dir, 'combined_train.json'), args.max_audio_sec)

    # Optimizer (only trainable LoRA params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    steps_per_epoch = len(train_ds) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    ts = time.strftime('%H:%M:%S')
    print(f'Train: {len(train_ds)} samples, {args.epochs} epochs, {total_steps} steps')
    print(f'LoRA: r={args.lora_r} alpha={args.lora_alpha} rsLoRA | LR: {args.lr} accum: {args.grad_accum}')
    print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB | Started: {ts}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    log_loss = 0.0
    log_n = 0

    for epoch in range(args.epochs):
        model.thinker.train()
        epoch_loss = 0.0
        epoch_n = 0
        t0 = time.time()
        optimizer.zero_grad()

        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            audio, transcript = train_ds[idx]
            if not transcript:
                continue

            try:
                full = prompt_text + transcript + processor.tokenizer.eos_token
                inputs = processor(text=full, audio=[audio], return_tensors='pt')
                inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
                inputs['labels'] = inputs['input_ids'].clone()

                # Mask prompt
                p_inputs = processor(text=prompt_text, audio=[audio], return_tensors='pt')
                plen = p_inputs['input_ids'].shape[-1]
                inputs['labels'][0, :plen] = -100

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model.thinker(**inputs)
                    loss = out.loss / args.grad_accum

                loss.backward()
                lv = loss.item() * args.grad_accum
                epoch_loss += lv
                epoch_n += 1
                log_loss += lv
                log_n += 1

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            except Exception as e:
                if i < 3:
                    print(f'  err[{i}]: {e}')
                continue

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg = log_loss / max(log_n, 1)
                    lr = scheduler.get_last_lr()[0]
                    vram = torch.cuda.memory_allocated() / 1e9
                    elapsed = time.time() - t0
                    done_in_epoch = global_step - (epoch * steps_per_epoch)
                    if done_in_epoch > 0:
                        eta = elapsed / done_in_epoch * (steps_per_epoch - done_in_epoch)
                    else:
                        eta = 0
                    print(f'  step={global_step}/{total_steps} loss={avg:.4f} lr={lr:.2e} vram={vram:.1f}G eta={eta/60:.0f}m')
                    log_loss = 0.0
                    log_n = 0

                if global_step > 0 and global_step % args.save_steps == 0:
                    sp = output_dir / f'checkpoint-{global_step}'
                    model.thinker.save_pretrained(str(sp))
                    print(f'  SAVED {sp}')

        avg = epoch_loss / max(epoch_n, 1)
        elapsed = time.time() - t0
        print(f'EPOCH {epoch+1}/{args.epochs} loss={avg:.4f} n={epoch_n} time={elapsed:.0f}s')
        sp = output_dir / f'epoch-{epoch+1}'
        model.thinker.save_pretrained(str(sp))
        print(f'  SAVED {sp}')

    model.thinker.save_pretrained(str(output_dir / 'final'))
    ts = time.strftime('%H:%M:%S')
    print(f'DONE! Adapter: {output_dir / "final"} | {ts}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/workspace/data')
    p.add_argument('--output_dir', default='/workspace/checkpoints/qwen3-v10')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lora_r', type=int, default=64)
    p.add_argument('--lora_alpha', type=int, default=128)
    p.add_argument('--grad_accum', type=int, default=4)
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--max_audio_sec', type=float, default=30.0)
    a = p.parse_args()
    train(a)
