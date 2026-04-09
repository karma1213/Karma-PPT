import os
import json
import argparse
import warnings
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from torchvision.io import read_video
import torchvision.transforms as T

def setup_warnings(quiet_warnings: bool = True):
    if not quiet_warnings:
        return

    # Silence known non-fatal deprecation warnings from dependency internals.
    warnings.filterwarnings(
        "ignore",
        message=r".*pynvml package is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*torchvision\.transforms\._functional_video.*deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*torchvision\.transforms\._transforms_video.*deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*torchvision\.transforms\.functional_tensor.*deprecated.*",
        category=UserWarning,
    )


class TSLVideoDataset(Dataset):
    def __init__(self,
        json_file: str,
        video_root: str,
        tokenizer,
        max_length: int = 512,
        num_frames: int = 16,
    ): 
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.video_root = video_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_frames = num_frames

        self.samples: List[Dict[str, Any]] = []
        for item in raw:
            rel_path = item.get("visual_input", "")
            video_path = os.path.join(video_root, rel_path)
            if os.path.exists(video_path):
                self.samples.append(item)

        self.transform = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if len(self.samples) == 0:
            raise ValueError("No valid samples found. Check json and video paths.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, video_path: str) -> torch.Tensor:
        video, _, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")
        if video.shape[0] == 0:
            raise ValueError("Empty video")

        indices = torch.linspace(0, video.shape[0] - 1, self.num_frames).long()
        video = video[indices].float() / 255.0  # [T, C, H, W]
        video = self.transform(video)
        video = video.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
        return video

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        attempts = 0
        cur_idx = idx
        while attempts < 10:
            item = self.samples[cur_idx]
            video_path = os.path.join(self.video_root, item["visual_input"])
            try:
                video = self._load_video(video_path)

                prompt_query = f"USER: <video>\n{item['instruction']} ASSISTANT:"
                prompt_answer = f" {item['output']}</s>"

                q_ids = self.tokenizer(prompt_query, add_special_tokens=False).input_ids
                a_ids = self.tokenizer(prompt_answer, add_special_tokens=False).input_ids

                input_ids = (q_ids + a_ids)[: self.max_length]
                labels = ([-100] * len(q_ids) + a_ids)[: self.max_length]

                pad_len = self.max_length - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len

                return {
                    "video": video,
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            except Exception:
                attempts += 1
                cur_idx = (cur_idx + 1) % len(self.samples)

        raise RuntimeError("Failed to fetch a valid sample after multiple attempts.")


def find_lora_targets(model: torch.nn.Module) -> List[str]:
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj"}
    names = set()
    for module_name, _ in model.named_modules():
        short = module_name.split(".")[-1]
        if short in wanted:
            names.add(short)
    if not names:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return sorted(names)


def build_collate_fn(tokenizer):
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        videos = torch.stack([b["video"] for b in batch])
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        return {
            "images": videos,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_visible_devices", type=str, default="2")
    parser.add_argument("--model_path", type=str, default="/data/test/python/models/Video-LLaVA/Video-LLaVA-7B")
    parser.add_argument("--prompt_json", type=str, default="/data/test/python/dataset/TSL-300/tsl300_REORDERED_UPDATED.json")
    parser.add_argument("--video_root", type=str, default="/data/test/python/dataset/TSL-300")
    parser.add_argument("--output_dir", type=str, default="/data/test/python/projectlyq/videollava_tsl_final_v7")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--quiet_warnings", action="store_true", default=True)
    parser.add_argument("--no_quiet_warnings", action="store_false", dest="quiet_warnings")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_warnings(args.quiet_warnings)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    model.get_model().get_video_tower().load_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_targets = find_lora_targets(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_dataset = TSLVideoDataset(
        json_file=args.prompt_json,
        video_root=args.video_root,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_frames=args.num_frames,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=True,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=build_collate_fn(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()