"""
QLoRA fine-tuning script for cybersecurity LLM.
4-bit quantization + LoRA for efficient training on single GPU.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Optional, Dict, List
import wandb
from pathlib import Path
import json


class CyberQLoRATrainer:
    """QLoRA trainer for cybersecurity instruction tuning"""

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-2-7b-hf",
        output_dir: str = "./models/qlora_cyber",
        use_wandb: bool = True
    ):
        """
        Initialize QLoRA trainer.

        Args:
            base_model: Base model to fine-tune
            output_dir: Output directory for checkpoints
            use_wandb: Enable W&B logging
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        if use_wandb:
            wandb.init(
                project="cyber-llm-qlora",
                name=f"qlora-{base_model.split('/')[-1]}",
                config={
                    "base_model": base_model,
                    "method": "qlora",
                    "bits": 4
                }
            )

    def load_model_and_tokenizer(self):
        """Load model with 4-bit quantization and prepare for QLoRA"""
        print(f"Loading base model: {self.base_model}")

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        print("✓ Model and tokenizer loaded")

    def setup_lora(
        self,
        r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ):
        """
        Configure LoRA adapter.

        Args:
            r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability
            target_modules: Modules to apply LoRA (None = auto-detect)
        """
        if target_modules is None:
            # Default for Llama
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        print("✓ LoRA configured")

    def load_datasets(
        self,
        custom_datasets: Optional[List[str]] = None,
        max_samples: Optional[int] = None
    ) -> Dataset:
        """
        Load and combine cybersecurity instruction datasets.

        Args:
            custom_datasets: Paths to custom JSONL datasets
            max_samples: Limit total samples (for testing)

        Returns:
            Combined dataset
        """
        datasets = []

        # Load open cybersecurity datasets
        try:
            # Example: Load from HuggingFace
            print("Loading AttackQA dataset...")
            attack_qa = load_dataset("cyberseek/attackqa", split="train")
            datasets.append(attack_qa)
        except Exception as e:
            print(f"Warning: Could not load AttackQA: {e}")

        try:
            print("Loading SecQA dataset...")
            sec_qa = load_dataset("tiiuae/SecQA", split="train")
            datasets.append(sec_qa)
        except Exception as e:
            print(f"Warning: Could not load SecQA: {e}")

        # Load custom datasets
        if custom_datasets:
            for dataset_path in custom_datasets:
                print(f"Loading custom dataset: {dataset_path}")
                custom_data = load_dataset('json', data_files=dataset_path, split='train')
                datasets.append(custom_data)

        if not datasets:
            raise ValueError("No datasets loaded! Provide at least one dataset.")

        # Combine datasets
        combined = concatenate_datasets(datasets)

        if max_samples:
            combined = combined.select(range(min(max_samples, len(combined))))

        print(f"✓ Loaded {len(combined)} training examples")
        return combined

    def format_prompt(self, example: Dict) -> str:
        """
        Format example as instruction prompt (Llama-2 chat format).

        Args:
            example: Dataset example with 'instruction', 'input', 'output'

        Returns:
            Formatted prompt string
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        # Cybersecurity-specific system prompt
        system = (
            "You are a cybersecurity expert AI assistant. "
            "Provide accurate, detailed answers about threats, vulnerabilities, "
            "and defensive techniques."
        )

        if input_text:
            prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction}\n{input_text} [/INST] {output}</s>"
        else:
            prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST] {output}</s>"

        return prompt

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Tokenize and prepare dataset for training.

        Args:
            dataset: Raw dataset

        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            # Format prompts
            if isinstance(examples['instruction'], list):
                prompts = [self.format_prompt({
                    "instruction": inst,
                    "input": inp,
                    "output": out
                }) for inst, inp, out in zip(
                    examples['instruction'],
                    examples.get('input', [''] * len(examples['instruction'])),
                    examples['output']
                )]
            else:
                prompts = [self.format_prompt(examples)]

            # Tokenize
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                max_length=2048,
                padding="max_length"
            )

            # Labels = input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )

        return tokenized_dataset

    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 3,
        per_device_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500
    ):
        """
        Train model with QLoRA.

        Args:
            dataset: Training dataset
            num_epochs: Number of training epochs
            per_device_batch_size: Batch size per GPU
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
        """
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            fp16=True,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            report_to="wandb" if self.use_wandb else "none",
            gradient_checkpointing=True,
            group_by_length=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        print("\nStarting training...")
        trainer.train()

        print("✓ Training completed")

    def save_adapter(self, adapter_path: Optional[str] = None):
        """Save LoRA adapter weights"""
        if adapter_path is None:
            adapter_path = self.output_dir / "adapter"

        adapter_path = Path(adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        print(f"✓ LoRA adapter saved to {adapter_path}")

        return adapter_path


def main():
    """Main training pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for cyber LLM")
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/qlora_cyber",
        help="Output directory"
    )
    parser.add_argument(
        "--custom-datasets",
        nargs="+",
        help="Paths to custom JSONL datasets"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")

    args = parser.parse_args()

    # Initialize trainer
    trainer = CyberQLoRATrainer(
        base_model=args.base_model,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb
    )

    # Load model
    trainer.load_model_and_tokenizer()

    # Setup LoRA
    trainer.setup_lora(r=args.lora_r)

    # Load datasets
    dataset = trainer.load_datasets(custom_datasets=args.custom_datasets)

    # Preprocess
    tokenized_dataset = trainer.preprocess_dataset(dataset)

    # Train
    trainer.train(
        dataset=tokenized_dataset,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size
    )

    # Save adapter
    trainer.save_adapter()

    print("\n✓ Fine-tuning complete!")


if __name__ == "__main__":
    main()
