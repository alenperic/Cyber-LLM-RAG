"""
Continual Pretraining (CPT) for domain adaptation.
Short CPT pass on cybersecurity corpora before instruction tuning.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, concatenate_datasets
from pathlib import Path
import json
from typing import List, Dict, Any
import wandb


class CyberCPTTrainer:
    """Continual pretraining on cybersecurity domain data"""

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-2-7b-hf",
        output_dir: str = "./models/cpt_cyber",
        use_wandb: bool = True
    ):
        """
        Initialize CPT trainer.

        Args:
            base_model: Base model for CPT
            output_dir: Output directory
            use_wandb: Enable W&B logging
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        if use_wandb:
            wandb.init(
                project="cyber-llm-cpt",
                name=f"cpt-{base_model.split('/')[-1]}",
                config={"base_model": base_model, "method": "cpt"}
            )

    def load_model_and_tokenizer(self, use_flash_attention: bool = True):
        """Load model and tokenizer for CPT"""
        print(f"Loading base model: {self.base_model}")

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto"
        }

        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            **model_kwargs
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✓ Model and tokenizer loaded")

    def prepare_attack_corpus(self, data_dir: Path) -> List[str]:
        """
        Extract text from ATT&CK STIX for CPT.

        Args:
            data_dir: Directory with raw ATT&CK data

        Returns:
            List of text documents
        """
        texts = []
        attack_dir = data_dir / "attack"

        if not attack_dir.exists():
            print(f"Warning: ATT&CK directory not found: {attack_dir}")
            return texts

        for stix_file in attack_dir.glob("*.json"):
            with open(stix_file) as f:
                data = json.load(f)

            for obj in data.get("objects", []):
                # Extract descriptions from various object types
                if obj.get("type") in ["attack-pattern", "course-of-action", "intrusion-set"]:
                    name = obj.get("name", "")
                    description = obj.get("description", "")

                    if description:
                        text = f"{name}\n{description}"
                        texts.append(text)

                    # Also include x_mitre fields
                    for key, value in obj.items():
                        if key.startswith("x_mitre_") and isinstance(value, str):
                            texts.append(value)

        print(f"  Extracted {len(texts)} documents from ATT&CK")
        return texts

    def prepare_cwe_capec_corpus(self, data_dir: Path) -> List[str]:
        """
        Extract text from CWE/CAPEC XML for CPT.

        Args:
            data_dir: Directory with raw CWE/CAPEC data

        Returns:
            List of text documents
        """
        import xml.etree.ElementTree as ET
        texts = []

        # CWE
        cwe_dir = data_dir / "cwe"
        if cwe_dir.exists():
            for xml_file in cwe_dir.glob("*.xml"):
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()

                    # Extract all text content from XML
                    for elem in root.iter():
                        if elem.text and len(elem.text.strip()) > 50:
                            texts.append(elem.text.strip())
                except Exception as e:
                    print(f"Warning: Could not parse {xml_file}: {e}")

        # CAPEC
        capec_dir = data_dir / "capec"
        if capec_dir.exists():
            for xml_file in capec_dir.glob("*.xml"):
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()

                    for elem in root.iter():
                        if elem.text and len(elem.text.strip()) > 50:
                            texts.append(elem.text.strip())
                except Exception as e:
                    print(f"Warning: Could not parse {xml_file}: {e}")

        print(f"  Extracted {len(texts)} documents from CWE/CAPEC")
        return texts

    def prepare_nvd_corpus(self, data_dir: Path) -> List[str]:
        """
        Extract text from NVD CVE JSON for CPT.

        Args:
            data_dir: Directory with raw NVD data

        Returns:
            List of text documents
        """
        texts = []
        nvd_dir = data_dir / "nvd"

        if not nvd_dir.exists():
            print(f"Warning: NVD directory not found: {nvd_dir}")
            return texts

        for json_file in nvd_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)

            for item in data.get("CVE_Items", []):
                cve = item.get("cve", {})
                cve_id = cve.get("CVE_data_meta", {}).get("ID")

                # Get description
                descriptions = cve.get("description", {}).get("description_data", [])
                if descriptions:
                    desc = descriptions[0].get("value", "")
                    if desc:
                        text = f"CVE {cve_id}: {desc}"
                        texts.append(text)

        print(f"  Extracted {len(texts)} documents from NVD")
        return texts

    def prepare_sigma_corpus(self, data_dir: Path) -> List[str]:
        """
        Extract text from Sigma rules for CPT.

        Args:
            data_dir: Directory with Sigma rules

        Returns:
            List of text documents
        """
        import yaml
        texts = []
        sigma_dir = data_dir / "sigma" / "rules"

        if not sigma_dir.exists():
            sigma_dir = data_dir / "sigma"

        if not sigma_dir.exists():
            print(f"Warning: Sigma directory not found: {sigma_dir}")
            return texts

        for yaml_file in sigma_dir.rglob("*.yml"):
            try:
                with open(yaml_file) as f:
                    rule = yaml.safe_load(f)

                if not rule or not isinstance(rule, dict):
                    continue

                title = rule.get("title", "")
                description = rule.get("description", "")

                if description:
                    text = f"Sigma Detection Rule: {title}\n{description}"
                    texts.append(text)

            except Exception:
                continue

        print(f"  Extracted {len(texts)} documents from Sigma")
        return texts

    def build_cpt_dataset(self, data_dir: Path) -> Dataset:
        """
        Build CPT dataset from all sources.

        Args:
            data_dir: Directory with raw data

        Returns:
            HuggingFace Dataset
        """
        print("Building CPT corpus from cybersecurity data sources...")

        all_texts = []

        # Collect from all sources
        all_texts.extend(self.prepare_attack_corpus(data_dir))
        all_texts.extend(self.prepare_cwe_capec_corpus(data_dir))
        all_texts.extend(self.prepare_nvd_corpus(data_dir))
        all_texts.extend(self.prepare_sigma_corpus(data_dir))

        print(f"✓ Total CPT documents: {len(all_texts)}")

        # Create HF dataset
        dataset = Dataset.from_dict({"text": all_texts})

        return dataset

    def preprocess_dataset(self, dataset: Dataset, max_length: int = 2048) -> Dataset:
        """
        Tokenize dataset for CPT.

        Args:
            dataset: Raw text dataset
            max_length: Maximum sequence length

        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing CPT dataset"
        )

        return tokenized

    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 1,
        per_device_batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 1e-5,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500
    ):
        """
        Run continual pretraining.

        Args:
            dataset: Tokenized dataset
            num_epochs: Training epochs (typically 1 for CPT)
            per_device_batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation
            learning_rate: Learning rate (lower than initial pretraining)
            warmup_steps: Warmup steps
            logging_steps: Logging frequency
            save_steps: Checkpoint frequency
        """
        # Data collator for language modeling (MLM/CLM)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM (not masked)
        )

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            bf16=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="wandb" if self.use_wandb else "none",
            gradient_checkpointing=True,
            dataloader_pin_memory=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        print("\nStarting continual pretraining...")
        trainer.train()

        print("✓ CPT completed")

    def save_model(self, model_path: Optional[str] = None):
        """Save CPT model"""
        if model_path is None:
            model_path = self.output_dir / "final_model"

        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

        print(f"✓ CPT model saved to {model_path}")

        return model_path


def main():
    """Main CPT pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Continual pretraining on cyber data")
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with raw cybersecurity data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/cpt_cyber",
        help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")

    args = parser.parse_args()

    # Initialize trainer
    trainer = CyberCPTTrainer(
        base_model=args.base_model,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb
    )

    # Load model
    trainer.load_model_and_tokenizer()

    # Build CPT dataset
    dataset = trainer.build_cpt_dataset(Path(args.data_dir))

    # Preprocess
    tokenized_dataset = trainer.preprocess_dataset(dataset)

    # Train
    trainer.train(
        dataset=tokenized_dataset,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size
    )

    # Save model
    trainer.save_model()

    print("\n✓ Continual pretraining complete!")
    print("Next: Run QLoRA fine-tuning on this CPT model for instruction alignment.")


if __name__ == "__main__":
    main()
