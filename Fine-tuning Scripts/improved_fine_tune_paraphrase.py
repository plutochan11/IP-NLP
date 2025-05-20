import os
import torch
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    logging,
)
from trl import SFTTrainer

# --- Disable wandb which is causing issues with distutils.spawn ---
os.environ["WANDB_DISABLED"] = "true"

# --- Handle the distutils issue in Python 3.12 ---
# This creates a fake distutils module to prevent import errors 
# with dependencies that still try to use it
if sys.version_info >= (3, 12):
    import importlib.metadata
    import types
    
    # Create a fake distutils module if it doesn't exist
    if 'distutils' not in sys.modules:
        distutils_module = types.ModuleType('distutils')
        sys.modules['distutils'] = distutils_module
        
        # Create nested modules if needed
        version_module = types.ModuleType('distutils.version')
        sys.modules['distutils.version'] = version_module
        
        # Create a placeholder LooseVersion class
        class LooseVersion:
            def __init__(self, version_str):
                self.version_str = str(version_str)
                
            def __str__(self):
                return self.version_str
                
            def __repr__(self):
                return f"LooseVersion('{self.version_str}')"
                
            def __eq__(self, other):
                if isinstance(other, str):
                    return self.version_str == other
                return self.version_str == str(other)
                
            def __lt__(self, other):
                if isinstance(other, str):
                    return self.version_str < other
                return self.version_str < str(other)
                
        # Add LooseVersion to the version module
        version_module.LooseVersion = LooseVersion

# --- Configuration ---

# Base model to fine-tune
MODEL_ID = "roneneldan/TinyStories-33M"

# Input data file - use the clean, high-quality dataset
DATA_FILE_PATH = "Data/fine-tuning-dataset.jsonl"

# Output directory for the fine-tuned model
OUTPUT_DIR = "./improved-paraphrase-model"

# --- Training Hyperparameters ---

MICRO_BATCH_SIZE = 4       # Increase if your hardware allows
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5       # Lower learning rate for better stability
NUM_TRAIN_EPOCHS = 5       # More epochs for better convergence
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.5        # Lower for more stable training
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
SAVE_STEPS = 50            # Save checkpoints more frequently
EVAL_STEPS = 50            # Evaluate more frequently
MAX_SEQ_LENGTH = 256       # Reduced to focus on shorter paraphrases
EVAL_RATIO = 0.15          # Slightly larger evaluation set

# --- Data Preprocessing ---

def preprocess_dataset(dataset):
    """
    Additional preprocessing to ensure high-quality training data
    """
    # Filter examples with empty outputs or inputs
    filtered = dataset.filter(lambda x: x['output'] and x['input'] and len(x['output'].strip()) > 0 and len(x['input'].strip()) > 0)
    
    # Filter examples where output is the same as input (no paraphrase)
    filtered = filtered.filter(lambda x: x['output'].strip() != x['input'].strip())
    
    # Filter examples that are too long
    filtered = filtered.filter(lambda x: len(x['output'].split()) <= 50 and len(x['input'].split()) <= 50)
    
    print(f"Original dataset size: {len(dataset)}, Filtered size: {len(filtered)}")
    return filtered

# Helper function to format instructions
def format_instruction(instruction, input_text, output_text):
    """
    Format the instruction, input and output as a prompt for the model.
    We explicitly make the paraphrasing intent clear.
    """
    # Check if input already has the prefix
    prefix = "Paraphrase: "
    if input_text.startswith(prefix):
        input_text = input_text[len(prefix):].strip()
    
    # Add the prefix back to clarify the task
    input_with_prefix = f"Paraphrase: {input_text}"
    
    # Use a very clear instruction template
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_with_prefix}

### Response:
{output_text}{tokenizer.eos_token}"""

# --- Device Selection ---
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS is available! Using MPS backend.")
        return "mps"
    elif torch.cuda.is_available():
        print("CUDA is available! Using GPU.")
        return "cuda"
    else:
        print("Using CPU backend (Training will be slow).")
        return "cpu"

# --- Main Fine-tuning Logic ---
def main():
    device = get_device()
    
    print(f"Starting fine-tuning for paraphrasing model: {MODEL_ID}")
    print(f"Using data file: {DATA_FILE_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    # 1. Load and Preprocess Dataset
    print("Loading and preprocessing dataset...")
    try:
        dataset = load_dataset("json", data_files=DATA_FILE_PATH, split="train")
        print(f"Raw dataset loaded with {len(dataset)} examples.")
        
        # Apply preprocessing
        dataset = preprocess_dataset(dataset)
        
        # Split dataset
        dataset = dataset.shuffle(seed=42)
        split_dataset = dataset.train_test_split(test_size=EVAL_RATIO)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"Train dataset: {len(train_dataset)}, Eval dataset: {len(eval_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Load Tokenizer
    global tokenizer  # Make tokenizer global for format_instruction function
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 3. Load Model
    print("Loading base model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Setup Training Arguments
    print("Configuring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        report_to="tensorboard",  # Only use tensorboard for reporting, not wandb
        remove_unused_columns=False,
        group_by_length=True,
        no_cuda=True if device != "cuda" else False,
    )

    # 5. Configure SFT Trainer
    print("Creating SFT Trainer...")
    
    # Use SFTTrainer for simplified fine-tuning
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=lambda example: format_instruction(example['instruction'], example['input'], example['output']),
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    
    # 6. Start Training
    print("\n*** Starting Training ***")
    print(f"Training will run for {NUM_TRAIN_EPOCHS} epochs")
    
    try:
        # Start training
        train_result = trainer.train()
        
        # Log and save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate model
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        # Save the final model
        print("Saving final model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"Training complete. Model saved to {OUTPUT_DIR}")
        
        # Print final metrics
        print("\nFinal Training Metrics:")
        print(f"  Train Loss: {metrics['train_loss']:.4f}")
        print(f"  Train Runtime: {metrics['train_runtime']:.2f} seconds")
        print("\nFinal Evaluation Metrics:")
        print(f"  Eval Loss: {eval_metrics['eval_loss']:.4f}")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    
    print("Fine-tuning process complete.")

if __name__ == "__main__":
    main()