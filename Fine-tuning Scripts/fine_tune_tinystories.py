import os
# Disable wandb to prevent the distutils import error in Python 3.12
os.environ["WANDB_DISABLED"] = "true"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    logging,
)
from trl import SFTTrainer # Simplified Fine-tuning Trainer
import datetime

# --- Configuration ---

MODEL_ID = "roneneldan/TinyStories-33M"

DATA_FILE_PATH = "Data/fine-tuning-dataset.jsonl" # Using the fixed dataset file

TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

OUTPUT_DIR = f"Fine-tuned Results/Tiny Stories Paraphrase_{TIMESTAMP}" # <--- CHANGE THIS IF DESIRED

# Training Hyperparameters (Adjust as needed, expect slower training on CPU/MPS)
MICRO_BATCH_SIZE = 2       # Keep batch size small for CPU/Macbook memory
GRADIENT_ACCUMULATION_STEPS = 8 # Increase accumulation to compensate for small micro-batch
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 5
WEIGHT_DECAY = 0.01        # Weight decay for regularization
MAX_GRAD_NORM = 1.0        # Max gradient norm for clipping (can be higher than QLoRA)
LR_SCHEDULER_TYPE = "cosine" # Learning rate scheduler type
WARMUP_RATIO = 0.1         # Proportion of training steps for linear warmup (adjust if needed)
LOGGING_STEPS = 20         # Log training metrics every N steps
SAVE_STEPS = 100           # Save a checkpoint every N steps (adjust based on time/disk space)
MAX_SEQ_LENGTH = 512       # Max sequence length (adjust based on data/memory, 33M model handles shorter sequences well)
EVAL_RATIO = 0.2           # Proportion of data to use for evaluation

# --- Helper Function to Format Data ---

# Creates the prompt string from the instruction, input, and output
def format_instruction(instruction, input_text, output_text):
    # Using the same template as before
    if input_text and input_text.strip():
         prefix = "Paraphrase: "
         if input_text.startswith(prefix):
             input_text = input_text[len(prefix):].strip()

         # IMPORTANT: Add EOS token *here* when formatting for SFTTrainer
         return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}{tokenizer.eos_token}""" # Add EOS token after the output
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output_text}{tokenizer.eos_token}""" # Add EOS token after the output


# --- Check for MPS Availability ---
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS is available! Using MPS backend.")
        return "mps"
    else:
        print("MPS not available. Using CPU backend (Training will be slow).")
        return "cpu"

# --- Main Fine-tuning Logic ---

def main():
    device = get_device()

    print(f"--- Starting fine-tuning process for model: {MODEL_ID} on device: {device} ---")
    print(f"--- Using data file: {DATA_FILE_PATH} ---")
    print(f"--- Output directory: {OUTPUT_DIR} ---")
    print("----------------------------")

    # 1. Load Dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset("json", data_files=DATA_FILE_PATH, split="train")
        # Ensure the dataset is not empty
        if len(dataset) == 0:
             print(f"Error: The dataset file '{DATA_FILE_PATH}' is empty or failed to load correctly.")
             return
        print(f"Dataset loaded successfully with {len(dataset)} examples.")
        
        # Split dataset into train and validation sets
        dataset = dataset.shuffle(seed=42)  # Shuffle dataset before splitting
        split_dataset = dataset.train_test_split(test_size=EVAL_RATIO)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(eval_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Load Tokenizer
    global tokenizer # Make tokenizer global for the formatting function
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        # Set padding token if necessary (often EOS token for Causal LMs)
        if tokenizer.pad_token is None:
            print("Tokenizer does not have a pad token, using EOS token as pad token.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Ensure padding is on the right
        print(f"Tokenizer loaded successfully. EOS token: '{tokenizer.eos_token}', PAD token: '{tokenizer.pad_token}'")
    except Exception as e:
        print(f"Error loading tokenizer for model {MODEL_ID}: {e}")
        return

    # 3. Load Model
    print("Loading base model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            # No quantization_config needed for CPU/MPS
            # No device_map needed, Trainer handles device placement
        )
        # Important for Mac M1/M2/M3: Ensure model uses float32 if MPS has issues with lower precision
        # model = model.to(torch.float32) # Uncomment if you face precision issues on MPS

        print("Base model loaded successfully.")
    except Exception as e:
        print(f"Error loading model {MODEL_ID}: {e}")
        return

    # 4. Define Training Arguments for CPU/MPS
    print("Defining training arguments...")
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim="adamw_torch", # Standard optimizer
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=False, # Disable FP16 for CPU/MPS standard training
        bf16=False, # Disable BF16
        max_grad_norm=MAX_GRAD_NORM,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        group_by_length=True,
        report_to="tensorboard", # or "none"
        save_strategy="steps",
        no_cuda=True, # Explicitly disable CUDA
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    print("Training arguments defined.")

    # 5. Create SFT Trainer
    print("Creating SFT Trainer...")
    # SFTTrainer handles the formatting and training loop
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # No PEFT config needed for full fine-tuning
        formatting_func=lambda example: format_instruction(example['instruction'], example['input'], example['output']),
        # max_seq_length=MAX_SEQ_LENGTH,
        processing_class=tokenizer,
        args=training_arguments,
        # packing=False, # Keep packing False unless you specifically prepare data for it
    )
    print("SFT Trainer created.")

    # 6. Start Training
    print("\n*** Starting Training ***")
    try:
        train_result = trainer.train()
        print("*** Training Complete ***")

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Save the final FULL model
        print("Saving final model...")
        trainer.save_state()
        trainer.save_model(OUTPUT_DIR) # Saves the full fine-tuned model
        tokenizer.save_pretrained(OUTPUT_DIR) # Save tokenizer with the model
        print(f"Model saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback

    print("Fine-tuning script finished.")


if __name__ == "__main__":
    main()