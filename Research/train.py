import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Load the Synthetic Golden Dataset you just built
print("📥 Loading synthetic training data...")
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# 2. Hardware Optimization: 4-Bit Quantization for RTX 3050 (6GB)
print("🗜️ Loading base model in 4-bit precision...")
model_name = "Qwen/Qwen2.5-3B" # (You can swap this to a Mamba model later)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Note: In production, you would pass quantization_config=BitsAndBytesConfig(...) here
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    torch_dtype=torch.float16, 
    load_in_4bit=True  # <-- Crucial for 6GB VRAM
)

# 3. Apply LoRA (Low-Rank Adaptation)
print("🧠 Attaching LoRA Adapters...")
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"], # Target the Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. The Training Engine
print("🔥 Initializing PyTorch Trainer...")
training_args = TrainingArguments(
    output_dir="./resume_extractor_lora",
    per_device_train_batch_size=1, # Keep this at 1 to survive 6GB VRAM!
    gradient_accumulation_steps=4, # Simulate a batch size of 4
    optim="paged_adamw_32bit",
    logging_steps=5,
    learning_rate=2e-4,
    max_steps=100, # Just a quick test run
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="input", # Points to the text in your .jsonl
    max_seq_length=1024,
    args=training_args,
)

# 5. Ignite the GPU
print("🚀 Starting Fine-Tuning on RTX 3050...")
trainer.train()

# 6. Save the custom brain
trainer.model.save_pretrained("custom_resume_model")
print("✅ Training complete. Custom LoRA weights saved to disk!")
