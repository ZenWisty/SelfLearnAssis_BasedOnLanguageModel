# require unsloth  
# require xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124
# require .\triton-2.0.0-cp310-cp310-win_amd64.whl


from huggingface_hub import login
from unsloth import FastLanguageModel
import wandb

hf_token = "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"
login(hf_token)

wb_token = "e29e763288c639eabaa9d34f0eef8064ebf3ea85"
wandb.login(key=wb_token)

# 托管
run = wandb.init(
    project='Fine-tune-DeepSeek-R1-Distill-Llama-8B on Medical COT Dataset',
    job_type="training",
    anonymous="allow"
)

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>{}"""

train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""


if __name__ == "__main__":
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        token = hf_token,
    )

    question = "A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, what would cystometry most likely reveal about her residual volume and detrusor contractions?"

    FastLanguageModel.for_inference(model)
    inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    print(response[0].split("### Response:")[1])

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


    def formatting_prompts_func(examples):
        inputs = examples["Question"]
        cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = []
        for input, cot, output in zip(inputs, cots, outputs):
            text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    from datasets import load_dataset
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","en", split = "train[0:500]",trust_remote_code=True)
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    dataset["text"][0]

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            # Use num_train_epochs = 1, warmup_ratio for full training runs!
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    trainer_stats = trainer.train()
    new_model_local = "DeepSeek-R1-Medical-COT"
    model.save_pretrained(new_model_local)
    tokenizer.save_pretrained(new_model_local)

    model.save_pretrained_merged(new_model_local, tokenizer, save_method = "merged_16bit",)

    question = "A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, what would cystometry most likely reveal about her residual volume and detrusor contractions?"


    FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!
    inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    print(response[0].split("### Response:")[1])