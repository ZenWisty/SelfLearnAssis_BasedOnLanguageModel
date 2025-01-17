# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")