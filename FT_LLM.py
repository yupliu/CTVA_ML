from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch
import pandas as pd
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Prepare dataset
def load_dataset(file_path, tokenizer, block_size=128):
    #df = pd.read_xml(file_path)
    #texts = df['text'].tolist()
    # read the file and put all text into a list
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    texts = [text.strip() for text in texts]
    tokenizer.pad_token = tokenizer.eos_token
    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=block_size)
    dataset = TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=block_size)
    return dataset
train_dataset = load_dataset(file_path='/data/camd/Database/Pubtator3/output/BioCXML/10.BioC.XML', tokenizer=tokenizer, block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # GPT-2 is not a masked language model
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)
trainer.train()
model.save_pretrained('./fine_tuned_gpt2')
tokenizer.save_pretrained('./fine_tuned_gpt2') # Save tokenizer too 
# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_gpt2')
# Generate text
input_text = "What is pulmonary tuberculosis?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
attention_mask = torch.ones_like(input_ids)  # Create attention mask

output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=50,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id  # Set pad_token_id
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text) # Print generated text
