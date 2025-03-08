import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-chat-hf')
#model = get_model('castorini/repllama-v1-7b-lora-passage')

instr = 'write a short poem about dizzyness'
instr_tokenized = tokenizer(f'{instr}</s>', return_tensors='pt')

gen_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-13b-chat-hf')
# Run the model forward to compute embeddings and query-passage similarity score
with torch.no_grad():
    # compute query embedding
    query_outputs = gen_model(**instr_tokenized, output_hidden_states=True)
    query_embedding = query_outputs.hidden_states[-1][-1][-1]
    print(query_embedding.shape)
    print(tokenizer.batch_decode(gen_model.generate(**instr_tokenized, max_new_tokens=32, do_sample=False)))
    query_embedding = query_embedding.unsqueeze(0).unsqueeze(2)
    #query_embedding = query_embedding.unsqueeze(0)
    print(tokenizer.batch_decode(gen_model.generate(inputs_embeds=query_embedding, max_new_tokens=32, do_sample=False)))


