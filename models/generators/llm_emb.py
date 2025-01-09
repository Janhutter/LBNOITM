from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from models.generators.generator import Generator
import torch.nn.functional as F
from utils import get_doc_embeds_from_dataset, get_embeddings_dataset, get_index_path

class LLMEmb(Generator):
    def __init__(self, 
                 model_name=None, 
                 max_new_tokens=1, 
                 max_doc_len=100,
                 max_length=None,
                 prompt=None,
                 ):
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.bos_token
        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype='bfloat16',
                            bnb_4bit_use_double_quant=False,
                        )
        self.max_doc_len = max_doc_len
        self.device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map='auto', 
            quantization_config=quant_config, 
            attn_implementation="flash_attention_2"
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.max_new_tokens = max_new_tokens
        self.model.eval()
        self.prompt = prompt
    
    def generate(self, model_input):
        input_ids = model_input['input_ids']
        attention_mask = model_input['attention_mask']
        inputs_embeds =  self.model.get_input_embeddings()(input_ids.to('cuda'))
        generated_ids =  self.model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return generated_response
    
    def collate_fn(self, examples,**kwargs):
        q_ids = [e['q_id'] for e in examples]
        instr = [self.format_instruction(e) for e in examples]
        label = [[e['label']] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)
        model_input = self.tokenizer(
            instr, 
            padding=True, 
            return_tensors="pt", 
            truncation=True,
            max_length=self.max_length,
            )
        data_dict = {
            'model_input': model_input,
            'q_id': q_ids, 
            'query': query, 
            'instruction': instr,
            'label': label, 
            'ranking_label': ranking_label,
        }
        return data_dict