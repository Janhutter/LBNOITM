from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from models.generators.generator import Generator
from utils import get_doc_embeds_from_dataset, get_embeddings_dataset, get_index_path
from peft import PeftConfig, AutoPeftModelForCausalLM
import time 

class LLMUnified(Generator):
    def __init__(self, 
            model_name=None, 
            max_new_tokens=1, 
            max_doc_len=100,
            max_length=None,
            prompt=None
            ):
        self.max_length = max_length
        self.model_name = model_name
        self.prompt = prompt

        # get tokenizer of lora adapter if exists else use models' tokenizer
        try:
            config = PeftConfig.from_pretrained(model_name)
            tokenizer_name = config.base_model_name_or_path
            model_class = AutoPeftModelForCausalLM
        except:
            tokenizer_name = self.model_name
            model_class = AutoModelForCausalLM

        # for training we pad right
        self.tokenizer_right = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='right')
        self.tokenizer_right.pad_token = self.tokenizer_right.bos_token
        # for evaluation we pad left
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.bos_token


        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type='nf4',
                            bnb_4bit_compute_dtype='bfloat16',
                        )
        self.model = model_class.from_pretrained(
            self.model_name, 
            quantization_config=quant_config, 
            attn_implementation="flash_attention_2",
            device_map='auto',
        )
        # self.model.merge_and_unload()
        self.max_doc_len = max_doc_len
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.model = self.model.bfloat16()
        self.max_new_tokens = max_new_tokens
        self.model.eval()
        # get embedding dim 
        embedding_dim = self.model.config.hidden_size
        #doc_embeds_path = get_index_path(self.index_folder, doc_dataset_name, self.retriever.get_clean_model_name(), 'doc')
        doc_embeds_path = '/local/calmar/rag/indexes/kilt-100w_doc_castorini_repllama-v1-7b-lora-passage/'
        self.embeds_dataset = get_embeddings_dataset(doc_embeds_path, embedding_dim)
        
        emb = self.model.get_input_embeddings()
        # Clone the embedding layer
        self.emb = torch.nn.Embedding(emb.num_embeddings, emb.embedding_dim)
        self.emb.weight = torch.nn.Parameter(emb.weight.clone().cpu(), requires_grad=False)

        self.pad_emb = self.emb(torch.LongTensor([self.tokenizer_right.pad_token_id])).detach()

    def prepare_input(self, doc_embeds=None, input_ids=None, label_ids=None, ignore_index=-100, padding_side='right'):
        # Prepares input for the model, concatenates embeddings, creates attention mask, pads label_ids
        query_embeds =  self.emb(input_ids).detach()

        # end of instruction emebddings 
        end_instr = self.emb(torch.LongTensor(self.get_response_template_ids())).detach()

        if padding_side == 'right':
           
            # getting query lengths without padding
            query_lengths = (input_ids != self.tokenizer_right.pad_token_id).sum(dim=1) + 1 # + 1 because we pad with bos which counts as input token
        else:
             query_lengths = torch.LongTensor([len(i) for i in input_ids])

        if doc_embeds == None:
            doc_embeds = torch.empty(0, 0)


        if label_ids == None:
            label_embeds = torch.empty(0, 0)
            label_lengths = [0] * query_lengths.size(0)
        else:
            label_lengths = (label_ids != self.tokenizer_right.pad_token_id).sum(dim=1)
            label_embeds = self.emb(label_ids).detach()
        
        # input tensor with combining query and doc embeds filled with padding
        inputs_embeds = self.pad_emb.repeat(query_embeds.size(0), query_embeds.size(1) + doc_embeds.size(1) + label_embeds.size(1) + end_instr.size(0), 1)
        # fill with query and doc embeds

        for i in range(query_lengths.size(0)):
            doc_embeds_size = doc_embeds[i].size(0) if doc_embeds.size(0) != 0 else 0
            # fill with unpadded query
            inputs_embeds[i, :query_lengths[i]] = query_embeds[i, :query_lengths[i]]
            # fil in doc embeds
            if doc_embeds_size != 0:
                # fill with doc embeds + eos
                inputs_embeds[i, query_embeds[i].size(0):query_embeds[i].size(0)+doc_embeds_size] = doc_embeds[i]
            
            inputs_embeds[i, query_lengths[i] + doc_embeds_size : query_lengths[i]+ end_instr.size(0) + doc_embeds_size] = end_instr

            # fill in label embeds
            if label_embeds.size(0) != 0:
                inputs_embeds[i, query_lengths[i]+ doc_embeds_size:query_lengths[i]+doc_embeds_size+label_lengths[i]] = label_embeds[i, :label_lengths[i]]
            
        # make label ids
        if label_ids is not None:
            label_ids_with_ignore = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), ignore_index, dtype=torch.long)
            # make labels 
            for i in range(query_lengths.size(0)):
                query_and_doc_length = query_lengths[i] + doc_embeds_size if doc_embeds.size(0) != 0 else query_lengths[i]
                label_ids_with_ignore[i, query_and_doc_length:query_and_doc_length+label_lengths[i]] = label_ids[i][:label_lengths[i]]

        input_length = [doc_embeds.size(1) + query_lengths[i] + label_lengths[i] + end_instr.size(0) for i in range(len(input_ids))]

        if padding_side == 'right':
            attention_mask = torch.LongTensor([[1]* input_length[i] + [0]* (inputs_embeds.size(1) - input_length[i]) for i in range(len(input_length))])
        else:
            # attention mask
            attention_mask = torch.LongTensor([[0]* (inputs_embeds.size(1) - input_length[i]) +  [1]* input_length[i] for i in range(len(input_length))])
              
        return (inputs_embeds, attention_mask, label_ids_with_ignore if label_ids != None else None)
        
    def prediction_step(self, model, model_input, label_ids=None):
        inputs_embeds, attention_mask = model_input['inputs_embeds'], model_input['attention_mask']
        output = model(inputs_embeds=inputs_embeds.to('cuda').bfloat16(), attention_mask=attention_mask.to('cuda'), labels=label_ids.to('cuda'))
        return output.logits, output.loss
    
    def get_response_template_ids(self):
        response_template =  self.get_response_template()
        return self.tokenizer.encode(response_template, add_special_tokens=False)

    def get_response_template(self):
        return "[/INST]"
    
    def generate(self, model_input):
        inputs_embeds, attention_mask = model_input['inputs_embeds'], model_input['attention_mask']
        #if doc_embeds is None:
        if False:
            query_tokenized = model_input['tokenized_query']
            output_ids =  self.model.generate(
                **query_tokenized,
                max_new_tokens=self.max_new_tokens, 
                do_sample=False,
            )
            prompt_len = query_tokenized.input_ids.size(1)

        else:
            output_ids =  self.model.generate(
                inputs_embeds=inputs_embeds.to('cuda').bfloat16(), 
                attention_mask=attention_mask.to('cuda'), 
                max_new_tokens=self.max_new_tokens, 
                do_sample=False,
                )
            #prompt_len = doc_embeds.size(1)
        #generated_ids = output_ids[:, prompt_len:].squeeze()
        #generated_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return generated_response

    def collate_fn(self, examples, eval=False, **kwargs):
            q_ids = [e['q_id'] for e in examples]
            labels = [[e['label']] if isinstance(e['label'], str) else e['label'] for e in examples]
            first_labels_only = [e['label'] if isinstance(e['label'], str) else e['label'][0] for e in examples]

            # add \n add remove it after tokenization, necessary because of llama's contexualized tokenizer
            first_labels_only_eos = ['\n' + lab + self.tokenizer.eos_token for lab in first_labels_only]
            # tokenize labels
            label_ids = self.tokenizer_right(
                first_labels_only_eos, 
                padding=True, 
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt", 
                add_special_tokens=False,
                ).input_ids[:, 2:]


            query = [e['query'] for e in examples]
            ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)

            instr = [None] * len(examples)
            if 'd_idx' in examples[0]:
                d_idxs = [e['d_idx'] for e in examples]
                # if embeds_dataset is provided fetch embeddings
                doc_embeds = get_doc_embeds_from_dataset(d_idxs, self.embeds_dataset)
            else:
                doc_embeds=None

            tokenizer = self.tokenizer if eval else self.tokenizer_right
            instr = [self.format_instruction(e) for e in examples]

            # remove last </INSTR> add in prepare function
            instr = [inst.replace(self.get_response_template(), ' ') for inst in instr]
            # tokenize query
            instr_tokenized = tokenizer(
                instr, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                add_special_tokens=False,
                max_length=self.max_length,
                )
            
            input_ids = instr_tokenized.input_ids
        
            inputs_embeds, attention_mask, label_ids_processed = self.prepare_input(
                doc_embeds=doc_embeds, 
                input_ids=input_ids, 
                label_ids=None if eval else label_ids , 
                padding_side=tokenizer.padding_side,
                )

            data_dict = {
                'model_input': {
                    'inputs_embeds': inputs_embeds,
                    'attention_mask': attention_mask,     
                },
                'label_ids': label_ids_processed,
                'q_id': q_ids, 
                'query': query, 
                'd_idx': d_idxs if 'd_idx' in examples[0] else None,
                'label': labels, 
                'ranking_label': ranking_label,
                'instruction': instr,
            }
            if eval:
                data_dict['model_input']['tokenized_query'] =  instr_tokenized
            return data_dict

    def format_instruction(self, sample):
        # will be injected into formatted prompt string
        question = sample['query']
        # in case we have previously retrieved documents
        if 'doc' in sample:
            docs = ''
            for i, doc in enumerate(sample['doc']):
                doc = ' '.join(doc.split()[:self.max_doc_len])
                docs += f"Document {i+1}: {doc}\n"
            compiled_prompt = self.compile_prompt(self.prompt.system, self.prompt.user, question, docs)
        else:
            # without retrieval we don't put documents in the prompt
            compiled_prompt = self.compile_prompt(self.prompt.system_without_docs, self.prompt.user_without_docs, question)
        return compiled_prompt