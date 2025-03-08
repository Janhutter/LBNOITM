from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
import torch
from models.generators.generator import Generator
import torch.nn.functional as F
#from accelerate import Accelerator
import warnings
import numpy as np
from utils import prepare_labels
from peft import AutoPeftModelForCausalLM, PeftConfig
import random
import os
import json
import torch
random.seed(42)
class LLM(Generator):
    def __init__(self, 
                model_name=None, 
                max_new_tokens=1, 
                max_doc_len=100,
                max_length=None,
                prompt=None,
                quantization=None,
                visualize_attention=False
                 ):

        # device_index = Accelerator().process_index
        # device_map = {"": device_index}

        self.max_length = max_length
        self.model_name = model_name
        self.max_doc_len = max_doc_len
        self.quantization = quantization

        self.visualize= visualize_attention

         # get tokenizer of lora adapter if exists else use models' tokenizer

        if quantization == "no":
            warnings.warn(f"Could not find PeftConfig for {model_name}. Using regular model.")
            tokenizer_name = self.model_name
            model_class = AutoModelForCausalLM
        else:
            try:
                config = PeftConfig.from_pretrained(model_name)
                tokenizer_name = config.base_model_name_or_path
                model_class = AutoPeftModelForCausalLM
                print("loading adaptor")
            except:
                warnings.warn(f"Could not find PeftConfig for {model_name}. Using regular model.")
                tokenizer_name = self.model_name
                model_class = AutoModelForCausalLM
        print(tokenizer_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:

            config_dict = os.path.join(tokenizer_name, 'config.json')
            with open(config_dict, 'r') as f:
                config = json.load(f)
            tokenizer_name = config['_name_or_path']
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.bos_token


        if quantization == "int8":
            quant_config = BitsAndBytesConfig(
                llm_int8_enable_fp32_cpu_offload=True
            )
            if self.visualize:
                self.model = model_class.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    attn_implementation="eager",
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                    output_attentions=True
                )
            else:
                self.model = model_class.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                )


        elif quantization == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
            )
            if self.visualize:
                self.model = model_class.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    attn_implementation="eager",
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                    output_attentions=True
                )
            else:
                self.model = model_class.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                )
        elif self.visualize:
            self.model = model_class.from_pretrained(
                self.model_name,
                device_map='auto',
                output_attentions=True
            )
        else:
            self.model = model_class.from_pretrained(
                self.model_name,
                device_map='auto',
            )

        # self.model.merge_and_unload()
        #self.model.config.use_cache = False
        self.model = self.model.bfloat16()
        self.model.eval()
        self.model.config.pretraining_tp = 1
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt

    def get_response(self):
        return '\nResponse:\n'

    def get_response_template_ids(self):
        response_template =  self.get_response()
        return self.tokenizer.encode(response_template, add_special_tokens=False)

    def prediction_step(self, model, model_input, label_ids=None):
        output = model(**model_input, labels=label_ids)
        return output.logits, output.loss

    def tokenize_instruction(self, instruction, input_length):
        # Split the instruction into individual sections
        # so we get [instruction, docs1, doc2, ..., question]
        splitted = instruction.split("\nDocument")
        
        for i, doc in enumerate(splitted):
            if i == 0:
                continue
            splitted[i] = "Document" + doc
            splitted[i-1] = splitted[i-1] + "\n"
        
        last_doc, question = splitted[-1].split("Question:")
        question = "Question:" + question
        splitted[-1] = last_doc
        splitted.append(question)

        # Create bins for each section
        bins = []
        start = 0
        for i, doc in enumerate(splitted):
            tokenized_doc = self.tokenizer.encode(doc)
            if i != 0:
                tokenized_doc = tokenized_doc[1:]
            if i != len(splitted) - 1:
                tokenized_doc = tokenized_doc[:-1]
            bins.append((start, start + len(tokenized_doc)))
            start += len(tokenized_doc)

        # replace the end of the last bin with the end of the output tokens
        bins[-1] = (bins[-1][0], input_length + 1)

        return bins

    def get_normalized_attention(self, attentions, bins):
        # take the mean over the sequence length
        document_scores = []
        for start, end in bins:
            doc_attention = attentions[start:end]
            # take the mean over the output tokens
            document_scores.append(doc_attention.mean())

        # normalizing
        document_scores = [round(float(i)/sum(document_scores), 4) for i in document_scores]
        return document_scores
    

    def visualize_attention(self, instr_tokenized, instr_untokenized):
        input_ids = instr_tokenized['input_ids'].to("cuda")
        attention_mask = instr_tokenized['attention_mask'].to("cuda")
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True
        )
        # (attention per generated token, layers, batch_size, num_heads, sequence_length, sequence_length)
        # 
        # take the attention while generating the first token
        output_attentions = output['attentions']

        all_layers = []

        bins = self.tokenize_instruction(instr_untokenized[0], input_ids.shape[1])
        output_attentions_list = [
            torch.stack(output_attentions[attn_index], dim=0)[:, :, :, :, :input_ids.shape[1]]
            for attn_index in range(1, len(output_attentions))
        ]

        print('instr_tokenized', instr_tokenized)
        print('instr_untokenized', instr_untokenized)

        print('input_ids.shape:', input_ids.shape)

        output_ids = output['sequences']

        prompt_len = instr_tokenized['input_ids'].size(1)

        generated_ids = output_ids[:, prompt_len:]

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        print('response:', decoded[0])
        attentions = torch.stack(output_attentions_list, dim=0)
        print('attentions.shape:', attentions.shape)

        attentions = attentions.to(torch.float32)
        attentions = attentions.mean(axis=0)
        amount_of_layers = attentions.size(0)
        used_layers = [0, amount_of_layers // 2, amount_of_layers - 1]
        print('used_layers:', used_layers)
        for layer in used_layers:
            all_document_scores = []
            # take the mean over the heads
            layer_attentions = attentions[layer][0, :, :, :]
        

            print('again attentions.shape', layer_attentions.shape)
            # take the mean over the heads
            layer_attentions = layer_attentions.mean(axis=0).squeeze(0).detach().cpu().numpy()

            document_scores = self.get_normalized_attention(layer_attentions, bins)

            all_document_scores.append(document_scores)
            

            # go through all_document_scores and determine the average score for each position
            all_document_scores = torch.tensor(all_document_scores)
            all_document_scores = all_document_scores.mean(axis=0).tolist()
            all_layers.append(all_document_scores)
            print('all_document_scores:', all_document_scores)
            print('doc position with max value:', all_document_scores[1:-1].index(max(all_document_scores[1:-1])))

        return [str([decoded[0], all_layers])]


    def generate(self, instr_tokenized):
        input_ids = instr_tokenized['input_ids'].to("cuda")
        attention_mask = instr_tokenized['attention_mask'].to("cuda")
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )

        prompt_len = instr_tokenized['input_ids'].size(1)
        generated_ids = output_ids[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return decoded

    def collate_fn(self, examples, eval=False, **kwargs):
        ignore_index = -100
        q_ids = [e['q_id'] for e in examples]

        input_ids_list = [e["tokenized_input"]["input_ids"][0] for e in examples]
        attention_mask_list = [e["tokenized_input"]["attention_mask"][0] for e in examples]

        label = [e['label'] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)
        instr = [e["formatted_instruction"] for e in examples]

        # Determine the maximum sequence length from input_ids
        max_length = max(len(ids) for ids in input_ids_list)

        # Perform left padding manually for input_ids
        input_ids_tensor = torch.stack([
            torch.cat([torch.full((max_length - len(ids),), self.tokenizer.pad_token_id, dtype=torch.long),
                       torch.tensor(ids, dtype=torch.long)])
            for ids in input_ids_list
        ])

        # Assuming 0 is the appropriate padding value for attention_mask
        attention_mask_tensor = torch.stack([
            torch.cat(
                [torch.full((max_length - len(mask),), 0, dtype=torch.long), torch.tensor(mask, dtype=torch.long)])
            for mask in attention_mask_list
        ])
        model_input = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
        }
        data_dict = {}
        # prepare labels only for training 
        if not eval:

            response_token_ids = self.get_response_template_ids()
            label_ids = prepare_labels(model_input['input_ids'], response_token_ids[1:], ignore_index=ignore_index)
            data_dict['label_ids'] =  label_ids

        
        data_dict.update({
            'model_input': model_input,
            'q_id': q_ids, 
            'query': query, 
            'instruction': instr,
            'label': label, 
            'ranking_label': ranking_label,
        })

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
        return compiled_prompt + self.get_response()
