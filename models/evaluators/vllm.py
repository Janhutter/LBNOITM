from transformers import AutoTokenizer

from tqdm import tqdm
import torch
import re
import numpy as np

from vllm import LLM as vllm
from vllm import  SamplingParams


class LLM:
    def __init__(self, model_name, batch_size=1, custom_format_instruction=None, pos_word=None, neg_word=None):
        self.batch_size = batch_size
        self.custom_format_instruction = custom_format_instruction

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.quantization = None
        if self.quantization is None:
            self.model = vllm(model=self.model_name,tensor_parallel_size=torch.cuda.device_count(),gpu_memory_utilization=0.9,max_model_len=1,enforce_eager=False,kv_cache_dtype="fp8_e5m2")        
        else:
            self.model = vllm(model=self.model_name,tensor_parallel_size=torch.cuda.device_count(),quantization=self.quantization)
        self.sampling_params =  SamplingParams(temperature=1,best_of=1, top_p=1, top_k=-1,logprobs=1, max_tokens=1)

        self.pos_word = pos_word if pos_word != None else 'true' 
        self.neg_word = neg_word if neg_word != None else 'false'
        
        self.pos_tokenid, self.neg_tokenid = self.tokenizer.encode(f'\n{self.pos_word}', add_special_tokens=False)[-1], self.tokenizer.encode(f'\n{self.neg_word}', add_special_tokens=False)[-1]



    def format_instruction(self, sample):
        reference = sample['reference']
        if isinstance(reference, str):
            reference = [reference]
        # reference = ', '.join(reference)
        return f"""Assess whether the candidate answer effectively answers the question in comparison to at least one of the provided reference answers. Consider factors such as relevance, correctness, and completeness in your
Question: {sample['question']}
Reference Answers: {reference}
Candidate Answer: {sample['candidate']}
Output: {{"""

    def collate_fn(self, examples, max_length=512):
        instr = [self.format_instruction(sample) if self.custom_format_instruction == None else self.custom_format_instruction(sample) for sample in examples]  # Add prompt to each text
        #instr_tokenized = self.tokenizer(instr, padding=True, truncation=True, return_tensors="pt")
        return instr #instr_tokenized, instr

    @torch.no_grad()
    def __call__(self, predictions, references, questions):
        # Loading the TensorFlow Hub model
        assert len(predictions) == len(references) == len(questions)
        examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i]}  for i in range(len(predictions))]
        
        instrs = self.collate_fn(examples)
        # The outputs are raw logits.
        scores = list()
        # Perform batch inference
        for i in tqdm(range(0, len(instrs), self.batch_size), desc=f'LLM evaluation with {self.model_name}...'):
            outputs = self.model.generate(instrs[i:i+self.batch_size], self.sampling_params)
            decoded = [output.outputs[0].text for output in outputs]
            print(decoded)
            model_scores = [ outputs.outputs[0].logprobs for output in outputs]
            print(model_scores)
            negscore =  [ model_score[self.neg_tokenid] for model_score in model_scores]
            posscore = [ model_score[self.pos_tokenid] for model_score in model_scores]
            print (negscore,posscore)
            pos_prob = torch.softmax(torch.as_tensor([negscore,posscore]), 1)[:,1]
            print (pos_prob)
            sss
            for i, score in enumerate(pos_prob):
                scores.append(score.float())

        torch.cuda.empty_cache()
        return np.mean(scores), scores

