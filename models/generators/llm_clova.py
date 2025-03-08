# -*- coding: utf-8 -*-
import os
from models.generators.generator import Generator
import time
import base64
import json
import http.client
import os

class CompletionExecutor:
    def __init__(self, host, client_id, client_secret, access_token=None):
        self._host = host
        # client_id and client_secret are used to issue access_token.
        # You should not share this with others.
        self._client_id = client_id
        self._client_secret = client_secret
        # Base64Encode(client_id:client_secret)
        self._encoded_secret = base64.b64encode('{}:{}'.format(self._client_id, self._client_secret).encode('utf-8')).decode('utf-8')
        self._access_token = access_token

    def _refresh_access_token(self):
        headers = {
            'Authorization': 'Basic {}'.format(self._encoded_secret)
        }

        conn = http.client.HTTPSConnection(self._host)
        # If existingToken is true, it returns a token that has the longest expiry time among existing tokens.
        conn.request('GET', '/v1/auth/token?existingToken=true', headers=headers)
        response = conn.getresponse()
        body = response.read().decode()
        conn.close()

        token_info = json.loads(body)
        self._access_token = token_info['result']['accessToken']

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            #'Accept': 'text/event-stream',
            'Authorization': 'Bearer {}'.format(self._access_token)
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/v1/chat-completions/HCX-003', json.dumps(completion_request), headers)
        response = conn.getresponse().read().decode('utf-8')
        result = json.loads(response) 
        conn.close()
        return result

    def execute(self, completion_request):
        if self._access_token is None:
            self._refresh_access_token()

        res = self._send_request(completion_request)
        if res['status']['code'] == '40103':
            # Check whether the token has expired and reissue the token.
            self._access_token = None
            return self.execute(completion_request)
        elif res['status']['code'] == '20000':
            return res['result']['message']
        else:
            return 'Error'



class Clova(Generator):
    def __init__(self, 
                model_name="gpt-3.5-turbo", 
                max_new_tokens=1, 
                max_doc_len=100,
                max_length=None,
                prompt=None
                 ):
        #self.client = openai.OpenAI(api_key = os.environ.get("OPENAI_API_KEY"),)
        self.client = CompletionExecutor(
            host='api-hyperclova.navercorp.com',
            client_id=os.environ.get("CLIENT_ID"),
            client_secret=os.environ.get("CLIENT_SECRET") 
            )
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.chat_template = True
        self.max_doc_len = max_doc_len
        self.max_length = max_length

    def run_llm(self,client, messages):
        responses=[]
        for  msg in messages:
            request_data = {
            'messages': msg,
            'maxTokens': 300,
            'temperature': 0.5,
            'topK': 0,
            'topP': 0.8,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': True
            }
            response = client.execute(request_data)    
            responses.append(response['content'])
        return responses

    
    def generate(self, messages):
        response = self.run_llm(self.client, messages=messages)
        time.sleep(1)
        return response

    # only required for training
    def prediction_step(self, model, model_input, label_ids=None):
        # e.g.       
        # output = model(**model_input, labels=label_ids)
        # return output.logits, output.loss
        pass
    
    def collate_fn(self, examples, eval=False, **kwargs):
        q_ids = [e['q_id'] for e in examples]
        instr = [self.format_instruction(e) for e in examples]

        label = [e['label'] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)

        data_dict = {}
        # for inference just format and tokenize instruction 
        instr = [self.format_instruction(e) for e in examples]
        model_input =  instr
        
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
            compiled_prompt = self.compile_prompt(self.prompt.system_without_docs, self.prompt.user_without_docs,question)
        return compiled_prompt        



    def compile_prompt(self, system_prompt, user_prompt, question, docs=None):
        # check if chat template allows for system prompts

        # if has chat_template e.g. gamma does not use it
        if self.chat_template == None:
            user_prompt_with_values = eval(user_prompt).replace(':\ ', ': ')
            return f"{system_prompt}\n{user_prompt_with_values}"
        else:
            if True: #'system' in self.tokenizer.chat_template:
                instr_prompt = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": eval(user_prompt).replace(':\ ', ': ')}
                ]
            # if no system prompts are allowed just append system prompt to user prompt
            else:
                user_prompt_with_values = eval(user_prompt).replace(':\ ', ': ')
                instr_prompt = [
                    {"role": "user", "content": f"{system_prompt}\n{user_prompt_with_values}"}
                ]    
            return instr_prompt #self.tokenizer.apply_chat_template(instr_prompt, tokenize=False)