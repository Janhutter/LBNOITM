# -*- coding: utf-8 -*-

import base64
import json
import http.client
from tqdm import tqdm 
import numpy as np 
import os
import time

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
        conn.request('POST', '/v1/chat-completions/HCX-002', json.dumps(completion_request), headers)
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



def run_llm(client, messages):
    request_data = {
        'messages': messages,
        'maxTokens': 300,
        'temperature': 0.5,
        'topK': 0,
        'topP': 0.8,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True
    }
    time.sleep(1)
    response = client.execute(request_data)    
    try :response['content']
    except: 
        print(response)
        sss
    #print (response)
    return response['content']



def create_instruction(question,answer,prediction):
    prefix =  [{'role': 'system',
             'content': "You are an evaluation tool. Just answer by [Yes] or [No]."}]
    prefix.extend([{'role': 'user',
             'content': f"Here is a question, a golden answer and an AI-generated answer. Can you judge whether the AI-generated answer is correct according to the question and golden answer, simply answer Yes or No.\n Question: {question}. \ngolden answer: {answer} \n Generated answer: {prediction}"}
             ]
             )
    return prefix    


# for evaluation
class ClovaAI():
    def __init__(self):
        self.client = CompletionExecutor(
            host='api-hyperclova.navercorp.com',
            client_id=os.environ.get("CLIENT_ID"),
            client_secret=os.environ.get("CLIENT_SECRET") 
            )

    def __call__(self, predictions, references, questions):
        scores=list()
        for q,r,p in (tq:= tqdm(zip(questions,references,predictions),total=len(questions),desc=f"score:  0.0%")):
            prompt = create_instruction(q,r[0],p)
            response = run_llm(self.client,prompt)
            score = 1 if "yes" in response.lower() else 0       
            scores.append(score)
            tq.set_description(f"score: {np.mean(scores)* 100:4.1f}%")

        return np.mean(scores), scores
