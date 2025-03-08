from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from models.reducers.reducer import Reducer
class CrossEncoder(Reducer):
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()
        self.reverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self.sep_token = self.tokenizer.sep_token_id

    def collate_fn(self, examples):
        query = [e['query'] for e in examples]
        doc = [e['doc'] for e in examples]
        d_id = [e['d_id'] for e in examples]
        query_reformed = []
        #query is a list of strings, d_ids are nested list, need to duplicate for each query wih respect to the lenth of doc in the query
        for i in range(len(query)):
            query_reformed += [query[i]] * len(doc[i])
        query = query_reformed

        #doc and d_ids are nested list, need to flat
        doc = [item for sublist in doc for item in sublist]
        d_id = [item for sublist in d_id for item in sublist]

        inp_dict = self.tokenizer(query, doc, padding=True, truncation="only_second", return_tensors='pt')
        inp_dict['d_id'] = d_id
        return inp_dict


    def __call__(self, kwargs):
        #score = self.model(**kwargs.to('cuda')).logits
        outputs = self.model(**kwargs.to('cuda')).logits
        print(outputs)
        return {}
        #return {
                #"score": score
            #}

    def reduce_fn(self, doc_embeds, combineds_tokenized, topk=None):
        docs_reduced = []
        for doc_embed, combined_tokenized in zip(doc_embeds["embedding"], combineds_tokenized):
            sep_index = (combined_tokenized == self.sep_token).nonzero(as_tuple=True)[0][0]

            # Split the tensor into question and document parts
            question_tokenized = combined_tokenized[:sep_index]
            doc_tokenized = combined_tokenized[sep_index + 1:]

            doc_rep = doc_embed.squeeze()
            col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
            weights = doc_rep[col].cpu().tolist()
            d = {k: v for k, v in zip(col, weights)}
            sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True) if k in doc_tokenized}
            if topk is None:
                topk = len(sorted_d)

            top_k = list(sorted_d.keys())[:topk]
            reduced_tokenized = [token.item() for token in doc_tokenized if token in top_k]
            doc_reduced = self.tokenizer.decode(reduced_tokenized)

            print("question: ", self.tokenizer.decode(question_tokenized))
            print("doc: ", self.tokenizer.decode(doc_tokenized))
            print("reduced doc: ", doc_reduced)
            docs_reduced.append(doc_reduced)
        return docs_reduced