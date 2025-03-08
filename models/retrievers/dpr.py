from transformers import AutoModel, AutoTokenizer, DefaultDataCollator
import torch
from models.retrievers.retriever import Retriever

class DPR(Retriever):
    def __init__(self, model_name=None):

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)

    # Mean pooling
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        content_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return content_embeddings

    def __call__(self, kwargs):
        kwargs = {key: value.to(self.device) for key, value in kwargs.items()}
        outputs = self.model(**kwargs)
        # pooling over hidden representations
        emb = self.mean_pooling(outputs[0], kwargs['attention_mask'])
        return {
                "embedding": emb
            }

    def collate_fn(self, batch, query_or_doc=None):
        content = [sample['content'] for sample in batch]
        return_dict = self.tokenizer(content, padding=True, truncation=True, return_tensors='pt')
        return return_dict
    
    def similarity_fn(self, query_embds, doc_embds):
        return torch.mm(query_embds, doc_embds.t())
    