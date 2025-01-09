from transformers import AutoModel, AutoTokenizer
import torch
from models.retrievers.retriever import Retriever

class RetroMAE(Retriever):
    def __init__(self, model_name=None):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch.float16, add_pooling_layer=False)
        self.model.eval()

        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)

    def collate_fn(self, batch, query_or_doc=None):
        content = [sample['content'] for sample in batch]
        return_dict = self.tokenizer(content, padding=True, truncation=True, return_tensors='pt')
        return return_dict

    def __call__(self, kwargs):
        kwargs = {key: value.to(self.device) for key, value in kwargs.items()}
        outputs = self.model(**kwargs)
        # pooling over hidden representations
        emb = outputs[0][:,0]
        return {
                "embedding": emb
            }

    def similarity_fn(self, query_embds, doc_embds, eps=1e-8):
        query_embds_n, doc_embds_n = query_embds.norm(dim=1)[:, None], doc_embds.norm(dim=1)[:, None]
        query_embds_norm = query_embds / torch.max(query_embds_n, eps * torch.ones_like(query_embds_n))
        doc_embds_norm = doc_embds / torch.max(doc_embds_n, eps * torch.ones_like(doc_embds_n))
        scores = torch.mm(query_embds_norm, doc_embds_norm.t())
        return scores