from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
from hydra.utils import instantiate
from datasets import Dataset

# reranking
class Reduce():
    def __init__(self, init_args=None, batch_size=1, topk=None):
        self.batch_size = batch_size
        self.init_args = init_args
        self.model = instantiate(self.init_args)
        self.topk = topk

    @torch.no_grad()
    def reduce(self, dataset):
        self.model.model.to('cuda')
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.model.collate_fn,
            num_workers=4
        )

        reduced_docs_map = defaultdict(list)
        proportions_overall = []
        # run inference on the dataset
        for batch in tqdm(dataloader, desc=f'Reducing: {self.model.model_name}'):
            batch_d_ids = [item for item in batch['d_id']]  # Flatten the list of d_ids
            output_embeds = self.model(batch)
            doc_tokenized = batch['input_ids']
            doc_reduced_batch, proportions = self.model.reduce_fn(output_embeds, doc_tokenized, self.topk)
            proportions_overall.extend(proportions)
            # Map each reduced doc to its d_id
            for d_id, reduced_doc in zip(batch_d_ids, doc_reduced_batch):
                reduced_docs_map[d_id] = reduced_doc

        # Update dataset docs based on the d_id mapping
        updated_dataset_dict = defaultdict(list)
        for item in dataset:
            for key in item:
                if key == "doc":
                    new_docs = [reduced_docs_map[d_id] for d_id in item['d_id']]
                    updated_dataset_dict[key].append(new_docs)
                else:
                    updated_dataset_dict[key].append(item[key])

        # Create a new Dataset from the updated records
        updated_dataset = Dataset.from_dict(updated_dataset_dict)
        print(f"Average proportion of tokens kept: {sum(proportions_overall) / len(proportions_overall)}"   )
        return updated_dataset
