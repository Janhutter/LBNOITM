from modules.rag import RAG
from utils import move_finished_experiment
 
class DefaultRAG(RAG):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, dataset_split):

        dataset = self.datasets[dataset_split]
        query_dataset_name = self.datasets[dataset_split]['query'].name
        doc_dataset_name = self.datasets[dataset_split]['doc'].name

        # retrieve
        if self.retriever != None:
            query_ids, doc_ids, _ = self.retrieve(
                    dataset, 
                    query_dataset_name, 
                    doc_dataset_name,
                    dataset_split, 
                    self.retrieve_top_k,
                    )  
        else:
            query_ids, doc_ids = None, None
        # rerank
        if self.reranker !=  None:
            query_ids, doc_ids, _ = self.rerank(
                dataset, 
                query_dataset_name, 
                doc_dataset_name,
                dataset_split, 
                query_ids, 
                doc_ids,
                self.rerank_top_k,
                )

        # generate
        if self.generator !=  None:
            questions, _, predictions, references = self.generate(
                dataset, 
                dataset_split,
                query_ids, 
                doc_ids,
                )
            # eval metrics
            self.eval_metrics(
                dataset_split, 
                questions, 
                predictions, 
                references
                )

        move_finished_experiment(self.experiment_folder)
