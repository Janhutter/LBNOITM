# RAG Benchmark

## Requirements

To create a new env:
```bash
conda create -n "rag" python=3.10 
```


Please use `Python 3.10`. To install all required python packages run:

```bash
pip3 install -r requirements.txt
```

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
```


## Quick Start
Example experiment with retrieval using `BM25`, reranking using `MiniLM6`, generation using `tinyllama-chat` in debug mode (using 15 queries) on `kilt_nq`.
```bash
  python3 main.py retriever="bm25" reranker="minilm6" generator='tinyllama-chat' dataset='kilt_nq' +debug=True
```

## Overview
While `retriever`, `reranker` and `generator` are optional and can be `None`, the `dataset` argument must always be provided. 

Datasets will be downloaded, pre-processed, indexed and saved if they do not exist yet, otherwise they will be loaded from `dataset_folder` and `index_folder` respectively. All datasets can be overwritten by adding `+overwrite_datasets=True` as an argument (`Caution`: This might overwrite collections that take long long to encode). In case the indexing is interrupted you can continue encoding a collection from batch 1000 by additionally using the argument `+continue_batch=1000`.

Retrieval, reranking runs will be loaded from files if they already exist in `runs`, otherwise they will be created. 

Retrieval will only be evaluated if the `query` dataset contains the field `ranking_label`.

Experiments are saved under `experiments_folder`. The experiments folder is named after the hash of the config, unless the experiment is finished the folder name will contain the prefix `tmp_`. The script will be aborted if an experiment with the exact same parameters has been run before. To overwrite the experiment add `+overwrite_exp=True` as an argument.

To overwrite an existing index (and subsequently the ranking run) add `+overwrite_index=True` as an argument.

To print the results in a table run. By default this will print all experiments that contain generation metric files in `experiments/` and sort them by the `generator`.

```bash
python3 print_results.py --folder experiments/
```


## Code Structure
```
|-- config
|   |-- dataset/
|   |-- generator/
|   |-- prompt/
|   |-- reranker/
|   |-- retriever/
|   |-- train/
|   |-- rag.yaml
|-- evaluation
|-- eval.py
|-- main.py
|-- models/
|   |-- evaluators/
|   |-- generators/
|   |-- rerankers/
|   |-- retrievers/
|-- modules/
|   |-- dataset_processor.py
|   |-- evaluation.py
|   |-- generate.py
|   |-- rag.py
|   |-- rerank.py
|   |-- retrieve.py
|   `-- utils.py
|-- scripts/
|-- README.md
|-- requirements.py
|-- utils.py

```

## Evaluation
Non-neural metrics will be calculated automatically. Neural metrics such as `BEM` and `LLM` need to be evoked seperately.

By default `eval.py` will scan all folders in `experiments/` and evaluate them sequentially. To evaluate a single folder pass the folder using `--folder`. To avoid running out of memory either run `BEM` using `--bem` or run `LLM` using `--llm`. A csv file will automatically be saved to `results/` containing the table in `csv` format.

```bash
python3 eval.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --llm
```



## Training
For training a model add a training config e.g. `train=lora` as an argument, e.g.

```bash
python3 main.py retriever='bm25' generator='llama-2-7b-chat' dataset='kilt_nq' train='lora'
```

For training the `dev` dataset split that is defined in the config is split in `train` and `test` splits ( default test size: `0.01`). The best model (according to the newly generated `test` split) is loaded after the training and evaluated on the `dev`  dataset split.


## Output files
Example files generated for split `dev` using `naver_splade-cocondenser-selfdistil` as a retriever.
- `config.yaml` The parameters of the experiment in yaml format.
- `eval_dev_generation_time.json` The generation time in json format.
- `eval_dev_metrics.json` Generation evaluation metrics in json format.
- `eval_dev_out.json` Output of the generation, contains `q_id` (str), `response` `(str)` the generated response, `label` `(list (str))` the answer reference (multiple possible), `instruction` `(str)` the instruction given to the generator, `ranking_label` `(list(list(str)), optional)` ids of reference paragraph (again multiple references possible).
- `run.retrieve.top_5.kilt_nq.dev.naver_splade-cocondenser-selfdistil.trec` The retrieval run in `trec` format.
- `eval_dev_ranking_metrics.json` Retrieval evaluation metrics in json format.

## Printing Results Table

Simply run:
```bash
python3 print_results.py --folder experiments/
```

# Extend Library
 
## Add Retriever
Retrievers inherit from the abstract `Retriever` class and thus needs to follow this structure:

```python
from models.retrievers.retriever import Retriever

class NewRetriever(Retriever):
  def __init__(self, model_name=None):
    self.model_name = 'new_retriever'

  @abstractmethod
  def __call__(self, kwargs):
    # model inference e.g. return model(**kwargs)
    pass

  @abstractmethod
  def collate_fn(self, batch, query_or_doc=None):
    # implement collate_fn here
    pass 

  @abstractmethod
  def similarity_fn(self, q_embds, doc_embs):
    # similarity fn to use e.g. torch.mm.(q_embs, doc_embs.t())
    pass 
```
We save it under `models/retrievers/new_retriever.py`.

As the second step create a config for this model under `config/retrievers/new_retriever.yaml`. 

```yaml
init_args: 
  _target_: models.retrievers.new_retriever.NewRetriever
  model_name: "new_retriever"
batch_size: 1024
batch_size_sim: 256
```

To use the model add the argument `retriever='new_retriever'`:

```python
python3 main.py retriever='new_retriever'
```


## Add Reranker
Rerankers inherit from the abstract `Reranker` class and thus needs to follow this structure:

```python
from models.rerankers.reranker import Reranker

class NewReranker(Reranker):
  def __init__(self, model_name=None):
    self.model_name = 'new_reranker'

  @abstractmethod
  def __call__(self, kwargs):
    # model inference e.g. self.model(**kwargs)
    pass

  @abstractmethod
  def collate_fn(self, batch, query_or_doc=None):
    # implement collate function 
    pass

```

We save it under `models/rerankers/new_reranker.py`.

As the second step create a config for this model under `config/rerankers/new_reranker.yaml`. 

```yaml
init_args: 
  _target_: models.rerankers.new_reranker.NewReranker
  model_name: "new_reranker"
batch_size: 2048
```

To use the model add the argument `reranker='new_reranker'`:

```python
python3 main.py reranker='new_reranker'
```

### Add Generator
The Generator inherits from the abstract `Generator` class and thus needs to follow this structure:

```python
from models.generators.generator import Generator

class NewGenerator(Generator):
  def __init__(self, model_name=None):
    self.model_name = 'new_generator'

  @abstractmethod
  def generate(self, inp):
    # generation e.g. self.model(**inp)
    pass
  @abstractmethod
  def collate_fn(self, inp):
    pass

  # only required for training
  @abstractmethod
  def prediction_step(self, model, model_input, label_ids=None):
      # e.g.       
      # output = model(**model_input, labels=label_ids)
      # return output.logits, output.loss
      pass 

```

We save it under `models/generators/new_generator.py`.

As the second step create a config for this model under `config/generators/new_generator.yaml`.


```yaml
defaults:
  - prompt: basic
init_args: 
  _target_: models.generators.new_generator.NewGenerator
  model_name: "new_generator"
  max_new_tokens: 128
batch_size: 32
max_inp_length: null
```


To use the model add the argument `generator='new_generator'`:

```python
python3 main.py generator='new_generator'
```


## Add Dataset
A dataset config contains two entries: `doc` for the collection and `query` for the queries.

A query dataset **must** contain the fields **`id`**, `wikipedia_id` (optional), **`content`** after the processing. 

A document dataset **must** contain the fields **`id`**, and **`content`** after the processing.

Define a new dataset class in `modules/dataset_processor.py`

```python
class NewDataset(Processor):

  def __init__(self, *args, **kwargs):
    # name under which the dataset will be saved 'datasets/new_dataset_{split}' (default)
    dataset_name = 'new_dataset'
    super().__init__(*args, **kwargs, dataset_name = dataset_name)

  def process(self):
    # load model 
    # e.g. for hf hub 
    #dataset = datasets.load_dataset('hf_dataset_name')
    def map_fn(example):
      # do some mapping
      return example

    dataset = dataset.map(map_fn, num_proc=self.num_proc)
    return dataset
```

To use the dataset add a new dataset config e.g. `config/dataset/new_config.yaml` using the new class `NewDataset` for the collection (`doc` field). As a query we are using an already existing Dataset `KILTNQProcessor`. Additinally, add the field `split` which defines which split within the dataset should be used. 

```yaml
test:
    doc: null
    query: null
dev:
  doc: 
    init_args:
    _target_: modules.dataset_processor.NewDataset
    split: "full"
query:
  init_args:
    _target_: modules.dataset_processor.KILTNQProcessor
    split: "validation"
train:
    doc: null
    query: null
```




## Add Prompt
Prompts are stored in `config/prompt/` via the argument `prompt`.

Create a new prompt `new_prompt` under  `config/prompt/new_prompt.yaml`
An exmaple prompt could look like this. THE local variables (e.g. `query`) will insterted into the formatted string within the respective models' `format_instruction()` function.
`Important`: empty spaces after a colon within the formatted string need to be escaped like to `Question:\ `.

```yaml
system: "You are a helpful assistant. Your task is to extract relevant information from the provided documents and to answer questions accordingly."
user: f"Background:\ {docs}\n\nQuestion:\ {question}\nAnswer:"
system_without_docs: "You are a helpful assistant."
user_without_docs: f"Question:\ {question}\nAnswer:"
```

To use the prompt pass it as an argument: 

```bash
python3 main.py generator='tinyllama-chat' prompt='new_prompt'

```


# Oracle
## Oracle Answer

Using the oracle answers instead of generating using a LLM.

For running the generation simply use the generator `oracle_answer`. For example: 

```python
python3 main.py dataset='kilt_nq' generator='oracle_answer'
```

## Oracle Provenances

To generate all oracle runs (trec runs) and save them in `runs` execute the script `scripts/kilt_generate_oracle.py` once.


### Oracle Provenances as Input to LLM

Generating answers using Llama with the oracle provenances as documents. 

For running the generation with e.g. `llama-2-7b-chat` simply select `orcale_provenance` as a retriever. For example: 

```python
python3 main.py dataset='kilt_nq' retriever='oracle_provenance' generator='llama-2-7b-chat'
```


### Oracle Provenances as Answer
Generating answers using oracle provenances directly as an answer. 

For running the generation simply selectn as the retriever and the generator `oracle_provenance`. For example: 

```python
python3 main.py dataset='kilt_nq' retriever='oracle_provenance' generator='oracle_provenance'
```
# Testing

To run all tests run:

To run all tests in the `tests` folder run:

```bash
pytest tests/
```

To run a single test (e.g. `tinyonly`) run: 
```bash 
pytest tests/ -k "tinyonly"
```

