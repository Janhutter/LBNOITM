# Lost but Not Only in the Middle
The code is based on [BERGEN](https://github.com/naver/bergen) by Naver.

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
Example experiment in the `relevant distractor` setting on `kilt_nq` using the `solar 10.7B` model, where the oracle document is at the first position `i=0` out of the 5 total amount of documents `run=5`.
```bash
 model='vllm_SOLAR-107B'
 modelname='solar107b'
 i=0
 run=5
 dataset='kilt_nq'
 # task='random'
 # task='relevant_not_correct'
 task='relevant'
 ./run.sh $run $i $model $modelname $dataset $task
```

## Overview
Using the trec files, from the runs folder, the different contexts are constructed. These trec files are build using the lost_in_the_middle.py script.
The Bergen code has been adapted to read specific trec files and skips the retrieval and reranking stages. 

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
|-- figures/
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
|-- figures.py
|-- figures2.py
|-- correct_answering_attention.py
|-- combine_datasets.py
|-- check_tokens.py
|-- testing.py
|-- visualize_attention.py
|-- lost_in_middle.py
```
More documentation on BERGEN can be found [here](https://github.com/naver/bergen)

