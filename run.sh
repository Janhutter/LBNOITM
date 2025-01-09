#!/bin/bash

python3 main.py retriever='oracle_provenance' generator="${3}" dataset="${5}" run_name="${4}_top${1}${6}_oracle${2}_${5}" trec_file="${5}_top${1}${6}_oracle_at_${2}" retrieve_top_k=${1} generation_top_k=${1} rerank_top_k=${1}
