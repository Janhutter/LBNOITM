cd ..



# oracle_provenance
generator='oracle_provenance'
for dataset in kilt_triviaqa kilt_eli5; do
#for dataset in kilt_nq kilt_hotpotqa kilt_triviaqa kilt_eli5 kilt_wow; do
    # without generator
    python3 main.py retriever=${generator} generator=${generator} dataset=${dataset} retrieve_top_k=1000
done

exit
# oracle answer
generator='oracle_answer'
for dataset in kilt_nq kilt_hotpotqa kilt_triviaqa kilt_eli5 kilt_wow; do
    # without generator
    python3 main.py generator=${generator} dataset=${dataset}
done
