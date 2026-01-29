# Exploring Relational Reasoning Capabilities in LLMs with REL

This repository contains the code and data for the paper "Exploring Relational Reasoning Capabilities in Large Language Models". We include the code for building the datasets, running the experiments, and analyzing the results.

## REL-A
* RPMs for REL-A1, REL-A2, REL-A3, and REL-A4 were generated using the code available at `https://github.com/IBM/raven-large-language-models`. We use no confounders or noise.
* We will make code for the RPTs in REL-A5, REL-A6, and REL-A7 available upon publication of the final manuscript.

## REL-B
* All questions along with model outputs are in `bio_data/`. Due to the size of the data it is split up but once you pull you can run rebuild_csv.sh to rebuild the csv file.
* Code to build these datasets and run the LLMs on the benchmark is available in `bio_benchmark/`

## REL-C
* The three questions are available at `chem_data/dataset_[c1,c2,c3].jsonl`.
* Code to build the datasets and run the LLMs on the benchmark is available in `chem_benchmark/`.
* Paper results are available in `paper_results/chemistry`.