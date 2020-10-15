#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create HPO annotations without propagation from raw file.

Output format:
{ protein_id1: [ hpo_term1, hpo_term2, ... ],
  protein_id2: [ hpo_term1, hpo_term2, ... ],
  ...
}
"""
import json
from collections import defaultdict
from src.utils.file_reader import gene2uniprot


with open("../../config/preprocessing/create_annotation.json") as fp:
    config = json.load(fp)

# load mapping of gene id to uniprot id
gene2protein = gene2uniprot(config["mapping"], gene_column=0, uniprot_column=1)

# load hpo annotations without propagation
annotation = defaultdict(list)
with open(config["raw_annotation"]) as fp:
    for line in fp:
        if line.startswith('#'):
            continue
        gene_id, _, _, hpo_term = line.strip().split('\t')
        for protein_id in gene2protein[gene_id]:
            annotation[protein_id].append(hpo_term)

# output annotation
with open(config["processed_annotation"], 'w') as fp:
    json.dump(annotation, fp, indent=2)
