#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make datasets along with the time.

Output files:
    - HPO annotations: old annotations + added new annotations
    - mask: indicator mask of training set and test set
    - protein: list of proteins
    - term: list of HPO terms
"""
import json
import os
from collections import defaultdict
from functools import reduce
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.file_reader import gene2uniprot
from src.utils.ontology import HumanPhenotypeOntology, get_root


if __name__ == "__main__":
    with open("../../../config/preprocessing/temporalpa/split_temporal_dataset_pa.json") as fp:
        config = json.load(fp)

    # ---------------------- HPO -----------------------
    # load old HPO
    old_ontology = HumanPhenotypeOntology(config["old_ontology"]["path"],
                                          version=config["old_ontology"]["version"])
    # load new HPO
    new_ontology = HumanPhenotypeOntology(config["new_ontology"]["path"],
                                          version=config["new_ontology"]["version"])

    # ------------------- Annotation -------------------
    # load old mapping of gene id to uniprot id
    old_gene2protein = gene2uniprot(config["old_mapping"], gene_column=0, uniprot_column=1)
    # load old hpo annotations without propagation
    old_annotation = defaultdict(list)
    with open(config["old_annotation"]) as fp:
        for line in fp:
            if line.startswith('#'):
                continue
            gene_id, _, _, hpo_term = line.strip().split('\t')
            for protein_id in old_gene2protein[gene_id]:
                old_annotation[protein_id].append(hpo_term)
    # propagate old annotations
    for protein in old_annotation:
        old_annotation[protein] = list(old_ontology.transfer(old_annotation[protein]))

    # load new mapping of gene id to uniprot id
    new_gene2protein = gene2uniprot(config["new_mapping"], gene_column=0, uniprot_column=1)
    # load new hpo annotations without propagation
    new_annotation = defaultdict(list)
    with open(config["new_annotation"]) as fp:
        for line in fp:
            if line.startswith('#'):
                continue
            gene_id, _, hpo_term, *_ = line.strip().split('\t')
            for protein_id in new_gene2protein[gene_id]:
                new_annotation[protein_id].append(hpo_term)

    # adapt new annotations to old ones
    new_annotation_adapted = defaultdict(list)
    for protein in new_annotation:
        for hpo_term in new_annotation[protein]:
            # if "veteran" HPO term, just copy down
            if hpo_term in old_ontology:
                new_annotation_adapted[protein].append(hpo_term)
            # else if has old, alternative HPO terms, replace it
            elif hpo_term in new_ontology.alt_ids:
                for alternative in new_ontology.alt_ids[hpo_term]:
                    print("Replace (%s, %s) to (%s, %s)" % (protein, hpo_term, protein, alternative))
                    new_annotation_adapted[protein].append(alternative)
            # if not found, then discard
            else:
                print("Discard (%s, %s)" % (protein, hpo_term))

    # propagate new annotations according to the old ontology
    for protein in new_annotation_adapted:
        new_annotation_adapted[protein] = list(old_ontology.transfer(
            new_annotation_adapted[protein]))

    # only remain HPO annotations in PA sub-ontology and root
    old_annotation_tmp = defaultdict(list)
    for protein in old_annotation:
        for hpo_term in old_annotation[protein]:
            if old_ontology[hpo_term].ns == "pa":
                old_annotation_tmp[protein].append(hpo_term)
    old_annotation = old_annotation_tmp
    for protein in old_annotation:
        old_annotation[protein].append(get_root())
    new_annotation_adapted_tmp = defaultdict(list)
    for protein in new_annotation_adapted:
        for hpo_term in new_annotation_adapted[protein]:
            if old_ontology[hpo_term].ns == "pa":
                new_annotation_adapted_tmp[protein].append(hpo_term)
    new_annotation_adapted = new_annotation_adapted_tmp

    # proteins in test positive set
    test_proteins = set(old_annotation.keys()) & set(new_annotation_adapted.keys())

    # make up test annotations (newly added)
    test_annotation = defaultdict(list)
    for protein in test_proteins:
        for hpo_term in new_annotation_adapted[protein]:
            # newly added annotations
            if hpo_term not in old_annotation[protein]:
                test_annotation[protein].append(hpo_term)

    # ------------- Output necessary files -------------
    # set of proteins
    proteins = list(set(old_annotation) | set(test_annotation))
    # set of HPO terms
    terms = list(reduce(lambda a, b: set(a) | set(b), old_annotation.values()) |
                 reduce(lambda a, b: set(a) | set(b), test_annotation.values()))

    # create training set indicator mask
    mlb = MultiLabelBinarizer()
    df_train_mask = pd.DataFrame(mlb.fit_transform(old_annotation.values()),
                                 columns=mlb.classes_,
                                 index=old_annotation.keys()).reindex(
                                 columns=terms, index=proteins, fill_value=0)

    # create test set indicator mask
    df_test_mask = pd.DataFrame(np.ones((len(proteins), len(terms)), dtype=int),
                                columns=terms, index=proteins)
    df_test_mask = (df_test_mask - df_train_mask).clip(lower=0, upper=1)

    # output annotation
    precessed_annotation = defaultdict(list)
    for protein in old_annotation:
        precessed_annotation[protein] = old_annotation[protein]
    for protein in test_annotation:
        for hpo_term in test_annotation[protein]:
            if hpo_term not in precessed_annotation[protein]:
                precessed_annotation[protein].append(hpo_term)

    # write to the file
    with open(config["output"]["processed_annotation"], 'w') as fp:
        json.dump(precessed_annotation, fp, indent=2)

    # write the masks
    if os.path.exists(config["output"]["mask"]):
        os.remove(config["output"]["mask"])
    df_train_mask.to_hdf(config["output"]["mask"], key="train")
    df_test_mask.to_hdf(config["output"]["mask"], key="test")

    # write proteins and terms to files
    with open(config["output"]["protein"], "w") as fp:
        json.dump(proteins, fp)
    with open(config["output"]["term"], "w") as fp:
        json.dump(terms, fp)
