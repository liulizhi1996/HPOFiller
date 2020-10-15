#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create useful auxiliary files only in PA sub-ontology, including
    - protein list: all proteins
    - term list: all HPO terms used to annotate proteins
"""
import json
from functools import reduce
import pandas as pd
from src.utils.ontology import HumanPhenotypeOntology
from src.utils.file_reader import load_annotation


if __name__ == "__main__":
    with open("../../../config/preprocessing/cvpa/create_auxiliary_file_pa.json") as fp:
        config = json.load(fp)

    # load HPO
    ontology = HumanPhenotypeOntology(config["ontology"]["path"],
                                      version=config["ontology"]["version"])
    # load annotations only in PA
    # { protein1: [ hpo_term1, hpo_term2, ... ], ... }
    hpo_annotation = load_annotation(config["annotation"], ontology,
                                     split=True, keep_root=True, propagate=True)
    hpo_annotation = hpo_annotation["pa"]
    # all proteins
    proteins = list(hpo_annotation.keys())
    # all HPO terms
    terms = list(reduce(lambda x, y: set(x) | set(y), hpo_annotation.values()))

    # write to files
    with open(config["output"]["protein"], "w") as fp:
        json.dump(proteins, fp)
    with open(config["output"]["term"], "w") as fp:
        json.dump(terms, fp)
