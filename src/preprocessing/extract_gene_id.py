#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract entrez gene ids from
ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt.

Download gene annotations file from
http://compbio.charite.de/jenkins/job/hpo.annotations.monthly/
with all sources and all frequencies. The first column of file is what we want.
Extract and save them into a txt file in one column. This file will be uploaded
to Uniprot ID Mapping Tool (http://www.uniprot.org/mapping/) to get gene2uniprot
mapping file.
"""

import json


with open("../../config/preprocessing/extract_gene_id.json") as fp:
    config = json.load(fp)

gene_set = set()
with open(config["anno_file"]) as fp:
    for line in fp:
        if line.startswith("#"):    # pass the header
            continue
        gene, *_ = line.split('\t')
        gene_set.add(gene)

with open(config["gene_list"], "w") as fp:
    for gene in gene_set:
        fp.write("%s\n" % gene)
