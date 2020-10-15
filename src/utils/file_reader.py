#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Readers of files with different formats.
"""
import json
from collections import defaultdict
from src.utils.ontology import get_ns_id, get_root, get_subontology


def gene2uniprot(file_path, gene_column, uniprot_column):
    """Mapping entrez gene id to uniprot protein id.

    :param file_path: path to mapping file
    :param gene_column: the column index of gene id
    :param uniprot_column: the column index of uniprot id
    :return: a dict with key being gene id and value being list of uniprot ids
    { gene_id: [uniprot_id1, uniprot_id2, ...] }
    """
    gene_to_protein = defaultdict(list)
    with open(file_path) as fp:
        for line in fp:
            if line.startswith("y"):    # omit the header line
                continue
            entries = line.strip().split('\t')
            # multi-genes mapped to the same protein
            if ',' in entries[gene_column]:
                genes = entries[gene_column].split(',')
                protein = entries[uniprot_column]
                for gene in genes:
                    gene_to_protein[gene].append(protein)
            # one gene mapped to one protein
            else:
                gene, protein = entries[gene_column], entries[uniprot_column]
                gene_to_protein[gene].append(protein)
    return gene_to_protein


def load_annotation(file_path, ontology,
                    split=True, keep_root=False, propagate=True):
    """Get propagated HPO annotations.
    :param file_path: path to raw annotation
    :param ontology: instance of HumanPhenotypeOntology
    :param split: split annotations into sub-ontologies if True, and False not
    :param keep_root: True then include roots, False then remove them
    :param propagate: True then propagate HPO terms according to
        true-path-rule, False then only keep leaves
    :return: annotations after propagation
    split=True: { ns: {protein1: [ hpo_term1, hpo_term2, ... ] ... } ... }
    split=False: { protein1: [ hpo_term1, hpo_term2, ... ] ... }
    """
    # load raw annotations without propagation
    with open(file_path) as fp:
        leaf_annotation = json.load(fp)

    # retrieve root terms
    ns_id = get_ns_id(ontology.version)
    root = {get_root()}
    subontology = set(get_subontology(ontology.version))

    if propagate:
        # propagate annotations and discard roots (All & sub-ontology)
        if split:
            propagated_annotation = dict([(ns, defaultdict(list)) for ns in ns_id])
            for ns in ns_id:
                for protein in leaf_annotation:
                    filtered = list(filter(lambda t: ontology[t].ns == ns,
                                           leaf_annotation[protein]))
                    if len(filtered) > 0:
                        if not keep_root:
                            propagated_annotation[ns][protein] = list(
                                ontology.transfer(filtered) - root - subontology)
                        else:
                            propagated_annotation[ns][protein] = list(
                                ontology.transfer(filtered))
        else:
            propagated_annotation = dict()
            for protein in leaf_annotation:
                leaves = leaf_annotation[protein]
                if len(leaves) > 0:
                    if not keep_root:
                        propagated_annotation[protein] = list(
                            ontology.transfer(leaves) - root - subontology)
                    else:
                        propagated_annotation[protein] = list(
                            ontology.transfer(leaves))
        return propagated_annotation
    else:
        # only keep leaves
        if split:
            leaves_annotation = dict([(ns, defaultdict(list)) for ns in ns_id])
            for ns in ns_id:
                for protein in leaf_annotation:
                    filtered = list(filter(lambda t: ontology[t].ns == ns,
                                           leaf_annotation[protein]))
                    if len(filtered) > 0:
                        if not keep_root:
                            leaves_annotation[ns][protein] = list(
                                set(filtered) - root - subontology)
                        else:
                            leaves_annotation[ns][protein] = filtered
        else:
            leaves_annotation = dict()
            for protein in leaf_annotation:
                leaves = leaf_annotation[protein]
                if len(leaves) > 0:
                    if not keep_root:
                        leaves_annotation[protein] = list(
                            set(leaves) - root - subontology)
                    else:
                        leaves_annotation[protein] = list(leaves)
        return leaves_annotation
