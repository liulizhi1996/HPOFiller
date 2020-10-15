#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPO semantic similarity measures listed in papers:
    1. HPOSim: An R Package for Phenotypic Similarity Measure and Enrichment
       Analysis Based on the Human Phenotype Ontology
    2. A new method to measure the semantic similarity from query phenotypic
       abnormalities to diseases based on the human phenotype ontology

Supported HPO similarity measures are (valid parameter in config["method"]):
    - resnik: Resnik measure
    - lin: Lin measure
    - jc: Jiang-Conrath measure
    - rel: Relevance measure
    - ic: Information coefficient measure
    - graphic: graph IC measure
"""
import sys
import json
import math
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.ontology import HumanPhenotypeOntology
from src.utils.file_reader import load_annotation


# global variable, ancestors of each HPO term
ancestors = dict()
# global variable, frequency of terms
freq = None
# global variable, information content of HPO terms
ic = None


def resnik_sim(x):
    """
    Resnik measure, see Resnik P. Using information content to evaluate
    semantic similarity in a taxonomy. In: Proceedings of the 14th
    international joint conference on artificial intelligence (IJCAI-95);
    1995.
    :param x: tuple of index name, i.e. (row_term, col_term)
    :return: similarity
    """
    global ancestors
    global ic

    term_a, term_b = x
    # set values on the diagonal to 1
    if term_a == term_b:
        return 1
    # ancestors of term_a
    ancestors_a = ancestors[term_a]
    # ancestors of term_b
    ancestors_b = ancestors[term_b]
    # all common ancestors of term_a and term_b (and also in terms)
    common_ancestors = list(ancestors_a & ancestors_b)
    # information content of most informative common ancestor
    ic_mica = ic[common_ancestors].max()
    # similarity between term_a and term_b
    sim = ic_mica
    return sim


def lin_sim(x):
    """
    Lin measure, see Lin D. An information-theoretic definition of
    similarity. In: ICML, vol. Vol. 98, no. 1998; 1998. p. 296–304.
    :param x: tuple of index name, i.e. (row_term, col_term)
    :return: similarity
    """
    global ancestors
    global ic

    term_a, term_b = x
    # set values on the diagonal to 1
    if term_a == term_b:
        return 1
    # ancestors of term_a
    ancestors_a = ancestors[term_a]
    # ancestors of term_b
    ancestors_b = ancestors[term_b]
    # all common ancestors of term_a and term_b (and also in terms)
    common_ancestors = list(ancestors_a & ancestors_b)
    # information content of most informative common ancestor
    ic_mica = ic[common_ancestors].max()
    # similarity between term_a and term_b
    sim = 2 * ic_mica / (ic[term_a] + ic[term_b])
    return sim


def jc_sim(x):
    """
    Jiang-Conrath measure, see Jiang JJ, Conrath DW. Semantic similarity
    based on corpus statistics and lexical taxonomy. In: Proc of 10th
    international conference on research in computational linguistics,
    ROCLING’97; 1997
    :param x: tuple of index name, i.e. (row_term, col_term)
    :return: similarity
    """
    global ancestors
    global ic

    term_a, term_b = x
    # set values on the diagonal to 1
    if term_a == term_b:
        return 1
    # ancestors of term_a
    ancestors_a = ancestors[term_a]
    # ancestors of term_b
    ancestors_b = ancestors[term_b]
    # all common ancestors of term_a and term_b (and also in terms)
    common_ancestors = list(ancestors_a & ancestors_b)
    # information content of most informative common ancestor
    ic_mica = ic[common_ancestors].max()
    # similarity between term_a and term_b
    sim = 1 / (1 + ic[term_a] + ic[term_b] - 2 * ic_mica)
    return sim


def rel_sim(x):
    """
    Relevance measure, see Schlicker A, Domingues FS, Rahnenführer J,
    Lengauer T. A new measure for functional similarity of gene products
    based on gene ontology. BMC Bioinforma. 2006;7(1):302.
    :param x: tuple of index name, i.e. (row_term, col_term)
    :return: similarity
    """
    global ancestors
    global freq
    global ic

    term_a, term_b = x
    # set values on the diagonal to 1
    if term_a == term_b:
        return 1
    # ancestors of term_a
    ancestors_a = ancestors[term_a]
    # ancestors of term_b
    ancestors_b = ancestors[term_b]
    # all common ancestors of term_a and term_b (and also in terms)
    common_ancestors = list(ancestors_a & ancestors_b)
    # information content of most informative common ancestor
    ic_mica = ic[common_ancestors].max()
    # frequency of most informative common ancestor
    freq_mica = freq[common_ancestors].min()
    # similarity between term_a and term_b
    sim = (2 * ic_mica / (ic[term_a] + ic[term_b])) * (1 - freq_mica)
    return sim


def ic_sim(x):
    """
    Information coefficient measure, see Li, B., Wang, J. Z., Feltus,
    F. A., Zhou, J., & Luo, F. (2010). Effectively integrating information
    content and structural relationship to improve the GO-based similarity
    measure between proteins. arXiv preprint arXiv: 1001.0958.
    :param x: tuple of index name, i.e. (row_term, col_term)
    :return: similarity
    """
    global ancestors
    global ic

    term_a, term_b = x
    # set values on the diagonal to 1
    if term_a == term_b:
        return 1
    # ancestors of term_a
    ancestors_a = ancestors[term_a]
    # ancestors of term_b
    ancestors_b = ancestors[term_b]
    # all common ancestors of term_a and term_b (and also in terms)
    common_ancestors = list(ancestors_a & ancestors_b)
    # information content of most informative common ancestor
    ic_mica = ic[common_ancestors].max()
    # similarity between term_a and term_b
    sim = (2 * ic_mica / (ic[term_a] + ic[term_b])) * (1 - 1 / (1 + ic_mica))
    return sim


def graphic_sim(x):
    """
    Graph IC measure, see Pesquita C, Faria D, Bastos H, Falcao A, Couto F.
    Evaluating GO-based semantic similarity measures. In: Proc. 10th annual
    bio-ontologies meeting, vol. 37, no. 40; 2007. p. 38.
    :param x: tuple of index name, i.e. (row_term, col_term)
    :return: similarity
    """
    global ancestors
    global ic

    term_a, term_b = x
    # set values on the diagonal to 1
    if term_a == term_b:
        return 1
    # ancestors of term_a
    ancestors_a = ancestors[term_a]
    # ancestors of term_b
    ancestors_b = ancestors[term_b]
    # similarity between term_a and term_b
    intersection = list(ancestors_a & ancestors_b)
    union = list(ancestors_a | ancestors_b)
    sim = ic[intersection].sum() / ic[union].sum()
    return sim


if __name__ == "__main__":
    conf_path = "../../config/HPOSim/hpo_sim.json"
    if len(sys.argv) > 1 and len(sys.argv[1]) > 0:
        conf_path = sys.argv[1]

    with open(conf_path) as fp:
        config = json.load(fp)

    # load HPO
    ontology = HumanPhenotypeOntology(config["ontology"]["path"],
                                      version=config["ontology"]["version"])
    # load propagated annotations { protein1: [ term1, term2, ... ], ... }
    hpo_annotation = load_annotation(config["annotation"], ontology,
                                     split=False, keep_root=True, propagate=True)
    # load all proteins
    with open(config["protein"]) as fp:
        proteins = json.load(fp)
    # load all HPO terms
    with open(config["term"]) as fp:
        terms = json.load(fp)
    # transform hpo_annotation to dataframe like
    #           func1  func2  func3  func4
    # protein1      1      1      0      0
    # protein2      0      1      0      1
    # protein3      0      0      0      0
    mlb = MultiLabelBinarizer()
    df_hpo_annotation = pd.DataFrame(mlb.fit_transform(hpo_annotation.values()),
                                     columns=mlb.classes_,
                                     index=hpo_annotation.keys()).reindex(
                                     columns=terms, index=proteins, fill_value=0)

    if config["mode"] == "cv":
        for i in range(config["n_folds"]):
            print("Fold", i)

            # load train mask
            train_mask = pd.read_hdf(config["mask"][i], "train").reindex(columns=terms, index=proteins)
            # apply mask to retain train HPO annotations (known associations)
            df_train_hpo_annotation = df_hpo_annotation * train_mask

            # remove columns containing only zeros
            df_train_hpo_annotation = df_train_hpo_annotation.loc[:, (df_train_hpo_annotation != 0).any(axis=0)]
            # remove rows containing only zeros
            df_train_hpo_annotation = df_train_hpo_annotation[(df_train_hpo_annotation.T != 0).any()]

            # total number of proteins
            total_protein = len(df_train_hpo_annotation.index)
            # sum over the proteins to calculate the frequency of terms
            freq = df_train_hpo_annotation.sum(axis=0) / total_protein
            # information content of each HPO term
            ic = -freq.apply(math.log2)

            # get ancestors of each term
            left_terms = list(df_train_hpo_annotation.columns)
            for term in left_terms:
                ancestors[term] = ontology.get_ancestors([term])

            # initialize an empty similarity matrix with 0s
            similarity = pd.DataFrame(0, index=left_terms, columns=left_terms)
            # calculate similarity of each pair of terms
            similarity = similarity.stack()
            if config["method"] == "resnik":
                similarity.loc[:] = similarity.index.map(resnik_sim)
            elif config["method"] == "lin":
                similarity.loc[:] = similarity.index.map(lin_sim)
            elif config["method"] == "jc":
                similarity.loc[:] = similarity.index.map(jc_sim)
            elif config["method"] == "rel":
                similarity.loc[:] = similarity.index.map(rel_sim)
            elif config["method"] == "ic":
                similarity.loc[:] = similarity.index.map(ic_sim)
            elif config["method"] == "graphic":
                similarity.loc[:] = similarity.index.map(graphic_sim)
            similarity = similarity.unstack()

            # write to the json file
            similarity = similarity.to_dict(orient="index")
            with open(config["output"][i], "w") as fp:
                json.dump(similarity, fp, indent=2)
    else:
        # load train mask
        train_mask = pd.read_hdf(config["mask"], "train").reindex(columns=terms, index=proteins)
        # apply mask to retain train HPO annotations (known associations)
        df_train_hpo_annotation = df_hpo_annotation * train_mask

        # remove columns containing only zeros
        df_train_hpo_annotation = df_train_hpo_annotation.loc[:, (df_train_hpo_annotation != 0).any(axis=0)]
        # remove rows containing only zeros
        df_train_hpo_annotation = df_train_hpo_annotation[(df_train_hpo_annotation.T != 0).any()]

        # total number of proteins
        total_protein = len(df_train_hpo_annotation.index)
        # sum over the proteins to calculate the frequency of terms
        freq = df_train_hpo_annotation.sum(axis=0) / total_protein
        # information content of each HPO term
        ic = -freq.apply(math.log2)

        # get ancestors of each term
        left_terms = list(df_train_hpo_annotation.columns)
        for term in left_terms:
            ancestors[term] = ontology.get_ancestors([term])

        # initialize an empty similarity matrix with 0s
        similarity = pd.DataFrame(0, index=left_terms, columns=left_terms)
        # calculate similarity of each pair of terms
        similarity = similarity.stack()
        if config["method"] == "resnik":
            similarity.loc[:] = similarity.index.map(resnik_sim)
        elif config["method"] == "lin":
            similarity.loc[:] = similarity.index.map(lin_sim)
        elif config["method"] == "jc":
            similarity.loc[:] = similarity.index.map(jc_sim)
        elif config["method"] == "rel":
            similarity.loc[:] = similarity.index.map(rel_sim)
        elif config["method"] == "ic":
            similarity.loc[:] = similarity.index.map(ic_sim)
        elif config["method"] == "graphic":
            similarity.loc[:] = similarity.index.map(graphic_sim)
        similarity = similarity.unstack()

        # write to the json file
        similarity = similarity.to_dict(orient="index")
        with open(config["output"], "w") as fp:
            json.dump(similarity, fp, indent=2)
