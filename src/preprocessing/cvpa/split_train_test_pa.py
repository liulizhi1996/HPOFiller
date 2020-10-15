#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split training set and test set. After selecting the test positive
annotation between protein p and HPO term t, all positive annotations
between protein p and t's descendants are removed from the training data.
Then, for the test negative annotations, all negative annotations
between protein p and t's ancestors are removed from the training data.
Finally, we generate two masks:
    1) train mask: m_ij = 1 if (i, j) is in training dataset, 0 otherwise
    2) test mask: m_ij = 1 if (i, j) is in test dataset, 0 otherwise
"""
import json
import random
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from src.utils.ontology import HumanPhenotypeOntology
from src.utils.file_reader import load_annotation


if __name__ == "__main__":
    with open("../../../config/preprocessing/cvpa/split_train_test_pa.json") as fp:
        config = json.load(fp)

    # load HPO
    ontology = HumanPhenotypeOntology(config["ontology"]["path"],
                                      version=config["ontology"]["version"])
    # load propagated annotations only in PA like
    # { protein1: [ term1, term2, ... ], ... }
    hpo_annotation = load_annotation(config["annotation"], ontology,
                                     split=True, keep_root=True, propagate=True)
    hpo_annotation = hpo_annotation["pa"]
    # load all protein ids
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

    # number of folds
    n_folds = config["n_folds"]

    # extract indices of positive samples
    non_zeros = df_hpo_annotation.values.nonzero()
    # [ (row_id1, col_id1), (row_id1, col_id2), ..., (row_id2, col_id3), ... ]
    pos_adjacency = list(zip(non_zeros[0], non_zeros[1]))
    # shuffle the pos_adjacency
    random.shuffle(pos_adjacency)
    # split positive folds
    pos_folds = list()
    for k in range(n_folds):
        pos_folds.append(pos_adjacency[k::n_folds])

    # extract indices of negative samples
    zeros = (df_hpo_annotation.values == 0).nonzero()
    # [ (row_id1, col_id1), (row_id1, col_id2), ..., (row_id2, col_id3), ... ]
    neg_adjacency = list(zip(zeros[0], zeros[1]))
    # shuffle the neg_adjacency
    random.shuffle(neg_adjacency)
    # split negative folds
    neg_folds = list()
    for k in range(n_folds):
        neg_folds.append(neg_adjacency[k::n_folds])

    # process training and test set
    for k in tqdm(range(n_folds), desc="fold"):
        # first, we process positive samples
        # fold of positive samples for test
        pos_test_fold = pos_folds[k]
        # the rest for training
        pos_train_fold = list()
        for t in [x for x in range(n_folds) if x != k]:
            pos_train_fold += pos_folds[t]

        # convert list of tuples to dict of list
        # { protein1: [ term1, term2, ... ], ... }
        # for training set
        pos_train_fold_d = dict()
        for x in pos_train_fold:
            pos_train_fold_d.setdefault(proteins[x[0]], []).append(terms[x[1]])
        # same for test set
        pos_test_fold_d = dict()
        for x in pos_test_fold:
            pos_test_fold_d.setdefault(proteins[x[0]], []).append(terms[x[1]])

        # remove descendant annotations of positive samples
        # for each protein in pos_test_fold
        for protein in pos_test_fold_d:
            # if protein has annotations in training set
            if protein in pos_train_fold_d:
                # get descendants of HPO terms of protein
                descendants = ontology.get_descendants(pos_test_fold_d[protein])
                # remove annotations of descendants in training set
                pos_train_fold_d[protein] = list(set(pos_train_fold_d[protein]) - set(descendants))

        # create positive train set indicator mask
        mlb = MultiLabelBinarizer()
        df_pos_train_mask = pd.DataFrame(mlb.fit_transform(pos_train_fold_d.values()),
                                         columns=mlb.classes_,
                                         index=pos_train_fold_d.keys()).reindex(
                                         columns=terms, index=proteins, fill_value=0)
        # create positive test set indicator mask
        mlb = MultiLabelBinarizer()
        df_pos_test_mask = pd.DataFrame(mlb.fit_transform(pos_test_fold_d.values()),
                                        columns=mlb.classes_,
                                        index=pos_test_fold_d.keys()).reindex(
                                        columns=terms, index=proteins, fill_value=0)

        # now, we process negative samples
        # fold of negative samples for test
        neg_test_fold = neg_folds[k]
        # the rest for training
        neg_train_fold = list()
        for t in [x for x in range(n_folds) if x != k]:
            neg_train_fold += neg_folds[t]

        # convert list of tuples to dict of list
        # { protein1: [ term1, term2, ... ], ... }
        # for training set
        neg_train_fold_d = dict()
        for x in neg_train_fold:
            neg_train_fold_d.setdefault(proteins[x[0]], []).append(terms[x[1]])
        # same for test set
        neg_test_fold_d = dict()
        for x in neg_test_fold:
            neg_test_fold_d.setdefault(proteins[x[0]], []).append(terms[x[1]])

        # remove parent annotations of negative samples
        # for each protein in neg_test_fold
        for protein in neg_test_fold_d:
            # if protein has annotations in training set
            if protein in neg_train_fold_d:
                # get ancestors of HPO terms of protein
                ancestors = ontology.get_ancestors(neg_test_fold_d[protein])
                # remove annotations of ancestors in training set
                neg_train_fold_d[protein] = list(set(neg_train_fold_d[protein]) - set(ancestors))

        # create negative train set indicator mask
        mlb = MultiLabelBinarizer()
        df_neg_train_mask = pd.DataFrame(mlb.fit_transform(neg_train_fold_d.values()),
                                         columns=mlb.classes_,
                                         index=neg_train_fold_d.keys()).reindex(
                                         columns=terms, index=proteins, fill_value=0)
        # create negative test set indicator mask
        mlb = MultiLabelBinarizer()
        df_neg_test_mask = pd.DataFrame(mlb.fit_transform(neg_test_fold_d.values()),
                                        columns=mlb.classes_,
                                        index=neg_test_fold_d.keys()).reindex(
                                        columns=terms, index=proteins, fill_value=0)

        # create training set indicator mask, 1 in training set, 0 not
        df_train_mask = df_pos_train_mask + df_neg_train_mask
        # create test set indicator mask, 1 in test set, 0 not
        df_test_mask = df_pos_test_mask + df_neg_test_mask

        # write the masks
        if os.path.exists(config["mask"][k]):
            os.remove(config["mask"][k])
        df_train_mask.to_hdf(config["mask"][k], key="train")
        df_test_mask.to_hdf(config["mask"][k], key="test")
