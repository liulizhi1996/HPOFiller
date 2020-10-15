#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score, roc_auc_score
from src.utils.ontology import HumanPhenotypeOntology
from src.utils.file_reader import load_annotation


def apk(y_true, y_score, k):
    """
    Average precision at top-k.
    @param y_true: list of true labels.
    @param y_score: list of target scores.
    @param k: threshold
    @return: AP@k
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_score, np.ndarray):
        y_score = np.array(y_score)
    ind = np.argsort(y_score)[::-1][:k]
    score, num_hits = 0.0, 0.0
    for i, idx in enumerate(ind):
        if y_true[idx] == 1:
            num_hits += 1
            score += num_hits / (i + 1.0)
    return score / k


if __name__ == "__main__":
    conf_path = "../../config/utils/evaluation.json"
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

    # for each type of prediction
    for i in range(len(config["prediction"])):
        print(config["prediction"][i]["name"])
        # for cross-validation, evaluate each fold
        if config["mode"] == "cv":
            for j in range(config["n_folds"]):
                # load test indicator mask (i.e. entries need to be considered)
                test_mask = pd.read_hdf(config["mask"][j], "test").reindex(columns=terms, index=proteins)
                # load predicted results
                # { protein1: { hpo_term1: score1, hpo_term2: score2, ... }, ... }
                with open(config["prediction"][i]["result"][j]) as fp:
                    predicted = json.load(fp)
                # transform predicted to dataframe
                df_predicted = pd.DataFrame.from_dict(predicted, orient="index").fillna(0)\
                    .reindex(columns=terms, index=proteins, fill_value=0)
                # evaluate based on propagated annotations
                y_true, y_score = list(), list()
                # for each protein
                for protein, row in test_mask.iterrows():
                    # indexes of considered HPO terms
                    non_zeros = row.to_numpy().nonzero()[0]
                    # extract entries to be evaluated
                    y_true += df_hpo_annotation.loc[protein][non_zeros].tolist()
                    y_score += df_predicted.loc[protein][non_zeros].tolist()
                # calculate metrics
                aupr = average_precision_score(y_true, y_score)
                auroc = roc_auc_score(y_true, y_score)
                ap5000 = apk(y_true, y_score, k=5000)
                ap10000 = apk(y_true, y_score, k=10000)
                ap20000 = apk(y_true, y_score, k=20000)
                ap50000 = apk(y_true, y_score, k=50000)

                print("Fold %d: %.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t" % (j, auroc, aupr, ap5000, ap10000, ap20000, ap50000))
        # for other scenario, only evaluate one task each time
        else:
            # load test indicator mask (i.e. entries need to be considered)
            test_mask = pd.read_hdf(config["mask"], "test").reindex(columns=terms, index=proteins)
            # load predicted results
            # { protein1: { hpo_term1: score1, hpo_term2: score2, ... }, ... }
            with open(config["prediction"][i]["result"]) as fp:
                predicted = json.load(fp)
            # transform predicted to dataframe
            df_predicted = pd.DataFrame.from_dict(predicted, orient="index").fillna(0) \
                .reindex(columns=terms, index=proteins, fill_value=0)

            # evaluate based on propagated annotations
            y_true, y_score = list(), list()
            # for each protein
            for protein, row in test_mask.iterrows():
                # indexes of considered HPO terms
                non_zeros = row.to_numpy().nonzero()[0]
                # extract entries to be evaluated
                y_true += df_hpo_annotation.loc[protein][non_zeros].tolist()
                y_score += df_predicted.loc[protein][non_zeros].tolist()
            # calculate metrics
            aupr = average_precision_score(y_true, y_score)
            auroc = roc_auc_score(y_true, y_score)
            ap5000 = apk(y_true, y_score, k=5000)
            ap10000 = apk(y_true, y_score, k=10000)
            ap20000 = apk(y_true, y_score, k=20000)
            ap50000 = apk(y_true, y_score, k=50000)

            print("%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t" % (auroc, aupr, ap5000, ap10000, ap20000, ap50000))
