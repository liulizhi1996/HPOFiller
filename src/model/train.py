#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPOFiller training and prediction phase are performed here.
We provide two modes:
    1) Cross-validation
    2) Temporal validation
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch as t
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from src.model.model import GCMC
from torch import nn, optim

from src.model.utils import rwr
from src.utils.file_reader import load_annotation
from src.utils.ontology import HumanPhenotypeOntology

# set random generator seed to allow reproducibility
t.manual_seed(12345)
np.random.seed(12345)

# assign cuda device ID
device = "cuda:1"


def tst_metric(label, input, idx):
    """
    Monitor the performance on test data set.
    :param label: ground-truth, i.e. completed protein-HPO matrix
    :param input: predicted score, i.e. predicted protein-HPO matrix
    :param idx: identify which entries are in the test set
    :return: AUC and AUPR on test set
    """
    score = input.detach().cpu().numpy()[idx]
    return roc_auc_score(label, score), average_precision_score(label, score)


def fit(model, train_data, optimizer):
    """
    Predict full protein-HPO association matrix.
    :param model: instance of model
    :param train_data: assembled training data
    :param optimizer: instance of optimizer
    :return: predicted scores, i.e. predicted protein-HPO association matrix
    """
    # turn to training mode
    model.train()
    # use MSE as the loss function
    criterion = nn.MSELoss(reduction='sum')
    # let learning_rate decrease by 50% at 500, 1000 and 2000-th epoch
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [500, 1000, 2000], gamma=0.5)

    def train_epoch(i):
        """
        Conduct i-th training iteration
        :param i: which iteration is going on
        :return: loss on training set, AUC & AUPR on test set
        """
        model.zero_grad()

        score = model(train_data["protein_feature"],
                      train_data["phenotype_feature"],
                      train_data["protein_sim"],
                      train_data["phenotype_sim"],
                      train_data["relation"])
        trn_loss = criterion(train_data["train_annotation"], score)
        # print log info every 25 iterations
        if i % 25 == 0:
            tst_auc, tst_aupr = tst_metric(train_data["test_label"], score, train_data["test_idx"])
        else:
            tst_auc, tst_aupr = 0, 0
        trn_loss.backward()
        optimizer.step()
        scheduler.step()
        return trn_loss, tst_auc, tst_aupr

    # conduct training for total of 3000 iterations
    for epoch in range(3000):
        trn_loss, tst_auc, tst_aupr = train_epoch(epoch)
        print("Epoch", epoch, "\t", trn_loss.item(), "\t", tst_auc, "\t", tst_aupr)

    # return the final predicted score
    return model(train_data["protein_feature"],
                 train_data["phenotype_feature"],
                 train_data["protein_sim"],
                 train_data["phenotype_sim"],
                 train_data["relation"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", dest="path_to_config",
                        default="../../config/model/cvpa/HPOFiller_cvpa.json",
                        help="path to config file")
    args = parser.parse_args()

    conf_path = args.path_to_config
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
    # the number of proteins and HPO terms
    m_prot, n_term = df_hpo_annotation.shape

    # load protein similarity
    with open(config["prot_sim"]) as fp:
        raw_sim = json.load(fp)
    # only select sub-graph with vertex in proteins
    protein_sim = pd.DataFrame(raw_sim).fillna(0).reindex(
        columns=proteins, index=proteins, fill_value=0).values

    # extract protein features by Random Walk with Restart
    protein_features = rwr(protein_sim, 0.9)

    if config["mode"] == "cv":
        for i in range(config["n_folds"]):
            print("Fold", i)

            # load phenotype similarity
            with open(config["pheno_sim"][i]) as fp:
                raw_phenotype_sim = json.load(fp)
            # only select sub-graph with vertex in terms
            phenotype_sim = pd.DataFrame(raw_phenotype_sim).fillna(0).reindex(
                columns=terms, index=terms, fill_value=0).values

            # extract phenotype features by Random Walk with Restart
            phenotype_features = rwr(phenotype_sim, 0.9)

            # load masks
            train_mask = pd.read_hdf(config["mask"][i], "train").reindex(
                columns=terms, index=proteins).values
            test_mask = pd.read_hdf(config["mask"][i], "test").reindex(
                columns=terms, index=proteins).values

            # apply mask to extract known annotations
            train_annotation = df_hpo_annotation.values * train_mask

            # construct protein-phenotype block matrix, size: (m + n, m + n)
            rel = np.concatenate((
                np.concatenate((np.zeros((m_prot, m_prot)), train_annotation), axis=1),
                np.concatenate((train_annotation.T, np.zeros((n_term, n_term))), axis=1)
            ), axis=0)

            # emphasize the positive labels
            train_annotation[train_annotation == 1] = 5

            # extract index of test entries
            test_idx = np.nonzero(test_mask)

            # assemble the training data
            train_data = {
                "protein_feature": t.FloatTensor(protein_features).to(device),
                "phenotype_feature": t.FloatTensor(phenotype_features).to(device),
                "protein_sim": t.FloatTensor(protein_sim).to(device),
                "phenotype_sim": t.FloatTensor(phenotype_sim).to(device),
                "train_annotation": t.FloatTensor(train_annotation).to(device),
                "relation": t.FloatTensor(rel).to(device),
                "test_label": df_hpo_annotation.values[test_idx],
                "test_idx": test_idx
            }

            # create our model
            model = GCMC(m_prot, n_term)
            model.to(device)
            # create optimizer
            optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1.)

            # make prediction
            Y_pred = fit(model, train_data, optimizer)
            Y_pred = Y_pred.detach().cpu().numpy()

            # convert it to DataFrame
            df_Y_pred = pd.DataFrame(Y_pred, index=proteins, columns=terms)
            # retain only predictions not in train set
            s = df_Y_pred.where(test_mask.astype(bool)).stack()
            prediction = {level: s.xs(level).to_dict() for level in s.index.levels[0]}
            # write to the json file
            path_to_output = config["output"].format(f=i+1)
            with open(path_to_output, "w") as fp:
                json.dump(prediction, fp, indent=2)
    else:
        # load phenotype similarity
        with open(config["pheno_sim"]) as fp:
            raw_phenotype_sim = json.load(fp)
        # only select sub-graph with vertex in terms
        phenotype_sim = pd.DataFrame(raw_phenotype_sim).fillna(0).reindex(
            columns=terms, index=terms, fill_value=0).values

        # extract phenotype features by Random Walk with Restart
        phenotype_features = rwr(phenotype_sim, 0.9)

        # load masks
        train_mask = pd.read_hdf(config["mask"], "train").reindex(
            columns=terms, index=proteins).values
        test_mask = pd.read_hdf(config["mask"], "test").reindex(
            columns=terms, index=proteins).values

        # apply mask to extract known annotations
        train_annotation = df_hpo_annotation.values * train_mask

        # construct protein-phenotype block matrix, size: (m + n, m + n)
        rel = np.concatenate((
            np.concatenate((np.zeros((m_prot, m_prot)), train_annotation), axis=1),
            np.concatenate((train_annotation.T, np.zeros((n_term, n_term))), axis=1)
        ), axis=0)

        # emphasize the positive labels
        train_annotation[train_annotation == 1] = 5

        # extract index of test entries
        test_idx = np.nonzero(test_mask)

        # assemble the training data
        train_data = {
            "protein_feature": t.FloatTensor(protein_features).to(device),
            "phenotype_feature": t.FloatTensor(phenotype_features).to(device),
            "protein_sim": t.FloatTensor(protein_sim).to(device),
            "phenotype_sim": t.FloatTensor(phenotype_sim).to(device),
            "train_annotation": t.FloatTensor(train_annotation).to(device),
            "relation": t.FloatTensor(rel).to(device),
            "test_label": df_hpo_annotation.values[test_idx],
            "test_idx": test_idx
        }

        # create our model
        model = GCMC(m_prot, n_term)
        model.to(device)
        # create optimizer
        optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1.)

        # make prediction
        Y_pred = fit(model, train_data, optimizer)
        Y_pred = Y_pred.detach().cpu().numpy()

        # convert it to DataFrame
        df_Y_pred = pd.DataFrame(Y_pred, index=proteins, columns=terms)
        # retain only predictions not in train set
        s = df_Y_pred.where(test_mask.astype(bool)).stack()
        prediction = {level: s.xs(level).to_dict() for level in s.index.levels[0]}
        # write to the json file
        with open(config["output"], "w") as fp:
            json.dump(prediction, fp, indent=2)
