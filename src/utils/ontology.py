#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from src.utils.obo_parser import GODag


def get_root():
    """Return the root term of HPO.
    :return: the root term
    """
    return 'HP:0000001'


def get_ns_id(version="201902"):
    """Return list of namespace id of each sub-ontology of HPO.
    :param version: version of HPO
    :return: namespace id list
    """
    if version == "201904":
        return ["cc", "cm", "mi", "pa", "freq", "bg", "pmh"]
    elif version == "201902":
        return ["cc", "cm", "mi", "pa", "freq", "bg"]
    elif version == "2018":
        return ["cc", "cm", "mi", "pa", "freq"]
    elif version == "2017":
        return ["cm", "ma", "mi", "pa", "freq"]
    else:
        raise ValueError("%s is not a valid version." % version)


def get_subontology(version="201902"):
    """Return term list of roots of each sub-ontology.
    :param version: version of HPO
    :return: root terms list
    """
    if version == "201904":
        return ['HP:0000005',   # Mode of inheritance
                'HP:0000118',   # Phenotypic abnormality
                'HP:0012823',   # Clinical modifier
                'HP:0031797',   # Clinical course
                'HP:0032223',   # Blood group
                'HP:0032443',   # Past medical history
                'HP:0040279'    # Frequency
                ]
    elif version == "201902":
        return ['HP:0000005',   # Mode of inheritance
                'HP:0000118',   # Phenotypic abnormality
                'HP:0012823',   # Clinical modifier
                'HP:0031797',   # Clinical course
                'HP:0032223',   # Blood group
                'HP:0040279'    # Frequency
                ]
    elif version == "2018":
        return ['HP:0000005',   # Mode of inheritance
                'HP:0000118',   # Phenotypic abnormality
                'HP:0012823',   # Clinical modifier
                'HP:0031797',   # Clinical course
                'HP:0040279'    # Frequency
                ]
    elif version == "2017":
        return ['HP:0000005',   # Mode of inheritance
                'HP:0000118',   # Phenotypic abnormality
                'HP:0012823',   # Clinical modifier
                'HP:0040006',   # Mortality/Aging
                'HP:0040279'    # Frequency
                ]
    else:
        raise ValueError("%s is not a valid version." % version)


def get_ns_id2hpo(version="201902"):
    """Return mapping of namespace to the root of sub-ontology.
    :param version: version of HPO
    :return: mapping of namespace to HPO term
    """
    if version == "201904":
        return {"cc": 'HP:0031797',
                "cm": 'HP:0012823',
                "mi": 'HP:0000005',
                "pa": 'HP:0000118',
                "bg": 'HP:0032223',
                "pmh": 'HP:0032443',
                "freq": 'HP:0040279'}
    elif version == "201902":
        return {"cc": 'HP:0031797',
                "cm": 'HP:0012823',
                "mi": 'HP:0000005',
                "pa": 'HP:0000118',
                "bg": 'HP:0032223',
                "freq": 'HP:0040279'}
    if version == "2018":
        return {"cc": 'HP:0031797',
                "cm": 'HP:0012823',
                "mi": 'HP:0000005',
                "pa": 'HP:0000118',
                "freq": 'HP:0040279'}
    elif version == "2017":
        return {"cm": 'HP:0012823',
                "ma": 'HP:0040006',
                "mi": 'HP:0000005',
                "pa": 'HP:0000118',
                "freq": 'HP:0040279'}
    else:
        raise ValueError("%s is not a valid version." % version)


def get_hpo2ns_id(version="201902"):
    """Return mapping of namespaces of roots of sub-ontology.
    :param version: version of HPO
    :return: mapping, key: root term of sub-ontology, value: namespace
    """
    if version == "201904":
        return {'HP:0000005': 'mi',
                'HP:0000118': 'pa',
                'HP:0012823': 'cm',
                'HP:0031797': 'cc',
                'HP:0032223': 'bg',
                'HP:0032443': 'pmh',
                'HP:0040279': 'freq'
                }
    if version == "201902":
        return {'HP:0000005': 'mi',
                'HP:0000118': 'pa',
                'HP:0012823': 'cm',
                'HP:0031797': 'cc',
                'HP:0032223': 'bg',
                'HP:0040279': 'freq'
                }
    elif version == "2018":
        return {'HP:0000005': 'mi',
                'HP:0000118': 'pa',
                'HP:0012823': 'cm',
                'HP:0031797': 'cc',
                'HP:0040279': 'freq'
                }
    elif version == "2017":
        return {'HP:0000005': 'mi',
                'HP:0000118': 'pa',
                'HP:0012823': 'cm',
                'HP:0040006': 'ma',
                'HP:0040279': 'freq'
                }
    else:
        raise ValueError("%s is not a valid version." % version)


class HPOTerm(object):
    def __init__(self, hpo_term):
        """Definition of an HPO term.
        Attributes:
            - id: accession of HPO term, e.g. HP:0000005
            - parents: set, parents of this term
            - name: name of HPO term
            - ns: namespace the term belongs to
            - children: set, children of this term
            - depth: the depth of term in the whole HPO
        :param hpo_term: instance of the HPO term
        :return: None
        """
        self.id = hpo_term.id
        self.parents = set([p.id for p in hpo_term.parents])
        if hasattr(hpo_term, 'relationship'):
            for parent in hpo_term.relationship.get('part_of', set()):
                if parent.namespace == hpo_term.namespace:
                    self.parents.add(parent.id)
        self.name = hpo_term.name
        self.ns = hpo_term.namespace
        self.children = set()
        self.depth = 0


class HumanPhenotypeOntology(dict):
    """Definition of HPO, it is inheritance of dict.
    Attributes:
        - version: version of HPO
        - alt_ids: dict, alternative ids of HPO terms, like
            { hpo_term1: [ alt_id1, alt_id2, ... ], ... }
        - root_term: root term of HPO, i.e. HP:0000001
        - subontology: list of root of subontology
        - dict of HPO terms, you can visit it like dict,
            e.g. ontology['HP:0000005']
    """
    def __init__(self, obo_file_path, version="201902"):
        """
        :param obo_file_path: path to obo file
        :return: None
        """
        super(HumanPhenotypeOntology, self).__init__()
        go_dag = GODag(obo_file_path, 'relationship')
        self.alt_ids = go_dag.alt_ids
        for hpo_id, hpo_term in go_dag.items():
            self[hpo_id] = HPOTerm(hpo_term)
        self._get_children()
        self.root_term = get_root()
        self._get_depth()
        self.version = version
        self.subontology = get_subontology(version)
        self._get_namespace()

    def _get_children(self):
        """Fill in children of each term.
        :return: None
        """
        for hpo_id in self:
            for parent in self[hpo_id].parents:
                self[parent].children.add(hpo_id)

    def _get_depth(self):
        """Fill in depth of each term.
        :return: None
        """
        self[self.root_term].depth = 1
        now = {self.root_term}
        while len(now) > 0:
            next = set()
            for hpo_term in now:
                for child in self[hpo_term].children:
                    if self[child].depth == 0:
                        next.add(child)
                        self[child].depth = self[hpo_term].depth + 1
            now = next

    def _get_namespace(self):
        """Fill in namespace of each term.
        :return: None
        """
        hpo2namespace = get_hpo2ns_id(self.version)
        self[get_root()].ns = 'all'
        for subontology in self.subontology:
            self[subontology].ns = hpo2namespace[subontology]
            now = {subontology}
            while len(now) > 0:
                next = set()
                for hpo_term in now:
                    for child in self[hpo_term].children:
                        if len(self[child].ns) == 0:
                            next.add(child)
                            self[child].ns = hpo2namespace[subontology]
                now = next

    def transfer(self, hpo_list):
        """Propagate HPO terms by true-path-rule.
        :param hpo_list: the HPO terms should be transferred
        :return: Propagated HPO terms of hpo_list
        """
        hpo_list = list(filter(lambda x: x in self, hpo_list))
        ancestors, now = set(hpo_list), set(hpo_list)
        while len(now) > 0:
            next = set()
            for hpo_term in now:
                if hpo_term in self:
                    next |= self[hpo_term].parents - ancestors
            now = next
            ancestors |= now
        return ancestors

    def transfer_scores(self, term_scores):
        """Keep the consistency of predictive scores - the score of the term
        must not be greater than the score of all its child nodes.
        :param term_scores: dict, predictive scores of HPO terms.
            { hpo_term1: score1, hpo_term2: score2, ... }
        :return: Scores that adhere to consistency constraints.
        """
        scores = defaultdict(float)
        for hpo_term in sorted(self.transfer(term_scores.keys()),
                              key=lambda x: self[x].depth, reverse=True):
            scores[hpo_term] = max(scores[hpo_term],
                                   term_scores.get(hpo_term, 0))
            for parent_id in self[hpo_term].parents:
                scores[parent_id] = max(scores[parent_id], scores[hpo_term])
        return scores

    def get_descendants(self, hpo_list):
        """Get all descendants of hpo_term.
        :param hpo_list: list of queried HPO terms
        :return: set(), all descendants of hpo_term
        """
        hpo_list = list(filter(lambda x: x in self, hpo_list))
        descendants, now = set(), set(hpo_list)
        while len(now) > 0:
            next = set()
            for hpo_term in now:
                if hpo_term in self:
                    next |= self[hpo_term].children - descendants
            now = next
            descendants |= now
        return descendants

    def get_ancestors(self, hpo_list):
        """Get all ancestors of hpo_term.
        :param hpo_list: list of queried HPO terms
        :return: set(), all ancestors of hpo_term
        """
        hpo_list = list(filter(lambda x: x in self, hpo_list))
        ancestors, now = set(), set(hpo_list)
        while len(now) > 0:
            next = set()
            for hpo_term in now:
                if hpo_term in self:
                    next |= self[hpo_term].parents - ancestors
            now = next
            ancestors |= now
        return ancestors
