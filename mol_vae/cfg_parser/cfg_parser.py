#!/usr/bin/env python3

import random
import logging
logger = logging.getLogger(__name__)

import nltk
from nltk.grammar import Nonterminal, Production

from rdkit import Chem

def canonicalize_smiles(smiles):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),True)
    return smiles


def renumber_independent_rings(smiles):
    """Renumbers rings.
    
    The SMILES specifications suggest that digits denoting ring closures can be reused.
    This is necessary for the parser in this library to not fail. 
    For example "C1CC1CC2CC2" will fail, but "C1CC1CC1CC1" will not.
    
    This function carries out this renumbering. It assumes that the first ring digit
    is 1, that no ring digits higher than 8 are present, and that all rings are 
    closed.
    
    Inputs
    ------
    smiles:   str, SMILES string
    
    Outputs
    ------
    smiles:   str, renumbered SMILES string
    
    
    Examples
    --------
    ```
    tests = [
        (
            "C1C(C1)OC2CC2",
            "C1CC1OC1CC1"
        ),
        (
            "Fc1c(nc(nc1)OCc2c(cc(cc2)Cl)F)C3CCN(CC3)Cc4n(c5c(n4)cccc5)C", 
             "Cn1c2ccccc2nc1CN1CCC(CC1)c1c(cnc(n1)OCc1ccc(cc1F)Cl)F"
        ),
        (
            "N2(CCC1(CNC1)CC2)C(=O)OCc3ccccc3",
            "c1ccc(cc1)COC(=O)N1CCC2(CC1)CNC2",
        ),
        (
            "Clc1nc2c(cn1)CCC32CCN(CC3)C(=O)OC(C)(C)C",
            "CC(C)(C)OC(=O)N1CCC2(CCc3c2nc(nc3)Cl)CC1",
        ),
        (
            "s1c(ccc1)C(=O)Oc2c(cc(cc2)/C=C/3\\N=C(OC3=O)c4ccc(cc4)NC(=O)OC(C)(C)C)OC", # Needs canonicalization
            "CC(C)(C)OC(=O)Nc1ccc(cc1)C1=NC(=Cc2ccc(c(c2)OC)OC(=O)c2cccs2)C(=O)O1",
        ),
        (
            "C1CC12CC2", 
            "C1CC12CC2",
        ),
        (
            "FC(F)(F)c1ccc(cc1)-c2ccc(cc2)[C@H](Nc3ncc(nc3)C(=O)NCCC(=O)O)CC(=O)N4C5(CCC4)CCCCC5",
            "c1cc(ccc1c1ccc(cc1)C(F)(F)F)C(CC(=O)N1CCCC12CCCCC2)Nc1cnc(cn1)C(=O)NCCC(=O)O",
        ),
    ]

    for (input_smiles, output_smiles) in tests:
        assert renumber_independent_rings(input_smiles) == output_smiles, f"Did not work for {input_smiles}"
    ```
    """   
    smiles = canonicalize_smiles(smiles)
    if sum(c.isdigit() for c in smiles) == 0:
        return smiles
    
    indices, digits = zip(*[(i,int(c)) for i,c in enumerate(smiles) if c.isdigit()])
    if len(digits) == 0: 
        return smiles
    if len(digits) %2 != 0:
        raise ValueError("There is an unclosed ring: " + smiles)
    if max(digits) > 8:
        raise ValueError("More than 8 rings: " + smiles)
    
    logger.debug(indices)
    logger.debug(digits)
    logger.debug(smiles)
    
    def find_opener(i):
        digit = digits[i]
        return indices[digits[:i].index(digit)]
    
    #available_digits = set(range(2, 9))
    i = 0
    old_digit = 0
    closer_index = indices[i]
    current_digit = digits[0]
    open_rings = {current_digit}
    available_digits = set(range(2,9))
    ring_index ={current_digit:1}
    for digit in digits[1:]:
        i += 1
        logging.debug("\n\nprocessing digit %d: %d", i, digit)
        
        if digit in open_rings:
            # close a ring
            opener_index = find_opener(i)
            closer_index = indices[i]
            logger.debug("Changing position %d", opener_index)
            smiles = smiles[:opener_index] + str(ring_index[digit]) + smiles[opener_index+1:]
            logger.debug(smiles)
            logger.debug("Changing position %d", closer_index)
            smiles = smiles[:closer_index] + str(ring_index[digit]) + smiles[closer_index+1:]
            logger.debug(smiles)
            logger.debug("Closed ring at position %d and released %d", closer_index, ring_index[digit])
            open_rings.remove(digit)
            available_digits.add(ring_index[digit])
            logger.debug("Open Rings: " + " ".join([str(x) for x in list(open_rings)]))
            logger.debug("Available Digits: " + " ".join([str(x) for x in list(available_digits)]))
            old_digit = digit
        
        else:
            # open a new ring:
            if indices[i] - closer_index == 1:
                temp_available_digits = available_digits.copy()
                temp_available_digits.remove(ring_index[old_digit])
                grabbed_digit = min(temp_available_digits)
            else:
                grabbed_digit = min(available_digits)
            ring_index[digit] = grabbed_digit
            open_rings.add(digit)
            available_digits.remove(grabbed_digit)
            logger.debug("Opening ring at position %d and grabbed %d", indices[i], ring_index[digit])
            logger.debug("Open Rings: " + " ".join([str(x) for x in list(open_rings)]))
            logger.debug("Available Digits: " + " ".join([str(x) for x in list(available_digits)]))
    
    return smiles       

class Grammar(object):
    def __init__(self, filepath=None):
        if filepath:
            self.load(filepath)

    def load(self, filepath):
        cfg_string = ''.join(list(open(filepath).readlines()))

        # parse from nltk
        cfg_grammar = nltk.CFG.fromstring(cfg_string)
        # self.cfg_parser = cfg_parser = nltk.RecursiveDescentParser(cfg_grammar)
        self.cfg_parser = cfg_parser = nltk.ChartParser(cfg_grammar)


        # our info for rule macthing
        self.head_to_rules = head_to_rules = {}
        self.valid_tokens = valid_tokens = set()
        rule_ranges = {}
        total_num_rules = 0
        first_head = None
        for line in cfg_string.split('\n'):
            if len(line.strip()) > 0:
                head, rules = line.split('->')
                head = Nonterminal(head.strip())  # remove space
                rules = [_.strip() for _ in rules.split('|')]  # split and remove space
                rules = [tuple([Nonterminal(_) if not _.startswith("'") else _[1:-1] for _ in rule.split()]) for rule in rules]
                head_to_rules[head] = rules

                for rule in rules:
                    for t in rule:
                        if isinstance(t, str):
                            valid_tokens.add(t)

                if first_head is None:
                    first_head = head

                rule_ranges[head] = (total_num_rules, total_num_rules + len(rules))
                total_num_rules += len(rules)

        self.first_head = first_head

        self.rule_ranges = rule_ranges
        self.total_num_rules = total_num_rules

    def generate(self):
        frontier = [self.first_head]
        while True:
            is_ended = not any(isinstance(item, Nonterminal) for item in frontier)
            if is_ended:
                break
            for i in range(len(frontier)):
                item = frontier[i]
                if isinstance(item, Nonterminal):
                    replacement_id = random.randint(0, len(self.head_to_rules[item]) - 1)
                    replacement = self.head_to_rules[item][replacement_id]
                    frontier = frontier[:i] + list(replacement) + frontier[i+1:]
                    break
        return ''.join(frontier)

    def tokenize(self, sent):
        # greedy tokenization
        # returns None is fails

        result = []
        n = len(sent)
        i = 0
        while i < n:
            j = i
            while j + 1 <= n and sent[i:j+1] in self.valid_tokens:
                j += 1
            if i == j:
                return None
            result.append(sent[i: j])
            i = j
        return result



class AnnotatedTree(object):
    '''Annotated Tree.

    It uses Nonterminal / Production class from nltk,
    see http://www.nltk.org/_modules/nltk/grammar.html for code.

    Attributes:
        symbol: a str object (for erminal) or a Nonterminal object (for non-terminal).
        children: a (maybe-empty) list of children.
        rule: a Production object.
        rule_selection_id: the 0-based index of which part of rule being selected. -1 for terminal.

    Method:
        is_leaf(): True iff len(children) == 0
    '''
    def __init__(self, symbol=None, children=None, rule=None, rule_selection_id=-1):
        symbol = symbol or ''
        children = children or []
        rule = rule or None
        # rule_selection_id = rule_selection_id or 0

        assert (len(children) > 0 and rule is not None) or (len(children) == 0 and rule is None)
        self.symbol = symbol
        self.children = children
        self.rule = rule
        self.rule_selection_id = rule_selection_id

    def is_leaf(self):
        return len(self.children) == 0

    def __str__(self):
        return '[Symbol = %s / Rule = %s / Rule Selection ID = %d / Children = %s]' % (
            self.symbol,
            self.rule,
            self.rule_selection_id,
            self.children
        )

    def __repr__(self):
        return self.__str__()


def parse(sent, grammar):
    '''Returns a list of trees
    (for it's possible to have multiple parse tree)

    Returns None if the parsing fails.
    '''
    # `sent` should be string
    assert isinstance(sent, str)
    sent = renumber_independent_rings(sent)

    sent = grammar.tokenize(sent)
    if sent is None:
        return None

    try:
        trees = list(grammar.cfg_parser.parse(sent))
    except ValueError:
        return None
    # print(trees)

    def _child_names(tree):
        names = []
        for child in tree:
            if isinstance(child, nltk.tree.Tree):
                names.append(Nonterminal(child._label))
            else:
                names.append(child)
        return names

    def _find_rule_selection_id(production):
        lhs, rhs = production.lhs(), production.rhs()
        assert lhs in grammar.head_to_rules
        rules = grammar.head_to_rules[lhs]
        for index, rule in enumerate(rules):
            if rhs == rule:
                return index
        assert False
        return 0


    def convert(tree):
        # convert from ntlk.tree.Tree to our AnnotatedTree

        if isinstance(tree, nltk.tree.Tree):
            symbol=Nonterminal(tree.label())
            children=list(convert(_) for _ in tree)
            rule=Production(Nonterminal(tree.label()), _child_names(tree))
            rule_selection_id = _find_rule_selection_id(rule)
            return AnnotatedTree(
                symbol=symbol,
                children=children,
                rule=rule,
                rule_selection_id=rule_selection_id
            )
        else:
            return AnnotatedTree(symbol=tree)

    trees = [convert(tree) for tree in trees]
    return trees
