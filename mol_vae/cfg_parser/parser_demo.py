#!/usr/bin/env python3

import cfg_parser as parser
import argparse

def print_parse_tree(parser, grammar, s='ClI=I=S(CBI)(-CN(C-N(N-C-F))I(S-I)C-C=I)'):
    try:
        ts = parser.parse(s, grammar)
        t = ts[0]
    except:
        raise ValueError(f"Invalid SMILES: {s}")

    print('(ugly) tree:')
    print(t)
    print()

    print('for root:')
    print('symbol is %s, is it non-terminal = %s, it\' value is %s (of type %s)' % (
        t.symbol,
        isinstance(t, parser.Nonterminal),
        t.symbol.symbol(),
        type(t.symbol.symbol())
    ))
    print('rule is %s, its left side is %s (of type %s), its right side is %s, a tuple '
    'which each element can be either str (for terminal) or Nonterminal (for nonterminal)' % (
       t.rule,
       t.rule.lhs(),
       type(t.rule.lhs()),
       t.rule.rhs(),
    ))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Parse SMILES string(s)")
    p.add_argument('--file', dest='input_file', help="File containing SMILES strings", default=None, type=str)
    p.add_argument('--smiles', dest='smi', help='SMILES string', default=None, type=str)
    args = p.parse_args()

    if (args.input_file == None) and (args.smi == None):
        raise ValueError("No input provided!")

    info_folder = '../../dropbox/context_free_grammars'
    grammar = parser.Grammar(info_folder + '/mol_zinc.grammar')

    if args.input_file:
        with open("valid_smiles", 'w') as output_file:
            with open(args.input_file, 'r') as input_file:
                for smi in input_file:
                    try:
                        smi = smi.strip()
                        print_parse_tree(parser, grammar, smi)
                        output_file.write(smi + "\n")
                    except ValueError as e:
                        print(e)
                    

    elif args.smi:
        print_parse_tree(parser, grammar, args.smi)

