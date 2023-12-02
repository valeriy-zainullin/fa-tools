#!/bin/bash

import fa_tools as fa
import io

#regex = fa.Regex("bb(a+b*)aa*(b*+a+bb*)")
regex = fa.Regex("ab*")

nfa = regex.make_nfa()
nfa.elim_eps_transitions()
nfa = nfa.copyndelete_unvisitable()

dfa = nfa.make_dfa()
cdfa = dfa.make_cdfa("abcd")

regex_str = cdfa.make_regex_str()
regex = fa.Regex(regex_str)
regex.print_ast()
print(regex_str)

regex.simplify()
print(regex.to_str())

# regex = fa.Regex("(c+dbd)***") TODO: figure out why this works.
regex.print_ast()
nfa = regex.make_nfa()
nfa.elim_eps_transitions()
nfa = nfa.copyndelete_unvisitable()
dfa = nfa.make_dfa()
fa.fa_to_popup_graphviz(dfa)
min_dfa = dfa.make_cdfa("abcd").make_minimized_cdfa()

# TODO: add 1 and 0 to regex syntax, support them in structures.
# TODO: support reapeated stars in syntax, it means just one star.
# TODO: implement minimization of DFA, DFA to regex.

fa.fa_to_popup_graphviz(min_dfa)

dfa = fa.DFA(4)
dfa.toggle_terminality(1)
fa.fa_to_popup_graphviz(dfa.make_cdfa("abcd").make_minimized_cdfa())
