#!/bin/bash

import fa_tools as fa
import io

regex = fa.Regex("(a+b)*+ad*")
# regex = fa.Regex("(c+dbd)***") TODO: figure out why this works.
regex.print_ast()
nfa = regex.make_equivalent_nfa()
nfa.elim_eps_transitions()
nfa = nfa.copyndelete_unvisitable()
dfa = nfa.convert_to_dfa()
fa.fa_to_popup_graphviz(dfa)
min_dfa = dfa.make_min_equiv_dfa()

# TODO: add 1 and 0 to regex syntax, support them in structures.
# TODO: support reapeated stars in syntax, it means just one star.
# TODO: implement minimization of DFA, DFA to regex.

fa.fa_to_popup_graphviz(min_dfa)

dfa = fa.DFA(4)
dfa.toggle_state_terminality(1)
fa.fa_to_popup_graphviz(dfa.make_min_equiv_dfa())
