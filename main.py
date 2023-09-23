import fa_tools as fa
import io

regex = fa.Regex("(a+b)*b*+a*+a*bb*(c+dbd)*")
# regex = fa.Regex("(c+dbd)***") TODO: figure out why this works.
regex.print_ast()
nfa = regex.make_equivalent_nfa()
nfa.elim_eps_transitions()
nfa = nfa.copyndelete_unvisitable()
dfa = nfa.convert_to_dfa()

# TODO: add 1 and 0 to regex syntax, support them in structures.
# TODO: support reapeated stars in syntax, it means just one star.
# TODO: implement minimization of DFA, DFA to regex.

fa.fa_to_popup_graphviz(dfa)
