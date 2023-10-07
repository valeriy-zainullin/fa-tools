import fa_tools as fa

regex = fa.Regex("(a+b)*baca*")
regex.print_ast()
# TODO: convert_to_nfa()
nfa = regex.make_equivalent_nfa()
nfa = nfa.copyndelete_unvisitable()

for character in "abaca":
	nfa.transit(character)
assert nfa.accepts()
nfa.reset()

for character in "abacad":
	nfa.transit(character)
assert not nfa.accepts()
nfa.reset()

nfa.elim_eps_transitions()
dfa = nfa.convert_to_dfa()

for character in "abaca":
	dfa.transit(character)
assert dfa.accepts()
dfa.reset()

for character in "abacad":
	dfa.transit(character)
assert not dfa.accepts()
dfa.reset()

min_dfa = dfa.make_min_equiv_dfa()

for character in "abaca":
	min_dfa.transit(character)
assert min_dfa.accepts()
min_dfa.reset()

for character in "abacad":
	dfa.transit(character)
assert not dfa.accepts()
dfa.reset()

dfa.convert_to_graphviz()
nfa.convert_to_graphviz()

dfa.transit('a')
assert not dfa.accepts()
dfa.reset()

# fa.fa_to_popup_graphviz(dfa)

dfa.add_transition(1, 'o', 2)
dfa.delete_transition(1, 'o', 2)
dfa.transit('o')
dfa.transit('o')

fa.Regex("(a+b)*(b+b)*baca*", alphabet="abcde")

dfa = fa.DFA(1)
dfa = dfa.make_min_equiv_dfa()
