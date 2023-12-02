import fa_tools as fa

regex = fa.Regex("(a+b)*baca*")
regex.print_ast()
# TODO: convert_to_nfa()
nfa = regex.make_nfa()
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
dfa = nfa.make_dfa()

for character in "abaca":
	dfa.transit(character)
assert dfa.accepts()
dfa.reset()

for character in "abacad":
	dfa.transit(character)
assert not dfa.accepts()
dfa.reset()

cdfa = dfa.make_cdfa("abc")
min_cdfa = cdfa.make_min_cdfa()

for character in "abaca":
	min_cdfa.transit(character)
assert min_cdfa.accepts()
min_cdfa.reset()

for character in "abacad":
	cdfa.transit(character)
assert not cdfa.accepts()
cdfa.reset()

cdfa.convert_to_graphviz()
nfa.convert_to_graphviz()

dfa.transit('a')
assert not dfa.accepts()
dfa.reset()

# fa.fa_to_popup_graphviz(dfa)

dfa.add_transition(1, 'o', 2)
dfa.delete_transition(1, 'o', 2)
dfa.transit('o')
dfa.transit('o')

fa.Regex("(a+b)*(b+b)*baca*")

dfa = fa.DFA(1)
cdfa = dfa.make_cdfa('a')
min_cdfa = cdfa.make_min_cdfa()

dfa = fa.DFA(2, 1)
dfa.add_transition(1, 'a', 2)
dfa.add_transition(1, 'a', 2)
dfa.add_transition(2, 'b', 1)
dfa.toggle_terminality(2)
assert dfa.count_transitions(None, None, 2) == 1

nfa = fa.NFA(2)
nfa.add_transition(1, '', 2)
assert "&epsilon;" in nfa.convert_to_graphviz()

nfa = fa.Regex("aba").make_nfa()
nfa.elim_eps_transitions()
dfa = nfa.make_dfa()
cdfa = dfa.make_cdfa("ab")
print(cdfa.make_regex_str())
regex = fa.Regex(cdfa.make_regex_str())
regex.simplify()
regex_str = regex.to_str()
assert regex_str == "aba", regex_str

nfa = fa.Regex("1").make_nfa()
nfa.elim_eps_transitions()
dfa = nfa.make_dfa()
regex_str = dfa.make_regex_str(empty_word_chr="1")
regex = fa.Regex(regex_str)
regex.simplify()
assert regex.to_str() == "1"
