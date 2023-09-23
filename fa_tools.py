# Finite automation may be interpreted
#   as a graph with labels on edges.
# So word state could be a synonim to
#   vertice in our case.
class FA:
	# Having the same start_state, character and end_state multiple times
	#   is pointless, it won't change the language of automation
	#   if we add or delete a duplicate transition (edge).
	# That's why transitions is a list of sets. It'll also allow to
	#   check if there's a transition quickier for consume function.
	# It's usually better for the indices of input to correspond
	#   to internal indices. If there's no additional cost for that
	#   or it's small enough. It's more convenient for debugging and
	#   formation of output.
	# If there's no transition at some point, this means the
	#   automation rejects the input. It goes to an infinite
	#   rejection loop, where it accepts any character of
	#   alphabet. We don't explicitly specify alphabet,
	#   it includes transition labels, but any other character
	#   makes automation reject the input. If we want a format fa,
	#   there's always a trash can vertice, that continuously
	#   rejects anything, if automation entered that state.
	# So the user may choose any alphabet one wants, if it
	#   includes the characters on transitions. But the
	#   language of automation will be the same, as it is
	#   linked to paths from initial state to terminal states
	#   in the automation 
	def __init__(self, num_states, initial_state=1):
		assert 1 <= initial_state <= num_states
		self.transitions = [None] + [{} for i in range(num_states)]
		self.initial_state = initial_state
		self.in_reject_state = False
		self.state_is_terminal = [None] + [False for i in range(num_states)]
		# If we're an NFA, we're in a set of states at every moment.
		self.cur_states = set([self.initial_state])
		# After every modification before running the NFA must
		#   be checked to have transitions from every state
		#   by every character of the alphabet, otherwise
		#   it may have no state to transit to.
		# The user specifies what alphabet one wants to run
		#   the FA on as an arg to prepare method. If one
		#   tricks the FA to accept the transitions, well,
		#   one will have to deal with that oneself. It's
		#   done with care for the user.
		self.alphabet_checked = False
	def add_states(self, num_new_states):
		self.transitions += [{} for i in range(num_new_states)]
		self.state_is_terminal += [False for i in range(num_new_states)]
		self.alphabet_checked = False
		self.reset()
	def reset(self):
		self.cur_states = set([self.initial_state])
		self.rejects = False
	def get_initial_state(self):
		return self.initial_state
	def get_terminal_states(self):
		return [state for state in range(1, self.get_num_states()+1) if self.check_state_is_terminal(state)]
	def get_num_states(self):
		return len(self.transitions) - 1
	def check_state_is_terminal(self, state):
		assert 1 <= state <= self.get_num_states()
		return self.state_is_terminal[state]
	def toggle_state_terminality(self, state):
		assert 1 <= state <= self.get_num_states()
		self.state_is_terminal[state] = not self.state_is_terminal[state]
		# This operation could make the FA accept a different set of
		#   words. Reset the FA, scan from the start please. It's
		#   done to avoid bugs.
		self.reset()
	def _ensure_chr_transition_set_exists(self, state, character):
		if character not in self.transitions[state]:
			self.transitions[state][character] = set()
	def _delete_chr_transition_set_if_empty(self, state, character):
		assert character in self.transitions[state]
		if len(self.transitions[state][character]) == 0:
			del self.transitions[state][character]
	def add_transition(self, start_state, character, end_state):
		assert len(character) <= 1
		assert 1 <= start_state <= self.get_num_states()
		assert 1 <= end_state   <= self.get_num_states()
		# We shouldn't disallow that, it's useful for nfa generation.
		#   I suppose, it'll also be useful in algorithms for deletion
		#   of eps transitions and others.
		# assert self.count_transitions(start_state, character, end_state) == 0, "Repeated edges are forbidden"
		self._ensure_chr_transition_set_exists(start_state, character)
		self.alphabet_checked = False
		self.transitions[start_state][character].add(end_state)
		# This operation could make the FA accept a different set of
		#   words. Reset the FA, scan from the start please. It's
		#   done to avoid bugs.
		self.reset()

	# None values stand for any possible value.
	def count_transitions(self, start_state=None, character=None, end_state=None):
		result = 0
		# tr_ stands for transition_
		for tr_start_state in range(1, self.get_num_states()+1):
			# Skip cases, when values are not what is requested
			# It's defensive programming, so called.
			if start_state is not None and tr_start_state != start_state:
				# If start_state is None, allow it's allowed to be any value.
				continue
			for tr_character, tr_end_states in self.transitions[tr_start_state].items():
				if character is not None and tr_character != character:
					continue
				for tr_end_state in tr_end_states:
					if end_state is not None and tr_end_state != end_state:
						continue
					result += 1
		return result

	def delete_transition(self, start_state, character, end_state):
		assert 1 <= start_state <= self.get_num_states()
		assert (character, end_state) in self.transitions[start_state]
		self.transitions[start_state].remove((character, end_state))
		self.alphabet_checked = False
		_delete_chr_transition_set_if_empty(start_state, character)
		# This operation could make the FA accept a different set of
		#   words. Reset the FA, scan from the start please. It's
		#   done to avoid bugs.
		self.reset()

	def iterate_transitions(self):
		for tr_start_state in range(1, self.get_num_states()+1):
			for tr_character, tr_end_states in self.transitions[tr_start_state].items():
				for tr_end_state in tr_end_states:
					yield (tr_start_state, tr_character, tr_end_state)
	def transit(self, character):
		if self.in_reject_state:
			return
		if self.count_transitions(self.cur_state, character, None) == 0:
			self.in_reject_state = True
			return
		old_states = self.cur_states
		new_states = set()
		for start_state in old_states:
			new_states |= self.transitions[start_state][character]
		self.cur_states = old_states

	def accepts(self):
		if self.in_reject_state:
			return False
		for cur_state in self.cur_states:
			if self.state_is_terminal[cur_state]:
				return True
		return False

	def convert_to_graphviz(self, node_shape="circle", term_node_shape="doublecircle"):
		result = []
		result.append("digraph G {")
		result.append("rankdir=LR;")
		result += ["node_start [shape = point];", "node_start -> node_%d;" % self.get_initial_state()]
		for state in range(1, self.get_num_states()+1):
			if self.check_state_is_terminal(state):
				result.append("node_%d [label=\"%d\", shape=\"%s\"];" % (state, state, term_node_shape))
			else:
				result.append("node_%d [label=\"%d\", shape=\"%s\"];" % (state, state, node_shape))
		for tr_start_state, tr_character, tr_end_state in self.iterate_transitions():
			character_repr = tr_character
			if tr_character == "":
				character_repr = "&epsilon;"
			result.append("node_%d -> node_%d [label=\"%s\"];" % (tr_start_state, tr_end_state, character_repr))
		result.append("}")
		return '\n'.join(result)

	# Warning! Will be very expensive operation, do this only once after you're done with all operations.
	#   Creates a new fa, where there are no unenterable states.
	def copyndelete_unvisitable(self, childclass):
		visited = [None] + [False for i in range(self.get_num_states())]
		# BFS, but all edges are of 0 weight, so we don't have to
		#   use queue, we can take any vertice from queue, they all
		#   have 0 distance.
		# Or it's just resursive operation to cover all accessible states.
		queue = set([self.initial_state])
		while queue:
			state = queue.pop()
			visited[state] = True
			for _, tr_end_states in self.transitions[state].items():
				for tr_end_state in tr_end_states:
					if visited[tr_end_state]:
						continue
					queue.add(tr_end_state)

		accessible_states = sorted([i for i in range(1, self.get_num_states()+1) if visited[i]])
		old_state_to_new_state = [None] + [None for i in range(self.get_num_states())]
		for index, accessible_state in enumerate(accessible_states):
			old_state_to_new_state[accessible_state] = index + 1
		assert self.initial_state in accessible_states, "Something went wrong.." # Initial state is always accessible from itself.
		print(len(accessible_states))
		print(self.initial_state)
		new_fa = childclass(len(accessible_states), old_state_to_new_state[self.initial_state])
		for tr_start_state, tr_character, tr_end_state in self.iterate_transitions():
			if old_state_to_new_state[tr_start_state] is None or old_state_to_new_state[tr_end_state] is None:
				continue
			new_fa.add_transition(old_state_to_new_state[tr_start_state], tr_character, old_state_to_new_state[tr_end_state])
		# Do not forget to mark accessible terminal states to be terminal in the new fa. 
		for old_state in range(1, self.get_num_states()+1):
			if self.check_state_is_terminal(old_state) and old_state_to_new_state[old_state] is not None:
				new_fa.toggle_state_terminality(old_state_to_new_state[old_state])
		return new_fa


class DFA(FA):
	def __init__(self, *args):
		super(DFA, self).__init__(*args)

	def add_transition(self, start_state, character, end_state):
		if self.count_transitions(start_state, character, end_state) > 0:
			# An already existing transition, skipping.
			return
		assert self.count_transitions(start_state, character, None) == 0, "For every starting state DFA must not have a pair of transitions with the same character in order to stay definite"
		super(DFA, self).add_transition(start_state, character, end_state)

	def copyndelete_unvisitable(self):
		return super(DFA, self).copyndelete_unvisitable(FA)


	def _image_of_state_set(self, state_set, char):
		result = set()
		for state in state_set:
			result |= self.transitions[state].get(char, set())
		return result

	def _preimage_of_state_set(self, state_set, char):
		result = set()
		for tr_start_state, tr_character, tr_end_state in self.iterate_transitions():
			if tr_end_state not in state_set:
				continue
			if tr_character == char:
				result.add(tr_start_state)
		return result

	def _class_is_split_by(self, eq_class, splitter):
		preimage = self._preimage_of_state_set(splitter[0], splitter[1])
		intersection = frozenset(preimage & eq_class)
		return 0 < len(intersection) < len(eq_class)

	def _split_class(self, eq_class, splitter):
		preimage = self._preimage_of_state_set(splitter[0], splitter[1])
		intersection = frozenset(preimage & eq_class)
		return intersection, eq_class - intersection

	# Hopcroft's algorithm to minimize a DFA.
	#   https://www.geeksforgeeks.org/minimization-of-dfa/
	#     (seems like it's Hopcroft's algorithm down the link)
	#   https://en.wikipedia.org/wiki/DFA_minimization
	#   http://i.stanford.edu/pub/cstr/reports/cs/tr/71/190/CS-TR-71-190.pdf
	#      Understood something from this source, but still not quite everything.
	#   http://www-igm.univ-mlv.fr/~berstel/Exposes/2009-06-08MinimisationLiege.pdf
	#      Page 26 saves the day.
	def make_min_equiv_dfa(self):
		# All characters that are present on edges + 'a' in case it's an automation without transitions.
		alphabet = set().union(*map(lambda d: set(d.keys()), self.transitions[1:])) | set(('a',))

		# P <- {F, F^c} (complement)
		partition = [set(), set()]
		for state in range(1, self.get_num_states()+1):
			if self.check_state_is_terminal(state):
				partition[1].add(state)
			else:
				partition[0].add(state)
		if set() in partition:
			# All of vertices were terminal, eliminate this case
			partition.remove(set())
		partition = set(map(frozenset, partition))
		waiting_set = set()
		
		smallest_partition_item = None
		for state_set in partition:
			if smallest_partition_item is None or len(state_set) < len(smallest_partition_item):
				smallest_partition_item = state_set
		for char in alphabet:
			waiting_set.add((smallest_partition_item, char))
		while waiting_set:
			# Slide 14 defines what a splitter is.
			splitter = waiting_set.pop()
			print("splitter =", splitter)
			visited_eq_classes = set()
			partition_old = set(partition)
			for eq_class in partition_old:
				if eq_class in visited_eq_classes:
					continue
				if not self._class_is_split_by(eq_class, splitter):
					print("Doesn't split", eq_class)
					continue
				part1, part2 = self._split_class(eq_class, splitter)
				print("part1 =", part1, ", ", "part2 =", part2)
				partition.remove(eq_class)
				partition.add(part1)
				partition.add(part2)
				for char in alphabet:
					if (eq_class, char) in waiting_set:
						waiting_set.remove((eq_class, char))
						waiting_set.add((part1, char))
						waiting_set.add((part2, char))
					else:
						smallest_part = None
						if len(part1) <= len(part2):
							smallest_part = part1
						else:
							smallest_part = part2
						waiting_set.add((smallest_part, char))
				visited_eq_classes.add(eq_class)
				break
		print("partition =", partition)
		partition = frozenset(map(frozenset, partition))
		eq_class_to_index = {}
		for eq_class in partition:
			eq_class_to_index[eq_class] = len(eq_class_to_index) + 1
		state_to_eq_class = [None] + [None for i in range(self.get_num_states())]
		for eq_class in partition:
			eq_class_index = eq_class_to_index[eq_class]
			for state in eq_class:
				state_to_eq_class[state] = eq_class_index
		initial_state_index = state_to_eq_class[self.initial_state]
		new_dfa = DFA(len(partition), initial_state_index)
		# Mark terminal states
		for eq_class in partition:
			for state in eq_class:
				if self.check_state_is_terminal(state):
					new_dfa.toggle_state_terminality(eq_class_to_index[eq_class])
					break
		for tr_start_state, tr_character, tr_end_state in self.iterate_transitions():
			new_dfa.add_transition(state_to_eq_class[tr_start_state], tr_character, state_to_eq_class[tr_end_state])
		return new_dfa


# Both with eps-edges and without,
#   eps edges can be eliminated
#   with a elim_eps_edges()
class NFA(FA):
	def __init__(self, *args):
		super(NFA, self).__init__(*args)

	def add_transition(self, start_state, character, end_state):
		super(NFA, self).add_transition(start_state, character, end_state)

	def copyndelete_unvisitable(self):
		return super(NFA, self).copyndelete_unvisitable(NFA)

	# Warning: language of automation may change! Only elim eps transitions
	#   should use it.
	def _delete_eps_transitions(self):
		for tr_start_state in range(1, self.get_num_states()+1):
			if '' in self.transitions[tr_start_state]:
				del self.transitions[tr_start_state]['']

	# https://www.lrde.epita.fr/dload/20070523-Seminar/delmon-eps-removal-vcsn-report.pdf
	# http://web.cecs.pdx.edu/~sheard/course/CS581/notes/NfaEpsilonDefined.pdf (main source, more concise)
	# https://www.cs.cornell.edu/courses/cs2800/2016sp/lectures/lec35-kleene.html
	# We want to compress paths that contain epsilon transitions by adding
	#   some more transitions, these would be a composition of some old transitions,
	#   and won't have eps on them (so they include a transition without eps), and
	#   eliminating eps transitions.
	# The first thing we do is we declare terminal vertices that have a reachable
	#   terminal by only eps edges. It doesn't change language of automation
	# Proof.
	# Second step.
	# Next thing is that we want for every path compress eps transitions in it.
	#   Every state has a set of eps-reachable states (reachable only by eps
	#   states), it now may now for every non-eps transition starting in this set
	#   transit straight to the end state, avoding this clutch of eps states. We
	#   add corresponding transitions.
	# It won't change the language of automation.
	#   Consider a word that was accepted previously (had a path leading to a terminal
	#   vertice with labels concatenated equal to the word), before modifications.
	#   If word's path is empty (as a sequence of transitions), then it leads to the
	#   starting vertice, word is eps, but then starting vertice was terminal and it's still
	#   terminal, we didn't do anything to it.
	#   If word's path only has eps transitions, then it's an empty word, we deleted
	#   those eps transitions. But before deletion starting vertice terminal, because it
	#   has a reachable terminal using only eps transitions. So the word is accepted now,
	#   an empty path accepts it.
	#   If the word's path has some non-eps transitions. The first thing is that
	#   we may eliminate eps transitions at the end of path. Because after any elimination
	#   end is still terminal, it has an eps-transition reachable terminal (we ordered
	#   such property before and that's the first thing we did to the automation).
	#   Now the path end (which is terminal) is visited by a non-eps transition.
	#   It's still a path in the automation before the second step.
	#   Now consider the initial state. If there are transitions at the start
	#   of path that are eps-transitions (empty labeled), we may delete them
	#   and go straight to the first state visited by a non-eps transition.
	#   There is a transition for that, intermediate vertices were eps-reachable
	#   from initial. Next we compress that first visited by a non-eps transition
	#   to reach the second visited by a non-eps transition. Repeating this step,
	#   we'll eventually visit our terminal. Yet we have a path in our new automation!
	# Done. Also we could delete vertices (states, I love to discuss automations in
	#   terms of graphs, it makes them more imaginable) that don't have any
	#   incoming transitions. They won't be visited by any words. Only eps-reachable
	#   states will be just like this and eps-elimination. And we may have a lot of
	#   eps transitions. 
	def elim_eps_transitions(self):
		# We need eps reachable states for both steps.
		# num_states iterations will be enough: shortest path to any other
		#   state may not need more than num_states transitions, otherwise
		#   we have a list of num_transitions intermediate states, which is
		#   more than num_states, so there's a state that's mentioned twice
		#   (otherwise length is not greater than num_states - 1, Dirichlet
		#   principle).
		eps_reachable_states = [None] + [set([state]) | (self.transitions[state][""] if "" in self.transitions[state] else set()) for state in range(1, self.get_num_states()+1)]
		new_eps_reachable_states = [None] + [set() for state in range(self.get_num_states())]
		# On the first iteration, eps_reachable states are all paths reachable in one transition
		#   new_eps_reachable states -- paths reachable in two transitions
		# On the second, eps_reachable states are all patha reachable in not more than four transitoins.
		iteration = 1
		while 2 ** (iteration - 1) < self.get_num_states():
			print(iteration)
			for state in range(1, self.get_num_states()+1):
				new_eps_reachable_states[state] = set(eps_reachable_states[state])
				for eps_reachable_state in eps_reachable_states[state]:
					new_eps_reachable_states[state] |= eps_reachable_states[eps_reachable_state]
			eps_reachable_states = [None] + [set(new_eps_reachable_states[i]) for i in range(1, self.get_num_states()+1)]
			iteration += 1
		# Step 1. Mark as terminal states that have eps-reachable terminal states.
		for state in range(1, self.get_num_states()+1):
			if self.check_state_is_terminal(state):
				# Already terminal, even if we mark, it doesn't change anything
				continue
			for eps_reachable_state in eps_reachable_states[state]:
				if self.check_state_is_terminal(eps_reachable_state):
					# Found a reachable terminal
					self.toggle_state_terminality(state)
					break
		# Step 2. Add "express" edges to states to shorthand moves from
		#   eps-reachables by a non-eps transition.
		new_transitions = set()
		for state in range(1, self.get_num_states()+1):
			for tr_start_state in eps_reachable_states[state]:
				if tr_start_state == state:
					# Any state is eps-reachable from itself.
					#   We don't need to readd these transitions.
					continue
				for tr_character, tr_end_states in self.transitions[tr_start_state].items():
					# Those edges we must not shorthand
					if tr_character == '':
						continue
					for tr_end_state in tr_end_states:
						new_transitions.add((state, tr_character, tr_end_state))
		for tr_start_state, tr_character, tr_end_state in new_transitions:
			self.add_transition(tr_start_state, tr_character, tr_end_state)

		# Deleting eps transitions.
		self._delete_eps_transitions()

	# In order to convert to dfa, we should assign a set of nfa states
	#   to a dfa state. And by a single character we transit from
	#   one set of nfa states to another set. We're in a set of nfa
	#   states at any moment.
	def convert_to_dfa(self):
		state_sets_queued = set()
		queue = set()
		queue.add(frozenset([self.initial_state]))
		possible_node_sets = {frozenset([self.initial_state]): [1, set()]}
		while queue:
			# These statse are a ste of states in nfa we're at the moment
			cur_states = queue.pop()
			# [Set] = [index in new graph, indices available from]
			cur_states_index = possible_node_sets[cur_states][0] # Index of the state in the new DFA, assigned when the state of DFA is discovered
			# If we move by one of these, we get a set of states.
			# If we try to move by not any of these, we get a rejected state.
			#    It's an imaginary vertice, that continuosly rejects any
			#    symbol of alphabet.
			# So our dfa will also have rejecting state and transitions to
			#   rejecting state that we won't write for the sake of simplicity.
			joined_transitions = {}
			for state in cur_states:
				for character in self.transitions[state].keys():
					if character not in joined_transitions:
						joined_transitions[character] = set()
					joined_transitions[character] |= self.transitions[state][character]
			assert '' not in joined_transitions, "must not have epsilon transitions, eliminate them first with elim_eps_transitions"
			for character in joined_transitions.keys():
				next_states = frozenset(joined_transitions[character])
				if next_states not in possible_node_sets:
					possible_node_sets[next_states] = [len(possible_node_sets)+1, set()]
				possible_node_sets[next_states][1].add((character, cur_states_index))
				if next_states in state_sets_queued:
					continue
				state_sets_queued.add(next_states)
				queue.add(next_states)
		# Now we know, what state sets are accessible in nfa. Now we can
		#   compress sets them into new states. A new state corresponds
		#   to a set of vertices
		dfa = DFA(len(possible_node_sets), 1)
		for possible_node_set in possible_node_sets.keys():
			index, accessible_from = possible_node_sets[possible_node_set]
			tr_end_state = index
			for tr_character, tr_start_state in accessible_from:
				dfa.add_transition(tr_start_state, tr_character, tr_end_state)
			is_terminal = False
			for nfa_state in possible_node_set:
				if self.check_state_is_terminal(nfa_state):
					is_terminal = True
					break
			if is_terminal:
				dfa.toggle_state_terminality(index)
		return dfa


class Scanner:
	def __init__(self, string, pos=0):
		self.string = string
		self.pos = 0
	def get_pos(self):
		return self.pos
	def peek(self):
		if self.pos < len(self.string):
			return self.string[self.pos]
		return None
	def read(self):
		result = self.peek()
		self.pos += 1
		return result

import subprocess
import sys
def fa_to_popup_graphviz(fa):
	graphviz_src = fa.convert_to_graphviz()
	with open("graphviz_tmp.dot", 'w') as stream:
		stream.write(graphviz_src)
	subprocess.check_output(["dot", "-q", "-Tpng", "-ographviz_tmp.png", "graphviz_tmp.dot"])
	subprocess.check_output(["xdg-open", "graphviz_tmp.png"])

# Syntax
# Very similar to parsing of numeric expression. So all tutorials
#   for recursive descends are valid.
# Implement parser for polynomials, multiply without stars,
#   and don't add raising to power, only plus.
#   Then make variables not just characters, but characters and optional
#   star after them or braced expression and optional star after them.
#   That's how I'll explain this to my friend tomorrow.
#   But before that he'd have to write recursive descent parser
# Not the most basic syntax for recursive descent, for beginners,
#   please, write something more basic to understand the concept.
#   For example, these:
#   https://acmp.ru/index.asp?main=task&id_task=288
#   https://acmp.ru/index.asp?main=task&id_task=937
#   These are my very first recursive descent tasks.
#   Also, be sure to check this first:
#     https://www.youtube.com/watch?v=QiZh2po6Xbk
#     https://www.youtube.com/watch?v=-_gEwmqKozQ
#   Every rule knows there's no paren at it's start
#   Regex syntax 
#   RegexImmChr = any chr of alphabet (not None, it's absence of chrs)
#     that is not special for regex syntax. We don't have to check
#     it's from alphabet if it's just not special chr, it's already there.
#   RegexChrRepOpt = RegexImmChr ['*']
#     # We can repeat a chr or a group, that's it, neither a sum nor a product.
#     # '*' is always related to the previous basic syntax item.
#   RegexProd  = (RegexChrRepOpt | RegexGroupRepOpt) {RegexChrRepOpt | RegexGroupRepOpt}
#     # If doesn't start with a paren, then it's a immediate chr, that could be
#     #   repeated. By induction. At the first encounter it's not a plus, plus
#     #   can't be without a left hand side. After that plus is always consumed
#     #   in regex sum and if we have a plus here, it means, a plus was empty.
#   RegexSum   = RegexProd {'+' RegexProd}
#   RegexGroup       = '(' (RegexGroup | RegexSum) ')'
#   RegexGroupRepOpt = RegexGroup ['*']
#   Regex            = RegexSum
# (a + b)* b* + a* + a*bb*(c+dbd)*
#   Regex (a + b)* b* + a* + a*bb*(c+dbd)*
#     RegexSum (a + b)* b* + a* + a*bb*(c+dbd)*
#       RegexProd (a + b)* b*
#         RegexGroupRepOpt (a + b)*
#         RegexChrRepOpt   b*
#       RegexProd a*
#         RegexChrRepOpt a*
#       RegexProd a*bb*(c+dbd)*
#         RegexChrRepOpt a*
#         RegexChrRepOpt b
#         RegexChrRepOpt b*
#         RegexGroupRepOpt (c+dbd)*

class RegexSyntaxItem:
	pass
class RegexSum(RegexSyntaxItem):
	# 4th (lowest) priority operation, parsed first
	# part1 + part2 + .. + partN
	def __init__(self, *parts):
		# TODO: check types, lhs and rhs are RegexSyntaxItems
		self.parts = parts
	def format_as_text_tree(self, indent=0):
		result = ' ' * indent + "RegexSum" + '\n'
		for part in self.parts:
			result += part.format_as_text_tree(indent + 1)
		return result
	def make_equivalent_nfa(self):
		result_nfa = NFA(2)
		result_nfa_init_state = 1
		result_nfa_term_state = 2
		result_nfa.toggle_state_terminality(2)

		for part in self.parts:
			part_nfa = part.make_equivalent_nfa()

			part_nfa_init_state = part_nfa.get_initial_state()
			part_nfa_term_state = part_nfa.get_terminal_states()
			try:
				assert len(part_nfa_term_state) == 1
			except:
				fa_to_popup_graphviz(part_nfa)
				raise
			part_nfa_term_state = part_nfa_term_state[0]

			state_index_shift = result_nfa.get_num_states()
			result_nfa.add_states(part_nfa.get_num_states())
			result_nfa.add_transition(result_nfa_init_state, '', state_index_shift + part_nfa_init_state)
			result_nfa.add_transition(state_index_shift + part_nfa_term_state, '', result_nfa_term_state)

			for part_nfa_tr_start, part_nfa_tr_chr, part_nfa_tr_end in part_nfa.iterate_transitions():
				result_nfa.add_transition(state_index_shift + part_nfa_tr_start, part_nfa_tr_chr, state_index_shift + part_nfa_tr_end)

		return result_nfa
class RegexProd(RegexSyntaxItem):
	# part1 part2 ... partN
	# regex abcd, for example
	#   It corresponds to language out of one word,
	#   {abcd}
	def __init__(self, *parts):
		self.parts = parts
	def format_as_text_tree(self, indent=0):
		result = ' ' * indent + "RegexProd" + '\n'
		for part in self.parts:
			result += part.format_as_text_tree(indent + 1)
		return result
	def make_equivalent_nfa(self):
		result_nfa = self.parts[0].make_equivalent_nfa()
		result_nfa_term_state = result_nfa.get_terminal_states()
		assert len(result_nfa_term_state) == 1
		result_nfa_term_state = result_nfa_term_state[0]

		for part_index in range(1, len(self.parts)):
			part = self.parts[part_index]

			part_nfa = part.make_equivalent_nfa()

			part_nfa_init_state = part_nfa.get_initial_state()
			part_nfa_term_state = part_nfa.get_terminal_states()
			assert len(part_nfa_term_state) == 1
			part_nfa_term_state = part_nfa_term_state[0]

			state_index_shift = result_nfa.get_num_states()
			result_nfa.add_states(part_nfa.get_num_states())
			result_nfa.add_transition(result_nfa_term_state, '', state_index_shift + part_nfa_init_state)
			result_nfa.toggle_state_terminality(result_nfa_term_state)
			result_nfa.toggle_state_terminality(state_index_shift + part_nfa_term_state)
			result_nfa_term_state = state_index_shift + part_nfa_term_state # We have a new terminal state

			for part_nfa_tr_start, part_nfa_tr_chr, part_nfa_tr_end in part_nfa.iterate_transitions():
				result_nfa.add_transition(state_index_shift + part_nfa_tr_start, part_nfa_tr_chr, state_index_shift + part_nfa_tr_end)

		return result_nfa
class RegexRep(RegexSyntaxItem):
	# group *
	# For example, a*, corresponds to language {eps, a, aa, ...}
	def __init__(self, repeated_part):
		self.repeated_part = repeated_part
	def format_as_text_tree(self, indent=0):
		result = ' ' * indent + "RegexRep" + '\n'
		result += self.repeated_part.format_as_text_tree(indent + 1)
		return result
	def make_equivalent_nfa(self):
		rep_nfa = self.repeated_part.make_equivalent_nfa()

		rep_nfa_init_state = rep_nfa.get_initial_state()
		rep_nfa_term_state = rep_nfa.get_terminal_states()
		assert len(rep_nfa_term_state) == 1
		rep_nfa_term_state = rep_nfa_term_state[0]

		rep_nfa.toggle_state_terminality(rep_nfa_term_state)
		# Empty character (aka empty word, eps) transition
		rep_nfa.add_transition(rep_nfa_term_state, "", rep_nfa_init_state)
		# Old initial state is now terminal
		rep_nfa.toggle_state_terminality(rep_nfa_init_state)

		return rep_nfa
class RegexImmChr(RegexSyntaxItem):
	# Immediate character
	def __init__(self, character):
		if character in ('*', '(', ')', '+'):
			raise Exception("Invalid character")
		self.character = character
	def format_as_text_tree(self, indent=0):
		result = ' ' * indent + "RegexImmChr '%c'" % self.character + '\n'
		return result
	def make_equivalent_nfa(self):
		result_nfa = NFA(2)
		result_nfa.add_transition(1, self.character, 2)
		result_nfa.toggle_state_terminality(2)
		return result_nfa


class InvalidRegexSyntaxException(Exception):
	def __init__(self, scanner):
		super().__init__("invalid syntax, position %d" % scanner.get_pos())

def parse_regex_chr_rep_opt(scanner):
	try:
		result = RegexImmChr(scanner.read())
	except:
		raise InvalidRegexSyntaxException(scanner)
	if scanner.peek() == "*":
		scanner.read()
		result = RegexRep(result)
	return result

def parse_regex_prod(scanner):
	result = None
	if scanner.peek() == '(':
		result = parse_regex_group_rep_opt(scanner)
	else:
		result = parse_regex_chr_rep_opt(scanner)
	parts = []
	# Is alphabet chr or is a open paren
	#   so just not a special char or an open paren
	while scanner.peek() is not None and (
		scanner.peek() not in ('*', '(', ')', '+') or
		scanner.peek() == '('
	):
		if scanner.peek() == '(':
			parts.append(parse_regex_group_rep_opt(scanner))
		else:
			parts.append(parse_regex_chr_rep_opt(scanner))
	if parts:
		result = RegexProd(result, *parts)
	return result

def parse_regex_sum(scanner):
	result = parse_regex_prod(scanner)
	parts = []
	while scanner.peek() == '+':
		scanner.read()
		parts.append(parse_regex_prod(scanner))
	if parts:
		result = RegexSum(result, *parts)
	return result

def parse_regex_group(scanner):
	result = None
	if scanner.read() != '(':
		raise InvalidRegexSyntaxException(scanner)
	if scanner.peek() == '(':
		result = parse_regex_group(scanner)
	else:
		result = parse_regex_sum(scanner)
	if scanner.read() != ')':
		raise InvalidRegexSyntaxException(scanner)
	if scanner.peek() == '*':
		scanner.read()
		result = RegexRep(result)
	return result

def parse_regex_group_rep_opt(scanner):
	result = parse_regex_group(scanner)
	if scanner.peek() == '*':
		scanner.read()
		result = RegexRep(result)
	return result

def parse_regex(scanner):
	result = parse_regex_sum(scanner)
	if scanner.peek() is not None:
		raise InvalidRegexSyntaxException(scanner)
	return result

class Regex:
	# If alphabet is None, it's all the characters in regex string
	#   except *, +, (, )
	# Alphabet may not contain these, if specified. If this case
	#   is not excluded, ambiguity in regex_str parsing arises.
	#   If '(' is a character or a control character (special),
	#   that groups items after it, we don't know.
	def __init__(self, regex_str, alphabet=None):
		special_chrs = ('*', '+', '(', ')')
		if alphabet is not None:
			for special_chr in special_chrs:
				if special_chr in alphabet:
					raise Exception("Found a forbidden alphabet character")
		regex_chrs = list(filter(lambda char: char not in special_chrs, regex_str))
		print(regex_chrs)
		if alphabet is None:
			alphabet = regex_chrs
		else:
			for regex_chr in regex_chrs:
				if regex_chr not in alphabet:
					raise Exception("Too small alphabet, regular expression wants characters outside of it")
		self.parse(regex_str)

	def parse(self, regex_str):
		scanner = Scanner(regex_str)
		self.ast = parse_regex(scanner)

	# Produces equivalent eps-nfa
	#   with exactly one starting
	#   state and one terminal state.
	def make_equivalent_nfa(self):
		# Constructing nfa by following regexp
		#   resursive syntax definition
		return self.ast.make_equivalent_nfa();

	def print_ast(self):
		print(self.ast.format_as_text_tree())
