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
	def __init__(self, num_states, initial_state=1):
		assert 1 <= initial_state <= num_states
		self.transitions = [None] + [{} for i in range(num_states)]
		self.initial_state = initial_state
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
	def get_initial_state(self):
		return self.initial_state
	def get_terminal_states(self):
		return [state for state in range(1, self.get_num_states()+1) if self.check_state_is_terminal(state)]
	# Warning! Will be very expensive operation, do this only once after you're done with all operations.
	#   Looks for an unenterable state, deletes transitions from it,
	#   looks for other unenterable. Then creates a new nfa.
	#def delete_unenterable_states(self):
	#	unenterable_states
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

	# Alphabet is enumeratable container of characters, which
	#   is used to obtain set of all words in theory of formal
	#   languages, FAs and regexps. If we want to obtain
	#   a language of an automation, it must have all edges
	#   for every character, otherwise some words may be
	#   neither accepted, nor rejected. And all edges must
	#   have their characters from alphabet for the automation
	#   to be formally correct FA over the alphabet.   
	def prepare(self, alphabet):
		for start_state in range(1, self.get_num_states()+1):
			for abt_character in alphabet:
				if self.count_transitions(start_state, abt_character, None) == 0:
					return False
		# Now every state has transitions for every character.
		for tr_start_state in range(1, self.get_num_states()+1):
			for tr_character, _ in self.transitions[tr_start_state]:
				if tr_character not in alphabet:
					return False
		# Now every transition character is from alphabet
		self.alphabet_checked = True
		return True

	def transit(self, character):
		if not self.alphabet_checked:
			raise Exception("Not alphabet checked")
		if self.count_transitions(self.cur_state, character, None) == 0:
			raise Exception("No such transition")
		old_states = self.cur_states
		new_states = set()
		for start_state in old_states:
			new_states |= self.transitions[start_state][character]
		self.cur_states = old_states

	def accepts(self):
		for cur_state in self.cur_states:
			if self.state_is_terminal[cur_state]:
				return True
		return False

	def fa_to_graphviz(self):
		result = []
		result.append("graph G {")
		for state in range(1, self.get_num_states()+1):
			result.append("node_%d [label=\"%d\" ];" % (state, state))
		for tr_start_state, tr_character, tr_end_state in self.iterate_transitions():
			result.append("node_%d -> node_%d [label=\"%s\"];" % (tr_start_state, tr_end_state, tr_character))
		result.append("}")
		return result.join('\n')

class DFA(FA):
	def __init__(self, *args):
		super(DFA, self).__init__(*args)

	def add_transition(self, start_state, character, end_state):
		assert self.count_transitions(start_state, character, None) == 0, "For every starting state DFA must not have a pair of transitions with the same character in order to stay definite"
		super().add_transition(self, start_state, character, end_state)

# Both with eps-edges and without,
#   eps edges can be eliminated
#   with a elim_eps_edges()
class NFA(FA):
	def __init__(self, *args):
		super(NFA, self).__init__(*args)

	def add_transition(self, start_state, character, end_state):
		super(NFA, self).add_transition(start_state, character, end_state)

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
			assert len(part_nfa_term_state) == 1
			part_nfa_term_state = part_nfa_term_state[0]

			state_index_shift = result_nfa.get_num_states()
			result_nfa.add_states(part_nfa.get_num_states())
			result_nfa.add_transition(result_nfa_init_state, '', state_index_shift + part_nfa_init_state)
			result_nfa.add_transition(state_index_shift + part_nfa_term_state, '', result_nfa_term_state)

			for part_nfa_tr_start, part_nfa_tr_chr, part_nfa_tr_end in part_nfa.iterate_transitions():
				print(part_nfa_tr_start, part_nfa_tr_chr, part_nfa_tr_end)
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

			part_nfa = part.make_equivant_nfa()

			part_nfa_init_state = rep_nfa.get_initial_state()
			part_nfa_term_state = rep_nfa.get_terminal_states()
			assert len(rep_nfa_term_state) == 1
			part_nfa_term_state = rep_nfa_term_state[0]

			state_index_shift = result_nfa.get_num_states()
			result_nfa.add_states(part_nfa.get_num_states())
			result_nfa.add_transition(result_nfa_term_state, '', state_index_shift + part_nfa_init_state)
			result_nfa.toggle_state_terminality(result_nfa_term_state)
			result_nfa.toggle_state_terminality(state_index_shift + part_nfa_term_state)

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
		result_nfa.add_transition(1, 'a', 2)
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
		regex_chrs = list(regex_str)
		for special_chr in special_chrs:
			regex_chrs.remove(special_chr)
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
