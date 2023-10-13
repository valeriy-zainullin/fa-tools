#!/bin/bash

import argparse
import sys
import os
import fa_tools as fa

cur_object = None
se_output = None

# Provides good --help and validates format. I'll parse
#   everything myself. This thing is not tailored for
#   my needs. It wants to make key-value dictionary,
#   but I need a different thing. Parsing myself with
#   the input validated is more concise than providing
#   custom actions to the library.
parser = argparse.ArgumentParser(prog="fa_tools", description="Transform a NFA, DFA or a regular expression. All operations executed in the order of specification")

input_args = parser.add_argument_group("Input", "Setting the input")
input_args.add_argument("input_type",   choices=["nfa", "dfa", "regex"],        help="object in input")
input_args.add_argument("input_format", choices=["json", "akht", "plain_text"], help="input format: json dump of fa_tools, akhtyamov format for the assignment, plain_text (only userful for regular expressions)")
input_args.add_argument("input_source", type=argparse.FileType('r'),            help="source of input (input file, or - if stdin)")

regex_ops = parser.add_argument_group("Regex operations", "Operating on regular expressions")
regex_ops.add_argument(
	"-rp",
	"--regex-parse",
	help="Parse regex from string",
	action="count"
)
regex_ops.add_argument(
	"-rs",
	"--regex-simplify",
	help="Simplify regex ast",
	action="count"
)
regex_ops.add_argument(
	"-rn",
	"--regex2nfa",
	help="Convert regex to nfa (with eps-transitions)",
	action="count"
)

fa_ops = parser.add_argument_group("FA operations", "Operating on any finite automations")
fa_ops.add_argument(
	"-du",
	"--delete-unreachable",
	help="Deletes unreachable states",
	action="count"
)
fa_ops.add_argument(
	"-at",
	"--add-transition",
	help="Adds a transition (format is (start,character,end))",
	action="append"
)
fa_ops.add_argument(
	"-dt",
	"--del-transition",
	help="Deletes a transition (format is (start,character,end))",
	action="append"
)
fa_ops.add_argument(
	"-te",
	"--toggle-terminality",
	help="Checks if word is accepted by the automation",
	action="append"
)

nfa_ops = parser.add_argument_group("NFA operations", "Operating on non-deterministic finite automation")
nfa_ops.add_argument(
	"-nd", 
	"--nfa2dfa",
	help="convert nfa to dfa",
	action="count"
)
nfa_ops.add_argument(
	"-ne",
	"--noepstr",
	help="eliminate eps transitions",
	action="count"
)

dfa_ops = parser.add_argument_group("DFA operations", "Operating on deterministic finite automation")
dfa_ops.add_argument(
	"-dn",
	"--dfa2nfa",
	help="convert dfa to nfa (only changes internal types, as dfa is an nfa also)",
	action="count"
)
dfa_ops.add_argument(
	"-dc",
	"--dfa2cdfa",
	help="convert dfa to a complete dfa",
	action="count"
)
dfa_ops.add_argument(
	"-cm",
	"--cdfa-minimize",
	help="minimize (in terms of number of states) complete dfa",
	action="count"
)
dfa_ops.add_argument(
	"-dr",
	"--dfa2regex",
	help="convert dfa to regex str (can be converted to a regex object with -rp)",
	action="count"
)
dfa_ops.add_argument(
	"-ds",
	"--delete-rejectors",
	help="delete rejectors (that are added to make an automation complete)",
	action="count"
)

se_ops = parser.add_argument_group("Sideffect operations", "Operations, that don't change the object, but output information to a dedicated file")
se_ops.add_argument(
	"-ra",
	"--regex-ast",
	help="print regex ast",
	action="count"
)
se_ops.add_argument(
	"-gs",
	"--get-num-states",
	help="Get number of states in the automation. Output is number of states on a separate line.",
	action="count"
)
se_ops.add_argument(
	"-lt",
	"--list-transitions",
	help="List transitions in the automation. Output is transitions each on a separate line.",
	action="count"
)
se_ops.add_argument(
	"-ct",
	"--check-transition",
	help="checks if transition is present the automation (format is (start,character,end)). Output is \"True\" or \"False\" on a separate line.",
	action="append"
)
se_ops.add_argument(
	"-ce",
	"--check-terminality",
	help="checks if specified state is terminal in the automation. Output is \"True\" or \"False\" on a separate line.",
	action="append"
)
se_ops.add_argument(
	"-cw",
	"--check-word",
	help="checks if word is accepted by the automation. Output is \"True\" or \"False\" on a separate line.",
	action="append"
)
se_ops.add_argument(
	"-so",
	"--sideffect-output",
	help="specifies file for sideffect output (make sure to set before executing sideffect operations, otherwise you'll get no output for previous requests)",
	type=argparse.FileType('w'),
	action="store"
)

# add output type, output format, output-dst


args = parser.parse_args()

cur_object = None
sd_output = args.get("so", os.devnull)

input_type = args["input_type"]
input_format = args["input_format"]
input_data = args["input_source"].read()
if input_type == "json":
	if input_format == "regex":
		cur_object = fa.Regex.from_json(input_data)
	elif input_format == "nfa":
		cur_object = fa.NFA.from_json(input_data)
	elif input_format == "dfa"
		cur_object = fa.DFA.from_json(input_data)
#elif input_type == "akht"
# todo: all input types, forward to from_{input type} for each type.

# get output file from argparse

#arg_types = [()] # arg sinonyms, number of args consumes after itself

# get a copy of argv, delete positional args;
# cycle through argv, process args by comparing keys, taking values
#   from argparse (parser object) instead of manually from argv.
#   Skip no of args that the argument wants


import io

#regex = fa.Regex("bb(a+b*)aa*(b*+a+bb*)")
regex = fa.Regex("ab*")
nfa = regex.make_nfa()
nfa.elim_eps_transitions()
nfa = nfa.copyndelete_unvisitable()
dfa = nfa.make_dfa()
dfa.make_cdfa("abcd")
regex_str = dfa.make_regex_str()
regex = fa.Regex(regex_str)
regex.print_ast()
print(regex_str)
regex.simplify()
print(regex.to_str())
# mfdfa = dfa.make_min_cdfa()
# # fa.fa_to_popup_graphviz(mfdfa)
# print(regex.to_str())
# print(type(mfdfa))
# regex2 = fa.Regex(mfdfa.make_regex_str())
# regex2.simplify()
# print(regex2.to_str())
exit(0)

# regex = fa.Regex("(c+dbd)***") TODO: figure out why this works.
regex.print_ast()
nfa = regex.make_nfa()
nfa.elim_eps_transitions()
nfa = nfa.copyndelete_unvisitable()
dfa = nfa.make_dfa()
fa.fa_to_popup_graphviz(dfa)
min_dfa = dfa.make_min_cdfa()

# TODO: add 1 and 0 to regex syntax, support them in structures.
# TODO: support reapeated stars in syntax, it means just one star.
# TODO: implement minimization of DFA, DFA to regex.

fa.fa_to_popup_graphviz(min_dfa)

dfa = fa.DFA(4)
dfa.toggle_terminality(1)
fa.fa_to_popup_graphviz(dfa.make_min_cdfa())
