import fa_tools as fa

regex = fa.Regex("(a+b)*")#b*")#+a*+a*bb*(c+dbd)*")
regex.print_ast()
nfa = regex.make_equivalent_nfa()

from subprocess import check_output
def fa_to_popup_graphviz(self, fa):
	graphviz_src = fa.to_graphviz()
	with open("graphviz_tmp.dot", 'w') as stream:
		stream.write(graphviz_src)
	check_output("dot -q -Tpng -ographviz_tmp.png graphviz_tmp.dot")
	check_output("xdg-open graphviz_tmp.png")
fa_to_popup_graphviz(nfa)
