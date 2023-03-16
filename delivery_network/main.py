from graph import Graph, graph_from_file, find, union, kruskal


data_path = "input/"
file_name = "network.01.in"

g= graph_from_file("input/network.1.in")
g_1=kruskal(g)
print(kruskal(g))
print(g_1.dfs())
print(g_1.get_power_and_path(1, 18))

"""
from graph import Graph, graph_from_file, kruskal, Union, Find


data_path = "input/"
file_name = "network.01.in"

g = graph_from_file(data_path + file_name)
print(g)
"""
