from graph import Graph, graph_from_file, find, union, kruskal
from graph import possible_trucks


data_path = "input/"
file_name = "network.01.in"

g= graph_from_file("input/network.1.in")

res = possible_trucks(g, "input/trucks.0.in", 1, 5)
print(res)

"""
g_1=kruskal(g)
print(kruskal(g))
print(g_1.dfs())
print(g_1.get_power_and_path(1, 18))
"""

"""
from graph import Graph, graph_from_file, kruskal, Union, Find


data_path = "input/"
file_name = "network.01.in"

g = graph_from_file(data_path + file_name)
print(g)
"""
