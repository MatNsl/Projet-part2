from graph import Graph, graph_from_file, find, union, kruskal
from graph import possible_trucks, useful_trucks, useful_trucks_listtest, route_x_out

g = kruskal(graph_from_file("input/network.1.in"))
print(g.dfsCMpro())

# route_x_out("input/network.1.in", "input/routes.1.in")


"""
data_path = "input/"
file_name = "network.01.in"

g = graph_from_file("input/network.1.in")

res = possible_trucks(g, "input/trucks.0.in", 1, 5)
print(res)

resbis = useful_trucks_listtest("input/trucks.1.in")
print(resbis)
"""

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
