from graph import Graph, graph_from_file, find, union, kruskal
from graph import route_x_out, preprocessing_test, greedy_knapsack, before_knapsack, etape_2


# route_x_out("input/network.1.in", "input/routes.1.in")

# print(preprocessing_test("input/trucks.1.in")) OK

# before_knapsack("input/network.1.in", "input/trucks.1.in", "input/routes.1.in")
route_x_out("input/network.1.in", "input/routes.1.in")
# print(etape_2("input/network.1.in","input/routes.1.in", "input/trucks.1.in"))

# print(greedy_knapsack("input/network.1.in","input/routes.1.in", "input/trucks.1.in"))
# etape_2(filename,filename_1,filename_2)
# print(len(greedy_knapsack("input/network.1.in","input/routes.1.in", "input/trucks.1.in")))

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
