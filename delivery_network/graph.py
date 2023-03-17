class Graph:
    """
    A class representing graphs as adjacency lists and implementing various algorithms on the graphs. Graphs in the class are not oriented. 
    Attributes: 
    -----------
    nodes: NodeType
        A list of nodes. Nodes can be of any immutable type, e.g., integer, float, or string.
        We will usually use a list of integers 1, ..., n.
    graph: dict
        A dictionnary that contains the adjacency list of each node in the form
        graph[node] = [(neighbor1, p1, d1), (neighbor1, p1, d1), ...]
        where p1 is the minimal power on the edge (node, neighbor1) and d1 is the distance on the edge
    nb_nodes: int
        The number of nodes.
    nb_edges: int
        The number of edges. 
    """

    def __init__(self, nodes=[]):
        """
        Initializes the graph with a set of nodes, and no edges. 
        Parameters: 
        -----------
        nodes: list, optional
            A list of nodes. Default is empty.
        """
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
    

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 
        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        if node1 not in self.graph:
            self.graph[node1] = []
            self.nb_nodes += 1
            self.nodes.append(node1)
        if node2 not in self.graph:
            self.graph[node2] = []
            self.nb_nodes += 1
            self.nodes.append(node2)

        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))
        self.nb_edges += 1
        
    

    def get_path_with_power(self, src, dest, power):
        visited={nodes:False for nodes in self.nodes}
        path=[]

        def visite(nodes,path):
            visited[nodes]=True
            path.append(nodes)
            if nodes==dest:
                return path
            elif nodes!=dest:
                for neighbor in self.graph[nodes]:
                    power_c,neighbor_id=neighbor[1],neighbor[0]
                    if visited[neighbor_id]==False and power_c<=power:
                        return visite(neighbor_id,path)
                    elif visited[neighbor_id]== True and nodes==dest:
                        return path
            return None 
        
        t=visite(src,path)
        return t
    

    def connected_components(self):
        liste=[]
        node_visited={nodes:False for nodes in self.nodes}

        def dfs(nodes):
            composant=[nodes]
            for voisin in self.graph[nodes]:
                voisin=voisin[0]
                if not node_visited[voisin]:
                    node_visited[voisin]=True
                    composant=composant+dfs(voisin)
            return composant 
        for node in self.nodes:
            if not node_visited[nodes]:
                liste.append(dfs(nodes))
        return liste 


    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    
    def min_power(self, src, dest):
        """
        Should return path, min_power. 
        """
        L=[]
        N=[]
        for nodes in self.nodes :
            N.append(nodes)
            for city in self.graph[nodes]:
                if city[0] not in N: #pour ne pas prendre deux fois la même puissance pour une même arrête
                   L.append(city[1])
        L.sort()
        i=0 
        while i<len(L) and self.get_path_with_power(src,dest,L[i])== None:
            i=i+1
        return L[i],self.get_path_with_power(src,dest,L[i])

    def dfs(self): #question 5 séance 2 
        """Finds the deoth of a node relative to an origin node."""
        depth=0
        depths={}
        parents={self.nodes[0]:[self.nodes[0],0]}
        visited=[]
        visited_1=[self.nodes[0]]

        def explore(node,depth):
            depths[node]=depth
            visited.append(node)
            for neighbor,power_min,dist in self.graph[node]:
                if neighbor not in visited:
                    explore(neighbor,depth+1)
                    parents[neighbor]=[node,power_min]
     
            return depths  
              
        depths=explore(self.nodes[0], 0)
        
  #      for nodes in self.nodes:
   #         for neighbor,power_min,dist in self.graph[nodes]:
    #            if neighbor not in visited_1 :
     #               visited_1.append(neighbor)
      #              parents[neighbor]=[nodes,power_min]
        
        self.depths = depths
        self.parents = parents


    def get_power_and_path(self,src,dest): #question 5 séance 2 
        depth_1=self.depths[src]
        depth_2=self.depths[dest]
        parent_1=src
        parent_2=dest
        path=[parent_1]
        L=[]
        list_power=[]

        if depth_1 > depth_2:
            while self.depths[parent_1]>depth_2:
                list_power.append(self.parents[parent_1][1])
                parent_1=self.parents[parent_1][0]
                path.append(parent_1)
                print(1)
            path.append(parent_1)
                
        elif depth_2 > depth_1 :
            while self.depths[parent_2]>depth_1:
                L=[parent_2]+L
                list_power.append(self.parents[parent_2][1])
                parent_2=self.parents[parent_2][0]
                
            L=[parent_2]+L
                
        while parent_1 != parent_2 :
            path.append(self.parents[parent_1][0])
            L=[self.parenst[parent_2][0]]+L
            list_power.append(self.parents[parent_1][1])
            list_power.append(self.parents[parent_2][1])
            parent_1= self.parents[parent_1][0]
            parent_2= self.parents[parent_2][0]
            print(3)

        path.pop()
        path=path+L
        print(path)
        print(list_power)

        return [max(list_power),path]


        


def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.
    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.
    Parameters: 
    -----------
    filename: str
        The name of the file
    Outputs: 
    -----------
    G: Graph
        An object of the class Graph with the graph from file_name.
    """
    with open(filename, "r") as file:
        n, m = map(int, file.readline().split())
        g = Graph(range(1, n+1))
        for _ in range(m):
            edge = list(map(int, file.readline().split()))
            if len(edge) == 3:
                node1, node2, power_min = edge
                g.add_edge(node1, node2, power_min) # will add dist=1 by default
            elif len(edge) == 4:
                node1, node2, power_min, dist = edge
                g.add_edge(node1, node2, power_min, dist)
            else:
                raise Exception("Format incorrect")
    return g

def find(nodes, link): #on veut trouver grâce à cette fonction dans quel graphe le noeud est.
        #si deux noeuds ont le même link alors ils sont dans le même graphe
        if link[nodes]==nodes: 
            return nodes
        return find(link[nodes],link)
     

def union(nodes_1,nodes_2,link,rank):
    root1=find(nodes_1,link)
    root2=find(nodes_2,link)
    if rank[root1]>rank[root2]: #on ajoute root2 au graphe contenant root1, rank sert juste à définir un ordre 
        link[root2]=root1
    elif rank[root1]<rank[root2]: #on ajoute root1 au graphe contenant root2 
        link[root1]=root2 
    else :
        link[root2]=root1
        rank[root1]+=1


def kruskal(g):
    liste_nodes=g.nodes
    g_mst= Graph(range(1, len(liste_nodes)+1))
    e=0
    i=0
    edges=[]
    rank={nodes:0 for nodes in liste_nodes}
    link={nodes:nodes for nodes in liste_nodes} # au début chaque noeud est dans un graphe dont il est le seul élément. 
        
    for nodes in liste_nodes : #on crée une liste contenant les arêtes ie une liste de sous-listes
        #où chaque sous liste comprend les deux sommets et la puissance minimale sur le noeud. 
        for neighbor in g.graph[nodes]:
            edges.append([nodes,neighbor[0],neighbor[1]])

    edges_sorted=sorted(edges, key=lambda item: item[2])

    while e < len(liste_nodes) - 1 and i<len(edges_sorted): #on sait que dans un arbre il y a au maximum nbres de nodes - 1 edges
        n_1,n_2,p_m = edges_sorted[i] 
        i = i + 1
        x = find(n_1, link)
        y = find(n_2, link)

        if x != y:
            e = e + 1
            g_mst.add_edge(n_1, n_2, p_m)
         #si les deux nodes ne font pas partie du même graphe connexe alors on ajoute l'edge entre les deux.
            union(x, y, link, rank)
        
    return g_mst
#la complexité de l'algorithme Kruskal est en O(Elog(V)) où V est le nombre de sommets et E
#le nombre d'arêtes.

import time

def estimated_time(filename,filename_1): #question 1 séance 2 
    #filename est le chemin vers le fichier routesx et filename_1 celui vers le fichier network 
    #associé
    g=graph_from_file(filename_1)
    with open(filename, "r") as file:
        n = map(int, file.readline())
        start=time.perf_counter()
        for i in range(20):
            src,dest,power=list(map(int, file.readline().split()))
            g.min_power(src,dest)
        end=time.perf_counter()
    return ((end-start)/10)*n

def estimation_2(filename,filename_1): #question 6 séance 2 
    #filename est le chemin associé à routex et filename_1 celui associé à network
    g=graph_from_file(filename_1)
    g_mst=kruskal(g)
    with open(filename, "r") as file:
        n = map(int, file.readline())
        start=time.perf_counter()
        for i in range(20):
            src,dest,power=list(map(int, file.readline().split()))
            g_mst.get_power_and_path(scr,dest)
        end=time.perf_counter()
    return ((end-start)/20)*n

def route_x_out(filename,filename_1): #question 6 

    g=graph_from_file(filename)
    g_mst=kruskal(g)
    f=open("input/route.x.out","a")
    with open(filename_1, "r") as file:
        n = map(int, file.readline())
        for j in range(n):
            src,dest,profit=list(map(int, file.readline().split()))
            power_min=g_mst.get_power_and_path(src,dest)[0]
            f.write(power_min)
        f.close()

def possible_trucks(self, file_truck, src, dest):
    """_summary_
    Args:
        file_truck (_type_): _description_
        src (_type_): _description_
        dest (_type_): _description_

    Returns:
        dictionary: find the trucks that are possible among those of a file trucks.x.in
        considering one graph and one journey in particular.
    """
    with open(file_truck, "r") as file:
        n = file.readline()
        n = int(n)
        d = dict()
        for i in range(n):
            truck = list(file.readline().split())
            power = int(truck[0])
            print(self.min_power(src, dest)[0])
            if int(self.min_power(src, dest)[0]) >= power:
                d[i] = True
            else:
                d[i] = False
    return d

def knapsack(nbr, truck):
    B = 25*(10^9)
    data_path = "input/"
    nbr = str(nbr)
    g = data_path + "network." + nbr + ".in"
    g = graph_from_file(g)
    filename = data_path + "routes." + nbr + ".in"





