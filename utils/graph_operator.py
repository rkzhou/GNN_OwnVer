import utils.datareader
import networkx as nx
import copy
import xgboost
import numpy as np



class Graph_self():
    def __init__(self, graph_data:utils.datareader.GraphData):
        edge_list = graph_data.adjacency.tolist()
        edge_list = self.reconstruct_edge_list(edge_list)
        self.data = copy.deepcopy(graph_data)
        self.graph = nx.Graph(edge_list)
        self.features = copy.deepcopy(graph_data.features).tolist()
        self.labels = copy.deepcopy(graph_data.labels).tolist()
        self.class_num = copy.deepcopy(graph_data.class_num)
        
    def reconstruct_edge_list(self, edge_list):
        new_edge_list = list()
        for i in range(len(edge_list[0])):
            source_node = edge_list[0][i]
            target_node = edge_list[1][i]
            node_pair = tuple([source_node, target_node])
            new_edge_list.append(node_pair)
        
        return new_edge_list
    
    def find_topk_nodes_with_adj(self, k):
        # find the top k nodes with the most edges in each class
        nodes_in_each_class = [list() for _ in range(self.class_num)]
        for i in self.data.train_nodes_index:
            nodes_in_each_class[self.labels[i]].append(i)
        
        nodes_edges = [dict() for _ in range(self.class_num)]
        topk_nodes = list()
        for class_index in range(self.class_num):
            for node_index in nodes_in_each_class[class_index]:
                node_edge_num = len(self.graph.edges(node_index))
                nodes_edges[class_index].update({node_index: node_edge_num})
        
        # sort dictionaries with the values
        new_node_edges = list()
        for class_node_edges in nodes_edges:
            class_node_edges = dict(sorted(class_node_edges.items(), key=lambda x:x[1], reverse=True))
            new_node_edges.append(class_node_edges)
        
        for i in range(self.class_num):
            topk_nodes.append(list(new_node_edges[i].keys())[:k])
        
        return topk_nodes