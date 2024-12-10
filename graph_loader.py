import random
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
import itertools
import ast
import numpy as np

from independent_cascade import IndependentCascadeModel
from influence_maximization import InfluenceMaximizationModel

class GraphLoader:
    operation_map = {
        'sample_edges': '_op_sample_edges',
        'filter_nodes': '_op_filter_nodes',
        'topk_largest_components': '_op_topk_largest_components'
    }

    def __init__(self, file_path, is_debug=False):
        self.file_path = file_path
        self.is_debug = is_debug
        self._initialize_graph()

    def _initialize_graph(self):
        self.raw_graph = self._load_graph()
        self.cur_graph = self.raw_graph.copy()
        self.community_cache = []
        self.community_colors_cache = {}
        self.influence_matrix_cache = None
        self.top_influencers_cache = []

    def reload_graph(self, file_path):
        self.file_path = file_path
        self._initialize_graph()

    @property
    def cur_node_list(self):
        return list(self.cur_graph.nodes())

    @property
    def raw_node_list(self):
        return list(self.raw_graph.nodes())
        
    @property
    def cur_edge_list(self):
        return list(self.cur_graph.edges())

    @property
    def raw_edge_list(self):
        return list(self.raw_graph.edges())

    def set_debug(self, is_debug):
        self.is_debug = is_debug

    def _debug_print(self, *args):
        if self.is_debug:
            print("[DEBUG] Print [SRART]:")
            print(*args)
            print("[DEBUG] Print [END]")
    
    def _debug_print_graph_info(self, use_raw=False, full_info=False):
        if self.is_debug:
            print("[DEBUG] Printing Graph Info [START]:")
            self.print_graph_info(use_raw=use_raw, full_info=full_info)
            print("[DEBUG] Printing Graph Info [END]")

    def _load_graph(self):
        with open(self.file_path, 'r') as f:
            edges = [tuple(map(int, line.strip().split())) for line in f.readlines()]
            # 颠倒一下
            edges = [(edge[1], edge[0]) for edge in edges]
        G = nx.DiGraph() 
        G.add_edges_from(edges)
        return G

    def visualize_graph(self, 
                        use_raw=False, seed=42, 
                        figsize=(10, 10),
                        font_size=8,
                        node_size=50, node_color='skyblue', edge_color='gray', with_labels=False, arrowsize=10):
        plt.figure(figsize=figsize)
        
        graph_to_display = self.raw_graph if use_raw else self.cur_graph
        plt_title = ("Raw" if use_raw else "Current") + "Network Structure Visualization"

        pos = nx.spring_layout(graph_to_display, seed=seed)  
        nx.draw(
            graph_to_display,
            pos,
            with_labels=with_labels,
            node_size=node_size,
            font_size=font_size,
            node_color=node_color,
            edge_color=edge_color,
            arrowsize=10
        )
        plt.title(plt_title, fontsize=16)
        return plt
    



    def apply_operations(self, operations):
        self.cur_graph = self.raw_graph.copy() 
        for op_name, params in operations.items(): 
            parsed_args = self._parse_arguments(params)
            
            if op_name in self.operation_map:
                op_func = getattr(self, self.operation_map[op_name])
                self._debug_print(f"Applying operation: {op_name} with args: {parsed_args}")
                op_func(*parsed_args)
                self._debug_print_graph_info()
            else:
                raise ValueError(f"Undefined Operation: {op_name}")

    def _parse_arguments(self, params):
        parsed_args = []
        for key, value in params.items():
            if isinstance(value, str):
                parsed_value = self._parse_condition(value)
            else:
                parsed_value = value  
            parsed_args.append(parsed_value)
        return parsed_args

    def _parse_condition(self, condition_str):
        try:
            condition = eval(condition_str)
            if callable(condition):
                return condition
            else:
                raise ValueError("Invalid condition function.")
        except Exception as e:
            raise ValueError(f"Error parsing condition: {e}")
        
    def _op_sample_edges(self, sample_size):
        sampled_edges = random.sample(self.cur_graph.edges(), sample_size)
        self.cur_graph = self.cur_graph.edge_subgraph(sampled_edges).copy()

    def _op_filter_nodes(self, condition):
        nodes_to_remove = [node for node in self.cur_graph.nodes if condition(self.cur_graph, node)]
        self.cur_graph.remove_nodes_from(nodes_to_remove)
        print(f"Removed {len(nodes_to_remove)} nodes.")

    def _op_topk_largest_components(self, topk=1):
        undirected_G = self.cur_graph.to_undirected()
        
        components = sorted(nx.connected_components(undirected_G), key=len, reverse=True)
        
        topk_components = components[:topk]

        nodes_to_keep = set(itertools.chain.from_iterable(topk_components))

        self.cur_graph = self.cur_graph.subgraph(nodes_to_keep).copy()

        self._debug_print(f"Kept {len(nodes_to_keep)} nodes in the top {topk} largest components.")

    def print_graph_info(self, use_raw=False, full_info=False):
        if use_raw:
            graph_to_display = self.raw_graph
            print("Raw Graph Info:")
        else:
            graph_to_display = self.cur_graph
            print("Current Graph Info:")

        num_nodes = graph_to_display.number_of_nodes()  
        num_edges = graph_to_display.number_of_edges()  

        in_degrees = [graph_to_display.in_degree(n) for n in graph_to_display.nodes()]
        out_degrees = [graph_to_display.out_degree(n) for n in graph_to_display.nodes()]
        
        avg_in_degree = sum(in_degrees) / num_nodes if num_nodes > 0 else 0
        avg_out_degree = sum(out_degrees) / num_nodes if num_nodes > 0 else 0
        
        max_in_degree = max(in_degrees) if in_degrees else 0
        max_out_degree = max(out_degrees) if out_degrees else 0
        
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")
        print(f"Average in-degree: {avg_in_degree:.2f}")
        print(f"Average out-degree: {avg_out_degree:.2f}")
        print(f"Max in-degree: {max_in_degree}")
        print(f"Max out-degree: {max_out_degree}")
        
        if full_info:
            density = nx.density(graph_to_display)
            is_strongly_connected = nx.is_strongly_connected(graph_to_display)
            is_weakly_connected = nx.is_weakly_connected(graph_to_display)
            print(f"Graph density: {density:.4f}")
            print(f"Is strongly connected: {is_strongly_connected}")
            print(f"Is weakly connected: {is_weakly_connected}")

    def detect_communities(self, use_raw=False):
        graph_to_display = self.raw_graph if use_raw else self.cur_graph
        undirected_G = graph_to_display.to_undirected()

        communities = list(greedy_modularity_communities(undirected_G))
        self.community_cache = communities

        colors = plt.cm.get_cmap('tab20', len(communities))
        for i, community in enumerate(communities):
            self.community_colors_cache[i] = colors(i)  

        print(f"Detected {len(communities)} communities:")
        for idx, community in enumerate(communities):
            print(f"Community {idx}: {len(community)} nodes")

    def visualize_communities(self, use_raw=False):
        graph_to_display = self.raw_graph if use_raw else self.cur_graph
        undirected_G = graph_to_display.to_undirected()

        communities = self.community_cache
        if not communities:
            raise ValueError("Communities not detected. Please run community detection first.")
        
        node_color_map = {}
        for i, community in enumerate(communities):
            for node in community:
                node_color_map[node] = self.community_colors_cache[i] 

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(graph_to_display, seed=42)
        nx.draw(
            graph_to_display,
            pos,
            node_size=50,
            node_color=[node_color_map[node] for node in graph_to_display.nodes()],
            with_labels=False,
            edge_color='gray',
            font_size=8,
            width=0.5,
            alpha=0.7
        )
        plt.title("Community Detection Visualization", fontsize=16)
        return plt

    def get_community_by_id(self, community_ids):
        if isinstance(community_ids, int):
            community_ids = [community_ids]
        for community_id in community_ids:
            if 0 <= community_id < len(self.community_cache):
                return list(self.community_cache[community_id])
            else:
                raise IndexError(f"Community ID {community_id} is out of range. Total communities: {len(self.community_cache)}.")

    def visualize_selected_communities(self, community_ids, use_raw=False):
        if isinstance(community_ids, int):
            community_ids = [community_ids]

        graph_to_display = self.raw_graph if use_raw else self.cur_graph
        undirected_G = graph_to_display.to_undirected()

        nodes_to_include = set()
        for community_id in community_ids:
            if 0 <= community_id < len(self.community_cache):
                nodes_to_include.update(self.community_cache[community_id])
            else:
                print(f"Warning: Community ID {community_id} out of range.")

        subgraph = graph_to_display.subgraph(nodes_to_include)

        node_color_map = {}
        for node in subgraph.nodes():
            for community_id, community in enumerate(self.community_cache):
                if node in community:
                    node_color_map[node] = self.community_colors_cache[community_id]

        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw(
            subgraph,
            pos,
            node_size=50,
            with_labels=False,
            node_color=[node_color_map[node] for node in subgraph.nodes()],
            edge_color='gray',
            font_size=8,
            width=0.5,
            alpha=0.7
        )
        plt.title(f"Selected Communities: {community_ids}", fontsize=16)
        return plt
    
    def independent_cascade(self, use_raw=False, p=0.1, iterations=10):
        graph_to_display = self.raw_graph if use_raw else self.cur_graph
        nodes = graph_to_display.nodes()
        self._debug_print(nodes)
        node2idx = {node: idx for idx, node in enumerate(nodes)}
        idx2node = {idx: node for idx, node in enumerate(nodes)}
        self._debug_print(node2idx)
        num_node = len(nodes)
        graph_matrix = np.full((num_node, num_node), p)
        


        for edge in graph_to_display.edges():
            # followee -> follower
            graph_matrix[node2idx[edge[1]], node2idx[edge[0]]] = p

        independent_cascade_model = IndependentCascadeModel(graph_matrix)
        self.influence_matrix_cache, self.top_influencers_cache = independent_cascade_model.run(iterations=iterations)
        self.top_influencers_cache = [idx2node[idx] for idx in self.top_influencers_cache]
        return self.top_influencers_cache
    
    def visualize_highlight_nodes(self, highlight_nodes, use_raw=False, seed=42, 
                                  figsize=(10, 10), font_size=8, node_size=50, 
                                  node_color='skyblue', edge_color='gray', 
                                  highlight_color='red', with_labels=False, arrowsize=10):
        plt.figure(figsize=figsize)
        
        graph_to_display = self.raw_graph if use_raw else self.cur_graph
        plt_title = ("Raw" if use_raw else "Current") + " Network Structure Visualization with Highlighted Nodes"

        pos = nx.spring_layout(graph_to_display, seed=seed)
        
        node_colors = [highlight_color if node in highlight_nodes else node_color for node in graph_to_display.nodes()]
        node_alpha = [1.0 if node in highlight_nodes else 0.2 for node in graph_to_display.nodes()]
        edge_alpha = [1.0 if edge[0] in highlight_nodes or edge[1] in highlight_nodes else 0.2 for edge in graph_to_display.edges()]
        
        nx.draw(
            graph_to_display,
            pos,
            with_labels=with_labels,
            node_size=node_size,
            font_size=font_size,
            node_color=node_colors,
            edge_color=edge_color,
            arrowsize=0,
            alpha=node_alpha
        )
        
        edges = nx.draw_networkx_edges(
            graph_to_display,
            pos,
            alpha=edge_alpha,
            edge_color=edge_color
        )
        
        plt.title(plt_title, fontsize=16)
        return plt


    def calculate_node_influence(self, node, p=0.7, max_depth=10):
        influence_score = 0
        visited = set()
        queue = [(node, 0)]
        
        while queue:
            current_node, depth = queue.pop(0)
            if depth > max_depth:
                break
            if current_node not in visited:
                visited.add(current_node)
                influence_score += p ** depth
                for neighbor in self.cur_graph.successors(current_node):
                    queue.append((neighbor, depth + 1))
        
        return influence_score
    
    def run_influence_maximization(self, k=10, p=0.5):
        influence_maximization_model = InfluenceMaximizationModel(self.cur_graph)
        influential_nodes = influence_maximization_model.run(k=k, p=p)
        return influential_nodes
    