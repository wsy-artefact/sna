import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class InfluenceMaximizationModel:
    def __init__(self, graph):
        self.graph = graph


    # 生成反向可达集 (RR Set)
    def generate_rr_set(self, graph, start_node, p):
        rr_set = set()
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            if node not in rr_set:
                rr_set.add(node)
                for predecessor in graph.predecessors(node):
                    if random.random() < p:
                        queue.append(predecessor)
        return rr_set


    # 构建反向可达集的超图 H
    def build_hypergraph(self, graph, r, p):
        rr_sets = []
        nodes = list(graph.nodes())
        for _ in range(r):
            start_node = random.choice(nodes)
            rr_set = self.generate_rr_set(graph, start_node, p)
            rr_sets.append(rr_set)
        return rr_sets

    # 选择具有最大影响力的k个节点
    def influence_maximization(self, graph, k, p, rr_iterations=1000):
        # 构建超图 H（即反向可达集的集合）
        rr_sets = self.build_hypergraph(graph, rr_iterations, p)
        print(rr_sets[:5])
        nodes = list(graph.nodes())
        selected_nodes = set()

        # 贪婪选择 k 个节点
        for i in tqdm(range(k), desc="Selecting Nodes"):
            marginal_gain = np.zeros(len(nodes))
            node_index_map = {node: idx for idx, node in enumerate(nodes)}  # 创建节点到索引的映射
            
            # 计算每个节点的边际收益
            for rr_set in rr_sets:
                if not rr_set.intersection(selected_nodes):
                    for node in rr_set:
                        if node in node_index_map:
                            marginal_gain[node_index_map[node]] += 1

            # 选择边际收益最大的节点
            next_node_idx = np.argmax(marginal_gain)
            next_node = nodes[next_node_idx]
            selected_nodes.add(next_node)

            # 移除包含该节点的所有反向可达集
            rr_sets = [rr_set for rr_set in rr_sets if next_node not in rr_set]
        
        return list(selected_nodes)
    # 读取输入数据
    def read_edges_from_file(self, file_path):
        edges = []
        with open(file_path, 'r') as file:
            for line in file:
                a, b = line.strip().split()
                edges.append((a, b))
        return edges

    # 可视化社交网络图
    def visualize_graph(self, graph, influential_nodes=None):
        pos = nx.spring_layout(graph)
        plt.figure(figsize=(10, 8))
        nx.draw(graph, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=80, font_size=10, font_weight='bold')
        if influential_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=influential_nodes, node_color='red', node_size=150)
        plt.title("社交网络图及最具影响力的节点")
        plt.show()

    def visualize_highlight_nodes(self, graph, highlight_nodes, use_raw=False, seed=42, 
                                    figsize=(10, 10), font_size=8, node_size=50, 
                                    node_color='skyblue', edge_color='gray', 
                                    highlight_color='red', with_labels=False, arrowsize=10):
            plt.figure(figsize=figsize)
            
            plt_title = ("Raw" if use_raw else "Current") + " Network Structure Visualization with Highlighted Nodes"

            pos = nx.spring_layout(graph, seed=seed)
            
            node_colors = [highlight_color if node in highlight_nodes else node_color for node in graph.nodes()]
            node_alpha = [1.0 if node in highlight_nodes else 0.2 for node in graph.nodes()]
            edge_alpha = [1.0 if edge[0] in highlight_nodes or edge[1] in highlight_nodes else 0.2 for edge in graph.edges()]
            
            nx.draw(
                graph,
                pos,
                with_labels=with_labels,
                node_size=node_size,
                font_size=font_size,
                node_color=node_colors,
                edge_color=edge_color,
                arrowsize=0,
                alpha=node_alpha
            )
            
            nx.draw_networkx_edges(
                graph,
                pos,
                alpha=edge_alpha,
                edge_color=edge_color
            )
            
            plt.title(plt_title, fontsize=16)
            plt.savefig('IM_k=10')
            plt.show()

    def run(self, k, p):
        influential_nodes = self.influence_maximization(self.graph, k=k, p=p)
        
        print(f"最具影响力的 {k} 个节点: {influential_nodes}")
        
        return influential_nodes
