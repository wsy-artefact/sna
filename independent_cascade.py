import numpy as np
from tqdm import tqdm

class IndependentCascadeModel:
    def __init__(self, graph_matrix):
        self.graph_matrix = graph_matrix
        self.num_nodes = graph_matrix.shape[0]
        self.influence_matrix = np.zeros_like(graph_matrix)

    def _independent_cascade_process(self, active_node):
        self.influence_matrix[active_node, active_node] += 1.0
        new_active_nodes = [active_node]
        all_active_nodes = [active_node]
        while new_active_nodes:
            current_node = new_active_nodes.pop(0)
            random_prob = np.random.uniform(0, 1)
            influenced_nodes = np.where(self.graph_matrix[current_node] > random_prob)[0].tolist()
            for node in influenced_nodes:
                if node not in all_active_nodes:
                    new_active_nodes.append(node)
                    all_active_nodes.append(node)
                    self.influence_matrix[active_node, node] += 1.0

    def _find_topk_influencers(self, topk=10):
        top_influencers = []
        best_node = np.argmax(self.influence_matrix.sum(axis=1))
        best_influence_row = self.influence_matrix[best_node]
        top_influencers.append(best_node)
        for _ in range(9):
            increase_matrix = np.maximum(self.influence_matrix - best_influence_row, 0)
            best_node = np.argmax(increase_matrix.sum(axis=1))
            best_influence_row = np.maximum(self.influence_matrix[best_node], best_influence_row)
            top_influencers.append(best_node)
        return top_influencers

    def run(self, iterations, topk=10):
        for node in tqdm(range(self.num_nodes)):
            for _ in range(iterations):
                self._independent_cascade_process(node)
        self.influence_matrix /= iterations
        top_influencers = self._find_topk_influencers(topk=topk)
        return self.influence_matrix, top_influencers
