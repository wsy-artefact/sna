import streamlit as st
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

st.set_page_config(layout="wide")
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis.network import Network
from graph_loader import GraphLoader
import os
import warnings
warnings.filterwarnings('ignore')

def read_edges_and_construct_elements(file_path):
    with open(file_path, 'r') as f:
        edges = [tuple(map(int, line.strip().split())) for line in f.readlines()]

        edges = [(edge[1], edge[0]) for edge in edges]
        
        nodes = []
        edges_list = []

        node_set = set()
        for edge in edges:
            node_set.add(edge[0])  # 被关注者
            node_set.add(edge[1])  # 施加影响力
        nodes = [{"data": {"id": node}} for node in node_set]

        for edge in edges:
            edges_list.append({"data": {"id": f"edge_{edge[0]}_{edge[1]}", "label": "followed by", "source": edge[0], "target": edge[1]}})

        elements = {"nodes": nodes, "edges": edges_list}
        
        return elements

elements = read_edges_and_construct_elements("../graph_edges.txt")






# 样式节点和边组
node_styles = [
    NodeStyle("PERSON", "#FF7F3E", "name", "person"),
    NodeStyle("POST", "#2A629A", "content", "description"),
]

edge_styles = [
    EdgeStyle("FOLLOWS", labeled=True, directed=True),
    EdgeStyle("POSTED", labeled=True, directed=True),
    EdgeStyle("QUOTES", labeled=True, directed=True),
]

# 渲染组件
st.markdown("社交网络可视化")
st_link_analysis(elements, "cose", node_styles, edge_styles)