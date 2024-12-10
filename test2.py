import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from graph_loader import GraphLoader
import warnings
import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader

# with open('./config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)
# warnings.filterwarnings('ignore')







# 数据加载
@st.cache_data
def load_data():
    return pd.read_csv('data/processed_drug_interactions.csv')

df_interact = load_data()

# 图加载
@st.cache_data
def load_graph(filepath):
    return GraphLoader(filepath)

graph_loader = load_graph('../graph_edges.txt')

# 页面标题
st.title('社交网络分析')

# 图对象
G = graph_loader.cur_graph

# --- 网络可视化 ---
st.subheader('社交网络图')


with_labels = st.checkbox('显示标签', value=False, key='with_labels')

plt = graph_loader.visualize_graph(
    use_raw=True,
    node_color='#00f2ff',
    edge_color='#9aaaab',
    with_labels=with_labels,
)

st.pyplot(plt)

# --- 影响力最大化 ---
st.subheader('影响力最大化算法')

im_topk = st.number_input('选择 TopK 节点数', value=5, min_value=1)
im_nodes = graph_loader.run_influence_maximization(k=im_topk)
im_plt = graph_loader.visualize_highlight_nodes(im_nodes)

st.write('影响力最大化的用户组：', ', '.join(map(str, im_nodes)))
st.pyplot(im_plt)

# --- 社区检测 ---
st.subheader('社区检测算法')

graph_loader.detect_communities()

communities = graph_loader.community_cache
st.write(f'检测到 {len(communities)} 个社区')

com_plt = graph_loader.visualize_communities()
st.pyplot(com_plt)

# --- 节点子树可视化 ---
st.subheader('子树可视化')
# --- 输入节点列表 ---
node_input = st.text_area('输入节点列表（用逗号分隔）', '1, 2, 3')

# 处理用户输入
try:
    node_list = [int(node.strip()) for node in node_input.split(',')]
except ValueError:
    st.warning("请确保输入的节点是有效的整数列表。")
    node_list = []

# --- 提取子图 ---
sg_with_labels = st.checkbox('显示标签', value=False, key='sg_with_labels')
if node_list:
    subgraph_nodes = set()  # 用于存储子图的所有节点
    for node in node_list:
        if node in G:
            # 提取每个节点的 ego graph（即子图）
            subgraph = nx.ego_graph(G, node, radius=30000)  
            subgraph_nodes.update(subgraph.nodes)

    if subgraph_nodes:
        # 创建包含所有节点的子图
        subgraph = G.subgraph(subgraph_nodes).copy()
        st.write(f"提取的子图包含 {len(subgraph.nodes)} 个节点和 {len(subgraph.edges)} 条边。")

        # 可视化
        # 高亮显示输入的节点
        node_colors = ['red' if node in node_list else 'skyblue' for node in subgraph.nodes]
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(subgraph)


        nx.draw(subgraph, pos, with_labels=sg_with_labels, ax=ax, node_color=node_colors, node_size=50,  edge_color='gray', arrowsize=10)
        st.pyplot(fig)
    else:
        st.warning("没有找到有效的节点或子图为空。")

else:
    st.write("请在输入框中输入一个节点列表。")

# --- Footer ---
st.markdown(
    """
    <br>
    <h6>使用测试数据生成。</h6>
    """, unsafe_allow_html=True
)




