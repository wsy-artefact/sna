import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis.network import Network
from graph_loader import GraphLoader
import os
import warnings
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle
import pickle
import json
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
st.set_page_config(layout="wide")

node_detail = json.load(open('./data/TwiBot-20_cleaned.json', 'r'))
node_detail_dict = {int(node['ID']): node for node in node_detail}

def get_user_info_by_id(user_id, node_detail_dict):
    
    # 查找用户信息
    user = node_detail_dict.get(user_id)
    if user:
        profile = user.get("profile", {})
        tweets = user.get("tweets", [])
        neighbor = user.get("neighbor", {})
        info = ''
        info += f"-  名称: {profile.get('name', '未知')}\n"
        info += f"-    用户名: {profile.get('screen_name', '未知')}\n"
        info += f"-    简介: {profile.get('description', '无')}\n"
        info += f"-    创建时间: {profile.get('created_at', '未知')}\n"
        info += f"-    位置: {profile.get('location', '未知')}\n"
        info += f"-    粉丝数: {profile.get('followers_count', '未知')}\n"
        info += f"-    关注数: {profile.get('friends_count', '未知')}\n"
        info += f"-    点赞数: {profile.get('fav_count', '未知')}\n"
        # info += f"推文:\n"
        # for tweet in tweets:
        #     info += f"- {tweet}\n"
        info += f"-    邻居:\n"
        info += f"-      关注: {neighbor.get('following', [])}\n"
        info += f"-      粉丝: {neighbor.get('follower', [])}\n"
        return info
    else:
        return "未找到该用户ID的信息"



community_dict = pickle.load(open('./data/community_dict.pkl', 'rb'))
# Set header title
st.title('社交网络分析')

graph_loader = GraphLoader('../graph_edges.txt')



st.subheader('社交网络图')

G = graph_loader.cur_graph


# top30.txt read
top30 = []
with open('top30.txt', 'r') as f:
    for line in f:
        top30.append(int(line.strip()))




# 创建一个选择框让用户选择是否使用原始图
# use_raw = st.checkbox('Use Raw Graph', value=False)

# 其他参数可以通过 Streamlit 的输入组件来获取
# seed = st.number_input('Seed', value=42, min_value=1)
# figsize = st.number_input('Figure Size (width)', value=10, min_value=1), st.number_input('Figure Size (height)', value=10, min_value=1)
# font_size = st.number_input('Font Size', value=8, min_value=1)
# node_size = st.number_input('Node Size', value=50, min_value=1)

with_labels = st.checkbox('With Labels', value=False)
# arrowsize = st.number_input('Arrow Size', value=10, min_value=1)

colors = [
    "#f0f8ff", "#e6f0ff", "#d9e6ff", "#cce0ff", "#b3d9ff",
    "#99ccff", "#80c2ff", "#66b3ff", "#4da3ff", "#3399ff",
    "#f2f2f2", "#e0e0e0", "#d0d0d0", "#c0c0c0", "#b0b0b0",
    "#d9f7f3", "#b3f1eb", "#8cebe3", "#66e5db", "#4ddfcb",
    "#33d9c3", "#1ad3bb", "#00cdb3", "#00c0a3", "#00b59b",
    "#ffeff0", "#ffd9e6", "#ffb3d9", #ffebb3", "#ffedba",
    "#dbd0b2", "#b8dbb2", "#dee6dc", "#bfddde", "#98eaed",
    "#98eaed", "#a4bbe0", "#f4bff5", "#dbc8db", "#f5a9be",
    "#f5bba9", "#f5d0a9", "#f5f1a9", "#e4f5a9", "#d9f5a9",
    "#a9f5c6", "#a9f5e8", "#a9f5f2", "#a9e1f5", "#a9c8f5",
]


raw_graph = graph_loader.raw_graph

# 创建 top30 的复选框 + 显示用户所选节点信息的框
st.write('Top30 用户')
top_nodes = st.multiselect('Top30 用户', top30)
st.write('您选择的用户：', top_nodes)
# 打印用户信息 get_user_info_by_id
for node in top_nodes:
    # 粗体
    st.markdown(f'**用户 {node}**')
    st.write(get_user_info_by_id(node, node_detail_dict))

st.write('\n')
st.write(f'当前图网络检测到{len(community_dict)}个社区')

def visualize_communities(raw_graph, community_dict, highlight_nodes):
        graph_to_display = raw_graph
        undirected_G = graph_to_display.to_undirected()
        pos = nx.spring_layout(graph_to_display, seed=42)
        
        communities = community_dict
        if not communities:
            raise ValueError("Communities not detected. Please run community detection first.")

        node_color_map = {}
        for i, community in communities.items():
            for node in community:
                # print(len(colors))
                # print(i)
                node_color_map[node] = colors[i] 

        for node in highlight_nodes:
            node_color_map[node] = 'red'
        # Highlight nodes 更加大，并且带有 label

            

        node_size = [100 if node in highlight_nodes else 50 for node in graph_to_display.nodes()]
        
        node_colors = ['red' if node in highlight_nodes else node_color_map[node] for node in graph_to_display.nodes()]
        node_alpha = [1.0 if node in highlight_nodes else 0.2 for node in graph_to_display.nodes()]
        if highlight_nodes:
            edge_alpha = [1.0 if edge[0] in highlight_nodes or edge[1] in highlight_nodes else 0.2 for edge in graph_to_display.edges()]
        else:
            edge_alpha = [0.2 for edge in graph_to_display.edges()]
        
        nx.draw(
            graph_to_display,
            pos,
            with_labels=with_labels,
            node_size=node_size,
            font_size=8,
            node_color=node_colors,
            edge_color='gray',
            arrowsize=0,
            alpha=node_alpha
        )
        
        edges = nx.draw_networkx_edges(
            graph_to_display,
            pos,
            alpha=edge_alpha,
            edge_color='gray'
        )
        
            
        for i, community in communities.items():
            plt.scatter([], [], c=[colors[i]], label=f'Community {i}', edgecolors='black')

        # 设置标题并显示图例
        plt.title("Community Detection Visualization", fontsize=16)
        # 图例字小一点
        
        # plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), prop={'size': 8})
        
        return plt


st.subheader('影响力最大化算法')


st.pyplot(visualize_communities(raw_graph, community_dict, top_nodes))   



# --- 节点子树可视化 ---
st.subheader('子树可视化')


# --- 提取子图 ---
sg_with_labels = st.checkbox('显示标签', value=False, key='sg_with_labels')
if top_nodes:
    subgraph_nodes = set()  # 用于存储子图的所有节点
    for node in top_nodes:
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
        node_colors = ['red' if node in top_nodes else 'skyblue' for node in subgraph.nodes]
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(subgraph)


        nx.draw(subgraph, pos, with_labels=sg_with_labels, ax=ax, node_color=node_colors, node_size=50,  edge_color='gray', arrowsize=10)
        st.pyplot(fig)
    else:
        st.warning("没有找到有效的节点或子图为空。")

else:
    st.write("请选择至少一个节点")


# Footer
st.markdown(
    """
    <br>
    <h6></h6>
    """, unsafe_allow_html=True
    )
