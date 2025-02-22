"""
Samara Canjura Galvez
CSULB ID: 013192541
CECS 427 Spring 2025 Section 01
Assignment 2: Scale and Large-Scale Networks
"""

import networkx as nx  # type: ignore
import matplotlib.pyplot as plt
import argparse
import sys
import os
import numpy as np
from scipy import stats 

def girvan_newman_components(G, n):
    """
    Partition the graph G into n connected components by iteratively removing
    the edge with the highest betweenness centrality.This simulates 
    community detection by progressively cutting bottleneck edges
    """
    components = list(nx.connected_components(G))
    while len(components) < n: 
        centrality = nx.edge_betweenness_centrality(G)
        if not centrality:
            break
        edge_to_remove = max(centrality, key=centrality.get)
        G.remove_edge(*edge_to_remove)
        components = list(nx.connected_components(G))
    return G

def compute_clustering_coefficients(G):
    """
    Compute the clustering coefficient for each node in G.
    This measures the degree to which nodes cluster together.
     """
    return nx.clustering(G)

def compute_neighborhood_overlap(G):
    """
    Compute neighborhood overlap for each node, which quantifies the similarity 
    of neighbor sets between connected nodes. This is useful for analyzing network density.
    """
    overlap = {}
    for node in G.nodes():
        neighbors = set(G.neighbors(node))
        if not neighbors:
            overlap[node] = 0.0
        else:
            sim_sum = sum(len(neighbors.intersection(set(G.neighbors(nbr)))) / 
                          len(neighbors.union(set(G.neighbors(nbr)))) 
                          for nbr in neighbors if len(neighbors.union(set(G.neighbors(nbr)))) > 0)
            overlap[node] = sim_sum / len(neighbors) if len(neighbors) > 0 else 0.0
    return overlap

def plot_graph(G, plot_option, clustering=None, overlap=None):
    """
     Plot the graph based on selected options, displaying node colors, edge signs, 
    and clustering or overlap if requested.
    
    - 'C': Highlights clustering coefficient by varying node size.
    - 'N': Highlights neighborhood overlap by varying node size.
    - 'P': Uses predefined node colors.
    """
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8,6))
    
    # Extract node colors
    node_colors = [data.get('color', 'skyblue') for _, data in G.nodes(data=True)]
    
    # Extract edge colors based on "color" attribute 
    edge_colors = ['green' if data.get('color') == 'g' else 'red' if data.get('color') == 'r' else 'black' 
                   for _, _, data in G.edges(data=True)]
    
    # Adjust node sizes based on selected plot options
    node_sizes = [500] * len(G.nodes)
    if 'C' in plot_option and clustering is not None:
        cluster_values = list(clustering.values())
        min_cluster, max_cluster = min(cluster_values), max(cluster_values) if cluster_values else (0, 1)
        node_sizes = [100 + (clustering[node] - min_cluster) / (max_cluster - min_cluster) * 900 if max_cluster != min_cluster else 500 for node in G.nodes()]
    if 'N' in plot_option and overlap is not None:
        overlap_values = list(overlap.values())
        min_overlap, max_overlap = min(overlap_values), max(overlap_values) if overlap_values else (0, 1)
        node_sizes = [100 + (overlap[node] - min_overlap) / (max_overlap - min_overlap) * 900 if max_overlap != min_overlap else 500 for node in G.nodes()]
    
    # Draw graph
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, edge_color=edge_colors, width=2)
    
    # Draw edge labels for "sign" attributes
    edge_labels = {(u, v): data.get('sign', '') for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color='blue')
    
    plt.title(f"Graph Visualization ({plot_option})")
    plt.show()


def verify_balanced_graph(G):
    """
    Check if the graph is balanced based on node attributes and edge signs.
    A graph is considered balanced if:
    - Nodes with the same color have a '+' edge sign between them.
    - Nodes with different colors have a '-' edge sign.
    """
    inconsistent_edges = [(u, v) for u, v, data in G.edges(data=True) if 'sign' in data and 'color' in G.nodes[u] and 'color' in G.nodes[v] and
                          ((G.nodes[u]['color'] == G.nodes[v]['color'] and data['sign'] != '+') or 
                           (G.nodes[u]['color'] != G.nodes[v]['color'] and data['sign'] != '-'))]
    
    if inconsistent_edges:
        print("Graph is not balanced. Inconsistent edges found:")
        for edge in inconsistent_edges:
            print("  Edge:", edge)
    else:
        print("Graph is balanced according to node attributes and edge signs.")

def main():
    parser = argparse.ArgumentParser(description="Graph Analysis Tool")
    parser.add_argument("graph_file", help="Input graph file in GML format")
    parser.add_argument("--components", type=int, help="Partition the graph into n components")  # ✅ Added
    parser.add_argument("--plot", help="Plot option: C for clustering, N for neighborhood overlap, P for node coloring")
    parser.add_argument("--verify_homophily", action="store_true", help="Test for homophily in the graph")
    parser.add_argument("--verify_balanced_graph", action="store_true", help="Check if the graph is balanced")
    parser.add_argument("--output", help="Output file to save the updated graph in GML format")  # ✅ Added
    args = parser.parse_args()

    # Verify input file exists
    if not os.path.exists(args.graph_file):
        print("Graph file does not exist:", args.graph_file)
        sys.exit(1)

    # Read the graph from GML
    try:
        G = nx.read_gml(args.graph_file)
    except Exception as e:
        print("Error reading GML file:", e)
        sys.exit(1)

    # Partition the graph into components if requested
    if args.components:
        if args.components < 1:
            print("Error: Number of components must be at least 1.")
            sys.exit(1)
        initial_components = nx.number_connected_components(G)
        if args.components <= initial_components:
            print(f"Graph already has {initial_components} or more components. No partitioning needed.")
        else:
            G = girvan_newman_components(G, args.components)
            print(f"Graph partitioned into {nx.number_connected_components(G)} components.")

    # Plot the graph if requested
    if args.plot:
        plot_option = args.plot.upper()
        if 'C' in plot_option:
            clustering = compute_clustering_coefficients(G)
            plot_graph(G, 'C', clustering=clustering)
        if 'N' in plot_option:
            overlap = compute_neighborhood_overlap(G)
            plot_graph(G, 'N', overlap=overlap)
        if 'P' in plot_option:
            plot_graph(G, 'P')

    # Run homophily verification if requested
    if args.verify_homophily:
        verify_homophily(G)

    # Run balanced graph verification if requested
    if args.verify_balanced_graph:
        verify_balanced_graph(G)

    # Save the updated graph if requested
    if args.output:
        try:
            nx.write_gml(G, args.output)
            print(f"Graph saved to {args.output}")
        except Exception as e:
            print("Error writing output graph:", e)

if __name__ == "__main__":
    main()
