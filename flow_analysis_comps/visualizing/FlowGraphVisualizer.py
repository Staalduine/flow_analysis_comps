import networkx as nx
import matplotlib.pyplot as plt

def visualize_fluid_graph_from_undirected(G_undirected, flow_pairs, pipe_width=2.0):
    """
    Converts an undirected graph with node positions and edge shape metadata into a MultiDiGraph,
    adds directional flow data, and visualizes it.

    Parameters:
        G_undirected (nx.Graph): Original graph with node positions and edge metadata.
        flow_pairs (list of dict): List of {'from_to': ..., 'to_from': ...} per edge.
        pipe_width (float): Visual thickness for all edges.

    Returns:
        None (shows matplotlib plot)
    """
    if len(flow_pairs) != G_undirected.number_of_edges():
        raise ValueError("flow_pairs must match the number of edges in G_undirected.")

    # Get node positions (either from attribute or direct dict)
    pos = nx.get_node_attributes(G_undirected, "pos")
    if not pos:
        raise ValueError("Node positions must be stored in 'pos' node attribute.")

    # Create new MultiDiGraph
    G_directed = nx.MultiDiGraph()
    G_directed.add_nodes_from(G_undirected.nodes(data=True))

    # Transfer edges with metadata and flow speeds
    for (u, v), flow in zip(G_undirected.edges(), flow_pairs):
        edge_data = G_undirected[u][v]

        # Common metadata (e.g., curvature, shape info)
        meta = {k: edge_data[k] for k in edge_data}

        # Add both directions with flow speeds and width
        G_directed.add_edge(u, v, **meta, speed_um_per_s=flow["from_to"], width=pipe_width)
        G_directed.add_edge(v, u, **meta, speed_um_per_s=flow["to_from"], width=pipe_width)

    # Draw nodes
    nx.draw_networkx_nodes(G_directed, pos, node_size=700, node_color="lightblue")
    nx.draw_networkx_labels(G_directed, pos, font_size=12)

    # Draw edges
    edges = G_directed.edges(keys=True, data=True)
    widths = [d["width"] for _, _, _, d in edges]

    # Use curvature if available; otherwise fallback
    curvature = 0.15
    nx.draw_networkx_edges(
        G_directed, pos, edgelist=edges, width=widths, edge_color="gray",
        arrows=True,
        connectionstyle=f"arc3,rad={curvature}"
    )

    # Draw edge labels (flow speed)
    edge_labels = {(u, v, k): f'{d["speed_um_per_s"]} µm/s' for u, v, k, d in edges}
    nx.draw_networkx_edge_labels(G_directed, pos, edge_labels=edge_labels, font_size=10)

    plt.title("Fluid Flow Visualization (with Positions & Edge Shape)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
