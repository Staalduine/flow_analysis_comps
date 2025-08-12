import networkx as nx
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
import scipy.sparse


def generate_set_node(graph_tab: pd.DataFrame) -> list:
    """
    Retrieve all nodes in a DataFrame graph

    Args:
        graph_tab (pd.DataFrame): DataFrame containing edges of a graph

    Returns:
        set: set of node id's
    """
    nodes = set()
    for _, row in graph_tab.iterrows():
        nodes.add(row["origin"])
        nodes.add(row["end"])
    return sorted(nodes)


def generate_nx_graph(graph_tab: pd.DataFrame) -> tuple[nx.Graph, dict]:
    """
    Create networkx graph from dataframe

    Args:
        graph_tab (pd.DataFrame): DataFrame with edge data

    Returns:
        tuple[nx.Graph, dict]: Graph in nx format, with a dictionary of positions
    """
    G = nx.Graph()
    pos = {}
    nodes = generate_set_node(graph_tab)
    for index, row in graph_tab.iterrows():
        identifier1 = nodes.index(row["origin"])
        identifier2 = nodes.index(row["end"])
        pos[identifier1] = np.array(row["origin"]).astype(np.int32)
        pos[identifier2] = np.array(row["end"]).astype(np.int32)
        info = {"weight": len(row["pixel_list"]), "pixel_list": row["pixel_list"]}
        G.add_edges_from([(identifier1, identifier2, info)])
    return G, pos


def extract_branches(doc_skel: dict):
    def get_neighbours(pixel):
        x = pixel[0]
        y = pixel[1]
        primary_neighbours = {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)}
        secondary_neighbours = {
            (x + 1, y - 1),
            (x + 1, y + 1),
            (x - 1, y + 1),
            (x - 1, y - 1),
        }
        num_neighbours = 0
        actual_neighbours = []
        for neighbour in primary_neighbours:
            if neighbour in non_zero_pixel:
                num_neighbours += 1
                xp = neighbour[0]
                yp = neighbour[1]
                primary_neighboursp = {
                    (xp + 1, yp),
                    (xp - 1, yp),
                    (xp, yp + 1),
                    (xp, yp - 1),
                }
                for neighbourp in primary_neighboursp:
                    secondary_neighbours.discard(neighbourp)
                actual_neighbours.append(neighbour)
        for neighbour in secondary_neighbours:
            if neighbour in non_zero_pixel:
                num_neighbours += 1
                actual_neighbours.append(neighbour)
        return (actual_neighbours, num_neighbours)

    pixel_branch_dic = {pixel: set() for pixel in doc_skel.keys()}
    is_node = {pixel: False for pixel in doc_skel.keys()}
    pixel_set = set(doc_skel.keys())
    non_zero_pixel = doc_skel
    new_index = 1
    non_explored_direction = set()
    while len(pixel_set) > 0:
        is_new_start = len(non_explored_direction) == 0
        if is_new_start:
            pixel = pixel_set.pop()
        else:
            pixel = non_explored_direction.pop()
        actual_neighbours, num_neighbours = get_neighbours(pixel)
        if is_new_start:
            if num_neighbours == 2:
                new_index += 1
                pixel_branch_dic[pixel] = {new_index}
        is_node[pixel] = num_neighbours in [0, 1, 3, 4]
        pixel_set.discard(pixel)
        #!!! This is to solve the two neighbours nodes problem
        if is_node[pixel]:
            for neighbour in actual_neighbours:
                if is_node[neighbour]:
                    new_index += 1
                    pixel_branch_dic[pixel].add(new_index)
                    pixel_branch_dic[neighbour].add(new_index)
            continue
        else:
            for neighbour in actual_neighbours:
                if neighbour in pixel_set:
                    non_explored_direction.add(neighbour)
                pixel_branch_dic[neighbour] = pixel_branch_dic[neighbour].union(
                    pixel_branch_dic[pixel]
                )
    return pixel_branch_dic, is_node, new_index


def get_neighbours2(pixel, xs, ys):
    x = pixel[0]
    y = pixel[1]
    primary_neighbours = {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)}
    secondary_neighbours = {
        (x + 1, y - 1),
        (x + 1, y + 1),
        (x - 1, y + 1),
        (x - 1, y - 1),
    }
    pixel_list = [(x, ys[i]) for i, x in enumerate(xs)]
    # num_neighbours = 0
    actual_neighbours = set()
    for neighbour in primary_neighbours:
        if neighbour in pixel_list:
            xp = neighbour[0]
            yp = neighbour[1]
            primary_neighboursp = {
                (xp + 1, yp),
                (xp - 1, yp),
                (xp, yp + 1),
                (xp, yp - 1),
            }
            for neighbourp in primary_neighboursp:
                secondary_neighbours.discard(neighbourp)
            actual_neighbours.add(neighbour)
    for neighbour in secondary_neighbours:
        if neighbour in pixel_list:
            actual_neighbours.add(neighbour)
    return actual_neighbours


def order_pixel(pixel_begin, pixel_end, pixel_list):
    ordered_list = [pixel_begin]
    current_pixel = pixel_begin
    precedent_pixel = pixel_begin
    xs = [pixel[0] for pixel in pixel_list]
    ys = [pixel[1] for pixel in pixel_list]

    while current_pixel != pixel_end:
        neighbours = get_neighbours2(current_pixel, np.array(xs), np.array(ys))
        neighbours.discard(precedent_pixel)
        precedent_pixel = current_pixel
        current_pixel = neighbours.pop()
        ordered_list.append(current_pixel)
    return ordered_list


def from_sparse_to_graph(doc_skel: dict) -> pd.DataFrame:
    column_names = ["origin", "end", "pixel_list"]
    graph = pd.DataFrame(columns=column_names)
    pixel_branch_dic, is_node, new_index = extract_branches(doc_skel)
    # nodes = []
    edges = {}
    for pixel in pixel_branch_dic:
        for branch in pixel_branch_dic[pixel]:
            right_branch = branch
            if right_branch not in edges.keys():
                edges[right_branch] = {"origin": [], "end": [], "pixel_list": [[]]}
            if is_node[pixel]:
                if len(edges[right_branch]["origin"]) == 0:
                    edges[right_branch]["origin"] = [pixel]
                else:
                    edges[right_branch]["end"] = [pixel]
            edges[right_branch]["pixel_list"][0].append(pixel)
    for branch in edges:
        if len(edges[branch]["origin"]) > 0 and len(edges[branch]["end"]) > 0:
            graph = pd.concat([graph, pd.DataFrame(edges[branch])])
    for index, row in graph.iterrows():
        row["pixel_list"] = order_pixel(row["origin"], row["end"], row["pixel_list"])
    return graph


def reconnect_degree_2(nx_graph: nx.Graph, pos: dict, has_width: bool = True) -> None:
    assert isinstance(nx_graph.degree, dict), "Graph must have nodes to reconnect."
    degree_2_nodes = [node for node in nx_graph.nodes if nx_graph.degree[node] == 2]
    while len(degree_2_nodes) > 0:
        node = degree_2_nodes.pop()
        neighbours = list(nx_graph.neighbors(node))
        right_n = neighbours[0]
        left_n = neighbours[1]
        right_edge = nx_graph.get_edge_data(node, right_n)["pixel_list"]
        left_edge = nx_graph.get_edge_data(node, left_n)["pixel_list"]
        if has_width:
            right_edge_width = nx_graph.get_edge_data(node, right_n)["width"]
            left_edge_width = nx_graph.get_edge_data(node, left_n)["width"]
        else:
            # Maybe change to Nan if it doesnt break the rest
            right_edge_width = 40
            left_edge_width = 40
        if np.any(right_edge[0] != pos[node]):
            right_edge = list(reversed(right_edge))
        if np.any(left_edge[-1] != pos[node]):
            left_edge = list(reversed(left_edge))
        pixel_list = left_edge + right_edge[1:]
        width_new = (
            right_edge_width * len(right_edge) + left_edge_width * len(left_edge)
        ) / (len(right_edge) + len(left_edge))
        info = {"weight": len(pixel_list), "pixel_list": pixel_list, "width": width_new}
        if right_n != left_n:
            connection_data = nx_graph.get_edge_data(right_n, left_n)
            if connection_data is None or connection_data["weight"] >= info["weight"]:
                if connection_data is not None:
                    nx_graph.remove_edge(right_n, left_n)
                nx_graph.add_edges_from([(right_n, left_n, info)])
        nx_graph.remove_node(node)
        degree_2_nodes = [node for node, degree in nx_graph.degree if degree == 2]
    degree_0_nodes = [node for node in nx_graph.nodes if nx_graph.degree[node] == 0]
    for node in degree_0_nodes:
        nx_graph.remove_node(node)


def remove_spurs(
    nx_g: nx.Graph, pos: dict, threshold: int = 100
) -> tuple[nx.Graph, dict]:
    assert isinstance(nx_g.degree, dict), "Graph must have nodes to remove spurs."
    found = True
    while found:
        spurs = []
        found = False
        for edge in nx_g.edges:
            edge_data = nx_g.get_edge_data(*edge)
            if (nx_g.degree[edge[0]] == 1 or nx_g.degree[edge[1]] == 1) and edge_data[
                "weight"
            ] < threshold:
                spurs.append(edge)
                found = True
        for spur in spurs:
            nx_g.remove_edge(spur[0], spur[1])
        reconnect_degree_2(nx_g, pos, has_width=False)
    return nx_g, pos


def orient(pixel_list, root_pos) -> list[tuple[int, int]]:
    """Orients a pixel list with the root position at the begining"""
    if np.all(root_pos == pixel_list[0]):
        return pixel_list
    else:
        return list(reversed(pixel_list))


def generate_index_along_sequence(n: int, resolution=3, offset=0) -> list[int]:
    """
    From the length `n` of the list, generate indexes at interval `resolution`
    with `offset` at the start and at the end.
    :param n: length of the sequence
    :param resolution: step between two chosen indexes
    :param offset: offset at the begining and at the end
    """
    x_min = offset
    x_max = n - 1 - offset
    # Small case
    if x_min > x_max:
        return [n // 2]
    # Normal case
    k_max = (x_max - x_min) // resolution
    line = [x_min + k * resolution for k in range(k_max + 1)]
    return line


def skeletonize_segmented_im(segmented: np.ndarray) -> tuple[nx.Graph, dict]:
    """
    Take segmented image and skeletonize it

    Args:
        segmented (np.ndarray): Segmented image

    Returns:
        tuple[nx.Graph, dict]: networkx graph and positions
    """
    skeletonized = skeletonize(segmented > 0)

    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph_pruned, pos = remove_spurs(nx_graph, pos, threshold=200)

    return nx_graph_pruned, pos
