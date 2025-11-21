import json
import networkx as nx
import pandas as pd
import os
from collections import deque

# ------------------------
# Load JSON safely
# ------------------------
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def extract_tuples(data):
    if isinstance(data, dict) and "#select" in data:
        return data["#select"]["tuples"]
    return data


# ------------------------
# Build Graph
# ------------------------
def build_graph(folder="."):
    G = nx.DiGraph()

    # Load files
    dto_data = load_json(os.path.join(folder, "result_dto_fields.json"))
    controller_data = extract_tuples(load_json(os.path.join(folder, "result_controller.json")))
    account_controller = extract_tuples(load_json(os.path.join(folder, "result_account_controller.json")))
    calls_data = extract_tuples(load_json(os.path.join(folder, "result_calls.json")))
    account_calls = extract_tuples(load_json(os.path.join(folder, "result_account_calls.json")))
    notif_calls = extract_tuples(load_json(os.path.join(folder, "result_notification_calls.json")))

    # Add controllers
    for t in controller_data + account_controller:
        pkg, cls, method, file, line = t[:5]
        ctrl_id = f"controller:{cls}"
        method_id = f"method:{cls}.{method}"
        G.add_node(ctrl_id, type="Controller", className=cls)
        G.add_node(method_id, type="Method", className=cls, methodName=method)
        G.add_edge(ctrl_id, method_id, label="EXPOSES")

    # Add method calls
    for t in calls_data + account_calls + notif_calls:
        callerC, callerM, calleeC, calleeM, file, line = t[:6]
        c1 = f"method:{callerC}.{callerM}"
        c2 = f"method:{calleeC}.{calleeM}"
        G.add_node(c1, type="Method", className=callerC, methodName=callerM)
        G.add_node(c2, type="Method", className=calleeC, methodName=calleeM)
        G.add_edge(c1, c2, label="CALLS")

    # DTOs
    for dto in dto_data:
        dto_id = f"dto:{dto['package']}.{dto['dto']}"
        G.add_node(dto_id, type="DTO", dto=dto["dto"])

        for field in dto["fields"]:
            f_id = f"field:{dto['dto']}.{field}"
            G.add_node(f_id, type="Field", field=field)
            G.add_edge(dto_id, f_id, label="HAS_FIELD")

    return G


# ------------------------
# Impact Analysis
# ------------------------
def impact_analysis(G, start, max_hops=4):
    # Normalize start
    start_nodes = []
    if start.startswith("method:") or start.startswith("controller:") or start.startswith("dto:"):
        if start in G.nodes:
            start_nodes.append(start)
    else:
        # Try as method class.method
        if "." in start:
            node = "method:" + start
            if node in G.nodes:
                start_nodes.append(node)

        # Try controller
        node = "controller:" + start
        if node in G.nodes:
            start_nodes.append(node)

        # Try methodName only
        for n, d in G.nodes(data=True):
            if d.get("methodName", "").lower() == start.lower():
                start_nodes.append(n)

    if not start_nodes:
        return None

    seen = set()
    queue = deque([(n, 0) for n in start_nodes])
    result_nodes, result_edges = [], []

    while queue:
        node, depth = queue.popleft()
        if node in seen or depth > max_hops:
            continue

        seen.add(node)
        result_nodes.append((node, G.nodes[node]))

        for nbr in G.successors(node):
            result_edges.append((node, nbr, G.edges[node, nbr]))
            queue.append((nbr, depth+1))

        for nbr in G.predecessors(node):
            result_edges.append((nbr, node, G.edges[nbr, node]))
            queue.append((nbr, depth+1))

    nodes_df = pd.DataFrame([{"node":n, **meta} for n,meta in result_nodes])
    edges_df = pd.DataFrame([{"from":u, "to":v, **meta} for u,v,meta in result_edges])

    return nodes_df, edges_df


if __name__ == "__main__":
    G = build_graph(".")
    print("Graph loaded with:", len(G.nodes()), "nodes,", len(G.edges()), "edges")

    nodes, edges = impact_analysis(G, "processPayment")
    print("Impacted nodes:", len(nodes))
    print("Impacted edges:", len(edges))

    nodes.to_csv("impact_nodes.csv", index=False)
    edges.to_csv("impact_edges.csv", index=False)

    print("Exported impact_nodes.csv and impact_edges.csv")
