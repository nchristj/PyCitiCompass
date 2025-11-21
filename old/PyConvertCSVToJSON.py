import pandas as pd
import json

nodes = pd.read_csv("impact_nodes.csv")
edges = pd.read_csv("impact_edges.csv")

payload = {
    "functional_requirement": "Add email notification when payment fails",  # Change for each FR
    "nodes": nodes.to_dict(orient="records"),
    "edges": edges.to_dict(orient="records")
}

with open("impact_payload.json", "w") as f:
    json.dump(payload, f, indent=2)

print("Created impact_payload.json")
