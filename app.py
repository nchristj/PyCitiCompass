import os
import re
import json
import logging
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from ollamacall import call_ollama_local

import pandas as pd
import networkx as nx
from collections import deque

from fastapi import FastAPI, HTTPException, Request
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from py2neo import Graph
import yaml

# ============================================================
# CONFIG LOADING
# ============================================================

CONFIG_PATH = os.getenv("PCC_CONFIG_PATH", "config.yaml")

if not os.path.exists(CONFIG_PATH):
    raise RuntimeError(f"Config file not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

WORK_DIR: str = cfg["work_dir"]
CODEQL_BIN: str = cfg.get("codeql_bin", "codeql")
QUERY_DIR: str = cfg["query_dir"]
SERVICES: List[Dict[str, Any]] = cfg["services"]
SRC_FOLDERS: List[str] = cfg["src_folders"]

DEFAULT_FUNCTIONAL_REQUIREMENT: str = cfg.get(
    "functional_requirement",
    "Add email notification when payment fails"
)
DEFAULT_IMPACT_START: str = cfg.get("impact_start", "processPayment")

neo4j_cfg = cfg.get("neo4j", {})
NEO4J_ENABLED: bool = bool(neo4j_cfg.get("enabled", True))
NEO4J_URI: str = neo4j_cfg.get("uri", "bolt://localhost:7687")
NEO4J_USER: str = neo4j_cfg.get("user", "neo4j")
NEO4J_PASSWORD: str = neo4j_cfg.get("password", "password")

scheduler_cfg = cfg.get("scheduler", {})
SCHED_HOUR: int = int(scheduler_cfg.get("hour", 0))
SCHED_MINUTE: int = int(scheduler_cfg.get("minute", 0))

ollama_cfg = cfg.get("ollama", {})
OLLAMA_ENABLED = ollama_cfg.get("enabled", True)
OLLAMA_LOCAL_MODE = ollama_cfg.get("local_mode", True)
OLLAMA_URL = ollama_cfg.get("url", "http://localhost:11434/api/generate")
OLLAMA_MODEL = ollama_cfg.get("model", "glm-4.6:cloud")

# ============================================================
# LOGGING
# ============================================================

os.makedirs(WORK_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(WORK_DIR, "fr_impact_pipeline.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

last_run_status: Dict[str, Any] = {
    "last_run_time": None,
    "last_run_success": None,
    "last_run_error": None,
}

neo4j_graph: Optional[Graph] = None

# ============================================================
# UTILS
# ============================================================

def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> None:
    logger.info("Running command: %s (cwd=%s)", " ".join(cmd), cwd or os.getcwd())
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        shell=False
    )
    if result.returncode != 0:
        logger.error("Command failed: %s", " ".join(cmd))
        logger.error("STDOUT:\n%s", result.stdout)
        logger.error("STDERR:\n%s", result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    else:
        if result.stdout:
            logger.info("STDOUT:\n%s", result.stdout)
        if result.stderr:
            logger.info("STDERR:\n%s", result.stderr)


def decode_bqrs_to_json(bqrs_path: str, json_path: str) -> None:
    cmd = [CODEQL_BIN, "bqrs", "decode", bqrs_path, "--format=json"]
    logger.info("Decoding BQRS %s -> %s", bqrs_path, json_path)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=False
    )
    if result.returncode != 0:
        logger.error("BQRS decode failed for %s", bqrs_path)
        logger.error("STDOUT:\n%s", result.stdout)
        logger.error("STDERR:\n%s", result.stderr)
        raise RuntimeError(f"BQRS decode failed for {bqrs_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)

# ============================================================
# STEP 1: CodeQL DB CREATION
# ============================================================

def step1_create_databases() -> None:
    logger.info("STEP 1: Creating CodeQL databases...")
    for svc in SERVICES:
        cmd = [
            CODEQL_BIN,
            "database",
            "create",
            "--language=java",
            f"--source-root={svc['project_root']}",
            "--command",
            "mvn clean install -DskipTests",
            svc["db_name"],
        ]
        run_cmd(cmd, cwd=svc["project_root"])
    logger.info("STEP 1 completed.")

# ============================================================
# STEP 2: CodeQL QUERIES + BQRS DECODE
# ============================================================

def step2_run_queries_and_decode() -> None:
    logger.info("STEP 2: Running CodeQL queries and decoding BQRS...")

    controller_ql = os.path.join(QUERY_DIR, "Controller_Extract.ql")
    methods_ql = os.path.join(QUERY_DIR, "Method_Extract.ql")
    calls_ql = os.path.join(QUERY_DIR, "Calls_Extract.ql")

    for svc in SERVICES:
        db_name = svc["db_name"]

        ctrl_bqrs = os.path.join(WORK_DIR, svc["controller_bqrs"])
        meth_bqrs = os.path.join(WORK_DIR, svc["methods_bqrs"])
        calls_bqrs = os.path.join(WORK_DIR, svc["calls_bqrs"])

        ctrl_json = os.path.join(WORK_DIR, svc["controller_json"])
        meth_json = os.path.join(WORK_DIR, svc["methods_json"])
        calls_json = os.path.join(WORK_DIR, svc["calls_json"])

        run_cmd([
            CODEQL_BIN, "query", "run",
            controller_ql,
            "--database", db_name,
            "--output", ctrl_bqrs
        ], cwd=svc["project_root"])

        run_cmd([
            CODEQL_BIN, "query", "run",
            methods_ql,
            "--database", db_name,
            "--output", meth_bqrs
        ], cwd=svc["project_root"])

        run_cmd([
            CODEQL_BIN, "query", "run",
            calls_ql,
            "--database", db_name,
            "--output", calls_bqrs
        ], cwd=svc["project_root"])

        decode_bqrs_to_json(ctrl_bqrs, ctrl_json)
        decode_bqrs_to_json(meth_bqrs, meth_json)
        decode_bqrs_to_json(calls_bqrs, calls_json)

    logger.info("STEP 2 completed.")

# ============================================================
# STEP 3: DTO EXTRACTION
# ============================================================

DTO_REGEX = re.compile(r"class\s+(\w+Dto|\w+Request|\w+Response|\w+Message)")
FIELD_REGEX = re.compile(r"private\s+[\w\<\>\[\]]+\s+(\w+);")


def step3_run_dto_extraction() -> None:
    logger.info("STEP 3: Running DTO extraction...")

    output: List[Dict[str, Any]] = []

    for src in SRC_FOLDERS:
        for root, dirs, files in os.walk(src):
            for file in files:
                if file.endswith(".java"):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    dto_match = DTO_REGEX.search(content)
                    if not dto_match:
                        continue

                    dto_name = dto_match.group(1)
                    fields = FIELD_REGEX.findall(content)

                    if fields:
                        package_match = re.search(r"package\s+([\w\.]+);", content)
                        package_name = package_match.group(1) if package_match else ""

                        output.append({
                            "dto": dto_name,
                            "package": package_name,
                            "fields": fields
                        })

    dto_json_path = os.path.join(WORK_DIR, "result_dto_fields.json")
    with open(dto_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    logger.info("Generated %s with %d DTOs.", dto_json_path, len(output))
    logger.info("STEP 3 completed.")

# ============================================================
# STEP 4: GRAPH BUILDING + CSV + PAYLOAD
# ============================================================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_tuples(data: Any) -> List:
    if isinstance(data, dict) and "#select" in data:
        return data["#select"]["tuples"]
    return data


def build_graph(folder: str) -> nx.DiGraph:
    logger.info("Building graph from JSON files in %s", folder)
    G = nx.DiGraph()

    dto_data = load_json(os.path.join(folder, "result_dto_fields.json"))

    controller_data = extract_tuples(load_json(os.path.join(folder, "result_controller.json")))
    account_controller = extract_tuples(load_json(os.path.join(folder, "result_account_controller.json")))

    calls_data = extract_tuples(load_json(os.path.join(folder, "result_calls.json")))
    account_calls = extract_tuples(load_json(os.path.join(folder, "result_account_calls.json")))
    notif_calls = extract_tuples(load_json(os.path.join(folder, "result_notification_calls.json")))

    for t in controller_data + account_controller:
        pkg, cls, method, file, line = t[:5]
        ctrl_id = f"controller:{cls}"
        method_id = f"method:{cls}.{method}"
        G.add_node(ctrl_id, type="Controller", className=cls)
        G.add_node(method_id, type="Method", className=cls, methodName=method)
        G.add_edge(ctrl_id, method_id, label="EXPOSES")

    for t in calls_data + account_calls + notif_calls:
        callerC, callerM, calleeC, calleeM, file, line = t[:6]
        c1 = f"method:{callerC}.{callerM}"
        c2 = f"method:{calleeC}.{calleeM}"
        G.add_node(c1, type="Method", className=callerC, methodName=callerM)
        G.add_node(c2, type="Method", className=calleeC, methodName=calleeM)
        G.add_edge(c1, c2, label="CALLS")

    for dto in dto_data:
        dto_id = f"dto:{dto['package']}.{dto['dto']}"
        G.add_node(dto_id, type="DTO", dto=dto["dto"])
        for field in dto["fields"]:
            f_id = f"field:{dto['dto']}.{field}"
            G.add_node(f_id, type="Field", field=field)
            G.add_edge(dto_id, f_id, label="HAS_FIELD")

    logger.info("Graph built: %d nodes, %d edges", len(G.nodes()), len(G.edges()))
    return G


def impact_analysis(G: nx.DiGraph, start: str, max_hops: int = 4) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    logger.info("Running impact analysis from start='%s', max_hops=%d", start, max_hops)

    start_nodes = []
    if start.startswith("method:") or start.startswith("controller:") or start.startswith("dto:"):
        if start in G.nodes:
            start_nodes.append(start)
    else:
        if "." in start:
            node = "method:" + start
            if node in G.nodes:
                start_nodes.append(node)

        node = "controller:" + start
        if node in G.nodes:
            start_nodes.append(node)

        for n, d in G.nodes(data=True):
            if d.get("methodName", "").lower() == start.lower():
                start_nodes.append(n)

    if not start_nodes:
        logger.warning("No start nodes found for '%s'", start)
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
            queue.append((nbr, depth + 1))

        for nbr in G.predecessors(node):
            result_edges.append((nbr, node, G.edges[nbr, node]))
            queue.append((nbr, depth + 1))

    nodes_df = pd.DataFrame([{"node": n, **meta} for n, meta in result_nodes])
    edges_df = pd.DataFrame([{"from": u, "to": v, **meta} for u, v, meta in result_edges])

    logger.info("Impact analysis result: %d nodes, %d edges", len(nodes_df), len(edges_df))
    return nodes_df, edges_df


def export_impact_to_csv_and_json(
        G: nx.DiGraph,
        fr_text: str = DEFAULT_FUNCTIONAL_REQUIREMENT,
        start: str = DEFAULT_IMPACT_START,
        folder: str = WORK_DIR
) -> None:
    res = impact_analysis(G, start)
    if res is None:
        raise RuntimeError(f"No impact analysis result from start='{start}'")

    nodes_df, edges_df = res

    nodes_csv = os.path.join(folder, "impact_nodes.csv")
    edges_csv = os.path.join(folder, "impact_edges.csv")

    nodes_df.to_csv(nodes_csv, index=False)
    edges_df.to_csv(edges_csv, index=False)

    logger.info("Exported %s and %s", nodes_csv, edges_csv)

    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    payload = {
        "functional_requirement": fr_text,
        "nodes": nodes.to_dict(orient="records"),
        "edges": edges.to_dict(orient="records"),
    }

    payload_path = os.path.join(folder, "impact_payload.json")
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Created impact payload JSON at %s", payload_path)


def step4_build_graph_and_payload() -> nx.DiGraph:
    logger.info("STEP 4: Building graph and consolidating JSON call map...")
    G = build_graph(WORK_DIR)
    export_impact_to_csv_and_json(G)
    logger.info("STEP 4 completed.")
    return G

# ============================================================
# STEP 5: Neo4j PERSISTENCE (py2neo, Option B)
# ============================================================

def get_snapshot_id() -> str:
    return datetime.utcnow().isoformat()


def ensure_neo4j_constraints(graph: Graph) -> None:
    graph.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) "
        "REQUIRE (n.id, n.snapshot) IS NODE KEY"
    )
    graph.run(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:ImpactSnapshot) "
        "REQUIRE s.snapshot IS UNIQUE"
    )


def save_graph_to_neo4j(G: nx.DiGraph, snapshot_id: str) -> None:
    if not NEO4J_ENABLED or neo4j_graph is None:
        logger.info("Neo4j is disabled or graph not initialized; skipping graph persistence.")
        return

    graph = neo4j_graph
    logger.info("STEP 5A: Saving graph to Neo4j with snapshot '%s'...", snapshot_id)

    ensure_neo4j_constraints(graph)

    for node_id, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "Unknown")
        props = {
            "id": node_id,
            "snapshot": snapshot_id,
            "type": node_type,
        }
        for k, v in attrs.items():
            if k not in ["type"]:
                props[k] = v

        graph.run(
            """
            MERGE (n:Node {id: $id, snapshot: $snapshot})
            SET n.type = $type,
                n += $props
            """,
            id=node_id,
            snapshot=snapshot_id,
            type=node_type,
            props=props
        )

    for u, v, attrs in G.edges(data=True):
        rel_type = attrs.get("label", "RELATED")
        props = {"snapshot": snapshot_id}
        for k, val in attrs.items():
            if k != "label":
                props[k] = val

        cypher = f"""
        MATCH (a:Node {{id: $from, snapshot: $snapshot}}),
              (b:Node {{id: $to, snapshot: $snapshot}})
        MERGE (a)-[r:{rel_type} {{snapshot: $snapshot}}]->(b)
        SET r += $props
        """

        graph.run(
            cypher,
            **{
                "from": u,
                "to": v,
                "snapshot": snapshot_id,
                "props": props
            }
        )

    logger.info("STEP 5A completed: graph stored in Neo4j.")


def save_impact_payload_to_neo4j(snapshot_id: str) -> None:
    if not NEO4J_ENABLED or neo4j_graph is None:
        logger.info("Neo4j is disabled or graph not initialized; skipping payload persistence.")
        return

    payload_path = os.path.join(WORK_DIR, "impact_payload.json")
    if not os.path.exists(payload_path):
        logger.warning("impact_payload.json not found at %s; skipping.", payload_path)
        return

    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    fr_text = payload.get("functional_requirement", "")
    node_count = len(payload.get("nodes", []))
    edge_count = len(payload.get("edges", []))

    logger.info(
        "STEP 5B: Saving impact payload to Neo4j: snapshot=%s, nodes=%d, edges=%d",
        snapshot_id, node_count, edge_count
    )

    graph = neo4j_graph
    graph.run(
        """
        MERGE (s:ImpactSnapshot {snapshot: $snapshot})
        SET s.functional_requirement = $fr,
            s.nodeCount = $nodeCount,
            s.edgeCount = $edgeCount,
            s.raw = $raw
        """,
        snapshot=snapshot_id,
        fr=fr_text,
        nodeCount=node_count,
        edgeCount=edge_count,
        raw=json.dumps(payload)
    )

    logger.info("STEP 5B completed: impact payload stored in Neo4j.")


def step5_save_to_neo4j(G: nx.DiGraph, snapshot_id: str) -> None:
    if not NEO4J_ENABLED:
        logger.info("Neo4j integration disabled via config.")
        return

    if neo4j_graph is None:
        logger.warning("Neo4j graph is not initialized; skipping Neo4j persistence.")
        return

    logger.info("STEP 5: Persisting analysis results to Neo4j...")
    save_graph_to_neo4j(G, snapshot_id)
    save_impact_payload_to_neo4j(snapshot_id)
    logger.info("STEP 5 completed.")

# ============================================================
# STEP 6: SNAPSHOT READ + DIFF + IMPACT HELPERS
# ============================================================

def get_snapshot_payload(snapshot_id: str) -> Dict[str, Any]:
    if not NEO4J_ENABLED or neo4j_graph is None:
        raise RuntimeError("Neo4j is not enabled or not connected")

    result = neo4j_graph.run(
        "MATCH (s:ImpactSnapshot {snapshot: $snapshot}) RETURN s.raw AS raw",
        snapshot=snapshot_id
    ).data()

    if not result:
        raise KeyError(f"Snapshot '{snapshot_id}' not found")

    raw_str = result[0]["raw"]
    return json.loads(raw_str)


def get_snapshot_graph(snapshot_id: str) -> Dict[str, Any]:
    if not NEO4J_ENABLED or neo4j_graph is None:
        raise RuntimeError("Neo4j is not enabled or not connected")

    graph = neo4j_graph

    nodes_res = graph.run(
        "MATCH (n:Node {snapshot: $snapshot}) RETURN n.id AS id, n.type AS type",
        snapshot=snapshot_id
    ).data()

    rels_res = graph.run(
        """
        MATCH (a:Node {snapshot: $snapshot})-[r {snapshot: $snapshot}]->(b:Node {snapshot: $snapshot})
        RETURN a.id AS from, b.id AS to, type(r) AS type
        """,
        snapshot=snapshot_id
    ).data()

    return {
        "snapshot": snapshot_id,
        "nodes": nodes_res,
        "edges": rels_res,
    }


def diff_snapshots(from_snapshot: str, to_snapshot: str) -> Dict[str, Any]:
    if not NEO4J_ENABLED or neo4j_graph is None:
        raise RuntimeError("Neo4j is not enabled or not connected")

    g = neo4j_graph

    nodes_from = g.run(
        "MATCH (n:Node {snapshot: $snapshot}) RETURN n.id AS id",
        snapshot=from_snapshot
    ).data()
    nodes_to = g.run(
        "MATCH (n:Node {snapshot: $snapshot}) RETURN n.id AS id",
        snapshot=to_snapshot
    ).data()

    edges_from = g.run(
        """
        MATCH (a:Node {snapshot: $snapshot})-[r {snapshot: $snapshot}]->(b:Node {snapshot: $snapshot})
        RETURN a.id AS from, b.id AS to, type(r) AS type
        """,
        snapshot=from_snapshot
    ).data()
    edges_to = g.run(
        """
        MATCH (a:Node {snapshot: $snapshot})-[r {snapshot: $snapshot}]->(b:Node {snapshot: $snapshot})
        RETURN a.id AS from, b.id AS to, type(r) AS type
        """,
        snapshot=to_snapshot
    ).data()

    set_nodes_from = {n["id"] for n in nodes_from}
    set_nodes_to = {n["id"] for n in nodes_to}

    nodes_added = sorted(list(set_nodes_to - set_nodes_from))
    nodes_removed = sorted(list(set_nodes_from - set_nodes_to))

    def edge_key(e: Dict[str, Any]) -> Tuple[str, str, str]:
        return (e["from"], e["to"], e["type"])

    set_edges_from = {edge_key(e) for e in edges_from}
    set_edges_to = {edge_key(e) for e in edges_to}

    edges_added = sorted(list(set_edges_to - set_edges_from))
    edges_removed = sorted(list(set_edges_from - set_edges_to))

    return {
        "from_snapshot": from_snapshot,
        "to_snapshot": to_snapshot,
        "nodes_added": nodes_added,
        "nodes_removed": nodes_removed,
        "edges_added": edges_added,
        "edges_removed": edges_removed,
    }


def impact_for_node(snapshot_id: str, node_id: str, max_hops: int = 4) -> Dict[str, Any]:
    if not NEO4J_ENABLED or neo4j_graph is None:
        raise RuntimeError("Neo4j is not enabled or not connected")

    g = neo4j_graph

    res = g.run(
        """
        MATCH (start:Node {id: $id, snapshot: $snapshot})
        MATCH path = (start)-[r*1..$max_hops]->(n:Node {snapshot: $snapshot})
        WITH collect(DISTINCT n) AS nodes, collect(DISTINCT r) AS rels, start
        RETURN start, nodes, rels
        """,
        id=node_id,
        snapshot=snapshot_id,
        max_hops=max_hops
    ).data()

    if not res:
        return {
            "snapshot": snapshot_id,
            "start": node_id,
            "nodes": [],
            "edges": []
        }

    row = res[0]
    start_node = row["start"]
    nodes_list = row["nodes"]
    rels_list = row["rels"]

    nodes_out = []
    seen_ids = set()
    for n in [start_node] + nodes_list:
        nid = n["id"]
        if nid in seen_ids:
            continue
        seen_ids.add(nid)
        nodes_out.append({
            "id": nid,
            "type": n.get("type"),
            "className": n.get("className"),
            "methodName": n.get("methodName"),
            "dto": n.get("dto"),
        })

    edges_out = []
    for rel_seq in rels_list:
        for r in rel_seq:
            edges_out.append({
                "type": type(r).__name__,
                "start": r.start_node["id"],
                "end": r.end_node["id"],
            })

    edge_set = {(e["type"], e["start"], e["end"]) for e in edges_out}
    edges_out = [{"type": t, "start": s, "end": e} for (t, s, e) in edge_set]

    return {
        "snapshot": snapshot_id,
        "start": node_id,
        "nodes": nodes_out,
        "edges": edges_out,
    }

# ============================================================
# FULL DAILY PIPELINE
# ============================================================

def run_daily_pipeline() -> None:
    logger.info("=== DAILY PIPELINE STARTED ===")
    last_run_status["last_run_time"] = datetime.utcnow().isoformat()
    last_run_status["last_run_success"] = None
    last_run_status["last_run_error"] = None

    snapshot_id = get_snapshot_id()
    try:
        step1_create_databases()
        step2_run_queries_and_decode()
        step3_run_dto_extraction()
        G = step4_build_graph_and_payload()
        step5_save_to_neo4j(G, snapshot_id)

        last_run_status["last_run_success"] = True
        last_run_status["snapshot_id"] = snapshot_id
        logger.info("=== DAILY PIPELINE COMPLETED SUCCESSFULLY (snapshot=%s) ===", snapshot_id)
    except Exception as e:
        last_run_status["last_run_success"] = False
        last_run_status["last_run_error"] = str(e)
        logger.exception("DAILY PIPELINE FAILED: %s", e)

# ============================================================
# FASTAPI + SCHEDULER + NEO4J LIFECYCLE & APIS
# ============================================================

app = FastAPI(title="FR Impact Prestep Microservice (Config-Driven)")

scheduler = BackgroundScheduler()


@app.on_event("startup")
def startup_event():
    global neo4j_graph
    logger.info("Starting scheduler...")

    if NEO4J_ENABLED:
        try:
            neo4j_graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            neo4j_graph.run("RETURN 1")
            logger.info("Connected to Neo4j (py2neo) at %s", NEO4J_URI)
        except Exception as e:
            neo4j_graph = None
            logger.exception("Failed to connect to Neo4j via py2neo: %s", e)

    scheduler.start()
    scheduler.add_job(run_daily_pipeline, CronTrigger(hour=SCHED_HOUR, minute=SCHED_MINUTE))
    logger.info("Scheduler job registered: daily at %02d:%02d", SCHED_HOUR, SCHED_MINUTE)


@app.on_event("shutdown")
def shutdown_event():
    global neo4j_graph
    logger.info("Shutting down scheduler...")
    scheduler.shutdown()
    if neo4j_graph is not None:
        logger.info("Clearing Neo4j graph reference...")
        neo4j_graph = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "neo4j_enabled": NEO4J_ENABLED,
        "neo4j_connected": neo4j_graph is not None if NEO4J_ENABLED else False
    }


@app.post("/run-now")
def run_now():
    run_daily_pipeline()
    return {
        "message": "Pipeline executed",
        "last_run": last_run_status
    }


@app.get("/last-run")
def last_run():
    return last_run_status


@app.get("/snapshot/{snapshot_id}/payload")
def api_get_snapshot_payload(snapshot_id: str):
    try:
        payload = get_snapshot_payload(snapshot_id)
        return payload
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Snapshot '{snapshot_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/snapshot/{snapshot_id}/graph")
def api_get_snapshot_graph(snapshot_id: str):
    try:
        graph_data = get_snapshot_graph(snapshot_id)
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/snapshot/diff")
def api_diff_snapshots(from_snapshot: str, to_snapshot: str):
    try:
        diff = diff_snapshots(from_snapshot, to_snapshot)
        return diff
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/snapshot/{snapshot_id}/impact/{node_id}")
def api_impact_for_node(snapshot_id: str, node_id: str, max_hops: int = 4):
    try:
        res = impact_for_node(snapshot_id, node_id, max_hops=max_hops)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel

class OllamaRequest(BaseModel):
    prompt: str
    model: str | None = None   # overwrite if needed


@app.post("/ai/ollama")
async def ai_ollama(request: Request):
    """
    Accepts plain text body and returns plain text AI response.
    """
    if not OLLAMA_ENABLED:
        return {"error": "Ollama is disabled in config.yaml"}

    prompt = await request.body()
    prompt = prompt.decode("utf-8").strip()

    result = call_ollama_local(prompt, OLLAMA_MODEL, OLLAMA_URL)

    return {"model": OLLAMA_MODEL, "response": result}