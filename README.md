# PyCitiCompass
## 📘 Overview
PyCitiCompass Service is an automated **CodeQL + Impact Analysis Microservice** that:
- Scans multiple Java microservices using **CodeQL**
- Extracts controllers, methods, call relationships, DTOs & fields
- Builds impact graphs using NetworkX
- Stores versioned snapshots into **Neo4j (py2neo)**
- Includes an **AI reasoning engine** using **Ollama + GLM-4.6**
- Provides endpoints for snapshots, graph analysis, diffs, and AI reasoning
- Runs nightly via **APScheduler at 00:00 AM**
Demo: https://drive.google.com/file/d/1WRqV0LVgbGJZOpASpK9KudCNrzZFyqlb/view?usp=drive_link 
---

## 🚀 Features
✔ Fully config-driven (`config.yaml`)  
✔ Zero hardcoded paths  
✔ CodeQL → DTO → Graph → Neo4j workflow  
✔ Swagger UI for easy API testing  
✔ Docker + docker-compose for deployment  
✔ GitHub Actions CI/CD  
✔ Supports plain-text AI input  
✔ Handles multiple microservices (payment/account/notification)

---

## 📁 Project Structure

```
PyCodeCompassService/
│── app.py
│── ollamacall.py
│── config.yaml
│── Dockerfile
│── docker-compose.yml
│── postman_collection.json
│── cypher_queries.txt
│── bloom_rules.txt
│── README.md
└── .github/workflows/ci-cd.yml
```

---
🧩 High-Level Architecture
```pgsql
           +--------------------------------------+
           |   REACT UI  (Functional Requirement) |
           +-------------+------------------------+
                         |
                         v
                         |
                         v
              +----------+----------+
              |   PyCodeCompass     |--------------+------------------------|
              |   Microservice      |                                       |
              +----------+----------+                                       |
                         |                                                  |
              +----------+----------+                                       |
              |   PyCodeCompass     |                                       |
              |      Schedular      |                                       |
              +----------+----------+                                       |--------+----------|
                         |                                                                      |
                  +------------+                                                                |
            ----- +  Repo's    +-------------------                                             |
            |     +------------+                  |                                             |
            |        Graph Creation               |                                             |
            v                                     v                                             v
     +------+-----+        +------+-----+   +-------+------+                +----------+--------------+------------------+
     | CodeQL DB  |        | CodeQL DB  |   |  DTO Scanner |                |     AI Impact Report Generation            |
     | Generation |----->  | Extration  |   | (Java Source)|   |----------->|   (FR + GLM-4.6 + Ollama + Impact Graph)   |
     +------+-----+        +-------+----+  +------+-------+    |            +----------+--------------+------------------+
                                   |              |            |                               |
                                   +--------------+            |                    +----------+---------+
                                           |                   |                    |  Impact Report     |
                                           v                   |                    +----------+---------+
                                    +------+-------+           |                               |
                                    | Repo   Graph |           |                               |
                                    |  (NetworkX)  |--------+-----                             |
                                    +---+----------+             |                             |
                                                                 |                             |                               
                                                                 |                             |
                                                                 |                             |
                                                       +---------+---------+                   |
                                                       |   Neo4j Storage   |                   |
                                                       |  Versioned Graphs |<-------------------
                                                       +-------------------+
```
## 🤖 AI Reasoning (GLM-4.6 + Ollama)

### Install Ollama
Download → https://ollama.com/download

### Pull the model
```bash
ollama pull glm-4.6:cloud
```

### Start Ollama server
```bash
ollama serve
```

### API URL
```
http://localhost:11434/api/generate
```

---

## 🚀 AI Endpoint — Plain Text Input

### Endpoint
```
POST /ai/ollama
```

### Example Request
```
Explain payment flow
```

### Example Response
```json
{
  "model": "glm-4.6:cloud",
  "response": "The payment workflow begins..."
}
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
work_dir: "C:/CodeIntelligence/work"
codeql_bin: "codeql"
query_dir: "C:/CodeIntelligence/DB"

services:
  - name: "payment"
    project_root: "C:/Users/chris/IdeaProjects/payment-service"
    db_name: "payment-service_db"
    controller_bqrs: "result_controller.bqrs"
    methods_bqrs: "result_methods.bqrs"
    calls_bqrs: "result_calls.bqrs"
    controller_json: "result_controller.json"
    methods_json: "result_methods.json"
    calls_json: "result_calls.json"
  - name: "account"
    project_root: "C:/Users/chris/IdeaProjects/account-service"
    db_name: "account-service_db"
    controller_bqrs: "result_account_controller.bqrs"
    methods_bqrs: "result_account_methods.bqrs"
    calls_bqrs: "result_account_calls.bqrs"
    controller_json: "result_account_controller.json"
    methods_json: "result_account_methods.json"
    calls_json: "result_account_calls.json"
  - name: "notification"
    project_root: "C:/Users/chris/IdeaProjects/notification-service"
    db_name: "notification-service_db"
    controller_bqrs: "result_notification_controller.bqrs"
    methods_bqrs: "result_notification_methods.bqrs"
    calls_bqrs: "result_notification_calls.bqrs"
    controller_json: "result_notification_controller.json"
    methods_json: "result_notification_methods.json"
    calls_json: "result_notification_calls.json"

src_folders:
  - "C:/Users/chris/IdeaProjects/payment-service/src/main/java"
  - "C:/Users/chris/IdeaProjects/account-service/src/main/java"
  - "C:/Users/chris/IdeaProjects/notification-service/src/main/java"

neo4j:
  enabled: true
  uri: "bolt://neo4j:7687"
  user: "neo4j"
  password: "password"

scheduler:
  hour: 0
  minute: 0

ollama:
  enabled: true
  local_mode: true
  url: "http://localhost:11434/api/generate"
  model: "glm-4.6:cloud"
```

---

## 🧠 Pipeline Stages

### **1️⃣ CodeQL DB Creation**
Builds databases for each microservice.

### **2️⃣ CodeQL Query Execution**
Extracts controllers, methods, and calls → JSON.

### **3️⃣ DTO Extraction**
Parses Java source directories for DTOs & fields.

### **4️⃣ Graph Building (NetworkX)**
Creates a directed impact graph.

### **5️⃣ Impact Artifacts**
Produces:
- `impact_nodes.csv`  
- `impact_edges.csv`  
- `impact_payload.json`

### **6️⃣ Neo4j Snapshot Storage**
Stores versioned graph snapshots.

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health |
| POST | `/run-now` | Run pipeline immediately |
| GET | `/last-run` | Last pipeline status |
| GET | `/snapshot/{id}/payload` | Impact payload |
| GET | `/snapshot/{id}/graph` | Graph nodes & edges |
| GET | `/snapshot/diff` | Compare snapshots |
| POST | `/ai/ollama` | **AI reasoning (plain text)** |

---

## 🐳 Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PCC_CONFIG_PATH=/app/config.yaml
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🐳 docker-compose.yml

```yaml
version: "3.9"

services:
  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/password
    ports:
      - "7474:7474"
      - "7687:7687"

  pycodecompass:
    build: .
    depends_on:
      - neo4j
    ports:
      - "8000:8000"
```

---

## 🔧 GitHub Actions CI/CD

```yaml
name: CI-CD

on:
  push:
    branches: ["main"]

jobs:
  build-test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: python -m compileall app.py
```

---

## 🧪 Neo4j Cypher Queries

```cypher
MATCH (s:ImpactSnapshot) RETURN s ORDER BY s.snapshot DESC;

MATCH (n:Node {snapshot:"<snapshot>"})-[r]->(m)
RETURN n, r, m;

MATCH p=(m:Node {id:"method:PaymentService.processPayment", snapshot:"<snapshot>"})
      -[:CALLS*1..6]->(x)
RETURN p;
```

---

## 🌳 Bloom Rules

```
SHOW Node WHERE type="Controller"
SHOW PATH FROM Node WHERE id="method:PaymentService.processPayment"
```

---

## 🎯 Summary
✔ Automated CodeQL → Graph → Neo4j pipeline  
✔ AI reasoning using GLM-4.6 + Ollama  
✔ Nightly scheduled ingestion  
✔ Full REST API suite  
✔ Deployment-ready (Docker, CI/CD)  

