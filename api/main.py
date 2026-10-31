"""FastAPI Application and Interactive Web Visualizer for Smart Factory Digital Twin."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from config import load_config
from database.database import DatabaseManager
from digital_twin.twin import DigitalTwin, TwinStatusIndicator
from simulator.factory_simulator import FactorySimulator
from mqtt.publisher import FactoryMqttPublisher
from mqtt.subscriber import FactoryMqttSubscriber
from ai.inference import AnomalyDetector
from analytics.health import HealthEvaluator
from analytics.risk import OperationalRiskEngine
from analytics.maintenance import PredictiveMaintenanceEngine
from analytics.oee import OEEEngine
from alerts.alert_manager import AlertManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DigitalTwinAPI")

# Load configuration
config = load_config()

# Core Services
twin = DigitalTwin(config=config)
db = DatabaseManager(config=config)
ai_detector = AnomalyDetector(config=config)
health_evaluator = HealthEvaluator(config=config)
risk_engine = OperationalRiskEngine(config=config)
maint_engine = PredictiveMaintenanceEngine()
oee_engine = OEEEngine()
alert_manager = AlertManager(config=config)

# Background MQTT Subscriber & Publisher
subscriber = FactoryMqttSubscriber(config=config)
publisher = FactoryMqttPublisher(config=config)

# Simulation state
is_simulator_active: bool = True
simulator_task: Optional[asyncio.Task] = None

# Active WebSocket clients
connected_clients: List[WebSocket] = []


def hydrate_twin_from_db():
    """Load latest machine state, health, risk, and production from SQLite into the in-memory Digital Twin."""
    try:
        # 1. Machines
        for m_id in list(twin.machines.keys()):
            telemetry_list = db.get_latest_telemetry(machine_id=m_id, limit=1)
            health_list = db.get_latest_machine_health(machine_id=m_id, limit=1)

            telemetry = telemetry_list[0] if telemetry_list else None
            health_record = health_list[0] if health_list else None

            if telemetry:
                health_eval = {
                    "health_score": health_record.get("health_score", 100.0) if health_record else 100.0,
                    "condition": health_record.get("condition", "EXCELLENT") if health_record else "EXCELLENT",
                }
                risk_eval = {
                    "risk_score": health_record.get("risk_score", 0.0) if health_record else 0.0,
                    "risk_level": health_record.get("risk_level", "LOW") if health_record else "LOW",
                    "primary_driver": health_record.get("primary_risk_driver", "None") if health_record else "None",
                }
                ai_pred = {
                    "anomaly_score": health_record.get("anomaly_score", 0.0) if health_record else 0.0,
                }
                twin.update_machine(
                    machine_id=m_id,
                    telemetry=telemetry,
                    ai_pred=ai_pred,
                    health_eval=health_eval,
                    risk_eval=risk_eval,
                )

        # 2. Production
        latest_prod = db.get_latest_production()
        if latest_prod:
            twin.update_production_summary(latest_prod)

        logger.info("Digital Twin successfully hydrated from SQLite database.")
    except Exception as e:
        logger.warning("Could not hydrate Digital Twin from SQLite: %s", e)


def handle_incoming_mqtt_telemetry(topic: str, data: Dict[str, Any]):
    """Master telemetry processor executing AI, Analytics, Alerts, Twin, and DB storage."""
    try:
        if topic == "factory/production/telemetry":
            db.insert_production(data)
            twin.update_production_summary(data)

        elif "/telemetry" in topic and "machine_id" in data:
            # 1. Persist Raw Telemetry
            db.insert_telemetry(data)

            # 2. PyTorch AI Anomaly Inference
            ai_pred = ai_detector.predict_and_store(data, db)

            # 3. Multi-Factor Health Evaluation
            health_eval = health_evaluator.evaluate_health(
                data, ai_anomaly_score=ai_pred["anomaly_score"]
            )

            # 4. Multi-Dimensional Operational Risk Matrix
            risk_eval = risk_engine.evaluate_risk(
                data, health_eval, ai_anomaly_score=ai_pred["anomaly_score"]
            )

            # 5. Persist Machine Health & Risk
            db.insert_machine_health(
                health_data=health_eval,
                risk_data=risk_eval,
                anomaly_score=ai_pred["anomaly_score"],
                timestamp=data.get("timestamp"),
            )

            # 6. Predictive Maintenance Diagnosis
            maint_diag = maint_engine.diagnose_and_recommend(data, health_eval, risk_eval)
            maint_engine.log_recommendation_to_db(maint_diag, db)

            # 7. OEE Calculation
            oee_metrics = oee_engine.calculate_machine_oee(data)

            # 8. Alert Generation & Cooldown Filter
            alert_manager.evaluate_and_dispatch(
                telemetry=data,
                ai_pred=ai_pred,
                health_eval=health_eval,
                risk_eval=risk_eval,
                maint_diag=maint_diag,
                db=db,
            )

            # 9. Digital Twin State Model Synchronization
            twin.update_machine(
                machine_id=data["machine_id"],
                telemetry=data,
                ai_pred=ai_pred,
                health_eval=health_eval,
                risk_eval=risk_eval,
                maint_diag=maint_diag,
                oee_metrics=oee_metrics,
            )

    except Exception as e:
        logger.error("Error processing MQTT telemetry in API worker: %s", e)


# Wire subscriber handler
subscriber.register_handler(handle_incoming_mqtt_telemetry)


async def live_simulation_worker():
    """Background async worker that generates live telemetry packets every second."""
    global is_simulator_active
    logger.info("Starting background continuous factory simulation publisher...")
    pub_ok = publisher.connect(timeout=3.0)
    if not pub_ok:
        logger.warning("Live publisher could not connect immediately to %s:%d.", publisher.broker_host, publisher.broker_port)

    while True:
        try:
            if is_simulator_active and publisher.is_connected:
                publisher.publish_step(dt=1.0)
            elif is_simulator_active and not publisher.is_connected:
                publisher.connect(timeout=1.0)
        except Exception as e:
            logger.error("Error in live simulation publisher tick: %s", e)
        await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan managing startup hydration, background MQTT subscriber, and live simulator."""
    global simulator_task
    # 1. Hydrate state from SQLite
    hydrate_twin_from_db()

    # 2. Start background MQTT subscriber (HiveMQ Cloud or Local broker)
    try:
        connected = subscriber.connect(timeout=2.0)
        if connected:
            logger.info("FastAPI background subscriber connected to MQTT Broker (%s:%d).", subscriber.broker_host, subscriber.broker_port)
        else:
            logger.warning("FastAPI background subscriber could not connect immediately to %s:%d. Will retry in background.", subscriber.broker_host, subscriber.broker_port)
    except Exception as e:
        logger.warning("Error starting background MQTT subscriber: %s", e)

    # 3. Start live background publisher loop
   # simulator_task = asyncio.create_task(live_simulation_worker())
    # 3. Start live background publisher loop
    simulator_task = asyncio.create_task(live_simulation_worker())
    logger.info("Live factory simulation publisher task started.")

    yield

    # Shutdown
    if simulator_task:
        simulator_task.cancel()
    try:
        publisher.disconnect()
        subscriber.disconnect()
    except Exception as e:
        logger.error("Error disconnecting MQTT publisher/subscriber: %s", e)


app = FastAPI(
    title="AI-Driven Digital Twin API",
    description="Real-time virtual representation, HiveMQ Cloud telemetry synchronization, and analytics for Smart Factory Operations",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local web clients and dashboard visualizers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """API health status endpoint."""
    return {
        "status": "online",
        "service": "Smart Factory Digital Twin",
        "mqtt_connected": subscriber.is_connected,
        "mqtt_broker": f"{subscriber.broker_host}:{subscriber.broker_port}",
        "mqtt_tls": subscriber.use_tls,
        "simulator_active": is_simulator_active,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/mqtt/status")
async def get_mqtt_status():
    """Get live MQTT Broker connection status and telemetry metrics."""
    return {
        "broker_host": subscriber.broker_host,
        "broker_port": subscriber.broker_port,
        "is_connected": subscriber.is_connected,
        "use_tls": subscriber.use_tls,
        "auth_enabled": bool(subscriber.username),
        "total_messages_received": subscriber.total_received,
        "valid_messages": subscriber.valid_messages,
        "topics": subscriber.topics,
        "simulator_active": is_simulator_active,
    }


@app.get("/api/twin/state")
async def get_digital_twin_state():
    """Retrieve full Digital Twin virtual state snapshot with broker metadata."""
    snapshot = twin.get_snapshot()
    snapshot["mqtt_status"] = {
        "broker": f"{subscriber.broker_host}:{subscriber.broker_port}",
        "is_connected": subscriber.is_connected,
        "tls": subscriber.use_tls,
        "total_received": subscriber.total_received,
        "simulator_active": is_simulator_active,
    }
    return snapshot


@app.post("/api/simulator/toggle")
async def toggle_simulator():
    """Toggle continuous simulation on or off."""
    global is_simulator_active
    is_simulator_active = not is_simulator_active
    return {"simulator_active": is_simulator_active}


@app.post("/api/simulator/degrade/{machine_id}")
async def inject_degradation_api(machine_id: str, rate: float = 0.35):
    """Inject gradual mechanical degradation into a machine simulator."""
    m_id = machine_id.upper()
    publisher.simulator.inject_degradation(m_id, rate=rate)
    return {"status": "success", "action": "degradation_injected", "machine_id": m_id, "rate": rate}


@app.post("/api/simulator/fault/{machine_id}")
async def inject_fault_api(machine_id: str):
    """Inject critical breakdown failure into a machine simulator."""
    m_id = machine_id.upper()
    publisher.simulator.inject_failure(m_id)
    return {"status": "success", "action": "failure_injected", "machine_id": m_id}


@app.post("/api/simulator/repair/{machine_id}")
async def repair_machine_api(machine_id: str):
    """Trigger automated maintenance recovery and repair on a machine."""
    m_id = machine_id.upper()
    publisher.simulator.repair_machine(m_id)
    return {"status": "success", "action": "machine_repaired", "machine_id": m_id}


@app.get("/api/twin/machines/{machine_id}")
async def get_machine_twin(machine_id: str):
    """Retrieve digital twin virtual state for a specific machine."""
    m_id = machine_id.upper()
    if m_id not in twin.machines:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found.")
    return twin.machines[m_id].to_dict()


@app.get("/api/telemetry/latest")
async def get_latest_telemetry(machine_id: Optional[str] = None, limit: int = 10):
    """Query recent telemetry from SQLite."""
    return db.get_latest_telemetry(machine_id=machine_id, limit=limit)


@app.get("/api/production/latest")
async def get_latest_production():
    """Query the latest factory production metrics from SQLite."""
    prod = db.get_latest_production()
    return prod if prod else {}


@app.get("/api/alerts")
async def get_alerts(
    machine_id: Optional[str] = None,
    severity: Optional[str] = None,
    unacknowledged_only: bool = False,
    limit: int = 50,
):
    """Retrieve active and historical factory alerts from SQLite."""
    return db.get_alerts(
        machine_id=machine_id,
        severity=severity,
        unacknowledged_only=unacknowledged_only,
        limit=limit,
    )


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    """Acknowledge an industrial alert."""
    success = db.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert with ID {alert_id} not found.")
    return {"status": "success", "alert_id": alert_id, "acknowledged": True}


@app.websocket("/ws/twin")
async def websocket_twin_stream(websocket: WebSocket):
    """WebSocket endpoint streaming real-time digital twin state updates."""
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info("New WebSocket client connected (Total: %d)", len(connected_clients))
    try:
        snapshot = twin.get_snapshot()
        snapshot["mqtt_status"] = {
            "broker": f"{subscriber.broker_host}:{subscriber.broker_port}",
            "is_connected": subscriber.is_connected,
            "tls": subscriber.use_tls,
            "total_received": subscriber.total_received,
            "simulator_active": is_simulator_active,
        }
        await websocket.send_text(json.dumps(snapshot))
        while True:
            await asyncio.sleep(1.0)
            snapshot = twin.get_snapshot()
            snapshot["mqtt_status"] = {
                "broker": f"{subscriber.broker_host}:{subscriber.broker_port}",
                "is_connected": subscriber.is_connected,
                "tls": subscriber.use_tls,
                "total_received": subscriber.total_received,
                "simulator_active": is_simulator_active,
            }
            await websocket.send_text(json.dumps(snapshot))
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info("WebSocket client disconnected (Remaining: %d)", len(connected_clients))
    except Exception as e:
        logger.error("WebSocket streaming error: %s", e)
        if websocket in connected_clients:
            connected_clients.remove(websocket)


@app.get("/", response_class=HTMLResponse)
async def get_digital_twin_dashboard():
    """Interactive modern HTML5/CSS3/JavaScript visualizer dashboard with HiveMQ, WebSocket & Live Charts."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Smart Factory Digital Twin</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-main: #0B0F19;
      --bg-card: #111827;
      --bg-card-hover: #1F2937;
      --border-color: #374151;
      --text-main: #F9FAFB;
      --text-muted: #9CA3AF;
      --color-green: #10B981;
      --color-yellow: #F59E0B;
      --color-red: #EF4444;
      --color-blue: #3B82F6;
      --color-purple: #8B5CF6;
      --color-cyan: #06B6D4;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-main);
      color: var(--text-main);
      font-family: 'Inter', sans-serif;
      padding: 24px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .header-title h1 { font-size: 24px; font-weight: 800; display: flex; align-items: center; gap: 10px; }
    .header-title p { color: var(--text-muted); font-size: 14px; margin-top: 4px; }
    
    .header-badges {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    
    .status-badge {
      padding: 8px 16px;
      border-radius: 9999px;
      font-weight: 700;
      font-size: 13px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .badge-normal { background: rgba(16, 185, 129, 0.15); color: var(--color-green); border: 1px solid var(--color-green); }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: var(--color-yellow); border: 1px solid var(--color-yellow); }
    .badge-critical { background: rgba(239, 68, 68, 0.15); color: var(--color-red); border: 1px solid var(--color-red); }
    .badge-info { background: rgba(59, 130, 246, 0.15); color: var(--color-blue); border: 1px solid var(--color-blue); }

    .pulse-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background-color: var(--color-green);
      display: inline-block;
      box-shadow: 0 0 0 rgba(16, 185, 129, 0.7);
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
      70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
      100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Control Bar */
    .control-bar {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 14px 20px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    .control-buttons {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .btn {
      padding: 8px 14px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.2s ease;
    }
    .btn-primary { background: var(--color-blue); color: white; }
    .btn-primary:hover { background: #2563EB; }
    .btn-warning { background: rgba(245, 158, 11, 0.2); color: var(--color-yellow); border-color: var(--color-yellow); }
    .btn-warning:hover { background: rgba(245, 158, 11, 0.4); }
    .btn-danger { background: rgba(239, 68, 68, 0.2); color: var(--color-red); border-color: var(--color-red); }
    .btn-danger:hover { background: rgba(239, 68, 68, 0.4); }
    .btn-success { background: rgba(16, 185, 129, 0.2); color: var(--color-green); border-color: var(--color-green); }
    .btn-success:hover { background: rgba(16, 185, 129, 0.4); }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .kpi-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 18px;
    }
    .kpi-label { font-size: 12px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 28px; font-weight: 800; margin-top: 6px; font-family: 'JetBrains Mono', monospace; }

    .section-title { font-size: 18px; font-weight: 700; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }

    .line-flow {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      overflow-x: auto;
      gap: 12px;
    }
    .flow-node {
      background: #1E293B;
      border: 1px solid var(--border-color);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      white-space: nowrap;
      text-align: center;
    }
    .flow-node.active-machine {
      border-color: var(--color-blue);
      background: rgba(59, 130, 246, 0.1);
    }
    .flow-arrow { color: var(--text-muted); font-weight: 800; }

    .machines-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }
    .machine-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 20px;
      transition: all 0.2s ease;
    }
    .machine-card:hover { border-color: var(--color-blue); }
    .machine-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .machine-title h3 { font-size: 18px; font-weight: 700; }
    .machine-title span { font-size: 13px; color: var(--text-muted); }

    .health-bar-container { margin-bottom: 16px; }
    .health-bar-header { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
    .health-bar-track { height: 10px; background: #1E293B; border-radius: 5px; overflow: hidden; }
    .health-bar-fill { height: 100%; border-radius: 5px; transition: width 0.4s ease; }

    .sensor-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 16px;
    }
    .sensor-box {
      background: #1E293B;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.05);
    }
    .sensor-name { font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
    .sensor-val { font-size: 16px; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }

    .diagnostics-box {
      background: rgba(30, 41, 59, 0.7);
      border-left: 3px solid var(--color-blue);
      padding: 10px 12px;
      border-radius: 4px;
      font-size: 12px;
      color: #D1D5DB;
    }
    .diagnostics-box.crit { border-left-color: var(--color-red); }
    .diagnostics-box.warn { border-left-color: var(--color-yellow); }

    /* Live Charts Grid */
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }
    .chart-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 14px;
      padding: 20px;
    }
    .chart-header {
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 14px;
      color: var(--text-main);
      display: flex;
      justify-content: space-between;
    }

    /* Alerts Section */
    .alerts-section {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      margin-top: 24px;
    }
    .alerts-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 250px;
      overflow-y: auto;
      margin-top: 12px;
    }
    .alert-item {
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 13px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #1E293B;
      border-left: 4px solid var(--color-blue);
    }
    .alert-item.CRITICAL { border-left-color: var(--color-red); }
    .alert-item.WARNING { border-left-color: var(--color-yellow); }
  </style>
</head>
<body>

  <header>
    <div class="header-title">
      <h1>🏭 Smart Factory Digital Twin</h1>
      <p id="lastSync"><span class="pulse-dot"></span> Live Telemetry Synchronization Active</p>
    </div>
    <div class="header-badges">
      <div id="mqttBadge" class="status-badge badge-info">📡 MQTT: Connecting...</div>
      <div id="factoryStatusBadge" class="status-badge badge-normal">🟢 LINE NORMAL</div>
    </div>
  </header>

  <!-- Interactive Simulation Control Bar -->
  <div class="control-bar">
    <div style="font-size: 14px; font-weight: 600;">
      🎮 Live Simulation Controls:
      <span id="simStateBadge" style="color: var(--color-green); font-family: 'JetBrains Mono'; margin-left: 6px;">[STREAMING LIVE 1.0s]</span>
    </div>
    <div class="control-buttons">
      <button class="btn btn-primary" onclick="toggleSimulator()">⏸️ Toggle Stream</button>
      <button class="btn btn-warning" onclick="injectDegradation('M01')">⚡ Degrade M01</button>
      <button class="btn btn-danger" onclick="injectFault('M01')">💥 Break M01</button>
      <button class="btn btn-success" onclick="repairMachine('M01')">🔧 Repair M01</button>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Overall Factory Health</div>
      <div id="avgHealth" class="kpi-value" style="color: var(--color-green);">--%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total Units Produced</div>
      <div id="totalProd" class="kpi-value">0</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Good vs Defective</div>
      <div id="goodDefectRatio" class="kpi-value">0 / 0</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Line Defect Rate</div>
      <div id="defectRate" class="kpi-value">0.00%</div>
    </div>
  </div>

  <div class="section-title">📍 Production Line Virtual Routing</div>
  <div class="line-flow">
    <div class="flow-node">📥 Raw Infeed</div>
    <div class="flow-arrow">➔</div>
    <div class="flow-node">Conveyor 1</div>
    <div class="flow-arrow">➔</div>
    <div class="flow-node active-machine">M01: CNC Milling</div>
    <div class="flow-arrow">➔</div>
    <div class="flow-node active-machine">M02: Stamping Press</div>
    <div class="flow-arrow">➔</div>
    <div class="flow-node active-machine">M03: Robotic Welder</div>
    <div class="flow-arrow">➔</div>
    <div class="flow-node">Assembly Station</div>
    <div class="flow-arrow">➔</div>
    <div class="flow-node">Quality Inspection</div>
    <div class="flow-arrow">➔</div>
    <div class="flow-node">📦 Finished Output</div>
  </div>

  <div class="section-title">🤖 Real-Time Machine Digital Twins</div>
  <div id="machinesGrid" class="machines-grid">
    <!-- Machines dynamically populated by WebSocket / REST -->
  </div>

  <!-- Real-Time Waveform Charts -->
  <div class="section-title">📈 Real-Time Multi-Machine Telemetry Waveforms</div>
  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-header">
        <span>🌡️ Machine Temperature (°C)</span>
        <span style="font-size: 12px; color: var(--text-muted);">Real-time rolling window</span>
      </div>
      <canvas id="tempChart" height="180"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-header">
        <span>〰️ Vibration (mm/s) & AI Health Trend</span>
        <span style="font-size: 12px; color: var(--text-muted);">Real-time rolling window</span>
      </div>
      <canvas id="vibChart" height="180"></canvas>
    </div>
  </div>

  <div class="alerts-section">
    <div class="section-title">🚨 Recent Industrial Alerts</div>
    <div id="alertsList" class="alerts-list">
      <div style="color: var(--text-muted); font-size: 13px;">No active critical alerts. All machines nominal.</div>
    </div>
  </div>

  <script>
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${window.location.host}/ws/twin`;
    let socket;

    // Charts Configuration
    const maxDataPoints = 20;
    const chartLabels = [];
    const m01Temp = [], m02Temp = [], m03Temp = [];
    const m01Vib = [], m01Health = [];

    const tempCtx = document.getElementById('tempChart').getContext('2d');
    const tempChart = new Chart(tempCtx, {
      type: 'line',
      data: {
        labels: chartLabels,
        datasets: [
          { label: 'M01 (CNC)', data: m01Temp, borderColor: '#3B82F6', tension: 0.3, borderWidth: 2 },
          { label: 'M02 (Press)', data: m02Temp, borderColor: '#10B981', tension: 0.3, borderWidth: 2 },
          { label: 'M03 (Welder)', data: m03Temp, borderColor: '#F59E0B', tension: 0.3, borderWidth: 2 },
        ]
      },
      options: {
        responsive: true,
        animation: false,
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9CA3AF' } },
          y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9CA3AF' } }
        },
        plugins: { legend: { labels: { color: '#F9FAFB' } } }
      }
    });

    const vibCtx = document.getElementById('vibChart').getContext('2d');
    const vibChart = new Chart(vibCtx, {
      type: 'line',
      data: {
        labels: chartLabels,
        datasets: [
          { label: 'M01 Vibration (mm/s)', data: m01Vib, borderColor: '#EF4444', tension: 0.3, borderWidth: 2, yAxisID: 'y' },
          { label: 'M01 Health (%)', data: m01Health, borderColor: '#10B981', tension: 0.3, borderWidth: 2, yAxisID: 'y1' }
        ]
      },
      options: {
        responsive: true,
        animation: false,
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9CA3AF' } },
          y: { type: 'linear', position: 'left', grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#EF4444' } },
          y1: { type: 'linear', position: 'right', min: 0, max: 100, grid: { drawOnChartArea: false }, ticks: { color: '#10B981' } }
        },
        plugins: { legend: { labels: { color: '#F9FAFB' } } }
      }
    });

    async function toggleSimulator() {
      await fetch('/api/simulator/toggle', { method: 'POST' });
      fetchInitialState();
    }

    async function injectDegradation(mId) {
      await fetch(`/api/simulator/degrade/${mId}`, { method: 'POST' });
    }

    async function injectFault(mId) {
      await fetch(`/api/simulator/fault/${mId}`, { method: 'POST' });
    }

    async function repairMachine(mId) {
      await fetch(`/api/simulator/repair/${mId}`, { method: 'POST' });
    }

    async function fetchInitialState() {
      try {
        const res = await fetch('/api/twin/state');
        if (res.ok) {
          const data = await res.json();
          renderDashboard(data);
        }
      } catch (err) {
        console.warn('Initial state fetch error:', err);
      }
      fetchAlerts();
    }

    async function fetchAlerts() {
      try {
        const res = await fetch('/api/alerts?limit=5');
        if (res.ok) {
          const alerts = await res.json();
          renderAlerts(alerts);
        }
      } catch (err) {
        console.warn('Alerts fetch error:', err);
      }
    }

    function renderAlerts(alerts) {
      const container = document.getElementById('alertsList');
      if (!alerts || alerts.length === 0) {
        container.innerHTML = '<div style="color: var(--text-muted); font-size: 13px;">No active critical alerts. All machines nominal.</div>';
        return;
      }
      let html = '';
      for (const a of alerts) {
        html += `
          <div class="alert-item ${a.severity}">
            <div>
              <strong>[${a.severity}] ${a.machine_id}:</strong> ${a.message}
            </div>
            <span style="color: var(--text-muted); font-size: 11px;">${a.timestamp.substring(11, 19)}</span>
          </div>
        `;
      }
      container.innerHTML = html;
    }

    function connectWebSocket() {
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        document.getElementById('lastSync').innerHTML = '<span class="pulse-dot"></span> 🟢 Real-time Stream: Active';
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        renderDashboard(data);
      };

      socket.onclose = () => {
        document.getElementById('lastSync').innerText = '⚠️ Real-time Stream: Reconnecting...';
        setTimeout(connectWebSocket, 2000);
      };
    }

    function renderDashboard(data) {
      // 1. MQTT Broker Status Badge
      if (data.mqtt_status) {
        const badge = document.getElementById('mqttBadge');
        const broker = data.mqtt_status.broker || 'MQTT';
        const isConn = data.mqtt_status.is_connected;
        const isTls = data.mqtt_status.tls;
        const count = data.mqtt_status.total_received || 0;
        badge.innerText = `📡 ${broker} (${isTls ? 'TLS' : 'TCP'}) : ${isConn ? 'ONLINE' : 'CONNECTING'} [${count} pkts]`;
        badge.className = 'status-badge ' + (isConn ? 'badge-normal' : 'badge-warning');

        const simBadge = document.getElementById('simStateBadge');
        if (simBadge) {
          simBadge.innerText = data.mqtt_status.simulator_active ? '[STREAMING LIVE 1.0s]' : '[PAUSED]';
          simBadge.style.color = data.mqtt_status.simulator_active ? 'var(--color-green)' : 'var(--color-yellow)';
        }
      }

      // 2. Factory Status Badge
      const statusBadge = document.getElementById('factoryStatusBadge');
      statusBadge.innerText = data.factory_status || '🟢 NORMAL';
      statusBadge.className = 'status-badge ' + (
        (data.factory_status || '').includes('CRITICAL') ? 'badge-critical' :
        (data.factory_status || '').includes('WARNING') ? 'badge-warning' : 'badge-normal'
      );

      // 3. Production KPIs
      const summary = data.production_summary || {};
      document.getElementById('totalProd').innerText = summary.total_production || 0;
      document.getElementById('goodDefectRatio').innerText = `${summary.total_good || 0} / ${summary.total_defects || 0}`;
      document.getElementById('defectRate').innerText = `${((summary.defect_rate || 0) * 100).toFixed(2)}%`;

      // 4. Update Charts
      const timeStr = new Date().toLocaleTimeString();
      chartLabels.push(timeStr);
      if (chartLabels.length > maxDataPoints) chartLabels.shift();

      const m = data.machines || {};
      if (m['M01']?.telemetry) {
        m01Temp.push(m['M01'].telemetry.temperature);
        m01Vib.push(m['M01'].telemetry.vibration);
        m01Health.push(m['M01'].health_score);
        if (m01Temp.length > maxDataPoints) m01Temp.shift();
        if (m01Vib.length > maxDataPoints) m01Vib.shift();
        if (m01Health.length > maxDataPoints) m01Health.shift();
      }
      if (m['M02']?.telemetry) {
        m02Temp.push(m['M02'].telemetry.temperature);
        if (m02Temp.length > maxDataPoints) m02Temp.shift();
      }
      if (m['M03']?.telemetry) {
        m03Temp.push(m['M03'].telemetry.temperature);
        if (m03Temp.length > maxDataPoints) m03Temp.shift();
      }
      tempChart.update();
      vibChart.update();

      // 5. Machine Twin Cards
      const container = document.getElementById('machinesGrid');
      let totalHealth = 0;
      let machineCount = 0;
      let html = '';

      for (const [mId, mObj] of Object.entries(m)) {
        totalHealth += mObj.health_score;
        machineCount++;

        const healthColor = mObj.health_score > 75 ? 'var(--color-green)' : (mObj.health_score > 50 ? 'var(--color-yellow)' : 'var(--color-red)');
        const diagClass = mObj.risk_level === 'CRITICAL' ? 'crit' : (mObj.risk_level === 'HIGH' ? 'warn' : '');

        html += `
          <div class="machine-card">
            <div class="machine-header">
              <div class="machine-title">
                <h3>${mObj.machine_id}: ${mObj.name}</h3>
                <span>State: <strong>${mObj.state}</strong></span>
              </div>
              <div class="status-badge ${mObj.risk_level === 'CRITICAL' ? 'badge-critical' : (mObj.risk_level === 'HIGH' ? 'badge-warning' : 'badge-normal')}">
                ${mObj.status_indicator}
              </div>
            </div>

            <div class="health-bar-container">
              <div class="health-bar-header">
                <span>Health: ${mObj.condition}</span>
                <span style="color: ${healthColor};">${(mObj.health_score || 0).toFixed(1)}%</span>
              </div>
              <div class="health-bar-track">
                <div class="health-bar-fill" style="width: ${mObj.health_score || 0}%; background: ${healthColor};"></div>
              </div>
            </div>

            <div class="sensor-grid">
              <div class="sensor-box">
                <div class="sensor-name">Temperature</div>
                <div class="sensor-val">${(mObj.telemetry?.temperature || 0).toFixed(1)} °C</div>
              </div>
              <div class="sensor-box">
                <div class="sensor-name">Vibration</div>
                <div class="sensor-val">${(mObj.telemetry?.vibration || 0).toFixed(2)} mm/s</div>
              </div>
              <div class="sensor-box">
                <div class="sensor-name">Motor Current</div>
                <div class="sensor-val">${(mObj.telemetry?.current || 0).toFixed(1)} A</div>
              </div>
              <div class="sensor-box">
                <div class="sensor-name">Spindle Speed</div>
                <div class="sensor-val">${(mObj.telemetry?.rpm || 0).toFixed(0)} RPM</div>
              </div>
              <div class="sensor-box">
                <div class="sensor-name">AI Anomaly Score</div>
                <div class="sensor-val">${(mObj.anomaly_score || 0).toFixed(3)}</div>
              </div>
              <div class="sensor-box">
                <div class="sensor-name">Operational Risk</div>
                <div class="sensor-val">${(mObj.risk_score || 0).toFixed(1)} (${mObj.risk_level})</div>
              </div>
            </div>

            <div class="diagnostics-box ${diagClass}">
              <strong>🔍 Diagnostics:</strong> ${mObj.maintenance_recommendation || 'Operating normally within tolerances'}
            </div>
          </div>
        `;
      }

      container.innerHTML = html;
      if (machineCount > 0) {
        const avg = (totalHealth / machineCount).toFixed(1);
        const avgEl = document.getElementById('avgHealth');
        avgEl.innerText = `${avg}%`;
        avgEl.style.color = avg > 75 ? 'var(--color-green)' : (avg > 50 ? 'var(--color-yellow)' : 'var(--color-red)');
      }
    }

    // Initial load + WebSocket connection + polling fallback
    fetchInitialState();
    connectWebSocket();
    setInterval(fetchAlerts, 4000);
    setInterval(fetchInitialState, 2000); // Polling fallback ensures continuous updates under any network condition
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
