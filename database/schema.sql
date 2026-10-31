-- ====================================================================
-- AI-Driven Digital Twin for Smart Factory Operations: SQLite Schema
-- Target Database: data/smart_factory.db
-- ====================================================================

PRAGMA foreign_keys = ON;

-- 1. Machines Metadata Table
CREATE TABLE IF NOT EXISTS machines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    nominal_temperature REAL NOT NULL,
    nominal_vibration REAL NOT NULL,
    nominal_current REAL NOT NULL,
    nominal_rpm REAL NOT NULL,
    nominal_pressure REAL NOT NULL,
    nominal_power REAL NOT NULL,
    nominal_cycle_time REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 2. Real-Time Equipment Telemetry Time-Series Table
CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    machine_state TEXT NOT NULL,
    temperature REAL NOT NULL,
    vibration REAL NOT NULL,
    current REAL NOT NULL,
    voltage REAL NOT NULL,
    rpm REAL NOT NULL,
    pressure REAL NOT NULL,
    power REAL NOT NULL,
    production_count INTEGER DEFAULT 0,
    good_products INTEGER DEFAULT 0,
    defect_count INTEGER DEFAULT 0,
    defect_rate REAL DEFAULT 0.0,
    cycle_time REAL DEFAULT 0.0,
    downtime_seconds REAL DEFAULT 0.0,
    degradation_level REAL DEFAULT 0.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE
);

-- 3. Production Line Aggregate Telemetry Table
CREATE TABLE IF NOT EXISTS production (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    factory_name TEXT NOT NULL,
    factory_status TEXT NOT NULL,
    total_production INTEGER NOT NULL,
    total_good INTEGER NOT NULL,
    total_defects INTEGER NOT NULL,
    defect_rate REAL NOT NULL,
    active_machines INTEGER NOT NULL,
    warning_machines INTEGER NOT NULL,
    failed_machines INTEGER NOT NULL,
    maintenance_machines INTEGER NOT NULL,
    idle_machines INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 4. Industrial Alerts and Events Table
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    machine_id TEXT,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,         -- 'INFO', 'WARNING', 'CRITICAL'
    message TEXT NOT NULL,
    acknowledged INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 5. Maintenance and Service History Table
CREATE TABLE IF NOT EXISTS maintenance_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    event_type TEXT NOT NULL,        -- 'SCHEDULED', 'BREAKDOWN', 'REPAIR', 'RECOVERY'
    description TEXT,
    downtime_minutes REAL DEFAULT 0.0,
    technician TEXT DEFAULT 'Automated System',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 6. AI Inference & Health Predictions Table (For Phase 7-8)
CREATE TABLE IF NOT EXISTS ai_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    anomaly_score REAL NOT NULL,
    health_score REAL NOT NULL,
    risk_level TEXT NOT NULL,        -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    is_anomaly INTEGER DEFAULT 0,
    recommendation TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE
);

-- 7. Machine Health & Operational Risk History Table
CREATE TABLE IF NOT EXISTS machine_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    health_score REAL NOT NULL,
    health_condition TEXT NOT NULL,       -- 'EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'CRITICAL'
    rule_health REAL NOT NULL,
    ai_health REAL NOT NULL,
    anomaly_score REAL NOT NULL,
    risk_score REAL NOT NULL,
    risk_level TEXT NOT NULL,              -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    primary_risk_driver TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE
);

-- Indices for rapid querying and real-time analytics
CREATE INDEX IF NOT EXISTS idx_telemetry_machine_time ON telemetry(machine_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry(timestamp);
CREATE INDEX IF NOT EXISTS idx_production_timestamp ON production(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_machine_time ON alerts(machine_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_machine_time ON ai_predictions(machine_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_machine_health_machine_time ON machine_health(machine_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_machine_health_timestamp ON machine_health(timestamp);

-- Views for time-series analytics and tabular visualizers
CREATE VIEW IF NOT EXISTS v_telemetry_with_ai AS
SELECT 
    t.id,
    t.timestamp,
    t.machine_id,
    t.machine_state,
    t.temperature,
    t.vibration,
    t.current,
    t.voltage,
    t.rpm,
    t.pressure,
    t.power,
    t.production_count,
    t.good_products,
    t.defect_count,
    t.defect_rate,
    t.cycle_time,
    t.downtime_seconds,
    t.degradation_level,
    COALESCE(p.health_score, 100.0) AS health_score,
    COALESCE(p.anomaly_score, 0.0) AS anomaly_score,
    COALESCE(p.risk_level, 'LOW') AS risk_level,
    COALESCE(p.is_anomaly, 0) AS is_anomaly,
    COALESCE(p.recommendation, 'Operating normally') AS recommendation
FROM telemetry t
LEFT JOIN ai_predictions p 
    ON t.machine_id = p.machine_id 
    AND t.timestamp = p.timestamp;

CREATE VIEW IF NOT EXISTS v_latest_machine_overview AS
SELECT 
    m.machine_id,
    m.name AS machine_name,
    m.type AS machine_type,
    v.machine_state,
    v.temperature,
    v.vibration,
    v.current,
    v.voltage,
    v.rpm,
    v.pressure,
    v.power,
    v.production_count,
    v.good_products,
    v.defect_count,
    v.defect_rate,
    v.cycle_time,
    v.downtime_seconds,
    v.health_score,
    v.anomaly_score,
    v.risk_level,
    v.is_anomaly,
    v.recommendation,
    v.timestamp
FROM machines m
LEFT JOIN (
    SELECT *
    FROM v_telemetry_with_ai
    WHERE id IN (SELECT MAX(id) FROM telemetry GROUP BY machine_id)
) v ON m.machine_id = v.machine_id;
