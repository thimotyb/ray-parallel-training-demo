# Ray Cluster Monitoring with Prometheus and Grafana

This guide explains how to set up and use the monitoring stack (Prometheus + Grafana) to get enhanced metrics visualization in the Ray dashboard.

## Table of Contents

- [Why Monitoring?](#why-monitoring)
- [Quick Setup](#quick-setup)
- [Architecture](#architecture)
- [Accessing the Tools](#accessing-the-tools)
- [Ray Dashboard Integration](#ray-dashboard-integration)
- [Available Metrics](#available-metrics)
- [Creating Custom Dashboards](#creating-custom-dashboards)
- [Troubleshooting](#troubleshooting)

## Why Monitoring?

Without Prometheus and Grafana, the Ray dashboard shows this message:

> "Set up Prometheus and Grafana for better Ray Dashboard experience. Time-series charts are hidden because either Prometheus or Grafana server is not detected."

Installing the monitoring stack enables:

- **Time-series charts** in Ray dashboard
- **Historical metrics** tracking over time
- **Resource utilization graphs** (CPU, memory, network)
- **Job performance metrics** (task duration, throughput)
- **Custom dashboards** in Grafana
- **Long-term metrics storage** (beyond Ray's in-memory storage)

## Quick Setup

### Install Monitoring Stack

```bash
./scripts/install-monitoring.sh
```

This single command will:
1. Deploy Prometheus for metrics collection
2. Deploy Grafana for visualization
3. Configure Prometheus to scrape Ray metrics
4. Configure Grafana with Prometheus datasource
5. Set up necessary RBAC permissions

**Installation time:** 2-3 minutes

### Verify Installation

```bash
# Check Prometheus
kubectl get pods -n ray-system | grep prometheus
# Expected: prometheus-xxx  1/1  Running

# Check Grafana
kubectl get pods -n ray-system | grep grafana
# Expected: grafana-xxx  1/1  Running

# Check services
kubectl get svc -n ray-system | grep -E "prometheus|grafana"
```

### Refresh Ray Dashboard

After installation completes:
1. Open the Ray dashboard: `http://<minikube-ip>:30265`
2. **Refresh the page** (F5 or Cmd+R)
3. Time-series charts should now appear

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Ray Cluster                          │
│                                                          │
│  ┌──────────────┐              ┌──────────────┐        │
│  │  Ray Head    │              │ Ray Workers  │        │
│  │              │              │              │        │
│  │ Port 8080    │              │ Port 8080    │        │
│  │ (metrics)    │              │ (metrics)    │        │
│  └──────┬───────┘              └──────┬───────┘        │
│         │                             │                 │
│         │        Scrapes metrics      │                 │
│         └─────────────┬───────────────┘                 │
│                       │                                 │
│                       ▼                                 │
│              ┌────────────────┐                         │
│              │  Prometheus    │                         │
│              │  Port 9090     │                         │
│              │  (storage)     │                         │
│              └────────┬───────┘                         │
│                       │                                 │
│                       │ Data source                     │
│                       ▼                                 │
│              ┌────────────────┐                         │
│              │   Grafana      │                         │
│              │   Port 3000    │◄────────────────────────┤
│              │   (NodePort    │  Queries metrics        │
│              │    30300)      │                         │
│              └────────────────┘                         │
│                       ▲                                 │
└───────────────────────┼─────────────────────────────────┘
                        │
                        │ Ray Dashboard queries
                        │ Prometheus & Grafana
                        │
                ┌───────┴────────┐
                │  Ray Dashboard │
                │  Port 8265     │
                │  (NodePort     │
                │   30265)       │
                └────────────────┘
```

### Components

1. **Ray Metrics Exporter** (port 8080)
   - Built into Ray head and worker nodes
   - Exports metrics in Prometheus format
   - Enabled via `metrics-export-port: '8080'`

2. **Prometheus** (port 9090)
   - Scrapes metrics from Ray nodes every 15 seconds
   - Stores time-series data
   - Provides query API (PromQL)

3. **Grafana** (port 3000, NodePort 30300)
   - Connects to Prometheus as datasource
   - Visualizes metrics with dashboards
   - Accessible via web browser

4. **Ray Dashboard** (port 8265, NodePort 30265)
   - Automatically detects Prometheus and Grafana
   - Embeds time-series charts from these services
   - Enhanced with historical metrics

## Accessing the Tools

### Ray Dashboard

**Primary access method:**
```bash
./scripts/open-dashboard.sh
# Or directly: http://<minikube-ip>:30265
```

After monitoring stack is installed, the dashboard will show:
- Real-time metrics (as before)
- **Time-series charts** (new)
- Historical data visualization
- No more "setup Prometheus" warning

### Grafana

**Access URL:**
```bash
minikube ip
# Open: http://<minikube-ip>:30300
```

**Default Credentials:**
- Username: `admin`
- Password: `admin`

**First login:**
1. You'll be prompted to change the password (can skip)
2. Grafana is pre-configured with Prometheus datasource
3. Ready to create dashboards

**Port forwarding alternative:**
```bash
kubectl port-forward -n ray-system svc/grafana 3000:3000
# Then access: http://localhost:3000
```

### Prometheus

**Direct access (for debugging):**
```bash
kubectl port-forward -n ray-system svc/prometheus 9090:9090
# Then access: http://localhost:9090
```

**Useful Prometheus URLs:**
- Query UI: `http://localhost:9090/graph`
- Targets: `http://localhost:9090/targets` (shows Ray nodes being scraped)
- Configuration: `http://localhost:9090/config`

## Ray Dashboard Integration

### How It Works

The Ray dashboard detects Prometheus and Grafana via environment variables:

```yaml
env:
  - name: RAY_GRAFANA_HOST
    value: "http://grafana:3000"
  - name: RAY_PROMETHEUS_HOST
    value: "http://prometheus:9090"
```

When these are set:
1. Dashboard queries Prometheus for historical metrics
2. Dashboard embeds Grafana charts for visualization
3. Time-series graphs appear automatically

### What Charts Appear

After setup, the Ray dashboard shows:

**Cluster Tab:**
- CPU utilization over time
- Memory usage over time
- Network I/O over time
- Disk I/O over time

**Jobs Tab:**
- Job execution timeline
- Tasks per second
- Task duration distribution

**Actors Tab:**
- Active actors over time
- Actor creation/destruction rate

**Metrics Tab:**
- Custom application metrics
- System metrics (if exported)

### Refresh to See Changes

**Important:** After installing monitoring stack, you **must refresh** the Ray dashboard page to see the new charts appear.

## Available Metrics

### Ray System Metrics

Prometheus scrapes these metrics from Ray:

**Resource Metrics:**
- `ray_node_cpu_utilization` - CPU usage per node
- `ray_node_mem_used` - Memory used per node
- `ray_node_mem_available` - Memory available per node
- `ray_node_disk_usage` - Disk usage per node
- `ray_node_network_sent` - Network bytes sent
- `ray_node_network_received` - Network bytes received

**Task Metrics:**
- `ray_tasks_total` - Total tasks executed
- `ray_tasks_pending` - Tasks waiting to execute
- `ray_tasks_running` - Currently executing tasks
- `ray_tasks_failed` - Failed tasks count

**Actor Metrics:**
- `ray_actors_total` - Total actors created
- `ray_actors_alive` - Currently alive actors
- `ray_actors_restarting` - Actors being restarted

**Object Store Metrics:**
- `ray_object_store_memory` - Object store memory usage
- `ray_object_store_available_memory` - Available object store memory

### Querying Metrics

**In Prometheus UI** (`http://localhost:9090/graph`):

```promql
# Average CPU usage across all Ray nodes
avg(ray_node_cpu_utilization)

# Memory usage by node
ray_node_mem_used{node_type="head"}
ray_node_mem_used{node_type="worker"}

# Task execution rate (tasks per second)
rate(ray_tasks_total[1m])
```

**In Grafana:**
- Use these same PromQL queries
- Create visualizations (graphs, gauges, tables)
- Build custom dashboards

## Creating Custom Dashboards

### In Grafana

1. **Login to Grafana**
   - URL: `http://<minikube-ip>:30300`
   - Credentials: admin / admin

2. **Create New Dashboard**
   - Click "+" → "Dashboard"
   - Click "Add visualization"

3. **Select Prometheus Datasource**
   - Should be pre-configured

4. **Add Metrics Query**
   - Enter PromQL query (e.g., `ray_node_cpu_utilization`)
   - Choose visualization type (Time series, Gauge, etc.)

5. **Customize**
   - Set title, units, colors
   - Add multiple panels
   - Set refresh interval

6. **Save Dashboard**
   - Name it (e.g., "Ray Training Metrics")
   - Can share JSON for others to import

### Example Dashboard Panels

**CPU Utilization Panel:**
```promql
# Query
ray_node_cpu_utilization

# Panel settings
- Type: Time series
- Unit: Percent (0-100)
- Legend: {{pod}}
```

**Memory Usage Panel:**
```promql
# Query
ray_node_mem_used / (1024*1024*1024)

# Panel settings
- Type: Time series
- Unit: GB
- Legend: {{pod}}
```

**Tasks Per Second:**
```promql
# Query
rate(ray_tasks_total[1m])

# Panel settings
- Type: Stat
- Unit: tasks/sec
```

**Training Job Duration:**
```promql
# Query (assuming you export this metric)
ray_job_duration_seconds

# Panel settings
- Type: Bar gauge
- Unit: seconds
```

### Export/Import Dashboards

**Export:**
1. Open dashboard in Grafana
2. Click "Share" icon
3. Click "Export" → "Save to file"
4. Saves as JSON

**Import:**
1. Click "+" → "Import"
2. Upload JSON file or paste JSON
3. Select Prometheus datasource
4. Click "Import"

## Troubleshooting

### Time-Series Charts Not Appearing

**Problem:** Ray dashboard still shows "setup Prometheus" message

**Solutions:**

1. **Verify Prometheus is running:**
   ```bash
   kubectl get pods -n ray-system | grep prometheus
   # Should show: Running
   ```

2. **Verify Grafana is running:**
   ```bash
   kubectl get pods -n ray-system | grep grafana
   # Should show: Running
   ```

3. **Check Ray environment variables:**
   ```bash
   kubectl get raycluster -n ray-system ray-training-cluster -o yaml | grep -A 5 env
   # Should show RAY_GRAFANA_HOST and RAY_PROMETHEUS_HOST
   ```

4. **Refresh Ray dashboard:**
   - Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
   - Or clear browser cache

5. **Check Ray head logs:**
   ```bash
   kubectl logs -n ray-system <ray-head-pod> | grep -i prometheus
   kubectl logs -n ray-system <ray-head-pod> | grep -i grafana
   ```

6. **Restart Ray cluster** (if needed):
   ```bash
   kubectl delete raycluster -n ray-system ray-training-cluster
   kubectl apply -f k8s/ray-cluster.yaml
   ```

### Prometheus Not Scraping Metrics

**Problem:** Prometheus shows no Ray targets or targets are down

**Check Prometheus targets:**
```bash
kubectl port-forward -n ray-system svc/prometheus 9090:9090
# Open: http://localhost:9090/targets
```

**Solutions:**

1. **Verify Ray metrics port:**
   ```bash
   kubectl get pods -n ray-system -o yaml | grep -A 3 "containerPort: 8080"
   # Should exist on head and workers
   ```

2. **Test metrics endpoint:**
   ```bash
   RAY_HEAD=$(kubectl get pods -n ray-system -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
   kubectl exec -n ray-system $RAY_HEAD -- curl http://localhost:8080/metrics
   # Should return Prometheus metrics
   ```

3. **Check Prometheus logs:**
   ```bash
   kubectl logs -n ray-system deployment/prometheus
   # Look for scrape errors
   ```

4. **Verify service account permissions:**
   ```bash
   kubectl get clusterrolebinding prometheus
   # Should exist and bind to prometheus service account
   ```

### Grafana Cannot Connect to Prometheus

**Problem:** Grafana datasource shows connection error

**Solutions:**

1. **Check Prometheus service:**
   ```bash
   kubectl get svc -n ray-system prometheus
   # Should show ClusterIP and port 9090
   ```

2. **Test connection from Grafana pod:**
   ```bash
   GRAFANA_POD=$(kubectl get pods -n ray-system -l app=grafana -o jsonpath='{.items[0].metadata.name}')
   kubectl exec -n ray-system $GRAFANA_POD -- curl http://prometheus:9090/api/v1/status/config
   # Should return Prometheus config
   ```

3. **Check Grafana logs:**
   ```bash
   kubectl logs -n ray-system deployment/grafana
   # Look for datasource errors
   ```

4. **Reconfigure datasource in Grafana:**
   - Login to Grafana
   - Go to Configuration → Data Sources
   - Click Prometheus
   - Verify URL: `http://prometheus:9090`
   - Click "Save & Test"

### High Memory Usage

**Problem:** Prometheus using too much memory

**Solutions:**

1. **Reduce scrape interval** (edit `k8s/prometheus-config.yaml`):
   ```yaml
   scrape_interval: 30s  # Increase from 15s
   ```

2. **Reduce retention time:**
   ```yaml
   # Add to Prometheus deployment args:
   - '--storage.tsdb.retention.time=6h'  # Default is 15d
   ```

3. **Increase resource limits:**
   ```yaml
   resources:
     limits:
       memory: "1Gi"  # Increase from 512Mi
   ```

4. **Apply changes:**
   ```bash
   kubectl apply -f k8s/prometheus-config.yaml
   ```

### Cannot Access Grafana

**Problem:** Browser cannot reach Grafana URL

**Solutions:**

1. **Verify Grafana service:**
   ```bash
   kubectl get svc -n ray-system grafana
   # Should show NodePort 30300
   ```

2. **Get correct Minikube IP:**
   ```bash
   minikube ip
   # Use this IP, not localhost
   ```

3. **Use port forwarding instead:**
   ```bash
   kubectl port-forward -n ray-system svc/grafana 3000:3000
   # Access: http://localhost:3000
   ```

4. **Check Grafana pod:**
   ```bash
   kubectl get pods -n ray-system -l app=grafana
   kubectl logs -n ray-system deployment/grafana
   ```

## Cleanup

### Remove Monitoring Stack Only

```bash
kubectl delete -f k8s/prometheus-config.yaml
kubectl delete -f k8s/grafana-config.yaml
```

### Remove Everything

```bash
./scripts/cleanup.sh
```

## Additional Resources

- [Ray Metrics Documentation](https://docs.ray.io/en/latest/ray-observability/ray-metrics.html)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Kubernetes Monitoring Best Practices](https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-metrics-pipeline/)
