# Ray Dashboard Guide

Complete guide to accessing and using the Ray Dashboard for monitoring your distributed training cluster.

## Table of Contents

- [Quick Start](#quick-start)
- [Access Methods](#access-methods)
- [Dashboard Features](#dashboard-features)
- [Monitoring Training Jobs](#monitoring-training-jobs)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Automated Access

The easiest way to access the dashboard:

```bash
./scripts/open-dashboard.sh
```

This script will:
- Verify that Minikube and Ray cluster are running
- Display the dashboard URL
- Show current cluster status
- Optionally open the dashboard in your browser

## Access Methods

### Method 1: NodePort (Recommended for Minikube)

The Ray dashboard is exposed via Kubernetes NodePort on port **30265**.

**Steps:**

1. Get your Minikube IP address:
   ```bash
   minikube ip
   ```

2. Open your browser to:
   ```
   http://<minikube-ip>:30265
   ```

   Example: `http://192.168.49.2:30265`

**Advantages:**
- Direct access without port forwarding
- Persistent connection
- Works across terminal sessions

**When to use:**
- Default method for Minikube clusters
- When you want persistent access

### Method 2: Port Forwarding (Localhost)

Access the dashboard via localhost using kubectl port forwarding.

**Steps:**

1. In a separate terminal, run:
   ```bash
   kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265
   ```

2. Open your browser to:
   ```
   http://localhost:8265
   ```

3. Keep the terminal running while you use the dashboard

**Advantages:**
- Access via localhost (more secure)
- No need to know Minikube IP
- Works well with SSH tunneling

**When to use:**
- When working on remote servers via SSH
- When you prefer localhost access
- For enhanced security

### Method 3: Minikube Service

Use Minikube's built-in service command:

```bash
minikube service ray-cluster-head-svc -n ray-system --url
```

This will display the URL to access the dashboard.

## Dashboard Features

### 1. Overview Tab

**What you'll see:**
- Cluster resource utilization (CPU, Memory, GPU)
- Number of nodes (head + workers)
- Total available resources
- Active jobs and tasks

**Use this for:**
- Quick cluster health check
- Resource allocation overview
- Identifying bottlenecks

### 2. Jobs Tab

**What you'll see:**
- List of submitted jobs
- Job status (Running, Succeeded, Failed)
- Job duration and resource usage
- Job logs and error messages

**Use this for:**
- Monitoring training job progress
- Debugging failed jobs
- Viewing job submission history

**Key Information:**
- **Job ID**: Unique identifier for each job
- **Status**: Current state of the job
- **Start/End Time**: When the job started and finished
- **Resources**: CPUs, memory, GPUs allocated

### 3. Actors Tab

**What you'll see:**
- Active Ray actors (stateful workers)
- Actor lifecycle and resource usage
- Actor placement across nodes

**Use this for:**
- Understanding distributed training workers
- Monitoring actor health
- Debugging actor failures

### 4. Tasks Tab

**What you'll see:**
- Individual tasks being executed
- Task execution timeline
- Task dependencies and scheduling

**Use this for:**
- Fine-grained performance analysis
- Understanding task distribution
- Identifying slow tasks

### 5. Logs Tab

**What you'll see:**
- Real-time logs from all cluster nodes
- Error messages and warnings
- Training output and progress

**Use this for:**
- Debugging errors and failures
- Monitoring training metrics
- Viewing print statements from your code

### 6. Cluster Tab

**What you'll see:**
- Node-level information
- Per-node resource usage
- Worker pod status

**Use this for:**
- Checking node health
- Balancing workload across nodes
- Identifying node failures

### 7. Metrics Tab

**What you'll see:**
- Time-series graphs of resource usage
- Custom metrics from your application
- System-level metrics

**Use this for:**
- Performance analysis over time
- Capacity planning
- Identifying resource trends

## Monitoring Training Jobs

### During Baseline Training

When running `baseline_training.py`:

1. Navigate to the **Jobs** tab
2. Look for the baseline training job
3. Click on the job to see:
   - Real-time logs with training progress
   - Epoch completion status
   - Loss and accuracy metrics
   - Resource usage

### During Ray Distributed Training

When running `ray_training.py`:

1. **Jobs Tab**:
   - Shows the distributed training job
   - Multiple workers processing in parallel

2. **Actors Tab**:
   - Shows TensorFlow workers (one per Ray worker)
   - Each actor represents a training process

3. **Cluster Tab**:
   - Shows workload distributed across 2 worker nodes
   - CPU/Memory usage per node

4. **Logs Tab**:
   - Filter logs by worker to see individual progress
   - View aggregated training metrics

### Key Metrics to Watch

**Resource Utilization:**
- CPU usage should be high (80-100%) during training
- Memory usage should be stable
- Check for resource bottlenecks

**Training Progress:**
- Epoch completion rate
- Loss decreasing over time
- Accuracy improving

**Worker Status:**
- All workers should be active
- No worker failures or restarts
- Even workload distribution

## Authentication and Security

### Default Configuration

**Important:** The Ray dashboard does NOT require authentication by default.

- No username or password needed
- Direct access once connected
- Suitable for local development and testing

### Security Considerations

For production or shared environments, consider:

1. **Port Forwarding** (Localhost only):
   ```bash
   kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265
   ```
   Only accessible from your local machine.

2. **SSH Tunneling** (Remote access):
   ```bash
   ssh -L 8265:localhost:8265 user@remote-server
   ```
   Securely tunnel through SSH.

3. **Ingress with Authentication**:
   - Set up Kubernetes Ingress
   - Add OAuth or basic auth
   - Use TLS certificates

4. **Network Policies**:
   - Restrict dashboard access to specific IPs
   - Use Kubernetes NetworkPolicy

### Production Security Best Practices

If deploying to production:

- **Never expose port 30265 to the public internet**
- Use authentication/authorization (OAuth, LDAP, etc.)
- Enable TLS/SSL encryption
- Implement role-based access control (RBAC)
- Use VPN or bastion host for remote access
- Monitor dashboard access logs

## Troubleshooting

### Dashboard Not Accessible

**Problem:** Cannot reach the dashboard URL

**Solutions:**

1. Verify Minikube is running:
   ```bash
   minikube status
   ```

2. Check Ray cluster is deployed:
   ```bash
   kubectl get raycluster -n ray-system
   kubectl get pods -n ray-system
   ```

3. Verify service is running:
   ```bash
   kubectl get svc -n ray-system ray-cluster-head-svc
   ```

4. Check if dashboard port is open:
   ```bash
   kubectl logs -n ray-system <ray-head-pod-name> | grep dashboard
   ```

### Port Forwarding Issues

**Problem:** Port forwarding command fails or times out

**Solutions:**

1. Check if port 8265 is already in use:
   ```bash
   lsof -i :8265  # Linux/macOS
   netstat -ano | findstr :8265  # Windows
   ```

2. Use a different local port:
   ```bash
   kubectl port-forward -n ray-system svc/ray-cluster-head-svc 9999:8265
   # Then access: http://localhost:9999
   ```

3. Verify you have kubectl access:
   ```bash
   kubectl auth can-i get pods -n ray-system
   ```

### Dashboard Shows "Disconnected" or "Connecting"

**Problem:** Dashboard loads but shows connection issues

**Solutions:**

1. Check Ray head pod status:
   ```bash
   kubectl get pods -n ray-system -l ray.io/node-type=head
   ```

2. Restart port forwarding (if using Method 2)

3. Check Ray GCS (Global Control Store) is running:
   ```bash
   kubectl logs -n ray-system <ray-head-pod-name> | grep GCS
   ```

4. Verify dashboard is running on head node:
   ```bash
   kubectl exec -it -n ray-system <ray-head-pod-name> -- curl http://localhost:8265
   ```

### No Jobs Showing in Dashboard

**Problem:** Dashboard loads but no training jobs appear

**Solutions:**

1. Jobs only appear when actively running or recently completed

2. Verify job was submitted to the cluster:
   ```bash
   kubectl logs -n ray-system <ray-head-pod-name>
   ```

3. Check if training script is running:
   ```bash
   kubectl exec -n ray-system <ray-head-pod-name> -- ps aux | grep python
   ```

### Browser Shows "Connection Refused"

**Problem:** Browser cannot connect to dashboard URL

**Solutions:**

1. For NodePort method, verify Minikube IP:
   ```bash
   minikube ip
   # Use this exact IP in browser
   ```

2. For port forwarding, ensure command is still running:
   ```bash
   # Keep this terminal open:
   kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265
   ```

3. Try accessing from incognito/private browser window

4. Check firewall settings

## Advanced Usage

### Accessing Dashboard from Remote Machine

If running on a remote server:

1. **SSH with Local Port Forwarding:**
   ```bash
   ssh -L 8265:localhost:8265 user@remote-server

   # On remote server:
   kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265

   # On local machine, open: http://localhost:8265
   ```

2. **SSH with Dynamic Port Forwarding (SOCKS proxy):**
   ```bash
   ssh -D 9090 user@remote-server

   # Configure browser to use SOCKS proxy: localhost:9090
   # Then access: http://<minikube-ip>:30265
   ```

### Custom Dashboard Port

To change the dashboard port, modify `k8s/ray-service.yaml`:

```yaml
ports:
  - name: dashboard
    port: 8265
    targetPort: 8265
    nodePort: 30999  # Change this to your preferred port (30000-32767)
```

Then reapply:
```bash
kubectl apply -f k8s/ray-service.yaml
```

### Persistent Dashboard Access

For long-running clusters, add dashboard to systemd or use screen/tmux:

```bash
# Using screen
screen -S ray-dashboard
kubectl port-forward -n ray-system svc/ray-cluster-head-svc 8265:8265
# Press Ctrl+A, D to detach

# Reattach later
screen -r ray-dashboard
```

## Useful Dashboard URLs

Once connected to the dashboard, you can bookmark these direct links:

- **Overview**: `http://<dashboard-url>/`
- **Jobs**: `http://<dashboard-url>/#/jobs`
- **Actors**: `http://<dashboard-url>/#/actors`
- **Logs**: `http://<dashboard-url>/#/logs`
- **Cluster**: `http://<dashboard-url>/#/cluster`
- **Metrics**: `http://<dashboard-url>/#/metrics`

## Additional Resources

- [Ray Dashboard Documentation](https://docs.ray.io/en/latest/ray-observability/ray-dashboard.html)
- [Ray Monitoring Guide](https://docs.ray.io/en/latest/ray-observability/getting-started.html)
- [Kubernetes Port Forwarding](https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster/)
- [KubeRay Documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
