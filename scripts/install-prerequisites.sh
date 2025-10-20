#!/bin/bash
set -e

echo "=========================================="
echo "Ray Parallel Training Demo - Prerequisites Installation"
echo "=========================================="
echo ""
echo "This script will install:"
echo "  - kubectl (Kubernetes CLI)"
echo "  - Minikube (Local Kubernetes)"
echo "  - Helm (Kubernetes package manager)"
echo ""

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Convert architecture names
case "$ARCH" in
    x86_64)
        ARCH="amd64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        ;;
    *)
        echo "Error: Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

echo "Detected OS: $OS"
echo "Detected Architecture: $ARCH"
echo ""

# Check if running with sufficient privileges
if [[ "$OS" == "linux" ]] && [[ $EUID -eq 0 ]]; then
    echo "Warning: Running as root. This script will install binaries to /usr/local/bin/"
    SUDO=""
elif [[ "$OS" == "linux" ]]; then
    echo "This script requires sudo privileges to install binaries to /usr/local/bin/"
    SUDO="sudo"
    # Test sudo access
    sudo -v || { echo "Error: sudo access required"; exit 1; }
else
    SUDO=""
fi

echo ""
echo "=========================================="
echo "1. Installing kubectl"
echo "=========================================="

if command -v kubectl &> /dev/null; then
    KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null || kubectl version --client 2>/dev/null | head -1)
    echo "kubectl is already installed: $KUBECTL_VERSION"
    read -p "Do you want to reinstall/upgrade? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping kubectl installation"
    else
        echo "Downloading kubectl..."
        KUBECTL_VERSION=$(curl -L -s https://dl.k8s.io/release/stable.txt)
        curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/${OS}/${ARCH}/kubectl"
        chmod +x kubectl
        $SUDO mv kubectl /usr/local/bin/kubectl
        echo "kubectl installed successfully!"
    fi
else
    echo "Downloading kubectl..."
    KUBECTL_VERSION=$(curl -L -s https://dl.k8s.io/release/stable.txt)
    curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/${OS}/${ARCH}/kubectl"
    chmod +x kubectl
    $SUDO mv kubectl /usr/local/bin/kubectl
    echo "kubectl installed successfully!"
fi

# Verify kubectl installation
kubectl version --client
echo ""

echo "=========================================="
echo "2. Installing Minikube"
echo "=========================================="

if command -v minikube &> /dev/null; then
    MINIKUBE_VERSION=$(minikube version --short 2>/dev/null || minikube version | head -1)
    echo "Minikube is already installed: $MINIKUBE_VERSION"
    read -p "Do you want to reinstall/upgrade? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping Minikube installation"
    else
        echo "Downloading Minikube..."
        curl -LO "https://storage.googleapis.com/minikube/releases/latest/minikube-${OS}-${ARCH}"
        chmod +x "minikube-${OS}-${ARCH}"
        $SUDO mv "minikube-${OS}-${ARCH}" /usr/local/bin/minikube
        echo "Minikube installed successfully!"
    fi
else
    echo "Downloading Minikube..."
    curl -LO "https://storage.googleapis.com/minikube/releases/latest/minikube-${OS}-${ARCH}"
    chmod +x "minikube-${OS}-${ARCH}"
    $SUDO mv "minikube-${OS}-${ARCH}" /usr/local/bin/minikube
    echo "Minikube installed successfully!"
fi

# Verify Minikube installation
minikube version
echo ""

echo "=========================================="
echo "3. Installing Helm"
echo "=========================================="

if command -v helm &> /dev/null; then
    HELM_VERSION=$(helm version --short 2>/dev/null || helm version | head -1)
    echo "Helm is already installed: $HELM_VERSION"
    read -p "Do you want to reinstall/upgrade? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping Helm installation"
    else
        echo "Downloading and installing Helm..."
        curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
        echo "Helm installed successfully!"
    fi
else
    echo "Downloading and installing Helm..."
    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
    echo "Helm installed successfully!"
fi

# Verify Helm installation
helm version
echo ""

echo "=========================================="
echo "4. Verifying Docker"
echo "=========================================="

if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo "Docker is installed: $DOCKER_VERSION"

    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        echo "Docker daemon is running ✓"
    else
        echo "Warning: Docker is installed but the daemon is not running"
        echo "Please start Docker before proceeding with Minikube setup"
    fi
else
    echo "Warning: Docker is not installed"
    echo "Docker is required as the Minikube driver"
    echo ""
    echo "Please install Docker from:"
    echo "  - Ubuntu/Debian: https://docs.docker.com/engine/install/ubuntu/"
    echo "  - macOS: https://docs.docker.com/desktop/install/mac-install/"
    echo "  - Windows: https://docs.docker.com/desktop/install/windows-install/"
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Installed versions:"
echo "  - kubectl: $(kubectl version --client --short 2>/dev/null || echo 'installed')"
echo "  - Minikube: $(minikube version --short 2>/dev/null || minikube version | head -1)"
echo "  - Helm: $(helm version --short 2>/dev/null || echo 'installed')"

if command -v docker &> /dev/null; then
    echo "  - Docker: $(docker --version)"
fi

echo ""
echo "Next steps:"
echo "  1. Ensure Docker is running"
echo "  2. Run: ./scripts/setup-minikube.sh"
echo "  3. Run: ./scripts/install-ray.sh"
echo "  4. Run: ./scripts/run-benchmark.sh"
echo ""
echo "=========================================="
