#!/usr/bin/env bash
# Remote Optuna optimization on Hetzner CCX63.
#
# Provisions an ephemeral server, runs `dojiwick optimize` with 48 workers,
# writes Optuna results back to local PostgreSQL via SSH reverse tunnel,
# and deletes the server on exit (including Ctrl+C).
#
# One-time setup:
#   1. op item create --category ssh-key --title "Dojiwick Deploy Key" --vault Personal --ssh-generate-key ed25519
#   2. Get public key: op read "op://Personal/Dojiwick Deploy Key/public key"
#   3. Add the PUBLIC key to GitHub repo → Settings → Deploy keys (read-only)
#   4. hcloud ssh-key create --name mykey --public-key-from-file ~/.ssh/id_ed25519.pub
#
# Usage:
#   ./scripts/hetzner-optimize.sh --config config.toml --start 2025-01-01 --end 2025-06-01
#   ./scripts/hetzner-optimize.sh --config config.toml --start 2025-01-01 --end 2025-06-01 --gate
#   ./scripts/hetzner-optimize.sh --config config.toml --start 2025-01-01 --end 2025-06-01 --dry-run

set -euo pipefail

# Defaults
CONFIG_PATH=""
START=""
END=""
GATE=false
WORKERS=48
TIMEOUT_HOURS=4
LOCAL_PG_PORT=5432
OP_DEPLOY_KEY_REF="op://Personal/Dojiwick Deploy Key/private key"
SSH_KEY_NAME=""
DRY_RUN=false

# Resource tracking (used by cleanup)
SERVER_NAME=""
SERVER_IP=""
FW_NAME=""
RESOURCE_FILE=""

# Helpers
die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

# parse_args
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)      CONFIG_PATH="$2"; shift 2 ;;
            --start)       START="$2"; shift 2 ;;
            --end)         END="$2"; shift 2 ;;
            --gate)        GATE=true; shift ;;
            --workers)     WORKERS="$2"; shift 2 ;;
            --timeout)     TIMEOUT_HOURS="$2"; shift 2 ;;
            --pg-port)     LOCAL_PG_PORT="$2"; shift 2 ;;
            --op-key-ref)  OP_DEPLOY_KEY_REF="$2"; shift 2 ;;
            --ssh-key)     SSH_KEY_NAME="$2"; shift 2 ;;
            --dry-run)     DRY_RUN=true; shift ;;
            *) die "Unknown argument: $1" ;;
        esac
    done

    [[ -n "$CONFIG_PATH" ]] || die "--config is required"
    [[ -n "$START" ]]       || die "--start is required"
    [[ -n "$END" ]]         || die "--end is required"
}

# preflight_checks
preflight_checks() {
    info "Running pre-flight checks"

    command -v hcloud >/dev/null 2>&1 || die "hcloud CLI not found. Install: brew install hcloud"
    hcloud server-type list >/dev/null 2>&1 || die "hcloud not authenticated. Run: hcloud context create"

    pg_isready -h localhost -p "$LOCAL_PG_PORT" >/dev/null 2>&1 \
        || die "PostgreSQL not accepting connections on localhost:$LOCAL_PG_PORT"

    [[ -f "$CONFIG_PATH" ]] || die "Config file not found: $CONFIG_PATH"

    [[ -n "${BINANCE_API_KEY:-}" ]]    || die "BINANCE_API_KEY not set in environment"
    [[ -n "${BINANCE_API_SECRET:-}" ]] || die "BINANCE_API_SECRET not set in environment"

    command -v op >/dev/null 2>&1 || die "1Password CLI (op) not found. Install: brew install 1password-cli"
    op read "$OP_DEPLOY_KEY_REF" --ssh-format openssh >/dev/null 2>&1 \
        || die "Cannot read deploy key from 1Password: $OP_DEPLOY_KEY_REF"

    # Auto-detect SSH key name if not provided
    if [[ -z "$SSH_KEY_NAME" ]]; then
        SSH_KEY_NAME=$(hcloud ssh-key list -o noheader -o columns=name | head -1)
        [[ -n "$SSH_KEY_NAME" ]] || die "No SSH keys found in Hetzner project. Run: hcloud ssh-key create"
    else
        hcloud ssh-key list -o noheader | grep -q "$SSH_KEY_NAME" \
            || die "SSH key '$SSH_KEY_NAME' not found in Hetzner project"
    fi

    info "Pre-flight checks passed"
}

# detect_public_ip
detect_public_ip() {
    info "Detecting public IP"
    local ip
    ip=$(curl -sf https://api.ipify.org || curl -sf https://ifconfig.me || curl -sf https://checkip.amazonaws.com) \
        || die "Failed to detect public IP"
    # Trim whitespace
    ip=$(echo "$ip" | tr -d '[:space:]')
    [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "Invalid IP format: $ip"
    MY_IP="$ip"
    info "Public IP: $MY_IP"
}

# cleanup — runs on EXIT (including Ctrl+C)
cleanup() {
    echo ""
    info "Cleaning up resources"

    # Cancel remote shutdown timer (best-effort)
    if [[ -n "${SERVER_IP:-}" ]]; then
        ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=accept-new \
            "root@$SERVER_IP" 'shutdown -c' 2>/dev/null || true
    fi

    # Delete server (stops billing immediately)
    if [[ -n "${SERVER_NAME:-}" ]]; then
        for _attempt in 1 2 3; do
            if hcloud server delete "$SERVER_NAME" 2>/dev/null; then
                info "Server '$SERVER_NAME' deleted"
                SERVER_NAME=""
                break
            fi
            sleep 2
        done
    fi

    # Delete firewall
    if [[ -n "${FW_NAME:-}" ]]; then
        for _attempt in 1 2 3; do
            if hcloud firewall delete "$FW_NAME" 2>/dev/null; then
                info "Firewall '$FW_NAME' deleted"
                FW_NAME=""
                break
            fi
            sleep 2
        done
    fi

    # Clean up local temp files
    rm -f "${RESOURCE_FILE:-}"

    info "Cleanup complete"
}

# create_firewall
create_firewall() {
    FW_NAME="dojiwick-fw-$(date +%s)"
    info "Creating firewall '$FW_NAME' (SSH from $MY_IP only)"

    hcloud firewall create --name "$FW_NAME"
    hcloud firewall add-rule "$FW_NAME" \
        --direction in --protocol tcp --port 22 \
        --source-ips "$MY_IP/32" \
        --description "SSH from caller"
}

# create_server
create_server() {
    SERVER_NAME="dojiwick-opt-$(date +%s)"
    info "Creating server '$SERVER_NAME' (ccx63, ubuntu-24.04)"

    local cloud_init_file
    cloud_init_file=$(mktemp)
    cat > "$cloud_init_file" <<'EOF'
#cloud-config
ssh_pwauth: false
disable_root: false
write_files:
  - path: /etc/ssh/sshd_config.d/99-hardening.conf
    content: |
      MaxAuthTries 3
      PermitRootLogin prohibit-password
  - path: /etc/sysctl.d/99-hardening.conf
    content: |
      net.ipv4.tcp_syncookies=1
runcmd:
  - sysctl --system
  - systemctl restart sshd
  - passwd -l root
EOF

    hcloud server create \
        --name "$SERVER_NAME" \
        --type ccx63 \
        --image ubuntu-24.04 \
        --ssh-key "$SSH_KEY_NAME" \
        --firewall "$FW_NAME" \
        --location fsn1 \
        --user-data-from-file "$cloud_init_file"
    rm "$cloud_init_file"

    SERVER_IP=$(hcloud server ip "$SERVER_NAME")
    info "Server IP: $SERVER_IP"

    # Write resource IDs for manual cleanup fallback
    RESOURCE_FILE="/tmp/dojiwick-hetzner-$(date +%s).txt"
    echo "server=$SERVER_NAME firewall=$FW_NAME ip=$SERVER_IP" > "$RESOURCE_FILE"
    info "Resource IDs saved to $RESOURCE_FILE"
}

# wait_ready
wait_ready() {
    info "Waiting for SSH"
    until ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
        "root@$SERVER_IP" true 2>/dev/null; do
        sleep 3
    done
    info "SSH ready, waiting for cloud-init"
    ssh -o StrictHostKeyChecking=accept-new "root@$SERVER_IP" 'cloud-init status --wait'
    info "Server ready"
}

# setup_project
setup_project() {
    local repo_slug
    repo_slug=$(git remote get-url origin | sed 's|.*github.com[:/]||; s|\.git$||')

    info "Fetching deploy key from 1Password and sending to server"
    # Key never touches local disk — piped directly from op to remote via SSH
    op read "$OP_DEPLOY_KEY_REF" --ssh-format openssh \
        | ssh -o StrictHostKeyChecking=accept-new "root@$SERVER_IP" \
            'cat > /root/.ssh/deploy_key && chmod 600 /root/.ssh/deploy_key && \
             cat >> /root/.ssh/config <<SSHEOF
Host github.com
    IdentityFile /root/.ssh/deploy_key
    StrictHostKeyChecking accept-new
SSHEOF'

    info "Installing uv, Python 3.14, cloning repo, syncing deps"
    # shellcheck disable=SC2029  # client-side expansion of $repo_slug is intentional
    ssh "root@$SERVER_IP" "
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH=\"/root/.local/bin:\$PATH\"
        uv python install 3.14
        git clone git@github.com:$repo_slug.git /root/dojiwick
        rm -f /root/.ssh/deploy_key /root/.ssh/config
        cd /root/dojiwick
        uv sync --extra optuna --extra exchange --extra postgres
    "
    info "Project setup complete (deploy key deleted from server)"
}

# upload_config
upload_config() {
    info "Uploading config.toml (DSN rewritten for tunnel)"
    local temp_config
    temp_config=$(mktemp)
    sed 's/@postgres:/@localhost:/g' "$CONFIG_PATH" > "$temp_config"
    scp -q "root@$SERVER_IP:/root/dojiwick/config.toml" < /dev/null 2>/dev/null || true
    scp -q "$temp_config" "root@$SERVER_IP:/root/dojiwick/config.toml"
    rm "$temp_config"
}

# run_optimization
run_optimization() {
    local gate_flag=""
    [[ "$GATE" == true ]] && gate_flag="--gate"

    info "Setting remote shutdown timer (${TIMEOUT_HOURS}h safety net)"
    # shellcheck disable=SC2029  # client-side expansion is intentional
    ssh "root@$SERVER_IP" "shutdown +$((TIMEOUT_HOURS * 60))" 2>/dev/null || true

    info "Writing .env on remote (secrets not in /proc/cmdline)"
    # Use set +x to avoid leaking secrets if debug mode is on
    set +x
    # shellcheck disable=SC2087  # client-side expansion is intentional — we want local env var values
    ssh "root@$SERVER_IP" "cat > /root/dojiwick/.env; chmod 600 /root/dojiwick/.env" <<EOF
BINANCE_API_KEY=$BINANCE_API_KEY
BINANCE_API_SECRET=$BINANCE_API_SECRET
EOF
    set -x 2>/dev/null || true

    info "Starting optimization (${WORKERS} workers, ${TIMEOUT_HOURS}h timeout)"
    info "Tunnel: remote 127.0.0.1:5432 → local localhost:${LOCAL_PG_PORT}"
    ssh \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o TCPKeepAlive=yes \
        -R "127.0.0.1:5432:localhost:${LOCAL_PG_PORT}" \
        "root@$SERVER_IP" \
        "cd /root/dojiwick && \
         timeout --kill-after=60 ${TIMEOUT_HOURS}h \
         /root/.local/bin/uv run dojiwick optimize \
             --config config.toml \
             --start $START --end $END \
             --workers $WORKERS \
             $gate_flag"
}

# main
main() {
    parse_args "$@"
    preflight_checks
    detect_public_ip

    if [[ "$DRY_RUN" == true ]]; then
        echo ""
        echo "DRY RUN — would execute:"
        echo "  Server:    ccx63 (48 vCPU, 192 GB RAM), ubuntu-24.04, location fsn1"
        echo "  SSH key:   $SSH_KEY_NAME"
        echo "  Firewall:  SSH from $MY_IP/32 only"
        echo "  Config:    $CONFIG_PATH (DSN rewritten @postgres: → @localhost:)"
        echo "  Command:   dojiwick optimize --config config.toml --start $START --end $END --workers $WORKERS $([ "$GATE" == true ] && echo '--gate')"
        echo "  Tunnel:    remote 127.0.0.1:5432 → local localhost:$LOCAL_PG_PORT"
        echo "  Timeout:   ${TIMEOUT_HOURS}h"
        echo "  Deploy key: $OP_DEPLOY_KEY_REF (1Password, deleted from server after clone)"
        echo ""
        echo "No resources created."
        exit 0
    fi

    trap cleanup EXIT

    create_firewall
    create_server
    wait_ready
    setup_project
    upload_config
    run_optimization

    info "Optimization complete"
}

main "$@"
