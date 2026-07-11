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
#   ./scripts/hetzner-optimize.sh --config config.toml --start 2019-09-08 --end 2026-03-17
#   ./scripts/hetzner-optimize.sh --config config.toml --start 2019-09-08 --end 2026-03-17 --gate
#   ./scripts/hetzner-optimize.sh --config config.toml --start 2019-09-08 --end 2026-03-17 --dry-run
#
# SSH: connections pin a single identity (--identity, default ~/.ssh/id_ed25519)
# with IdentitiesOnly=yes — the private key must match the Hetzner key (--ssh-key).
# The server's host key is generated locally and injected via cloud-init, so
# every connection verifies against a pre-known key (no trust-on-first-use
# on the channel that carries the Binance credentials).

set -euo pipefail

# Wrap hcloud with 1Password plugin (alias doesn't work in scripts)
hcloud() { op plugin run -- hcloud "$@"; }

# Pin the SSH identity: without IdentitiesOnly the agent offers every loaded
# key and the server disconnects on MaxAuthTries before the right one is tried.
# Host key verification is strict against the per-run known_hosts file.
ssh() {
    command ssh -o IdentitiesOnly=yes -i "$SSH_IDENTITY" \
        -o UserKnownHostsFile="$KNOWN_HOSTS_FILE" -o StrictHostKeyChecking=yes "$@"
}
scp() {
    command scp -o IdentitiesOnly=yes -i "$SSH_IDENTITY" \
        -o UserKnownHostsFile="$KNOWN_HOSTS_FILE" -o StrictHostKeyChecking=yes "$@"
}

# Defaults
CONFIG_PATH=""
START=""
END=""
GATE=false
WORKERS=48
TIMEOUT_HOURS=4
LOCAL_PG_PORT=5432
OP_DEPLOY_KEY_REF="op://Personal/Dojiwick Deploy Key/private key"
OP_BINANCE_API_KEY_REF="op://Personal/Binance API/api_key"
OP_BINANCE_API_SECRET_REF="op://Personal/Binance API/api_secret"
SSH_KEY_NAME=""
SSH_IDENTITY="$HOME/.ssh/id_ed25519"
DRY_RUN=false

# Pinned remote toolchain (matches the Dockerfile's uv)
UV_VERSION="0.11.28"
UV_INSTALLER_SHA256="b7b3fe80cad1142a2a5794050b7db7b3291d1bac1423b0732571dd9366e8ca8b"

# Resource tracking (used by cleanup)
SERVER_NAME=""
SERVER_IP=""
FW_NAME=""
RESOURCE_FILE=""
HOST_KEY_DIR=""
KNOWN_HOSTS_FILE=""

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
            --identity)    SSH_IDENTITY="$2"; shift 2 ;;
            --dry-run)     DRY_RUN=true; shift ;;
            *) die "Unknown argument: $1" ;;
        esac
    done

    [[ -n "$CONFIG_PATH" ]] || die "--config is required"
    [[ -n "$START" ]]       || die "--start is required"
    [[ -n "$END" ]]         || die "--end is required"

    # Input validation — prevent command injection via SSH
    [[ "$START" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]   || die "--start must be YYYY-MM-DD"
    [[ "$END" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]     || die "--end must be YYYY-MM-DD"
    [[ "$WORKERS" =~ ^[0-9]+$ ]]                       || die "--workers must be a positive integer"
    [[ "$TIMEOUT_HOURS" =~ ^[0-9]+$ ]]                 || die "--timeout must be a positive integer"
    [[ "$LOCAL_PG_PORT" =~ ^[0-9]+$ ]]                 || die "--pg-port must be a number"
}

# preflight_checks
preflight_checks() {
    info "Running pre-flight checks"

    command -v hcloud >/dev/null 2>&1 || die "hcloud CLI not found. Install: brew install hcloud"
    hcloud server-type list >/dev/null 2>&1 || die "hcloud not authenticated. Run: hcloud context create"

    pg_isready -h localhost -p "$LOCAL_PG_PORT" >/dev/null 2>&1 \
        || die "PostgreSQL not accepting connections on localhost:$LOCAL_PG_PORT"

    [[ -f "$CONFIG_PATH" ]] || die "Config file not found: $CONFIG_PATH"

    command -v op >/dev/null 2>&1 || die "1Password CLI (op) not found. Install: brew install 1password-cli"
    op read "$OP_BINANCE_API_KEY_REF" >/dev/null 2>&1 \
        || die "Cannot read Binance API key from 1Password"
    op read "$OP_BINANCE_API_SECRET_REF" >/dev/null 2>&1 \
        || die "Cannot read Binance API secret from 1Password"
    op read "${OP_DEPLOY_KEY_REF}?ssh-format=openssh" >/dev/null 2>&1 \
        || die "Cannot read deploy key from 1Password: $OP_DEPLOY_KEY_REF"

    # Auto-detect SSH key name if not provided
    if [[ -z "$SSH_KEY_NAME" ]]; then
        SSH_KEY_NAME=$(hcloud ssh-key list -o noheader -o columns=name | head -1)
        [[ -n "$SSH_KEY_NAME" ]] || die "No SSH keys found in Hetzner project. Run: hcloud ssh-key create"
    else
        hcloud ssh-key list -o noheader | grep -q "$SSH_KEY_NAME" \
            || die "SSH key '$SSH_KEY_NAME' not found in Hetzner project"
    fi

    [[ -f "$SSH_IDENTITY" ]] \
        || die "SSH identity file not found: $SSH_IDENTITY (pass --identity <path> matching Hetzner key '$SSH_KEY_NAME')"

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
        ssh -o ConnectTimeout=3 "root@$SERVER_IP" 'shutdown -c' 2>/dev/null || true
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

    # Clean up local temp files (host key material included)
    rm -f "${RESOURCE_FILE:-}"
    [[ -n "${HOST_KEY_DIR:-}" ]] && rm -rf "$HOST_KEY_DIR"

    info "Cleanup complete"
}

# generate_host_key — pre-generate the server's SSH host key so connections
# never trust-on-first-use; the private half reaches the server only through
# Hetzner's cloud-init channel.
generate_host_key() {
    HOST_KEY_DIR=$(mktemp -d)
    ssh-keygen -q -t ed25519 -N '' -C "dojiwick-ephemeral-host" -f "$HOST_KEY_DIR/host_key"
    KNOWN_HOSTS_FILE="$HOST_KEY_DIR/known_hosts"
    : > "$KNOWN_HOSTS_FILE"
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
    # ssh_keys installs the locally generated host key; ssh_deletekeys removes
    # the image's autogenerated ones, so the fingerprint is known before the
    # first connection.
    cat > "$cloud_init_file" <<EOF
#cloud-config
ssh_pwauth: false
disable_root: false
ssh_deletekeys: true
ssh_keys:
  ed25519_private: |
$(sed 's/^/    /' "$HOST_KEY_DIR/host_key")
  ed25519_public: $(cat "$HOST_KEY_DIR/host_key.pub")
write_files:
  - path: /etc/ssh/sshd_config.d/99-hardening.conf
    content: |
      MaxAuthTries 6
      PermitRootLogin prohibit-password
  - path: /etc/sysctl.d/99-hardening.conf
    content: |
      net.ipv4.tcp_syncookies=1
runcmd:
  - sysctl --system
  - systemctl restart sshd
  - passwd -l root
EOF

    local locations=("nbg1" "fsn1" "hel1")
    local created=false
    for loc in "${locations[@]}"; do
        if hcloud server create \
            --name "$SERVER_NAME" \
            --type ccx63 \
            --image ubuntu-24.04 \
            --ssh-key "$SSH_KEY_NAME" \
            --firewall "$FW_NAME" \
            --location "$loc" \
            --user-data-from-file "$cloud_init_file" 2>&1; then
            created=true
            info "Server created in $loc"
            break
        fi
        info "Location $loc unavailable, trying next..."
    done
    rm "$cloud_init_file"
    [[ "$created" == true ]] || die "No available location for ccx63"

    SERVER_IP=$(hcloud server ip "$SERVER_NAME")
    info "Server IP: $SERVER_IP"

    # Pin the pre-generated host key to this server's address
    printf '%s %s\n' "$SERVER_IP" "$(cut -d' ' -f1-2 "$HOST_KEY_DIR/host_key.pub")" > "$KNOWN_HOSTS_FILE"

    # Write resource IDs for manual cleanup fallback
    RESOURCE_FILE="/tmp/dojiwick-hetzner-$(date +%s).txt"
    install -m 600 /dev/null "$RESOURCE_FILE"
    echo "server=$SERVER_NAME firewall=$FW_NAME ip=$SERVER_IP" > "$RESOURCE_FILE"
    info "Resource IDs saved to $RESOURCE_FILE"
}

# wait_ready
wait_ready() {
    info "Waiting for SSH on $SERVER_IP (may take 1-2 min)..."
    local attempt=0
    until ssh -o ConnectTimeout=5 "root@$SERVER_IP" true 2>/dev/null; do
        attempt=$((attempt + 1))
        printf "  SSH attempt %d (${attempt}x3s elapsed)...\r" "$attempt"
        sleep 3
        if [[ $attempt -ge 40 ]]; then
            echo ""
            info "SSH debug after 2min timeout:"
            ssh -v -o ConnectTimeout=5 "root@$SERVER_IP" true 2>&1 | tail -5
            die "SSH connection failed after 40 attempts"
        fi
    done
    echo ""
    info "SSH connected after $attempt attempts"
    info "Waiting for cloud-init..."
    ssh "root@$SERVER_IP" 'cloud-init status --wait'
    info "Server ready"
}

# setup_project
setup_project() {
    local repo_slug
    repo_slug=$(git remote get-url origin | sed 's|.*github.com[:/]||; s|\.git$||')
    [[ "$repo_slug" =~ ^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$ ]] || die "Invalid repo slug: $repo_slug"

    info "Fetching deploy key from 1Password and sending to server"
    # Key never touches local disk — piped directly from op to remote via SSH
    op read "${OP_DEPLOY_KEY_REF}?ssh-format=openssh" \
        | ssh "root@$SERVER_IP" \
            'cat > /root/.ssh/deploy_key && chmod 600 /root/.ssh/deploy_key && \
             cat >> /root/.ssh/config <<SSHEOF
Host github.com
    IdentityFile /root/.ssh/deploy_key
    StrictHostKeyChecking accept-new
SSHEOF'

    info "Installing uv ${UV_VERSION} (sha256-verified), Python 3.14, cloning repo, syncing deps"
    # shellcheck disable=SC2029  # client-side expansion is intentional
    ssh "root@$SERVER_IP" "
        set -euo pipefail
        curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh -o /tmp/uv-install.sh
        echo '${UV_INSTALLER_SHA256}  /tmp/uv-install.sh' | sha256sum -c - >/dev/null
        sh /tmp/uv-install.sh && rm /tmp/uv-install.sh
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
    # Host AND port must point at the tunnel endpoint (remote 127.0.0.1:5432);
    # the local port difference is absorbed by the -R mapping.
    sed -E 's/@postgres:[0-9]+/@localhost:5432/g' "$CONFIG_PATH" > "$temp_config"
    scp -q "$temp_config" "root@$SERVER_IP:/root/dojiwick/config.toml"
    rm "$temp_config"
    ssh "root@$SERVER_IP" "chmod 600 /root/dojiwick/config.toml"
}

# run_optimization
run_optimization() {
    local gate_flag=""
    [[ "$GATE" == true ]] && gate_flag="--gate"

    info "Setting remote shutdown timer (${TIMEOUT_HOURS}h safety net)"
    # shellcheck disable=SC2029  # client-side expansion is intentional
    ssh "root@$SERVER_IP" "shutdown +$((TIMEOUT_HOURS * 60))" 2>/dev/null || true

    info "Writing .env on remote (secrets fetched from 1Password)"
    _bk=$(op read "$OP_BINANCE_API_KEY_REF" 2>/dev/null) || true
    _bs=$(op read "$OP_BINANCE_API_SECRET_REF" 2>/dev/null) || true
    # A failed/expired op session must never write empty credentials
    [[ -n "$_bk" && -n "$_bs" ]] || die "1Password returned empty Binance credentials (session expired?)"
    ssh "root@$SERVER_IP" "cat > /root/dojiwick/.env; chmod 600 /root/dojiwick/.env" <<EOF
BINANCE_API_KEY=$_bk
BINANCE_API_SECRET=$_bs
EOF
    unset _bk _bs

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
        echo "  Config:    $CONFIG_PATH (DSN rewritten @postgres:<port> → @localhost:5432)"
        echo "  Command:   dojiwick optimize --config config.toml --start $START --end $END --workers $WORKERS $([ "$GATE" == true ] && echo '--gate')"
        echo "  Tunnel:    remote 127.0.0.1:5432 → local localhost:$LOCAL_PG_PORT"
        echo "  Timeout:   ${TIMEOUT_HOURS}h"
        echo "  Deploy key: $OP_DEPLOY_KEY_REF (1Password, deleted from server after clone)"
        echo ""
        echo "No resources created."
        exit 0
    fi

    trap cleanup EXIT

    generate_host_key
    create_firewall
    create_server
    wait_ready
    setup_project
    upload_config
    run_optimization

    info "Optimization complete"
}

main "$@"
