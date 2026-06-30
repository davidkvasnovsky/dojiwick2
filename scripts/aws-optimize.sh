#!/usr/bin/env bash
# Remote Optuna optimization on AWS EC2 Graviton4 (c7g.metal).
#
# Provisions an ephemeral bare-metal ARM instance, runs `dojiwick optimize`
# with 64 workers, writes Optuna results back to local PostgreSQL via SSH
# reverse tunnel, and terminates the instance on exit (including Ctrl+C).
#
# One-time setup:
#   1. brew install awscli
#   2. Store AWS credentials in 1Password:
#      op item create --category "API Credential" --title "AWS CLI" --vault Personal \
#        'access_key_id=AKIA...' 'secret_access_key=...'
#   3. Import SSH public key to AWS:
#      aws ec2 import-key-pair --key-name dojiwick \
#        --public-key-material fileb://<(op read "op://Personal/Hetzner Server SSH Key/public key")
#   4. Ensure deploy key is set up (same as Hetzner script)
#
# Usage:
#   ./scripts/aws-optimize.sh --config config.toml --start 2019-09-08 --end 2026-03-18
#   ./scripts/aws-optimize.sh --config config.toml --start 2019-09-08 --end 2026-03-18 --gate
#   ./scripts/aws-optimize.sh --config config.toml --start 2019-09-08 --end 2026-03-18 --spot
#   ./scripts/aws-optimize.sh --config config.toml --start 2019-09-08 --end 2026-03-18 --dry-run

set -euo pipefail

# Defaults
CONFIG_PATH=""
START=""
END=""
GATE=false
SPOT=false
WORKERS=64
TIMEOUT_HOURS=4
LOCAL_PG_PORT=5432
INSTANCE_TYPE="c7g.metal"
AWS_REGION="eu-central-1"
AWS_KEY_NAME="dojiwick"
OP_DEPLOY_KEY_REF="op://Personal/Dojiwick Deploy Key/private key"
OP_BINANCE_API_KEY_REF="op://Personal/Binance API/api_key"
OP_BINANCE_API_SECRET_REF="op://Personal/Binance API/api_secret"
OP_AWS_ACCESS_KEY_REF="op://Personal/AWS CLI/access_key_id"
OP_AWS_SECRET_KEY_REF="op://Personal/AWS CLI/secret_access_key"
DRY_RUN=false

# Cached credentials (resolved once at startup)
_AWS_AK=""
_AWS_SK=""

# Resource tracking (used by cleanup)
INSTANCE_ID=""
SERVER_IP=""
SG_ID=""
RESOURCE_FILE=""

# Helpers
die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

# AWS CLI wrapper — uses cached credentials
awscli() {
    AWS_ACCESS_KEY_ID="$_AWS_AK" \
    AWS_SECRET_ACCESS_KEY="$_AWS_SK" \
    AWS_DEFAULT_REGION="$AWS_REGION" \
    aws "$@"
}

# parse_args
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)      CONFIG_PATH="$2"; shift 2 ;;
            --start)       START="$2"; shift 2 ;;
            --end)         END="$2"; shift 2 ;;
            --gate)        GATE=true; shift ;;
            --spot)        SPOT=true; shift ;;
            --workers)     WORKERS="$2"; shift 2 ;;
            --timeout)     TIMEOUT_HOURS="$2"; shift 2 ;;
            --pg-port)     LOCAL_PG_PORT="$2"; shift 2 ;;
            --instance)    INSTANCE_TYPE="$2"; shift 2 ;;
            --region)      AWS_REGION="$2"; shift 2 ;;
            --key-name)    AWS_KEY_NAME="$2"; shift 2 ;;
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
    [[ "$INSTANCE_TYPE" =~ ^[a-z0-9][a-z0-9.-]*$ ]]   || die "--instance must be a valid instance type"
    [[ "$LOCAL_PG_PORT" =~ ^[0-9]+$ ]]                 || die "--pg-port must be a number"
}

# resolve_credentials — cache 1Password secrets once at startup
resolve_credentials() {
    info "Resolving credentials from 1Password"
    _AWS_AK=$(op read "$OP_AWS_ACCESS_KEY_REF") \
        || die "Cannot read AWS access key from 1Password"
    _AWS_SK=$(op read "$OP_AWS_SECRET_KEY_REF") \
        || die "Cannot read AWS secret key from 1Password"
}

# preflight_checks
preflight_checks() {
    info "Running pre-flight checks"

    command -v aws >/dev/null 2>&1 || die "AWS CLI not found. Install: brew install awscli"
    command -v op >/dev/null 2>&1 || die "1Password CLI (op) not found. Install: brew install 1password-cli"

    resolve_credentials

    # Verify AWS credentials work
    awscli sts get-caller-identity >/dev/null 2>&1 \
        || die "AWS credentials invalid. Check 1Password 'AWS CLI' item."

    pg_isready -h localhost -p "$LOCAL_PG_PORT" >/dev/null 2>&1 \
        || die "PostgreSQL not accepting connections on localhost:$LOCAL_PG_PORT"

    [[ -f "$CONFIG_PATH" ]] || die "Config file not found: $CONFIG_PATH"

    op read "$OP_BINANCE_API_KEY_REF" >/dev/null 2>&1 \
        || die "Cannot read Binance API key from 1Password"
    op read "$OP_BINANCE_API_SECRET_REF" >/dev/null 2>&1 \
        || die "Cannot read Binance API secret from 1Password"
    op read "${OP_DEPLOY_KEY_REF}?ssh-format=openssh" >/dev/null 2>&1 \
        || die "Cannot read deploy key from 1Password: $OP_DEPLOY_KEY_REF"

    # Verify key pair exists in AWS
    awscli ec2 describe-key-pairs --key-names "$AWS_KEY_NAME" >/dev/null 2>&1 \
        || die "SSH key '$AWS_KEY_NAME' not found in AWS. Import with: aws ec2 import-key-pair"

    info "Pre-flight checks passed (region: $AWS_REGION, instance: $INSTANCE_TYPE)"
}

# detect_public_ip
detect_public_ip() {
    info "Detecting public IP"
    local ip
    ip=$(curl -sf https://api.ipify.org || curl -sf https://ifconfig.me || curl -sf https://checkip.amazonaws.com) \
        || die "Failed to detect public IP"
    ip=$(echo "$ip" | tr -d '[:space:]')
    [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "Invalid IP format: $ip"
    MY_IP="$ip"
    info "Public IP: $MY_IP"
}

# cleanup — runs on EXIT (including Ctrl+C)
cleanup() {
    echo ""
    info "Cleaning up AWS resources"

    local saved_instance_id="${INSTANCE_ID:-}"

    # Cancel remote shutdown timer (best-effort)
    if [[ -n "${SERVER_IP:-}" ]]; then
        ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=accept-new \
            "ubuntu@$SERVER_IP" 'sudo shutdown -c' 2>/dev/null || true
    fi

    # Terminate instance (stops billing)
    if [[ -n "$saved_instance_id" ]]; then
        for _attempt in 1 2 3; do
            if awscli ec2 terminate-instances --instance-ids "$saved_instance_id" >/dev/null 2>&1; then
                info "Instance '$saved_instance_id' terminated"
                INSTANCE_ID=""
                break
            fi
            sleep 2
        done
    fi

    # Wait for termination before deleting SG (SG can't be deleted while instance exists)
    if [[ -n "${SG_ID:-}" && -n "$saved_instance_id" ]]; then
        info "Waiting for instance termination before deleting security group..."
        awscli ec2 wait instance-terminated --instance-ids "$saved_instance_id" 2>/dev/null || true
        sleep 5
        for _attempt in 1 2 3; do
            if awscli ec2 delete-security-group --group-id "$SG_ID" 2>/dev/null; then
                info "Security group '$SG_ID' deleted"
                SG_ID=""
                break
            fi
            sleep 5
        done
    elif [[ -n "${SG_ID:-}" ]]; then
        # No instance was launched — delete SG immediately
        awscli ec2 delete-security-group --group-id "$SG_ID" 2>/dev/null \
            && info "Security group '$SG_ID' deleted" || true
        SG_ID=""
    fi

    rm -f "${RESOURCE_FILE:-}"
    info "Cleanup complete"
}

# create_security_group
create_security_group() {
    local sg_name="dojiwick-opt-$(date +%s)"
    info "Creating security group '$sg_name' (SSH from $MY_IP only)"

    SG_ID=$(awscli ec2 create-security-group \
        --group-name "$sg_name" \
        --description "Dojiwick optimization - ephemeral" \
        --query 'GroupId' --output text)

    awscli ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 \
        --cidr "$MY_IP/32" >/dev/null

    info "Security group: $SG_ID"
}

# find_ami — latest Ubuntu 24.04 ARM AMI
find_ami() {
    info "Finding latest Ubuntu 24.04 ARM AMI"
    AMI_ID=$(awscli ec2 describe-images \
        --owners 099720109477 \
        --filters \
            "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*" \
            "Name=state,Values=available" \
            "Name=architecture,Values=arm64" \
            "Name=root-device-type,Values=ebs" \
            "Name=virtualization-type,Values=hvm" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text)

    [[ "$AMI_ID" != "None" && -n "$AMI_ID" ]] || die "Could not find Ubuntu 24.04 ARM AMI"
    info "AMI: $AMI_ID"
}

# create_instance
create_instance() {
    # Guard against duplicate instances (billing leak)
    local existing
    existing=$(awscli ec2 describe-instances \
        --filters "Name=tag:Name,Values=dojiwick-opt" "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[].Instances[].InstanceId' --output text)
    if [[ -n "$existing" && "$existing" != "None" ]]; then
        die "Existing dojiwick instance(s) found: $existing. Terminate first."
    fi

    info "Launching $INSTANCE_TYPE instance"

    # Build cloud-init user data for hardening
    local userdata_file
    userdata_file=$(mktemp)
    cat > "$userdata_file" <<'CLOUDINIT'
#cloud-config
ssh_pwauth: false
write_files:
  - path: /etc/ssh/sshd_config.d/99-hardening.conf
    content: |
      MaxAuthTries 6
      PermitRootLogin no
      PasswordAuthentication no
  - path: /etc/sysctl.d/99-hardening.conf
    content: |
      net.ipv4.tcp_syncookies=1
runcmd:
  - sysctl --system
  - systemctl restart sshd
CLOUDINIT

    # Build spot options if requested
    local spot_opts=()
    if [[ "$SPOT" == true ]]; then
        spot_opts=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
        info "Using SPOT pricing"
    fi

    INSTANCE_ID=$(awscli ec2 run-instances \
        --instance-type "$INSTANCE_TYPE" \
        --image-id "$AMI_ID" \
        --key-name "$AWS_KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --tag-specifications \
            "ResourceType=instance,Tags=[{Key=Name,Value=dojiwick-opt},{Key=Purpose,Value=ephemeral-optimization}]" \
            "ResourceType=volume,Tags=[{Key=Name,Value=dojiwick-opt-root},{Key=Purpose,Value=ephemeral-optimization}]" \
        --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=1,HttpEndpoint=enabled,InstanceMetadataTags=enabled" \
        --instance-initiated-shutdown-behavior terminate \
        --maintenance-options "AutoRecovery=default" \
        --user-data "file://$userdata_file" \
        "${spot_opts[@]}" \
        --query 'Instances[0].InstanceId' --output text)
    rm "$userdata_file"

    info "Instance: $INSTANCE_ID (waiting for running state...)"
    awscli ec2 wait instance-running --instance-ids "$INSTANCE_ID"

    SERVER_IP=$(awscli ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

    [[ "$SERVER_IP" != "None" && -n "$SERVER_IP" ]] \
        || die "Instance has no public IP. Check default VPC subnet settings."

    info "Server IP: $SERVER_IP"

    # Write resource IDs for manual cleanup fallback
    RESOURCE_FILE="/tmp/dojiwick-aws-$(date +%s).txt"
    install -m 600 /dev/null "$RESOURCE_FILE"
    echo "instance=$INSTANCE_ID sg=$SG_ID ip=$SERVER_IP region=$AWS_REGION" > "$RESOURCE_FILE"
    info "Resource IDs saved to $RESOURCE_FILE"
}

# wait_ready
wait_ready() {
    ssh-keygen -R "$SERVER_IP" 2>/dev/null || true
    info "Waiting for SSH on $SERVER_IP (bare metal may take 5-10 min)..."
    local attempt=0
    until ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
        "ubuntu@$SERVER_IP" true 2>/dev/null; do
        attempt=$((attempt + 1))
        printf "  SSH attempt %d (%d×3s elapsed)...\r" "$attempt" "$attempt"
        sleep 3
        if [[ $attempt -ge 200 ]]; then
            echo ""
            info "SSH debug after 10min timeout:"
            ssh -v -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
                "ubuntu@$SERVER_IP" true 2>&1 | tail -5
            die "SSH connection failed after 200 attempts"
        fi
    done
    echo ""
    info "SSH connected after $attempt attempts"
    info "Waiting for cloud-init..."
    ssh -o StrictHostKeyChecking=accept-new "ubuntu@$SERVER_IP" 'cloud-init status --wait'
    info "Server ready"
}

# setup_project
setup_project() {
    local repo_slug
    repo_slug=$(git remote get-url origin | sed 's|.*github.com[:/]||; s|\.git$||')
    [[ "$repo_slug" =~ ^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$ ]] || die "Invalid repo slug: $repo_slug"

    info "Fetching deploy key from 1Password and sending to server"
    op read "${OP_DEPLOY_KEY_REF}?ssh-format=openssh" \
        | ssh -o StrictHostKeyChecking=accept-new "ubuntu@$SERVER_IP" \
            'mkdir -p ~/.ssh && cat > ~/.ssh/deploy_key && chmod 600 ~/.ssh/deploy_key && \
             cat >> ~/.ssh/config <<SSHEOF
Host github.com
    IdentityFile ~/.ssh/deploy_key
    StrictHostKeyChecking accept-new
SSHEOF'

    info "Installing uv, Python 3.14, cloning repo, syncing deps"
    # shellcheck disable=SC2029
    ssh "ubuntu@$SERVER_IP" "
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH=\"\$HOME/.local/bin:\$PATH\"
        uv python install 3.14
        git clone git@github.com:$repo_slug.git ~/dojiwick
        rm -f ~/.ssh/deploy_key ~/.ssh/config
        cd ~/dojiwick
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
    scp -q "$temp_config" "ubuntu@$SERVER_IP:~/dojiwick/config.toml"
    rm "$temp_config"
    ssh "ubuntu@$SERVER_IP" "chmod 600 ~/dojiwick/config.toml"
}

# run_optimization
run_optimization() {
    local gate_flag=""
    [[ "$GATE" == true ]] && gate_flag="--gate"

    info "Setting remote shutdown timer (${TIMEOUT_HOURS}h safety net)"
    # shellcheck disable=SC2029
    ssh "ubuntu@$SERVER_IP" "sudo shutdown +$((TIMEOUT_HOURS * 60))" 2>/dev/null || true

    info "Writing .env on remote (secrets fetched from 1Password)"
    {
        _bk=$(op read "$OP_BINANCE_API_KEY_REF")
        _bs=$(op read "$OP_BINANCE_API_SECRET_REF")
    } 2>/dev/null
    ssh "ubuntu@$SERVER_IP" "cat > ~/dojiwick/.env; chmod 600 ~/dojiwick/.env" <<EOF
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
        "ubuntu@$SERVER_IP" \
        "cd ~/dojiwick && \
         timeout --kill-after=60 ${TIMEOUT_HOURS}h \
         ~/.local/bin/uv run dojiwick optimize \
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
        echo "  Instance:  $INSTANCE_TYPE ($([[ $SPOT == true ]] && echo 'SPOT' || echo 'on-demand'))"
        echo "  Region:    $AWS_REGION"
        echo "  Key pair:  $AWS_KEY_NAME"
        echo "  Security:  SSH from $MY_IP/32 only"
        echo "  Config:    $CONFIG_PATH (DSN rewritten @postgres: → @localhost:)"
        echo "  Command:   dojiwick optimize --config config.toml --start $START --end $END --workers $WORKERS $([ "$GATE" == true ] && echo '--gate')"
        echo "  Tunnel:    remote 127.0.0.1:5432 → local localhost:$LOCAL_PG_PORT"
        echo "  Timeout:   ${TIMEOUT_HOURS}h"
        echo ""
        echo "No resources created."
        exit 0
    fi

    trap cleanup EXIT

    create_security_group
    find_ami
    create_instance
    wait_ready
    setup_project
    upload_config
    run_optimization

    info "Optimization complete"
}

main "$@"
