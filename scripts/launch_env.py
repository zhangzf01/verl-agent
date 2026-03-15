"""Launch VisualWebArena / WebArena Docker containers for PoisonClaw.

Usage:
    # Start VisualWebArena on port 9999
    python scripts/launch_env.py --env visualwebarena --port 9999

    # Start WebArena on port 8080
    python scripts/launch_env.py --env webarena --port 8080

    # Stop all running PoisonClaw containers
    python scripts/launch_env.py --stop_all
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("poisonclaw.launch_env")

# Docker image references for each environment
_DOCKER_IMAGES = {
    "visualwebarena": {
        "image": "ghcr.io/web-arena-x/visualwebarena:latest",
        "services": {
            "reddit": 9999,
            "shopping": 7770,
            "classifieds": 9980,
        },
        "docs": "https://github.com/web-arena-x/visualwebarena",
    },
    "webarena": {
        "image": "ghcr.io/web-arena-x/webarena:latest",
        "services": {
            "gitlab": 8023,
            "reddit": 9999,
            "shopping": 7770,
            "map": 3000,
        },
        "docs": "https://github.com/web-arena-x/webarena",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch web environment Docker containers")
    parser.add_argument(
        "--env",
        choices=list(_DOCKER_IMAGES.keys()),
        help="Environment to launch",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Host port for the main service",
    )
    parser.add_argument(
        "--stop_all",
        action="store_true",
        help="Stop all running PoisonClaw environment containers",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status of running containers",
    )
    return parser.parse_args()


def check_docker() -> bool:
    """Check if Docker is available.

    Returns:
        True if Docker is installed and running.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.error(
            "Docker is not available. Please install Docker to use web environments.\n"
            "For local development without Docker, use stub environments by not setting "
            "env.vwa_task_file in your config."
        )
        return False


def launch_env(env_name: str, port: int) -> None:
    """Launch the specified environment Docker container.

    Args:
        env_name: Environment name (``"visualwebarena"`` or ``"webarena"``).
        port: Host port for the main service.
    """
    if not check_docker():
        sys.exit(1)

    env_info = _DOCKER_IMAGES[env_name]
    image = env_info["image"]
    main_port = port or list(env_info["services"].values())[0]

    logger.info("Launching %s (image: %s) on port %d...", env_name, image, main_port)
    logger.info("Docs: %s", env_info["docs"])

    # Build port mapping args
    port_args = []
    services = env_info["services"]
    base_service_port = list(services.values())[0]
    for service, service_port in services.items():
        host_port = main_port + (service_port - base_service_port)
        port_args += ["-p", f"{host_port}:{service_port}"]

    cmd = [
        "docker", "run", "-d",
        "--name", f"poisonclaw-{env_name}",
        *port_args,
        image,
    ]
    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        container_id = result.stdout.strip()[:12]
        logger.info(
            "Container started: %s\n"
            "Access at: http://localhost:%d",
            container_id,
            main_port,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to start container: %s", exc.stderr)
        sys.exit(1)


def stop_all() -> None:
    """Stop all running PoisonClaw environment containers."""
    if not check_docker():
        return

    result = subprocess.run(
        ["docker", "ps", "--filter", "name=poisonclaw-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    containers = [c.strip() for c in result.stdout.splitlines() if c.strip()]

    if not containers:
        logger.info("No PoisonClaw containers running.")
        return

    for name in containers:
        subprocess.run(["docker", "stop", name], check=False)
        subprocess.run(["docker", "rm", name], check=False)
        logger.info("Stopped and removed: %s", name)


def show_status() -> None:
    """Show status of running PoisonClaw containers."""
    if not check_docker():
        return

    result = subprocess.run(
        [
            "docker", "ps",
            "--filter", "name=poisonclaw-",
            "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout or "No PoisonClaw containers running.")


def main() -> None:
    args = parse_args()

    if args.stop_all:
        stop_all()
    elif args.status:
        show_status()
    elif args.env:
        launch_env(args.env, args.port or 0)
    else:
        logger.error("Specify --env, --stop_all, or --status.")
        sys.exit(1)


if __name__ == "__main__":
    main()
