"""Private-network gate shared by the broker and the local deploy.

The exhibit is served from a fixed LAN address, so voice endpoints must accept
private-range clients while still refusing the public internet.  Networks and
host suffixes are configuration, never a single hard-coded address.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

# RFC1918 plus IPv6 unique-local. Loopback is handled separately.
DEFAULT_LAN_CIDRS = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def parse_networks(spec: tuple[str, ...] | list[str] | str) -> tuple[ipaddress._BaseNetwork, ...]:
    """Parse CIDRs, skipping malformed entries instead of failing the service."""
    if isinstance(spec, str):
        spec = tuple(p.strip() for p in spec.split(",") if p.strip())
    networks: list[ipaddress._BaseNetwork] = []
    for item in spec:
        try:
            networks.append(ipaddress.ip_network(str(item).strip(), strict=False))
        except ValueError:
            continue
    return tuple(networks)


def host_of(authority: str | None) -> str:
    """Host part of a Host header or origin authority (no port, no brackets)."""
    if not authority:
        return ""
    value = authority.strip()
    if "://" in value:
        value = urlsplit(value).hostname or ""
    elif value.startswith("["):
        value = value[1:].split("]", 1)[0]
    elif value.count(":") == 1:
        value = value.split(":", 1)[0]
    return value.lower().rstrip(".")


def is_lan_host(host: str, networks: tuple[ipaddress._BaseNetwork, ...]) -> bool:
    """True for loopback or an address inside the configured LAN networks."""
    if not host:
        return False
    if host in LOOPBACK:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in network for network in networks if network.version == address.version)


def host_allowed(host: str, networks: tuple[ipaddress._BaseNetwork, ...],
                 host_suffixes: tuple[str, ...] = ()) -> bool:
    """True for loopback/LAN addresses and any configured local host suffix."""
    if is_lan_host(host, networks):
        return True
    return any(host.endswith(suffix) for suffix in host_suffixes if suffix)


def origin_allowed(origin: str | None, networks: tuple[ipaddress._BaseNetwork, ...],
                   host_suffixes: tuple[str, ...] = ()) -> bool:
    """Browser origin check: http(s) from loopback/LAN IPs or configured suffixes."""
    if not origin:
        return True  # non-browser callers; budget and per-IP limits still apply
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    host = parts.hostname.lower().rstrip(".")
    return host_allowed(host, networks, host_suffixes)
