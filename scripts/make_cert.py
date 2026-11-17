"""Create a self-signed HTTPS certificate so phones on your Wi-Fi can use the camera and GPS.

Phone browsers only allow camera and location access on HTTPS pages (or localhost).

    python scripts/make_cert.py            # auto-detects this PC's LAN IP
    python scripts/make_cert.py 192.168.1.40

Writes certs/cert.pem and certs/key.pem (gitignored). The phone shows a one-time "not
private" warning because the certificate is self-signed. That is expected on your own network.
"""
from __future__ import annotations

import ipaddress
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

OUT = Path(__file__).resolve().parent.parent / "certs"


def lan_ip() -> str:
    # Opening a UDP socket picks the outbound interface without sending any packets.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]


def main() -> None:
    ips = sys.argv[1:] or [lan_ip()]
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "MoodBite local")])
    san = x509.SubjectAlternativeName(
        [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
        + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=365))
            .add_extension(san, critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    OUT.mkdir(exist_ok=True)
    (OUT / "key.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    (OUT / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"Wrote {OUT / 'cert.pem'} and key.pem for: localhost, 127.0.0.1, {', '.join(ips)}")
    for ip in ips:
        print(f"Open on your phone (same Wi-Fi): https://{ip}:8501")


if __name__ == "__main__":
    main()
