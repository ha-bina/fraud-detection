# src/ip_converter.py

def ip_to_int(ip: str) -> int:
    """
    Convert dotted IPv4 address to integer.
    """
    parts = list(map(int, ip.split('.')))
    return (16777216 * parts[0]) + (65536 * parts[1]) + (256 * parts[2]) + parts[3]
