def int_to_bytes(x: int) -> bytes:
        return x.to_bytes(4, 'big')

def int_from_bytes(xbytes: bytes) -> int:
    return int.from_bytes(xbytes, 'big')