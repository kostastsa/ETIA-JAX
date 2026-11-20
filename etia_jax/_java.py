"""Small Java helpers for the emulation of Tetrad's iteration orders."""


def java_string_hash(s):
    """java.lang.String.hashCode() as an unsigned 32-bit int."""
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h


def table_size_for(c):
    """java.util.HashMap.tableSizeFor: the smallest power of two >= c."""
    n = 1
    while n < c:
        n *= 2
    return n
