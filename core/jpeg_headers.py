"""Shared baseline-JPEG plumbing: bit reader, Huffman tables, header parsing.

Extracted from :mod:`core.jpeg` so that TWO decoders can sit on top of exactly
the same, already-proven parsing:

* :mod:`core.jpeg` -- DC-only, 1/8 scale, very cheap (the fallback).
* :mod:`core.jpeg_scaled` -- keeps low-frequency AC too, 1/2 scale, sharp.

Splitting this out rather than copying it was deliberate: header parsing is the
fiddly part (marker walking, 8/16-bit quant tables, sampling factors, restart
intervals), it already works on real generations, and a second hand-maintained
copy would drift from the first the moment either was touched.

Scope is unchanged: baseline sequential Huffman JPEG (SOF0/SOF1), 8-bit, 1-3
components. Anything else raises :class:`UnsupportedJPEG` so callers fall back
instead of rendering garbage.
"""
from __future__ import annotations

import struct

__all__ = [
    "UnsupportedJPEG", "BitReader", "JpegHeaders", "ZIGZAG",
    "build_huffman", "decode_symbol", "extend", "clamp", "parse_headers",
]

_MAX_COMPONENTS = 3

# Zig-zag order: coefficient k of the entropy-coded stream belongs at this
# (row, col) of the 8x8 block. The DC-only decoder never needed this -- it read
# coefficient 0 and skipped the rest -- but a decoder that KEEPS the low
# frequencies has to know where each one lands.
ZIGZAG = (
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
)


class UnsupportedJPEG(Exception):
    """The JPEG uses a feature this minimal decoder does not implement."""


class BitReader:
    """MSB-first bit reader over entropy-coded JPEG data.

    Handles byte stuffing (a literal 0xFF is encoded as ``FF 00``) and stops
    at any marker, which is how the end of a scan is detected.
    """

    def __init__(self, data: bytes, pos: int) -> None:
        self._data = data
        self._pos = pos
        self._bits = 0
        self._nbits = 0

    @property
    def pos(self) -> int:
        return self._pos

    def align(self) -> None:
        """Drop buffered bits (used at restart markers)."""
        self._bits = 0
        self._nbits = 0

    def skip_restart_marker(self) -> bool:
        """Consume an RSTn marker if one is next. Returns True if consumed."""
        d, p = self._data, self._pos
        if p + 1 < len(d) and d[p] == 0xFF and 0xD0 <= d[p + 1] <= 0xD7:
            self._pos = p + 2
            self.align()
            return True
        return False

    def bit(self) -> int:
        if self._nbits == 0:
            d = self._data
            p = self._pos
            if p >= len(d):
                # Ran past the data: feed zeros rather than raising, so a
                # truncated tail still yields a usable image.
                return 0
            byte = d[p]
            p += 1
            if byte == 0xFF:
                nxt = d[p] if p < len(d) else 0x00
                if nxt == 0x00:
                    p += 1          # stuffed FF
                else:
                    return 0        # hit a marker; pad with zeros
            self._pos = p
            self._bits = byte
            self._nbits = 8
        self._nbits -= 1
        return (self._bits >> self._nbits) & 1

    def bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bit()
        return v


def build_huffman(counts: bytes, symbols: bytes) -> dict[tuple[int, int], int]:
    """Map (code_length, code) -> symbol, per the JPEG canonical code order."""
    table: dict[tuple[int, int], int] = {}
    code = 0
    k = 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            table[(length, code)] = symbols[k]
            code += 1
            k += 1
        code <<= 1
    return table


def decode_symbol(br: BitReader, table: dict[tuple[int, int], int]) -> int:
    code = 0
    for length in range(1, 17):
        code = (code << 1) | br.bit()
        sym = table.get((length, code))
        if sym is not None:
            return sym
    raise UnsupportedJPEG("bad Huffman code in scan data")


def extend(value: int, nbits: int) -> int:
    """JPEG's signed-value EXTEND: turn a raw magnitude into a signed diff."""
    if nbits == 0:
        return 0
    return value if value >= (1 << (nbits - 1)) else value - (1 << nbits) + 1


def clamp(v: float) -> int:
    return 0 if v < 0 else (255 if v > 255 else int(v))


class JpegHeaders:
    """Everything the entropy decoder needs, read out of the marker segments."""

    __slots__ = (
        "quant", "huff_dc", "huff_ac", "comps", "width", "height",
        "restart_interval", "scan_pos", "scan_comps",
    )

    def __init__(self) -> None:
        self.quant: dict[int, list[int]] = {}
        self.huff_dc: dict[int, dict[tuple[int, int], int]] = {}
        self.huff_ac: dict[int, dict[tuple[int, int], int]] = {}
        self.comps: list[dict] = []
        self.width = 0
        self.height = 0
        self.restart_interval = 0
        self.scan_pos = -1
        self.scan_comps: list[tuple[int, int, int]] = []

    @property
    def hmax(self) -> int:
        return max(c["h"] for c in self.comps)

    @property
    def vmax(self) -> int:
        return max(c["v"] for c in self.comps)

    def mcu_grid(self) -> tuple[int, int]:
        """MCUs across and down, i.e. how many the scan actually contains."""
        return (
            -(-self.width // (8 * self.hmax)),
            -(-self.height // (8 * self.vmax)),
        )

    def quant_for(self, comp: dict) -> list[int]:
        q = self.quant.get(comp["tq"])
        if q is None:
            raise UnsupportedJPEG("scan references a missing quantisation table")
        return q

    def tables_for(self, comp: dict):
        dc = self.huff_dc.get(comp["dc_tbl"])
        ac = self.huff_ac.get(comp["ac_tbl"])
        if dc is None or ac is None:
            raise UnsupportedJPEG("scan references a missing Huffman table")
        return dc, ac


def parse_headers(data: bytes) -> JpegHeaders:
    """Walk the marker segments up to (and including) the start of scan."""
    if data[:2] != b"\xff\xd8":
        raise UnsupportedJPEG("not a JPEG (no SOI marker)")

    h = JpegHeaders()
    i = 2
    n = len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9:  # EOI
            break
        if i + 4 > n:
            break
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        body = data[i + 4:i + 2 + seg_len]

        if marker == 0xDB:                                  # DQT
            p = 0
            while p < len(body):
                pq_tq = body[p]; p += 1
                precision, tq = pq_tq >> 4, pq_tq & 0x0F
                if precision:                               # 16-bit table
                    tbl = list(struct.unpack(">64H", body[p:p + 128])); p += 128
                else:
                    tbl = list(body[p:p + 64]); p += 64
                h.quant[tq] = tbl
        elif marker in (0xC0, 0xC1):                        # SOF0/SOF1
            precision = body[0]
            if precision != 8:
                raise UnsupportedJPEG(f"{precision}-bit samples unsupported")
            h.height, h.width = struct.unpack(">HH", body[1:5])
            ncomp = body[5]
            if not 1 <= ncomp <= _MAX_COMPONENTS:
                raise UnsupportedJPEG(f"{ncomp} components unsupported")
            for c in range(ncomp):
                cid, samp, tq = body[6 + c * 3:9 + c * 3]
                h.comps.append({
                    "id": cid, "h": samp >> 4 or 1, "v": samp & 0x0F or 1,
                    "tq": tq,
                })
        elif marker in (0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB,
                        0xCD, 0xCE, 0xCF):
            raise UnsupportedJPEG("only baseline Huffman JPEG is supported")
        elif marker == 0xC4:                                # DHT
            p = 0
            while p < len(body):
                tc_th = body[p]; p += 1
                counts = body[p:p + 16]; p += 16
                total = sum(counts)
                symbols = body[p:p + total]; p += total
                tbl = build_huffman(counts, symbols)
                (h.huff_dc if (tc_th >> 4) == 0 else h.huff_ac)[tc_th & 0x0F] = tbl
        elif marker == 0xDD:                                # DRI
            h.restart_interval = struct.unpack(">H", body[:2])[0]
        elif marker == 0xDA:                                # SOS
            ns = body[0]
            for c in range(ns):
                cs, tables = body[1 + c * 2:3 + c * 2]
                h.scan_comps.append((cs, tables >> 4, tables & 0x0F))
            h.scan_pos = i + 2 + seg_len
            break
        i += 2 + seg_len

    if h.scan_pos < 0 or not h.comps or not h.width or not h.height:
        raise UnsupportedJPEG("incomplete JPEG: no baseline scan found")
    if len(h.scan_comps) != len(h.comps):
        raise UnsupportedJPEG("multi-scan (progressive-style) JPEG unsupported")

    for c in h.comps:
        sel = next((s for s in h.scan_comps if s[0] == c["id"]), None)
        if sel is None:
            raise UnsupportedJPEG("scan does not cover every component")
        c["dc_tbl"], c["ac_tbl"] = sel[1], sel[2]

    return h
