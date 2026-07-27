"""Pure-stdlib baseline JPEG -> 1/8-scale thumbnail, with no inverse DCT.

The image models return JPEG (verified on a live generation: the bytes start
``ff d8``), so the PNG-only shrink path in :mod:`core.png` never fired for a
real render and the panel kept inlining full-size payloads.

Decoding JPEG "properly" needs an inverse DCT -- a lot to hand-roll. This
exploits a property of the format instead: each 8x8 block's DC coefficient IS
that block's average value, so reading DC alone yields the image at exactly
1/8 scale for the cost of Huffman decoding. No IDCT, no third-party library,
64x fewer pixels -- and 1/8 of a render is about what a preview wants anyway.

Scope: baseline sequential DCT (SOF0/SOF1), Huffman, 8-bit, 1-3 components,
restart intervals -- what these models emit. Progressive (SOF2) and arithmetic
coding raise :class:`UnsupportedJPEG` so callers fall back instead of
receiving garbage.
"""
from __future__ import annotations

import struct

__all__ = ["UnsupportedJPEG", "decode_dc_thumbnail"]

# Zig-zag position 0 is the DC coefficient; the other 63 are AC. Only their
# lengths matter here (to advance the bitstream), never their values.
_MAX_COMPONENTS = 3


class UnsupportedJPEG(Exception):
    """The JPEG uses a feature this minimal decoder does not implement."""


class _BitReader:
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
                # truncated tail still yields a usable thumbnail.
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


def _build_huffman(counts: bytes, symbols: bytes) -> dict[tuple[int, int], int]:
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


def _decode_symbol(br: _BitReader, table: dict[tuple[int, int], int]) -> int:
    code = 0
    for length in range(1, 17):
        code = (code << 1) | br.bit()
        sym = table.get((length, code))
        if sym is not None:
            return sym
    raise UnsupportedJPEG("bad Huffman code in scan data")


def _extend(value: int, nbits: int) -> int:
    """JPEG's signed-value EXTEND: turn a raw magnitude into a signed diff."""
    if nbits == 0:
        return 0
    return value if value >= (1 << (nbits - 1)) else value - (1 << nbits) + 1


def _clamp(v: float) -> int:
    return 0 if v < 0 else (255 if v > 255 else int(v))


def decode_dc_thumbnail(data: bytes) -> tuple[list[bytes], int, int]:
    """Decode a baseline JPEG to a 1/8-scale RGB thumbnail.

    Returns ``(rows, width, height)`` where each row is ``width * 3`` bytes,
    matching :func:`core.png.encode_rgb`'s input so the two compose directly.

    Raises :class:`UnsupportedJPEG` for anything outside baseline Huffman
    JPEG, so the caller can fall back rather than display nonsense.
    """
    if data[:2] != b"\xff\xd8":
        raise UnsupportedJPEG("not a JPEG (no SOI marker)")

    quant: dict[int, list[int]] = {}
    huff_dc: dict[int, dict[tuple[int, int], int]] = {}
    huff_ac: dict[int, dict[tuple[int, int], int]] = {}
    comps: list[dict] = []
    width = height = 0
    restart_interval = 0
    scan_pos = -1
    scan_comps: list[tuple[int, int, int]] = []

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
                quant[tq] = tbl
        elif marker in (0xC0, 0xC1):                        # SOF0/SOF1
            precision = body[0]
            if precision != 8:
                raise UnsupportedJPEG(f"{precision}-bit samples unsupported")
            height, width = struct.unpack(">HH", body[1:5])
            ncomp = body[5]
            if not 1 <= ncomp <= _MAX_COMPONENTS:
                raise UnsupportedJPEG(f"{ncomp} components unsupported")
            for c in range(ncomp):
                cid, samp, tq = body[6 + c * 3:9 + c * 3]
                comps.append({
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
                tbl = _build_huffman(counts, symbols)
                (huff_dc if (tc_th >> 4) == 0 else huff_ac)[tc_th & 0x0F] = tbl
        elif marker == 0xDD:                                # DRI
            restart_interval = struct.unpack(">H", body[:2])[0]
        elif marker == 0xDA:                                # SOS
            ns = body[0]
            for c in range(ns):
                cs, tables = body[1 + c * 2:3 + c * 2]
                scan_comps.append((cs, tables >> 4, tables & 0x0F))
            scan_pos = i + 2 + seg_len
            break
        i += 2 + seg_len

    if scan_pos < 0 or not comps or not width or not height:
        raise UnsupportedJPEG("incomplete JPEG: no baseline scan found")
    if len(scan_comps) != len(comps):
        raise UnsupportedJPEG("multi-scan (progressive-style) JPEG unsupported")

    hmax = max(c["h"] for c in comps)
    vmax = max(c["v"] for c in comps)
    mcus_x = -(-width // (8 * hmax))
    mcus_y = -(-height // (8 * vmax))

    for c in comps:
        c["bw"] = mcus_x * c["h"]
        c["bh"] = mcus_y * c["v"]
        c["dc"] = [[0] * c["bw"] for _ in range(c["bh"])]
        c["pred"] = 0
        sel = next((s for s in scan_comps if s[0] == c["id"]), None)
        if sel is None:
            raise UnsupportedJPEG("scan does not cover every component")
        c["dc_tbl"], c["ac_tbl"] = sel[1], sel[2]

    br = _BitReader(data, scan_pos)
    mcu_count = 0
    for my in range(mcus_y):
        for mx in range(mcus_x):
            if restart_interval and mcu_count and mcu_count % restart_interval == 0:
                if br.skip_restart_marker():
                    for c in comps:
                        c["pred"] = 0
            for c in comps:
                dc_tbl = huff_dc.get(c["dc_tbl"])
                ac_tbl = huff_ac.get(c["ac_tbl"])
                if dc_tbl is None or ac_tbl is None:
                    raise UnsupportedJPEG("scan references a missing Huffman table")
                for by in range(c["v"]):
                    for bx in range(c["h"]):
                        t = _decode_symbol(br, dc_tbl)
                        c["pred"] += _extend(br.bits(t), t)
                        c["dc"][my * c["v"] + by][mx * c["h"] + bx] = c["pred"]
                        # Walk the 63 AC coefficients to advance the bitstream.
                        k = 1
                        while k < 64:
                            rs = _decode_symbol(br, ac_tbl)
                            run, size = rs >> 4, rs & 0x0F
                            if size == 0:
                                if run != 15:
                                    break       # EOB
                                k += 16
                            else:
                                k += run + 1
                                br.bits(size)
            mcu_count += 1

    out_w, out_h = -(-width // 8), -(-height // 8)
    grid_w, grid_h = mcus_x * hmax, mcus_y * vmax
    out_w, out_h = min(out_w, grid_w) or 1, min(out_h, grid_h) or 1

    y_c = comps[0]
    q_y = quant.get(y_c["tq"])
    if q_y is None:
        raise UnsupportedJPEG("scan references a missing quantisation table")
    chroma = comps[1:] if len(comps) == 3 else []
    q_ch = [quant.get(c["tq"]) for c in chroma]
    if any(q is None for q in q_ch):
        raise UnsupportedJPEG("scan references a missing quantisation table")

    rows: list[bytes] = []
    for oy in range(out_h):
        row = bytearray()
        for ox in range(out_w):
            # DC * quant / 8 recovers the block mean; +128 undoes the level shift.
            yy = y_c["dc"][min(oy * y_c["v"] // vmax, y_c["bh"] - 1)][
                min(ox * y_c["h"] // hmax, y_c["bw"] - 1)] * q_y[0] / 8.0 + 128.0
            if not chroma:
                v = _clamp(yy)
                row += bytes((v, v, v))
                continue
            cb_c, cr_c = chroma[0], chroma[1]
            cb = cb_c["dc"][min(oy * cb_c["v"] // vmax, cb_c["bh"] - 1)][
                min(ox * cb_c["h"] // hmax, cb_c["bw"] - 1)] * q_ch[0][0] / 8.0
            cr = cr_c["dc"][min(oy * cr_c["v"] // vmax, cr_c["bh"] - 1)][
                min(ox * cr_c["h"] // hmax, cr_c["bw"] - 1)] * q_ch[1][0] / 8.0
            row += bytes((
                _clamp(yy + 1.402 * cr),
                _clamp(yy - 0.344136 * cb - 0.714136 * cr),
                _clamp(yy + 1.772 * cb),
            ))
        rows.append(bytes(row))
    return rows, out_w, out_h
