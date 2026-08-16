"""Shared VP8L bitstream plumbing: bit writer, Huffman codes, image-data block.

Extracted from :mod:`core.webp` for the same reason :mod:`core.jpeg_headers`
was extracted from :mod:`core.jpeg`: the deploy validator enforces a 300-line
cap per file (see ``tests/panel_helpers.py``), and this is the fiddly,
already-verified part -- the exact bit-order rules that cost real debugging
time to get right (see :class:`_BitWriter`) live here ONCE, so a future
change to the predictor transform in :mod:`core.webp` cannot silently drift
from them.

Everything here was cross-validated pixel-for-pixel against Google's own
``dwebp`` reference decoder (see ``tests/test_webp_crossvalidate.py`` and
:mod:`core.webp`'s module docstring for the full story, including the
Huffman bit-order bug this design is built to prevent recurring).
"""
from __future__ import annotations

__all__ = [
    "UnsupportedForWebP", "BitWriter",
    "build_canonical_huffman", "encode_image_data",
]

# Order the "normal" code-length code stores its 19 secondary code lengths
# in -- fixed by the spec (kCodeLengthCodeOrder), not a design choice here.
_CODE_LENGTH_CODE_ORDER = (
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
)


class UnsupportedForWebP(Exception):
    """Raised for inputs outside this encoder's narrow, verified scope."""


class BitWriter:
    """Accumulates the VP8L bitstream. Two DIFFERENT bit orders matter here.

    Plain numeric fields (widths, flags, code-length values) are unpacked by
    a decoder's ``ReadBits(n)`` least-significant-bit-first -- use
    :meth:`write` for those. A canonical Huffman CODE is walked bit-by-bit as
    a binary tree, where the first stream bit is the code's most significant
    bit -- use :meth:`write_code` for those instead. Conflating the two
    produces a bitstream that LOOKS well-formed (same length, same chunk
    headers) but decodes to wrong symbols, or fails outright once real
    variable-length codes are in play -- this is what actually happened
    during development, twice: once for pixel data (masked by uniform test
    images, where bit order does not matter because every code has the same
    length) and once more when a meta-prefix bit was accidentally added to
    the wrong of the two spec-7.3 image-data grammar productions.
    """

    __slots__ = ("_bits",)

    def __init__(self) -> None:
        self._bits: list[int] = []

    def write(self, value: int, nbits: int) -> None:
        for i in range(nbits):
            self._bits.append((value >> i) & 1)

    def write_code(self, code: int, length: int) -> None:
        for i in range(length - 1, -1, -1):
            self._bits.append((code >> i) & 1)

    def to_bytes(self) -> bytes:
        bits = self._bits
        pad = (-len(bits)) % 8
        if pad:
            bits = bits + [0] * pad
        out = bytearray(len(bits) // 8)
        for i, bit in enumerate(bits):
            if bit:
                out[i >> 3] |= 1 << (i & 7)
        return bytes(out)


def build_canonical_huffman(freqs: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Return ``{symbol: (code, bit_length)}``, a canonical Huffman code.

    Package-merge / length-limiting is not implemented: alphabets here are
    at most 280 symbols (the green+length channel) over previews of a few
    hundred thousand pixels, which does not produce codes anywhere near the
    15-bit VP8L limit in practice. A pathological input hitting that ceiling
    raises :class:`UnsupportedForWebP` instead of emitting an invalid code.
    """
    import heapq

    used = [s for s, f in freqs.items() if f > 0]
    if not used:
        return {}
    if len(used) == 1:
        return {used[0]: (0, 0)}  # single-symbol code: zero bits, spec special-case

    heap = [[f, [[s, 0]]] for s, f in freqs.items() if f > 0]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1]:
            pair[1] += 1
        for pair in hi[1]:
            pair[1] += 1
        heapq.heappush(heap, [lo[0] + hi[0], lo[1] + hi[1]])
    lengths = {s: l for s, l in heap[0][1]}

    max_len = max(lengths.values())
    if max_len > 15:
        raise UnsupportedForWebP(f"Huffman code length {max_len} exceeds VP8L's 15-bit limit")

    bl_count = [0] * (max_len + 1)
    for l in lengths.values():
        bl_count[l] += 1
    next_code = [0] * (max_len + 2)
    code = 0
    for bits in range(1, max_len + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code

    codes: dict[int, tuple[int, int]] = {}
    for s in sorted(lengths):
        l = lengths[s]
        codes[s] = (next_code[l], l)
        next_code[l] += 1
    return codes


def _write_huffman_code_simple(bw: BitWriter, symbol: int) -> None:
    """Simple code length code (spec 5.2.2): exactly one symbol in use."""
    bw.write(1, 1)  # is_simple = 1
    bw.write(0, 1)  # num_symbols - 1 == 0 (one symbol)
    if symbol <= 1:
        bw.write(0, 1)
        bw.write(symbol, 1)
    else:
        bw.write(1, 1)
        bw.write(symbol, 8)


def _write_huffman_code_normal(bw: BitWriter, code_lengths: list[int]) -> None:
    """Normal code length code (spec 5.2.2): a secondary Huffman code over
    the 19 code-length symbols, used to transmit the primary code's lengths.

    RLE symbols 16/17/18 (repeat-previous / repeat-zero) are not emitted --
    every length is written as a literal. This costs a little bitstream
    size on long runs of equal/zero lengths; it is never a correctness gap,
    since literal-only is always a legal encoding of the same lengths.
    """
    bw.write(0, 1)  # is_simple = 0
    freqs = {i: 0 for i in range(19)}
    for l in code_lengths:
        freqs[l] += 1
    huff = build_canonical_huffman(freqs)

    order_lengths = [huff.get(sym, (0, 0))[1] for sym in _CODE_LENGTH_CODE_ORDER]
    num_code_lengths = 19
    while num_code_lengths > 4 and order_lengths[num_code_lengths - 1] == 0:
        num_code_lengths -= 1
    bw.write(num_code_lengths - 4, 4)
    for i in range(num_code_lengths):
        bw.write(order_lengths[i], 3)

    bw.write(0, 1)  # no max_symbol truncation -- decode the full alphabet
    for l in code_lengths:
        code, length = huff[l]
        bw.write_code(code, length)


def _emit_channel_code(bw: BitWriter, freqs: dict[int, int], alphabet_size: int) -> dict[int, tuple[int, int]]:
    """Write one channel's Huffman code header and return it for reuse below."""
    used = [s for s in range(alphabet_size) if freqs[s] > 0]
    if len(used) <= 1:
        symbol = used[0] if used else 0
        _write_huffman_code_simple(bw, symbol)
        return {symbol: (0, 0)}
    huff = build_canonical_huffman(freqs)
    lengths = [huff.get(s, (0, 0))[1] for s in range(alphabet_size)]
    _write_huffman_code_normal(bw, lengths)
    return huff


def encode_image_data(
    bw: BitWriter, pixels_rgba: list[tuple[int, int, int, int]], *, spatial: bool,
) -> None:
    """Write one VP8L image stream: five Huffman code headers (green+length,
    red, blue, alpha, distance) followed by the literal pixel stream. No
    LZ77 matches or colour cache are ever emitted, so every pixel is a plain
    4-symbol literal -- see :mod:`core.webp`'s module scope note.

    ``spatial`` selects which of the two spec-7.3 grammar productions this
    call site needs -- they are NOT interchangeable, and conflating them was
    a real bug here:

    * ``spatially-coded-image = color-cache-info meta-prefix data`` -- the
      main image (``spatial=True``): color-cache-info bit, THEN a
      meta-prefix bit, THEN the codes+pixels.
    * ``entropy-coded-image = color-cache-info data`` -- the tiny
      predictor-mode subimage (``spatial=False``): color-cache-info bit,
      THEN the codes+pixels directly, with NO meta-prefix bit at all.
    """
    bw.write(0, 1)  # color-cache-info: not used, own bit per spec 7.3
    if spatial:
        bw.write(0, 1)  # meta-prefix: no entropy image, one code group total

    green_freqs = {i: 0 for i in range(256 + 24)}  # +24 backward-ref length codes, unused
    red_freqs = {i: 0 for i in range(256)}
    blue_freqs = {i: 0 for i in range(256)}
    alpha_freqs = {i: 0 for i in range(256)}
    for r, g, b, a in pixels_rgba:
        green_freqs[g] += 1
        red_freqs[r] += 1
        blue_freqs[b] += 1
        alpha_freqs[a] += 1

    green_huff = _emit_channel_code(bw, green_freqs, 256 + 24)
    red_huff = _emit_channel_code(bw, red_freqs, 256)
    blue_huff = _emit_channel_code(bw, blue_freqs, 256)
    alpha_huff = _emit_channel_code(bw, alpha_freqs, 256)
    _write_huffman_code_simple(bw, 0)  # distance code: never used (no backward refs)

    for r, g, b, a in pixels_rgba:
        code, length = green_huff[g]
        bw.write_code(code, length)
        code, length = red_huff[r]
        bw.write_code(code, length)
        code, length = blue_huff[b]
        bw.write_code(code, length)
        code, length = alpha_huff[a]
        bw.write_code(code, length)
