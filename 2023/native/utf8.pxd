from libc.stdint cimport uint32_t


# Adapted from CPython, licensed under PSF2 (BSD-like)
cdef inline int ucs4_to_utf8_json(uint32_t ucs4, char *utf8) nogil:
    if ucs4 == 0:
        return 0
    if ucs4 == b"\\" or ucs4 == b'"':
        utf8[0] = b"\\"
        utf8[1] = ucs4
        return 2
    if ucs4 < 0x20:
        # Escape control chars
        utf8[0] = b"\\"
        utf8[1] = b"u"
        utf8[2] = b"0"
        utf8[3] = b"0"
        utf8[4] = b"0" if ucs4 < 0x10 else b"1"
        ucs4 &= 0x0F
        if ucs4 > 0x09:
            utf8[5] = (ucs4 - 0x0A) + ord(b"A")
        else:
            utf8[5] = ucs4 + ord(b"0")
        return 6
    if ucs4 < 0x80:
        # Encode ASCII
        utf8[0] = ucs4
        return 1
    if ucs4 < 0x0800:
        # Encode Latin-1
        utf8[0] = 0xc0 | (ucs4 >> 6)
        utf8[1] = 0x80 | (ucs4 & 0x3f)
        return 2
    if 0xD800 <= ucs4 <= 0xDFFF:
        return 0
    if ucs4 < 0x10000:
        utf8[0] = 0xe0 | (ucs4 >> 12)
        utf8[1] = 0x80 | ((ucs4 >> 6) & 0x3f)
        utf8[2] = 0x80 | (ucs4 & 0x3f)
        return 3
    # Encode UCS4 Unicode ordinals
    utf8[0] = 0xf0 | (ucs4 >> 18)
    utf8[1] = 0x80 | ((ucs4 >> 12) & 0x3f)
    utf8[2] = 0x80 | ((ucs4 >> 6) & 0x3f)
    utf8[3] = 0x80 | (ucs4 & 0x3f)
    return 4
