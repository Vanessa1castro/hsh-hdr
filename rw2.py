import struct, numpy as np

def _ifd(d, off):
    n = struct.unpack('<H', d[off:off+2])[0]; tags = {}
    for k in range(n):
        e = d[off+2+12*k: off+14+12*k]
        tag, typ, cnt, val = struct.unpack('<HHII', e)
        unit = {1:1,2:1,3:2,4:4,5:8,7:1,9:4,10:8}.get(typ, 1); sz = unit*cnt
        raw = e[8:8+sz] if sz <= 4 else d[val:val+sz]
        if typ == 3: v = struct.unpack('<'+'H'*cnt, raw)
        elif typ == 4: v = struct.unpack('<'+'I'*cnt, raw)
        elif typ == 5: v = [a/b if b else 0 for a,b in zip(*[iter(struct.unpack('<'+'I'*2*cnt, raw))]*2)]
        elif typ == 10: v = [a/b if b else 0 for a,b in zip(*[iter(struct.unpack('<'+'i'*2*cnt, raw))]*2)]
        elif typ == 2: v = raw.split(b'\0')[0].decode(errors='ignore')
        else: v = raw
        tags[tag] = v[0] if isinstance(v, (tuple, list)) and len(v) == 1 else v
    return tags

def read_meta(path):
    d = open(path, 'rb').read()
    t = _ifd(d, struct.unpack('<I', d[4:8])[0])
    ex = _ifd(d, t[0x8769])
    return d, t, ex

def decode(path):
    d, t, ex = read_meta(path)
    W, H = t[0x2], t[0x3]
    assert t[0xb] == 34316 and t[0x2d] == 6 and t[0xa] == 14, (t[0xb], t[0x2d], t[0xa])
    off = t[0x118]; ppb = 11; bpr = W // ppb; rb = bpr * 16
    buf = np.frombuffer(d, np.uint8, rb*H, off).reshape(H, bpr, 16).astype(np.uint32)
    b = [buf[..., i] for i in range(16)]
    pb = [None]*14
    pb[0] = (b[15] << 6) | (b[14] >> 2)
    pb[1] = (((b[14] & 3) << 12) | (b[13] << 4) | (b[12] >> 4)) & 0x3fff
    pb[2] = (b[12] >> 2) & 3
    pb[3] = ((b[12] & 3) << 8) | b[11]
    pb[4] = (b[10] << 2) | (b[9] >> 6)
    pb[5] = ((b[9] & 0x3f) << 4) | (b[8] >> 4)
    pb[6] = (b[8] >> 2) & 3
    pb[7] = ((b[8] & 3) << 8) | b[7]
    pb[8] = ((b[6] << 2) & 0x3fc) | (b[5] >> 6)
    pb[9] = ((b[5] << 4) | (b[4] >> 4)) & 0x3ff
    pb[10] = (b[4] >> 2) & 3
    pb[11] = ((b[4] & 3) << 8) | b[3]
    pb[12] = (((b[2] << 2) & 0x3fc) | (b[1] >> 6)) & 0x3ff
    pb[13] = ((b[1] << 4) | (b[0] >> 4)) & 0x3ff
    out = np.zeros((H, bpr, ppb), np.int64)
    z = np.zeros((H, bpr), np.int64)
    oddeven = [z.copy(), z.copy()]; nonzero = [z.copy(), z.copy()]
    pmul = z.copy(); pbase = z.copy(); k = 0
    for pix in range(ppb):
        if pix % 3 == 2:
            base = pb[k].astype(np.int64); k += 1
            base = np.where(base == 3, 4, base)
            pbase = 0x200 << base; pmul = 1 << base
        ep = pb[k].astype(np.int64); k += 1
        p = pix % 2
        has = oddeven[p] != 0
        # branch A: oddeven already set
        epA = ep * pmul
        add = (pbase < 0x2000) & (nonzero[p] > pbase)
        epA = np.where(add, epA + nonzero[p] - pbase, epA)
        # branch B: first value
        epB = np.where(ep != 0, ep, nonzero[p])
        newodd = np.where(has, oddeven[p], ep)
        newnz = np.where(has, epA, np.where(ep != 0, ep, nonzero[p]))
        e = np.where(has, epA, epB)
        oddeven[p] = newodd; nonzero[p] = newnz
        s = e - 0xf
        out[..., pix] = np.where((s >= 0) & (s <= 0xffff), s, np.where(e + 0x7ffffff1 < 0, 0x3fff, 0) * 0 + 0)
    raw = out.reshape(H, bpr*ppb)
    raw = np.clip(raw, 0, 16383).astype(np.uint16)
    meta = dict(W=W, H=H, top=t[0x4], left=t[0x5], bottom=t[0x6], right=t[0x7],
                cfa=t[0x9], black=(t[0x1c], t[0x1d], t[0x1e]), white=16383,
                wb=(t[0x24], t[0x25], t[0x26]),
                exposure=ex.get(0x829a), fnum=ex.get(0x829d), iso=ex.get(0x8827),
                bias=ex.get(0x9204), time=ex.get(0x9003), model=t[0x110])
    return raw, meta
