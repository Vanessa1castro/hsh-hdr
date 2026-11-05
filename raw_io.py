"""Reading RAW files: metadata (for grouping brackets) and linear image data.

Panasonic RW2 files from the Lumix S1 (14-bit, "v6" compression) are decoded by
our own decoder in rw2.py. Every other camera (Canon CR2/CR3, Nikon NEF, Sony ARW,
Fujifilm RAF, DNG, other Panasonic models...) goes through LibRaw via `rawpy`.
"""
import datetime
import os
import struct

import cv2
import numpy as np

from . import rw2

RAW_EXTS = {'.rw2', '.cr2', '.cr3', '.nef', '.nrw', '.arw', '.srf', '.sr2', '.raf',
            '.orf', '.dng', '.pef', '.rwl', '.3fr', '.iiq', '.erf', '.mef', '.mos', '.srw', '.x3f'}

# sRGB(D65) -> XYZ
_XYZ_RGB = np.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]])
# Adobe-style XYZ->camera matrix for the Panasonic DC-S1 (x10000)
_S1_CAM_XYZ = [9744, -3905, -779, -4899, 12807, 2324, -798, 1630, 5827]


def _cam_to_srgb(coeffs):
    cam_xyz = np.array(coeffs, float).reshape(3, 3) / 10000
    cam_rgb = cam_xyz @ _XYZ_RGB
    cam_rgb = cam_rgb / cam_rgb.sum(1, keepdims=True)
    return np.linalg.inv(cam_rgb).T.astype(np.float32)  # use as: rgb_row @ M


_M_S1 = _cam_to_srgb(_S1_CAM_XYZ)


def is_raw(name):
    return os.path.splitext(name)[1].lower() in RAW_EXTS


# ---------------------------------------------------------------- metadata --

def _tiff_tags(d, off, endian):
    """Minimal TIFF IFD reader (both byte orders). Returns {tag: value}."""
    tags = {}
    try:
        n = struct.unpack(endian + 'H', d[off:off + 2])[0]
    except struct.error:
        return tags
    if n > 1000:
        return tags
    for k in range(n):
        e = d[off + 2 + 12 * k: off + 14 + 12 * k]
        if len(e) < 12:
            break
        tag, typ, cnt = struct.unpack(endian + 'HHI', e[:8])
        unit = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}.get(typ, 1)
        sz = unit * cnt
        if sz <= 4:
            raw = e[8:8 + sz]
        else:
            ptr = struct.unpack(endian + 'I', e[8:12])[0]
            raw = d[ptr:ptr + min(sz, 4096)]
        try:
            if typ == 3:
                v = struct.unpack(endian + 'H' * min(cnt, len(raw) // 2), raw[:2 * min(cnt, len(raw) // 2)])
            elif typ == 4:
                v = struct.unpack(endian + 'I' * min(cnt, len(raw) // 4), raw[:4 * min(cnt, len(raw) // 4)])
            elif typ in (5, 10):
                fmt = 'I' if typ == 5 else 'i'
                m = min(cnt, len(raw) // 8)
                vals = struct.unpack(endian + fmt * 2 * m, raw[:8 * m])
                v = [a / b if b else 0.0 for a, b in zip(vals[0::2], vals[1::2])]
            elif typ == 2:
                v = raw.split(b'\0')[0].decode(errors='ignore')
            else:
                v = raw
        except struct.error:
            continue
        tags[tag] = v[0] if isinstance(v, (tuple, list)) and len(v) == 1 else v
    return tags


def read_info(path):
    """Best-effort metadata: exposure (s), capture time, focal length, make/model."""
    info = {'exposure': None, 'time': None, 'focal': None, 'model': '', 'make': ''}
    try:
        with open(path, 'rb') as fh:
            d = fh.read(4 * 1024 * 1024)  # EXIF lives near the start of TIFF-based raws
        if d[:4] == b'IIU\x00':  # Panasonic RW2
            endian = '<'
        elif d[:2] == b'II':
            endian = '<'
        elif d[:2] == b'MM':
            endian = '>'
        else:
            return info
        off = struct.unpack(endian + 'I', d[4:8])[0]
        t0 = _tiff_tags(d, off, endian)
        info['make'] = str(t0.get(0x10f, '')).strip()
        info['model'] = str(t0.get(0x110, '')).strip()
        exif_ptr = t0.get(0x8769)
        ex = _tiff_tags(d, exif_ptr, endian) if isinstance(exif_ptr, int) else {}
        info['exposure'] = ex.get(0x829a) if isinstance(ex.get(0x829a), float) else None
        info['focal'] = ex.get(0x920a) if isinstance(ex.get(0x920a), float) else None
        ts = ex.get(0x9003) or t0.get(0x132)
        if isinstance(ts, str):
            try:
                info['time'] = datetime.datetime.strptime(ts.strip(), '%Y:%m:%d %H:%M:%S')
            except ValueError:
                pass
    except OSError:
        pass
    return info


# ------------------------------------------------------------- image data --

def _rw2_supported(path):
    try:
        _, t, _ = rw2.read_meta(path)
        return t.get(0xb) == 34316 and t.get(0x2d) == 6 and t.get(0xa) == 14
    except Exception:
        return False


def _load_rw2(path):
    raw, m = rw2.decode(path)
    r = raw[m['top']:m['bottom'], m['left']:m['right']].astype(np.float32)
    del raw
    bl = float(np.mean(m['black']))
    r = np.clip((r - bl) / (m['white'] - bl), 0, 1)
    rgb = cv2.cvtColor((r * 65535).astype(np.uint16), cv2.COLOR_BayerBG2RGB_EA).astype(np.float32) / 65535
    del r
    wb = np.array(m['wb'], np.float32) / m['wb'][1]
    rgb = np.minimum(rgb * wb, 1.0)  # clip after WB so blown highlights stay neutral
    lin = np.clip(rgb.reshape(-1, 3) @ _M_S1, 0, None).reshape(rgb.shape)
    return lin.astype(np.float32)


def _load_libraw(path):
    import rawpy  # installed on the server; not needed for the Lumix S1
    with rawpy.imread(path) as raw:
        out = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16,
                              use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB,
                              highlight_mode=rawpy.HighlightMode.Clip, user_flip=None)
    return out.astype(np.float32) / 65535.0


def load_linear(path):
    """Linear-light sRGB float32 image in [0, 1+], white balance applied."""
    if path.lower().endswith('.rw2') and _rw2_supported(path):
        return _load_rw2(path)
    return _load_libraw(path)


def quick_brightness(path):
    """Cheap exposure proxy when EXIF is missing: mean raw level."""
    try:
        import rawpy
        with rawpy.imread(path) as raw:
            v = raw.raw_image_visible
            return float(v[::16, ::16].mean())
    except Exception:
        if path.lower().endswith('.rw2'):
            raw, _ = rw2.decode(path)
            return float(raw[::16, ::16].mean())
        return 0.0
