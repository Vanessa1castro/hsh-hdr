"""HSH HDR engine: group brackets -> align -> exposure fusion -> AutoHDR-style finish.

The look (brightness, white walls, honey floors, soft contrast, straight lines) was
tuned against reference edits supplied by HSH.
"""
import gc
import os

import cv2
import numpy as np

import raw_io

# ------------------------------------------------------------------ grouping --


def group_brackets(paths, per_set=0, gap_s=8.0, infos=None):
    """Split RAW files into bracket sets.

    per_set > 0  -> fixed-size sets in capture/file order.
    per_set == 0 -> automatic: a new set starts when there is a pause of more than
                    `gap_s` seconds, or the exposure repeats one already in the set.
    """
    items = []
    for p in paths:
        inf = infos[p] if infos and p in infos else raw_io.read_info(p)
        items.append({'path': p, **inf})
    items.sort(key=lambda it: ((it['time'] or 0) and it['time'].timestamp(), os.path.basename(it['path'])))
    if not items:
        return []
    if per_set and per_set > 0:
        return [[it for it in items[i:i + per_set]] for i in range(0, len(items), per_set)]
    if any(it['exposure'] is None for it in items):
        for it in items:  # fall back to raw brightness as the exposure proxy
            it['exposure'] = raw_io.quick_brightness(it['path']) or 1.0
    sets = [[items[0]]]
    for it in items[1:]:
        cur = sets[-1]
        prev = cur[-1]
        pause = (it['time'] - prev['time']).total_seconds() if (it['time'] and prev['time']) else 0
        e_new, e_last = it['exposure'] or 1, prev['exposure'] or 1
        repeated = any(abs(np.log(e_new / (o['exposure'] or 1))) < 0.18 for o in cur)
        # brackets run in one direction (bright->dark or dark->bright); a reversal starts a new set
        reversed_ = False
        if len(cur) >= 2 and pause > 2:  # shots fired within 2 s belong together in any order
            d_prev = np.sign(np.log(e_last / (cur[-2]['exposure'] or 1)))
            reversed_ = np.sign(np.log(e_new / e_last)) != d_prev
        if pause > gap_s or repeated or reversed_ or len(cur) >= 9:
            sets.append([it])
        else:
            cur.append(it)
    return sets


# ------------------------------------------------------------------ merging --

def _srgb_gamma(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def _align(frames):
    ref = frames[len(frames) // 2]
    rg = cv2.resize(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), None, fx=0.25, fy=0.25)
    out = []
    for f in frames:
        if f is ref:
            out.append(f)
            continue
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), None, fx=0.25, fy=0.25)
        g = g * (np.median(rg) / max(np.median(g), 1e-4))
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(np.clip(rg, 0, 1).astype(np.float32),
                                           np.clip(g, 0, 1).astype(np.float32), warp,
                                           cv2.MOTION_TRANSLATION,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
        except cv2.error:
            pass
        warp[:, 2] *= 4
        if np.abs(warp[:, 2]).max() > 0.3 and np.abs(warp[:, 2]).max() < 200:
            f = cv2.warpAffine(f, warp, (f.shape[1], f.shape[0]),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        out.append(f)
    return out


def _gauss_pyr(img, n):
    g = [img]
    for _ in range(n):
        g.append(cv2.pyrDown(g[-1]))
    return g


def _lap_pyr(img, n):
    g = _gauss_pyr(img, n)
    lap = [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0])) for i in range(n)]
    lap.append(g[-1])
    return lap


def _fuse(frames, n=9):
    """Exposure fusion. `frames` sorted brightest -> darkest, display-referred [0,1]."""
    ws = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        c = np.abs(cv2.Laplacian(gray, cv2.CV_32F)) + 1e-3
        s = f.std(2) + 1e-3
        e = np.exp(-((f - 0.5) ** 2) / (2 * 0.2 ** 2)).prod(2)
        ws.append((c ** 0.6) * (s ** 0.4) * e + 1e-12)
    W = np.sum(ws, 0)
    fr = [w / W for w in ws]
    del ws
    # window recovery: where the brightest frame is blown out, lean on the darkest one
    lb = cv2.resize(cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY), None, fx=0.25, fy=0.25,
                    interpolation=cv2.INTER_AREA)
    cm = cv2.GaussianBlur(np.clip((lb - 0.90) / 0.08, 0, 1), (0, 0), 1.0)
    cm = np.clip(cv2.resize(cm, (W.shape[1], W.shape[0]), interpolation=cv2.INTER_LINEAR), 0, 1) * 0.6
    fr = [w * (1 - cm) for w in fr]
    fr[-1] = fr[-1] + cm
    del cm, W
    acc = None
    for f, w in zip(frames, fr):
        gp = _gauss_pyr(w, n)
        lp = _lap_pyr(f, n)
        if acc is None:
            acc = [l * g[..., None] for l, g in zip(lp, gp)]
        else:
            for i in range(len(acc)):
                acc[i] += lp[i] * gp[i][..., None]
        del gp, lp
        gc.collect()
    out = acc[-1]
    for i in range(n - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(acc[i].shape[1], acc[i].shape[0])) + acc[i]
    return np.clip(out, 0, 1)


# ------------------------------------------------------------------ finish --

def _gf(I, p, r, eps):
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    bf = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = bf(I), bf(p)
    a = (bf(I * p) - mI * mp) / (bf(I * I) - mI * mI + eps)
    b = mp - a * mI
    return bf(a) * I + bf(b)


def _whites(img, lum_q=60):
    s = cv2.resize(img, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
    l = s.mean(2)
    sat = s.max(2) - s.min(2)
    cand = (l > np.percentile(l, lum_q)) & (l < np.percentile(l, 99.5))
    if cand.sum() < 50:
        cand = l > np.percentile(l, 50)
    m = cand & (sat <= np.percentile(sat[cand], 50))
    return s, m


def _lens(img, k1, k2):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    R = np.hypot(cx, cy)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx - cx) / R
    y = (yy - cy) / R
    del xx, yy
    r2 = x * x + y * y
    f = 1 + k1 * r2 + k2 * r2 * r2
    del r2
    sc = 1 / (1 + k1 + k2)  # corners map to corners: no empty borders
    mx = (x * f * sc * R + cx).astype(np.float32)
    my = (y * f * sc * R + cy).astype(np.float32)
    del x, y, f
    return cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _white_balance(img, target=(1.0, 0.993, 0.975)):
    s, m = _whites(img, lum_q=45)
    ref = s[m].mean(0)
    g = ref.mean() / np.maximum(ref, 1e-4) * np.array(target)
    return img * (g / g.max()).astype(np.float32)[None, None, :]


def _tone(img, white_target=0.835):
    s = cv2.resize(img, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
    bp = float(np.percentile(s.mean(2), 0.2)) * 0.9
    img = np.clip((img - bp) / (1 - bp), 0, 1)
    s, m = _whites(img)
    Lw = float(np.median(s.mean(2)[m]))
    k = float(np.clip((white_target ** 2.2) / max(Lw ** 2.2, 1e-4), 0.7, 8.0))
    Y = np.maximum(img.mean(2), 1e-4)
    h, w = Y.shape
    lg = np.log(Y)
    small = cv2.resize(lg, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    base = cv2.resize(_gf(small, small, 16, 0.01), (w, h), interpolation=cv2.INTER_LINEAR)
    base = _gf(lg.astype(np.float32), base.astype(np.float32), 6, 0.004)
    detail = lg - base
    B = np.exp(base)
    del base
    wm = np.clip((B - 1.6 * Lw) / (0.5 * Lw), 0, 1)  # windows / lights get much less gain
    lin = (B ** 2.2) * np.power(k, 1 - 0.85 * wm)
    del wm, B
    sh, sp = 0.62, 1.6  # filmic shoulder keeps window views from blowing out
    lin = np.where(lin > sh, sh + (1 - sh) * (1 - np.exp(-(lin - sh) / sp)), lin)
    Yn = np.power(lin, 1 / 2.2) * np.exp(detail)
    out = img * (Yn / Y)[..., None]
    return np.clip(out, 0, 1)


def _color(img, chroma_k=0.88):
    Y = img.mean(2, keepdims=True)
    c = img - Y
    mag = np.abs(c).max(2, keepdims=True)
    out = Y + c * (chroma_k / (1 + 1.2 * mag))  # tame strong casts (orange floors) more
    del c, mag
    sat = out.max(2, keepdims=True) - out.min(2, keepdims=True)
    w = np.clip((Y - 0.58) / 0.15, 0, 1) * np.clip((0.16 - sat) / 0.08, 0, 1)
    w = cv2.GaussianBlur(w[..., 0], (0, 0), 25)[..., None]
    neutral = Y * np.array([1.0, 0.993, 0.975], np.float32)
    return np.clip(out * (1 - 0.7 * w) + neutral * (0.7 * w), 0, 1)


def _sharpen(img, amt=0.45, sigma=1.0, lc=0.06):
    Y = img.mean(2)
    fine = Y - cv2.GaussianBlur(Y, (0, 0), sigma)
    h, w = Y.shape
    ys = cv2.resize(Y, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    base = cv2.resize(cv2.GaussianBlur(ys, (0, 0), w / 4 * 0.01), (w, h))
    return np.clip(img + (amt * fine + lc * (Y - base))[..., None], 0, 1)


def _verticals(img, max_deg=12):
    """Level vertical lines. Returns (image, 3x3 warp or None)."""
    h, w = img.shape[:2]
    sc = 1500 / w
    g = cv2.cvtColor((cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) * 255).astype(np.uint8),
                     cv2.COLOR_RGB2GRAY)
    lines = cv2.createLineSegmentDetector().detect(g)[0]
    if lines is None:
        return img, None
    L = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        ln = np.hypot(dx, dy)
        if ln < g.shape[0] * 0.06:
            continue
        ang = (np.degrees(np.arctan2(dx, dy)) + 90) % 180 - 90
        if abs(ang) <= max_deg:
            L.append((x1 / sc, y1 / sc, x2 / sc, y2 / sc, ln))
    if len(L) < 6:
        return img, None
    L = np.array(L)
    cx, cy = w / 2, h / 2
    p1 = np.stack([L[:, 0] - cx, L[:, 1] - cy, np.ones(len(L))], 1)
    p2 = np.stack([L[:, 2] - cx, L[:, 3] - cy, np.ones(len(L))], 1)
    ln = np.cross(p1, p2)
    ln /= np.linalg.norm(ln[:, :2], axis=1, keepdims=True)
    wts = L[:, 4]
    vp = None
    for _ in range(5):
        _, _, vt = np.linalg.svd(ln * wts[:, None])
        vp = vt[-1]
        r = np.abs(ln @ vp) / max(np.linalg.norm(vp[:2]), 1e-9)
        wts = L[:, 4] / (1 + (r / (np.median(r) + 1e-9)) ** 2)
    if abs(vp[2]) < 1e-9:
        return img, None
    vx, vy = vp[0] / vp[2], vp[1] / vp[2]
    if abs(vy) < h * 1.2:
        return img, None
    if vy < 0:
        vx, vy = -vx, -vy
    th = np.arctan2(vx, vy)
    Rm = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    d = (Rm @ np.array([vx, vy, 1.0]))[1]
    P = np.array([[1, 0, 0], [0, 1, 0], [0, -1 / d, 1]])
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    H = np.linalg.inv(T) @ P @ Rm @ T
    c2 = H @ np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
    c2 = (c2[:2] / c2[2]).T
    x0, x1 = max(c2[0, 0], c2[3, 0]), min(c2[1, 0], c2[2, 0])
    y0, y1 = max(c2[0, 1], c2[1, 1]), min(c2[2, 1], c2[3, 1])
    cw, ch = x1 - x0, y1 - y0
    if cw <= 0 or ch <= 0:
        return img, None
    if cw / ch > w / h:
        nw = ch * w / h
        x0 += (cw - nw) / 2
        cw = nw
    else:
        nh = cw * h / w
        y0 += (ch - nh) / 2
        ch = nh
    if cw < w * 0.8:
        return img, None
    S = np.array([[w / cw, 0, -x0 * w / cw], [0, h / ch, -y0 * h / ch], [0, 0, 1]])
    Mw = S @ H
    out = cv2.warpPerspective(img, Mw, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out, 0, 1), Mw


# Lens distortion measured for the Lumix S1 + its 14mm-wide zoom against an HSH reference.
_LENS_PROFILES = {'DC-S1': (-0.1116, 0.0131, 14.0)}


def _lens_params(model, focal):
    prof = _LENS_PROFILES.get(model)
    if not (prof and focal):
        return None
    k1, k2, f0 = prof
    kf = (f0 / max(focal, 1.0)) ** 2
    return k1 * kf, k2 * kf


def _finish(fused, model='', focal=None):
    """Returns (finished image, geometry function to apply the same lens/level fix elsewhere)."""
    img = fused
    lp = _lens_params(model, focal)
    if lp:
        img = _lens(img, *lp)
    img = _white_balance(img)
    img = _tone(img)
    img = _white_balance(img)
    img = _color(img)
    img = _sharpen(img)
    img, Mw = _verticals(img)

    def geometry(other):
        o = _lens(other, *lp) if lp else other
        if Mw is not None:
            o = cv2.warpPerspective(o, Mw, (o.shape[1], o.shape[0]), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)
        return o
    return img, geometry



# ------------------------------------------------------- window recovery --
# Windows and open doors are blown out in the brightest bracket. Where the darkest
# bracket still shows the view (foliage, sky, trees), that view is toned on its own
# and placed back. White frames, sills, walls and patios keep the main edit.

def _srgb_lin(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _window_regions(bright, dark_lin):
    h, w = bright.shape[:2]
    sc = 800 / w
    b = cv2.resize(bright, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    dl = cv2.resize(dark_lin.mean(2), (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    rel = dl / max(float(np.median(dl)), 1e-5)
    m = ((b > 0.93) & (rel > 3.0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lg = np.log(np.maximum(dl, 1e-5))
    tex = np.abs(lg - cv2.GaussianBlur(lg, (0, 0), 2))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for i in range(1, n):
        comp = lab == i
        # skip small highlights (lamps) and smooth sunlit plaster (ceilings, walls)
        if st[i, cv2.CC_STAT_AREA] >= b.size * 0.003 and float(np.median(tex[comp])) > 0.05:
            keep[comp] = 1
    return keep


def _window_pixels(bright, dark_lin, keep):
    h, w = bright.shape[:2]
    reg = cv2.resize(cv2.dilate(keep, np.ones((5, 5), np.uint8)).astype(np.float32), (w, h),
                     interpolation=cv2.INTER_LINEAR)
    blown = np.clip((bright - 0.90) / 0.07, 0, 1)
    dl = np.log(np.maximum(dark_lin.mean(2), 1e-5))
    sig = w / 1000.0
    tex = cv2.GaussianBlur(np.abs(dl - cv2.GaussianBlur(dl, (0, 0), 2 * sig)), (0, 0), 4 * sig)
    del dl
    m = reg * blown * np.clip((tex - 0.02) / 0.04, 0, 1)
    return np.clip(cv2.GaussianBlur(m, (0, 0), 1.2 * sig) * 1.3, 0, 1)


def _window_view(dark_lin, keep, hi=0.78, lc=0.22, sat=1.45):
    s = cv2.resize(dark_lin, (keep.shape[1], keep.shape[0]), interpolation=cv2.INTER_AREA)
    Y = s.mean(2)[keep > 0]
    p = float(np.percentile(Y, 92)) if Y.size else 0.1
    x = dark_lin * (hi / max(p, 1e-5))
    Yx = np.maximum(x.mean(2), 1e-6)
    Yt = Yx * (1 + Yx / 1.6 ** 2) / (1 + Yx) * 1.35
    x = x * (np.minimum(Yt, 1.0) / Yx)[..., None]
    del Yx, Yt
    out = np.clip(_srgb_gamma(np.clip(x, 0, 1)), 0, 1).astype(np.float32)
    del x
    Yo = out.mean(2)
    base = cv2.GaussianBlur(Yo, (0, 0), out.shape[1] * 0.006)
    out = np.clip(out + (lc * (Yo - base))[..., None], 0, 1)
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    # foliage: richer, deeper natural green (yellow-green pushed toward leaf green)
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
    g = np.clip(1 - np.abs(hue - 120) / 45, 0, 1)
    lab[..., 1:] *= (1 + (sat - 1) * (0.5 + 0.5 * g))[..., None]
    lab[..., 0] -= 4 * g * np.clip((lab[..., 0] - 30) / 30, 0, 1)
    lab[..., 2] -= 3 * g
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)


def _window_openings(keep, dark_lin, full_shape):
    """The whole opening around each blown window: inside its bounding box, every
    pixel lit by daylight (clearly brighter than the room in the darkest bracket)."""
    sc_h, sc_w = keep.shape
    dl = cv2.resize(dark_lin.mean(2), (sc_w, sc_h), interpolation=cv2.INTER_AREA)
    rel = dl / max(float(np.median(dl)), 1e-5)
    out = np.zeros(keep.shape, np.float32)
    n, lab, st, _ = cv2.connectedComponentsWithStats(keep)
    for i in range(1, n):
        x, y, w, h = st[i, :4]
        p = max(2, int(0.04 * max(w, h)))
        y0, y1, x0, x1 = max(0, y - p), min(sc_h, y + h + p), max(0, x - p), min(sc_w, x + w + p)
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], np.clip((rel[y0:y1, x0:x1] - 1.6) / 0.8, 0, 1))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    H, W = full_shape
    reg = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(cv2.GaussianBlur(reg, (0, 0), W / 1000.0 * 2), 0, 1)


def _recover_windows(img, geometry, bright, dark_lin):
    keep = _window_regions(bright, dark_lin)
    if keep.sum() == 0:
        return img
    reg = np.clip(geometry(_window_openings(keep, dark_lin, bright.shape)), 0, 1)
    win = geometry(_window_view(dark_lin, keep))
    sig = img.shape[1] / 1000.0
    labw = cv2.cvtColor(win, cv2.COLOR_RGB2LAB)
    Lw = labw[..., 0]
    Cw = np.hypot(labw[..., 1], labw[..., 2])
    hue = np.degrees(np.arctan2(labw[..., 2], labw[..., 1]))
    del labw
    lg = Lw / 100.0
    tex = cv2.GaussianBlur(np.abs(lg - cv2.GaussianBlur(lg, (0, 0), 2 * sig)), (0, 0), 3 * sig)
    # the view = leaf greens, bright sky, or strongly coloured things; anything near-neutral
    # (white frames and doors in shade, patios, thresholds, walls) keeps the main edit
    green = np.clip(1 - np.abs(hue - 125) / 45, 0, 1)
    sky = np.clip(1 - np.abs(hue + 95) / 35, 0, 1) * np.clip((Lw - 70) / 8, 0, 1)
    colour = np.maximum(np.maximum(green, sky) * np.clip((Cw - 8) / 4, 0, 1), np.clip((Cw - 16) / 5, 0, 1))
    dark = np.clip((42 - Lw) / 10, 0, 1) * np.clip((tex - 0.02) / 0.03, 0, 1) * green * np.clip((Cw - 2) / 3, 0, 1)
    view = np.clip(cv2.GaussianBlur(np.maximum(colour, dark), (0, 0), 1.5 * sig) * 1.5, 0, 1)
    del hue, tex, green, sky, colour, dark, Cw
    Lf = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)[..., 0]
    m = reg * view * np.clip((Lf - Lw + 3) / 8, 0, 1)
    del Lf, Lw, view, reg
    return np.clip(img * (1 - m[..., None]) + win * m[..., None], 0, 1)


def _wood(img, chroma=1.08, dl=9.0, lift=24.0, strip=512):
    """Natural oak: light boards a touch richer (not pale), honey-oak doors opened up.
    Worked out on a smoothed copy so whole surfaces shift evenly and grain is kept;
    dark low-colour woods (walnut vanities) are left as they are."""
    h, w = img.shape[:2]
    sm = cv2.resize(np.clip(img, 0, 1).astype(np.float32), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), 2)
    lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    m = np.clip(1 - np.abs(hue - 72) / 22, 0, 1) * np.clip((C - 10) / 6, 0, 1) * np.clip((L - 15) / 10, 0, 1)
    dL = -dl * np.clip((L - 62) / 14, 0, 1) + lift * np.clip((52 - L) / 22, 0, 1) * np.clip((C - 15) / 4, 0, 1)
    dL = cv2.GaussianBlur(m * dL, (0, 0), 3)
    fc = cv2.GaussianBlur(1 + (chroma - 1) * m, (0, 0), 3)
    dL = cv2.resize(dL, (w, h), interpolation=cv2.INTER_LINEAR)
    fc = cv2.resize(fc, (w, h), interpolation=cv2.INTER_LINEAR)
    del sm, lab, L, a, b, C, hue, m
    out = np.empty_like(img, dtype=np.float32)
    for y0 in range(0, h, strip):
        lab = cv2.cvtColor(np.clip(img[y0:y0 + strip], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        lab[..., 0] = np.clip(lab[..., 0] + dL[y0:y0 + strip], 0, 100)
        lab[..., 1] *= fc[y0:y0 + strip]
        lab[..., 2] *= fc[y0:y0 + strip]
        out[y0:y0 + strip] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    return out


# ------------------------------------------------------------------ public --

def process_set(bracket_set, preview_width=1200):
    """bracket_set: list of dicts from group_brackets().

    Returns (final uint8 RGB full size, before-preview uint8 RGB, after-preview uint8 RGB).
    """
    items = sorted(bracket_set, key=lambda it: -(it['exposure'] or 0))  # brightest first
    frames = []
    for it in items:
        lin = raw_io.load_linear(it['path'])
        frames.append(np.clip(_srgb_gamma(np.clip(lin * 1.6, 0, 1)), 0, 1).astype(np.float32))
        del lin
        gc.collect()
    # make all frames the same size (some cameras vary by a few pixels)
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    frames = [f[:h, :w] for f in frames]
    mid = frames[len(frames) // 2].copy()
    bright = dark_lin = None
    if len(frames) > 1:
        frames = _align(frames)
        bright = frames[0].max(2)
        dark_lin = _srgb_lin(frames[-1]) / 1.6
        fused = _fuse(frames)
    else:
        fused = frames[0]
    del frames
    gc.collect()
    first = items[0]
    img, geometry = _finish(fused, first.get('model', ''), first.get('focal'))
    del fused
    gc.collect()
    if bright is not None:
        try:
            img = _recover_windows(img, geometry, bright, dark_lin)
        except Exception:  # never lose a photo over the window step
            pass
        del bright, dark_lin
        gc.collect()
    img = _wood(img)
    before = cv2.resize(geometry(mid), (preview_width, int(preview_width * h / w)), interpolation=cv2.INTER_AREA)
    del mid
    out = (img * 255 + 0.5).astype(np.uint8)
    after = cv2.resize(out, (preview_width, int(preview_width * out.shape[0] / out.shape[1])),
                       interpolation=cv2.INTER_AREA)
    return out, (before * 255 + 0.5).astype(np.uint8), after
