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



def _deblotch(fused, frames, sigma_frac=0.035, verbose=False):
    """Make large-scale brightness and colour follow the true scene light (a radiance
    merge of all brackets); the exposure fusion keeps only texture and local detail."""
    h, w = fused.shape[:2]
    sw = 800; sh = int(round(sw * h / w))
    sm = [cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA) for f in frames]
    lin = [_srgb_lin(f) for f in sm]
    vmax = [f.max(2) for f in sm]
    # relative exposure between consecutive brackets, measured from the pixels themselves
    scale = [1.0]
    for k in range(len(sm) - 1):
        a, b = lin[k].mean(2), lin[k + 1].mean(2)
        m = (vmax[k] < 0.9) & (vmax[k] > 0.08) & (vmax[k + 1] > 0.04) & (vmax[k + 1] < 0.9)
        r = float(np.median(a[m] / np.maximum(b[m], 1e-5))) if m.sum() > 200 else 3.0
        scale.append(scale[-1] * max(r, 1.0))
    num = np.zeros_like(lin[0]); den = np.zeros((sh, sw, 1), np.float32)
    for k, (L, v) in enumerate(zip(lin, vmax)):
        wgt = np.exp(-((v - 0.5) ** 2) / (2 * 0.22 ** 2))
        if k < len(lin) - 1:
            wgt = wgt * (v < 0.95)
        wgt = np.maximum(wgt, 1e-4 if k == len(lin) - 1 else 0)[..., None]
        num += wgt * L * scale[k]; den += wgt
    R = num / np.maximum(den, 1e-6)
    Lr = np.maximum(R.mean(2), 1e-6)
    fs = cv2.resize(fused, (sw, sh), interpolation=cv2.INTER_AREA)
    Lf = np.maximum(fs.mean(2), 1e-4)
    # one global, smooth, increasing curve: true light level -> the fusion's typical brightness
    lr = np.log(Lr).ravel(); lf = Lf.ravel()
    qs = np.quantile(lr, np.linspace(0, 1, 41))
    cx, cy = [], []
    for a, b in zip(qs[:-1], qs[1:]):
        m = (lr >= a) & (lr <= b)
        if m.sum() > 20:
            cx.append(float(np.median(lr[m]))); cy.append(float(np.median(lf[m])))
    cy = np.maximum.accumulate(np.array(cy))
    G = np.interp(np.log(Lr), cx, cy).astype(np.float32)
    G = np.maximum(G, 1e-4)
    sig = sigma_frac * sw
    # windows / lamps keep the fusion's own (recovered) rendering
    keep = np.clip((Lf - 0.72) / 0.12, 0, 1)
    # mainly lift areas the fusion left too dark (shadow patches, halos); only darken very little,
    # so light floors and bright rooms keep their airy look
    raw_c = np.clip(np.log(G) - np.log(Lf), -0.8, 0.8)
    # truly dark things (black fixtures, dark wood, deep corners) are not lifted: lifting them
    # would raise the black point and make the whole photo duller
    # smooth the correction using only room pixels (no dark ring around windows)
    corr = cv2.GaussianBlur(raw_c * (1 - keep), (0, 0), sig) / np.maximum(cv2.GaussianBlur(1 - keep, (0, 0), sig), 1e-3)
    corr = corr * (1 - cv2.GaussianBlur(keep, (0, 0), 1.5))
    # edge-aware fine pass: removes the narrow dark/bright halos the fusion leaves along
    # door frames and window edges, while keeping real edges (guided by the true light map)
    fine = _gf(np.log(G).astype(np.float32), ((raw_c - corr) * (1 - keep)).astype(np.float32), 5, 0.01)
    corr = corr + fine * (1 - cv2.GaussianBlur(keep, (0, 0), 1.5))
    # colour: walls and ceilings lit by mixed daylight / bulbs get blue and orange zones.
    # Pull the large-scale tint of near-neutral surfaces back to neutral; real colours stay.
    lb = cv2.cvtColor(np.clip(fs * np.exp(corr)[..., None], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    ab_lo = cv2.GaussianBlur(lb[..., 1:], (0, 0), sig)
    C_lo = np.hypot(ab_lo[..., 0], ab_lo[..., 1])
    ref = np.median(ab_lo[(C_lo < 12) & (lb[..., 0] > 50)], axis=0) if ((C_lo < 12) & (lb[..., 0] > 50)).sum() > 100 else np.zeros(2)
    # near-neutral surfaces; bright painted surfaces (white doors / walls lit warm or cool) count up to higher colour
    cmax = 16 + 8 * np.clip((lb[..., 0] - 70) / 10, 0, 1)
    neutral = np.clip((cmax - C_lo) / 8, 0, 1) * np.clip((lb[..., 0] - 35) / 15, 0, 1) * (1 - keep)
    dab = -cv2.GaussianBlur((ab_lo - ref) * 0.8 * neutral[..., None], (0, 0), sig)
    # window views: outdoor shade reads blue next to warm room light; take most of the blue out
    win_soft = cv2.GaussianBlur(keep, (0, 0), 2)
    blue_shift = np.minimum(cv2.GaussianBlur(lb[..., 2], (0, 0), 4) - ref[1], 0)
    dab[..., 1] -= 0.65 * blue_shift * win_soft
    if verbose:
        print('scales', [round(s, 2) for s in scale], 'corr range', float(corr.min()), float(corr.max()))
    # second, sharper pass at higher resolution for the thin halos along door frames and windows:
    # wherever the fusion is darker/brighter than the true light right next to an edge, follow
    # the true light (edge-aware, so real edges and texture stay)
    mw = 2000; mh = int(round(mw * h / w))
    corr_m = cv2.resize(corr, (mw, mh), interpolation=cv2.INTER_CUBIC)
    smm = [cv2.resize(f, (mw, mh), interpolation=cv2.INTER_AREA) for f in frames]
    num = 0; den = 0
    for k, f in enumerate(smm):
        v = f.max(2)
        wgt = np.exp(-((v - 0.5) ** 2) / (2 * 0.22 ** 2))
        if k < len(smm) - 1:
            wgt = wgt * np.clip((0.95 - v) / 0.04, 0, 1)
        wgt = np.maximum(wgt, 1e-4 if k == len(smm) - 1 else 0)
        num = num + wgt * _srgb_lin(f).mean(2) * scale[k]; den = den + wgt
    del smm
    Lr_m = np.maximum(num / np.maximum(den, 1e-6), 1e-6)
    G_m = np.maximum(np.interp(np.log(Lr_m), cx, cy).astype(np.float32), 1e-4)
    Lf_m = np.maximum(cv2.resize(fused, (mw, mh), interpolation=cv2.INTER_AREA).mean(2), 1e-4)
    keep_m = cv2.resize(keep, (mw, mh), interpolation=cv2.INTER_LINEAR)
    res = np.clip(np.log(G_m) - np.log(Lf_m) - corr_m, -0.35, 0.35) * (1 - keep_m)
    lg = np.log(G_m).astype(np.float32)
    # edge zone: within a few pixels of a strong brightness edge in the true light map
    gx = cv2.Sobel(lg, cv2.CV_32F, 1, 0); gy = cv2.Sobel(lg, cv2.CV_32F, 0, 1)
    zone = cv2.GaussianBlur(cv2.dilate((np.hypot(gx, gy) > 0.6).astype(np.float32), np.ones((25, 25), np.uint8)), (0, 0), 6)
    fine2 = _gf(lg, (res * zone).astype(np.float32), 8, 0.02) * zone
    corr = cv2.resize(corr_m + fine2, (w, h), interpolation=cv2.INTER_CUBIC)
    del Lr_m, G_m, Lf_m, res, lg, gx, gy, zone, fine2, corr_m
    dab = cv2.resize(dab, (w, h), interpolation=cv2.INTER_CUBIC)
    out = np.empty_like(fused)
    for y0 in range(0, h, 512):
        f = np.clip(fused[y0:y0 + 512] * np.exp(corr[y0:y0 + 512])[..., None], 0, 1).astype(np.float32)
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)
        lab[..., 1:] += dab[y0:y0 + 512]
        out[y0:y0 + 512] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    # keep the black point where the fusion had it: the later tone step sets contrast from the
    # darkest pixels, and lifting them would flatten the whole photo
    pf = float(np.percentile(fs.mean(2), 0.2))
    pd = float(np.percentile(cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA).mean(2), 0.2))
    if pd > pf:
        for y0 in range(0, h, 512):
            Y = out[y0:y0 + 512].mean(2, keepdims=True)
            out[y0:y0 + 512] = np.clip(out[y0:y0 + 512] - (pd - pf) * np.clip(1 - Y / 0.5, 0, 1), 0, 1)
    return out



def _blend_render(frames, scales_from=None):
    """How an editor blends brackets: start from the brightest photo and, only where it is
    blown out, take the same spot from the next darker photo (scaled to the same exposure).
    Masks follow the blown areas with soft edges, so there are no halos and no dark-bracket
    noise or colour on walls and floors. Returns linear light."""
    frames = sorted(frames, key=lambda f: -float(f.mean()))
    h, w = frames[0].shape[:2]
    sw = 800; sh = int(round(sw * h / w))
    sm = [cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA) for f in frames]
    lin_s = [_srgb_lin(f) for f in sm]
    scale = [1.0]
    for k in range(len(sm) - 1):
        va, vb = sm[k].max(2), sm[k + 1].max(2)
        m = (va < 0.9) & (va > 0.15) & (vb > 0.04)
        r = float(np.median(lin_s[k].mean(2)[m] / np.maximum(lin_s[k + 1].mean(2)[m], 1e-5))) if m.sum() > 200 else 3.0
        scale.append(scale[-1] * max(r, 1.0))
    sig = w / 600.0
    ker = np.ones((int(sig * 3) | 1, int(sig * 3) | 1), np.uint8)
    out = _srgb_lin(frames[0])
    clipped = frames[0].max(2)
    for k in range(1, len(frames)):
        # soft luminosity mask (like an editor's): wide, gradual hand-off, so big bright walls
        # don't get patches where one photo meets the next
        cs = cv2.resize(clipped.astype(np.float32), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        mx = cv2.dilate(cs, np.ones((5, 5), np.uint8))
        ms = np.clip((cv2.GaussianBlur(mx, (0, 0), w / 4 / 120) - 0.70) / 0.22, 0, 1)
        m = cv2.resize(ms, (w, h), interpolation=cv2.INTER_LINEAR)
        m = np.maximum(m, np.clip((clipped - 0.88) / 0.06, 0, 1))
        del cs, mx, ms
        nxt = _srgb_lin(frames[k]) * scale[k]
        out = out * (1 - m[..., None]) + nxt * m[..., None]
        # what is still blown after this step: blown in this darker frame too
        clipped = np.where(m > 0.5, frames[k].max(2), np.minimum(clipped, 0.5))
        del nxt
    return out, scale

def _to_display(lin, fused_like_median=0.36):
    """Global exposure + highlight shoulder -> display (sRGB-like) values. The shoulder is
    set from the brightest real surfaces (bright doors, lit hallways), so they roll off
    gently and keep their detail instead of clipping to flat white."""
    s = cv2.resize(lin, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA).mean(2)
    med = float(np.median(s))
    k = (fused_like_median ** 2.2) / max(med, 1e-6)
    W = float(np.clip(np.percentile(s * k, 99.8) * 1.15, 1.2, 12.0))
    gain = 1 + 1 / W  # keeps mid-tones where they were
    out = np.empty_like(lin)
    for y0 in range(0, lin.shape[0], 512):
        x = lin[y0:y0 + 512] * k
        Y = np.maximum(x.mean(2), 1e-7)
        Yt = Y * (1 + Y / (W * W)) / (1 + Y) * gain
        # soft knee above 0.85 instead of a hard clip at 1.0
        Yt = np.where(Yt > 0.85, 0.85 + 0.15 * (1 - np.exp(-(Yt - 0.85) / 0.15)), Yt)
        x = x * (Yt / Y)[..., None]
        out[y0:y0 + 512] = np.clip(_srgb_gamma(np.clip(x, 0, 1)), 0, 1)
    return out.astype(np.float32)


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


def _clean(img, strip=512):
    """Smooth walls and ceilings: remove fine grain (noise from the dark brackets) where the
    surface is flat; real edges, wood grain and texture are kept."""
    h, w = img.shape[:2]
    sc = w / 6000.0
    out = np.empty_like(img)
    pad = 16
    for y0 in range(0, h, strip):
        a, b = max(0, y0 - pad), min(h, y0 + strip + pad)
        blk = img[a:b]
        Y = blk.mean(2)
        base = cv2.bilateralFilter(Y.astype(np.float32), int(max(5, 9 * sc)) | 1, 0.03, 4 * sc + 1)
        d = Y - base
        # soft coring: small wiggles (grain) are removed, larger detail stays
        d2 = np.sign(d) * np.maximum(np.abs(d) - 0.006, 0) * np.clip(np.abs(d) / 0.02, 0, 1)
        # walls and ceilings only; floors and furniture keep their full grain
        res = np.clip(blk + ((d2 - d) * np.clip((base - 0.55) / 0.15, 0, 1))[..., None], 0, 1)
        out[y0:y0 + strip] = res[y0 - a:y0 - a + min(strip, h - y0)]
    return out


def _brighten(img, lo=76.0, target=78.0):
    """Rooms that come out darker than the HSH look (median below ~76 L*) get a gentle lift."""
    s = cv2.resize(np.clip(img, 0, 1), None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA).astype(np.float32)
    med = float(np.median(cv2.cvtColor(s, cv2.COLOR_RGB2LAB)[..., 0]))
    if med >= lo:
        return img
    m = med / 100.0
    t = target / 100.0
    g = float(np.clip(np.log(t ** 2.4) / np.log(max(m ** 2.4, 1e-4)), 0.75, 1.0))  # gamma in linear-ish space
    Y = np.maximum(img.mean(2), 1e-5)
    Yn = np.power(Y, g)
    return np.clip(img * (Yn / Y)[..., None], 0, 1)


def _sharpen(img, amt=0.40, sigma=1.0, lc=0.0):
    Y = img.mean(2)
    fine = Y - cv2.GaussianBlur(Y, (0, 0), sigma)
    # sharpen real edges; leave flat walls/ceilings smooth
    gx = cv2.Sobel(cv2.GaussianBlur(Y, (0, 0), 1.2), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(cv2.GaussianBlur(Y, (0, 0), 1.2), cv2.CV_32F, 0, 1)
    gm = np.hypot(gx, gy)
    # sharpen texture and soft edges; very strong edges (door frames against bright rooms) get
    # little, so they don't grow dark outlines
    edge = np.clip((gm - 0.03) / 0.08, 0, 1) * np.clip((0.9 - gm) / 0.5, 0.15, 1)
    del gx, gy
    h, w = Y.shape
    ys = cv2.resize(Y, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    base = cv2.resize(cv2.GaussianBlur(ys, (0, 0), w / 4 * 0.01), (w, h))
    return np.clip(img + (amt * fine * edge + lc * (Y - base))[..., None], 0, 1)


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
    img = _brighten(img)
    img = _clean(img)
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
    # shade outdoors reads blue next to warm room light: white fences / walls back to white,
    # skies keep a soft blue
    bl = lab[..., 2] < 0
    lab[..., 2] = np.where(bl, lab[..., 2] * 0.45, lab[..., 2])
    lab[..., 1] = np.where(bl, lab[..., 1] * 0.6, lab[..., 1])
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


def _window_tint(img, geometry, bright, strip=512):
    """Window panes (blown in the brightest bracket): no blue/purple cast on the view.
    Shade outside reads blue-violet next to warm room light; real greens and soft sky stay."""
    m = cv2.GaussianBlur(np.clip((bright - 0.92) / 0.06, 0, 1).astype(np.float32), (0, 0), 3)
    m = np.clip(geometry(m), 0, 1)
    out = img.copy()
    for y0 in range(0, img.shape[0], strip):
        mm = m[y0:y0 + strip]
        if mm.max() < 0.02:
            continue
        lab = cv2.cvtColor(np.clip(img[y0:y0 + strip], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        a, b = lab[..., 1], lab[..., 2]
        cool = b < 0
        nb = np.where(cool, b * 0.4, b)
        na = np.where(cool & (a > 0), a * 0.2, a)  # violet -> neutral
        Cw = np.hypot(a, b)
        na = np.where((a < 0) & (b < 4) & (Cw < 22), na * 0.35, na)  # cyan -> neutral (greens keep b > 0)
        lab[..., 1] = a + (na - a) * mm
        lab[..., 2] = b + (nb - b) * mm
        out[y0:y0 + strip] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    return out


def _deflare(img, strip=512):
    """Remove soft lens-flare spots (ghost reflections of bright windows) from plain walls and
    ceilings: small, round, soft, slightly brighter blobs on an otherwise flat surface."""
    h, w = img.shape[:2]
    sw = 600
    sh = int(round(sw * h / w))
    s = cv2.resize(np.clip(img, 0, 1), (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    lab = cv2.cvtColor(s, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    bg = cv2.medianBlur(np.clip(L * 2.55, 0, 255).astype(np.uint8), 31).astype(np.float32) / 2.55
    res = cv2.GaussianBlur(L, (0, 0), 2) - bg
    tex = cv2.GaussianBlur(np.abs(L - cv2.GaussianBlur(L, (0, 0), 1.5)), (0, 0), 4)
    gx = cv2.Sobel(cv2.GaussianBlur(L, (0, 0), 2), cv2.CV_32F, 1, 0); gy = cv2.Sobel(cv2.GaussianBlur(L, (0, 0), 2), cv2.CV_32F, 0, 1)
    edges = cv2.dilate((np.hypot(gx, gy) > 12).astype(np.uint8), np.ones((5, 5), np.uint8))
    cand = ((res > 1.5) & (res < 14) & (bg > 60) & (tex < 1.2)).astype(np.uint8)
    n, lbl, st, cen = cv2.connectedComponentsWithStats(cand)
    mask = np.zeros((sh, sw), np.float32)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if not (25 <= area <= 0.004 * sw * sh) or not (0.5 < bw / max(bh, 1) < 2.0) or area < 0.55 * bw * bh * 0.785:
            continue
        comp = lbl == i
        if edges[comp].mean() > 0.15:  # sits on real edges (outlet, frame) -> not a flare
            continue
        mask[comp] = 1
    if mask.max() == 0:
        return img
    mask = cv2.GaussianBlur(cv2.dilate(mask, np.ones((5, 5), np.uint8)), (0, 0), 3)
    dl = cv2.resize(-(res * mask), (w, h), interpolation=cv2.INTER_CUBIC)
    m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    # also pull the spot's tint back to the surrounding wall colour
    ab_bg = cv2.GaussianBlur(lab[..., 1:] * (1 - mask[..., None]), (0, 0), 12) / np.maximum(cv2.GaussianBlur(1 - mask, (0, 0), 12), 1e-3)[..., None]
    dab = cv2.resize((ab_bg - lab[..., 1:]) * mask[..., None], (w, h), interpolation=cv2.INTER_LINEAR)
    out = img.copy()
    for y0 in range(0, h, strip):
        if m[y0:y0 + strip].max() < 0.01:
            continue
        l2 = cv2.cvtColor(np.clip(img[y0:y0 + strip], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        l2[..., 0] += dl[y0:y0 + strip]
        l2[..., 1:] += dab[y0:y0 + strip]
        out[y0:y0 + strip] = np.clip(cv2.cvtColor(l2, cv2.COLOR_LAB2RGB), 0, 1)
    return out


def _white_surfaces(img, strip=512):
    """White doors, trim and walls lit by warm bulbs read cream/yellow: bring bright, weakly
    coloured warm surfaces back to clean white (real wood, beige tile etc. are more coloured)."""
    out = np.empty_like(img)
    for y0 in range(0, img.shape[0], strip):
        lab = cv2.cvtColor(np.clip(img[y0:y0 + strip], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        C = np.hypot(a, b)
        hue = np.degrees(np.arctan2(b, a))
        cm = 24 + 16 * np.clip((L - 74) / 10, 0, 1)
        w = (np.clip((L - 70) / 8, 0, 1) * np.clip((cm - C) / 6, 0, 1)
             * np.clip(1 - np.abs(hue - 80) / 70, 0, 1))
        cool = (np.clip((L - 66) / 8, 0, 1) * np.clip((14 - C) / 4, 0, 1)
                * np.clip(1 - np.abs(hue + 95) / 50, 0, 1))
        f = 1 - np.maximum(0.88 * w, 0.95 * cool)
        lab[..., 1] = a * f
        lab[..., 2] = b * f
        out[y0:y0 + strip] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    return out


def _wall_marks(img, strip=512):
    """On white walls: warm/brown marks and smudges go neutral, and small dark fixtures
    (hinges, outlets' screws, handles) lose the orange cast from warm bulbs."""
    h, w = img.shape[:2]
    Lf = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)[..., 0]
    small = cv2.resize(Lf, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    surround = cv2.resize(cv2.dilate(cv2.GaussianBlur(small, (0, 0), 6), np.ones((9, 9), np.uint8)), (w, h),
                          interpolation=cv2.INTER_LINEAR)
    del Lf, small
    out = np.empty_like(img)
    for y0 in range(0, h, strip):
        lab = cv2.cvtColor(np.clip(img[y0:y0 + strip], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        C = np.hypot(a, b)
        sb = np.clip((surround[y0:y0 + strip] - 74) / 6, 0, 1)  # sits on/next to a white wall
        smudge = sb * np.clip((L - 45) / 10, 0, 1) * np.clip((36 - C) / 6, 0, 1)
        fixture = sb * np.clip((50 - L) / 8, 0, 1)
        f = 1 - np.maximum(0.85 * smudge, 0.8 * fixture)
        lab[..., 1] = a * f
        lab[..., 2] = b * f
        out[y0:y0 + strip] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    del surround
    return out


def _soft_contrast(img, strip=512):
    """HSH look (measured from HSH/AutoHDR finals): whites sit around 80-85 L* with their
    shading kept instead of washing out to paper-white; dark floors a bit lighter."""
    xs = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100], np.float32)
    ys = np.array([0, 15, 29, 40, 49, 56, 63, 70, 77.5, 80.5, 84, 89, 96], np.float32)
    out = np.empty_like(img)
    for y0 in range(0, img.shape[0], strip):
        lab = cv2.cvtColor(np.clip(img[y0:y0 + strip], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        lab[..., 0] = np.interp(lab[..., 0], xs, ys)
        out[y0:y0 + strip] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    return out


def _floor_reflections(img, strip=512):
    """Sky reflected in glossy floors shows as a bright blue patch: give it the floor's own
    colour and pull its brightness most of the way down to the floor around it."""
    h, w = img.shape[:2]
    sw = 600; sh = int(round(sw * h / w))
    lab = cv2.cvtColor(cv2.resize(np.clip(img, 0, 1), (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32), cv2.COLOR_RGB2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    hue = np.degrees(np.arctan2(b, a)); C = np.hypot(a, b)
    wood = (np.clip(1 - np.abs(hue - 60) / 40, 0, 1) * np.clip((C - 3) / 4, 0, 1) * (L > 15) * (L < 75)).astype(np.float32)
    near = cv2.GaussianBlur(cv2.dilate(wood, np.ones((121, 121), np.uint8)), (0, 0), 8)
    ref = cv2.GaussianBlur(lab * wood[..., None], (0, 0), 60) / np.maximum(cv2.GaussianBlur(wood, (0, 0), 60), 1e-4)[..., None]
    # inside the floor: wood all around (a patch on a wall, window or mirror has little wood around it)
    dens = cv2.GaussianBlur(((L > 12) & (L < 60)).astype(np.float32), (0, 0), 25)
    # floor on both sides (left/right) of the spot: a reflection streak, not a mark on the wall
    sides = cv2.blur(((L > 12) & (L < 60)).astype(np.float32), (61, 3))
    dens = np.minimum(dens, sides + 0.1)
    cool = np.clip((ref[..., 2] - b - 13) / 6, 0, 1) * np.clip((near - 0.5) / 0.3, 0, 1) * np.clip((dens - 0.17) / 0.06, 0, 1)
    # lower part of the photo only (floors), not windows/skies
    yy = np.linspace(0, 1, sh)[:, None]
    cool = cool * np.clip((yy - 0.55) / 0.08, 0, 1)
    cool = np.clip(cv2.GaussianBlur(cv2.dilate(cool.astype(np.float32), np.ones((7, 7), np.uint8)), (0, 0), 4) * 1.3, 0, 1)
    if cool.max() < 0.05:
        return img
    target_L = L - 0.45 * np.maximum(L - ref[..., 0], 0)
    d = np.dstack([target_L - L, ref[..., 1] - a, ref[..., 2] - b]) * cool[..., None]
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img.copy()
    for y0 in range(0, h, strip):
        l2 = cv2.cvtColor(np.clip(img[y0:y0 + strip], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        l2 += d[y0:y0 + strip]
        out[y0:y0 + strip] = np.clip(cv2.cvtColor(l2, cv2.COLOR_LAB2RGB), 0, 1)
    return out


def _flare_tint(img, geometry, bright, strip=512):
    """Lens-flare ghosts and daylight spill show as blue/violet blobs on white walls, ceilings
    and mirrors. Outside the windows themselves, bright surfaces lose that cool tint."""
    win = (bright > 0.92).astype(np.uint8)
    win = cv2.dilate(cv2.resize(win, (bright.shape[1] // 8, bright.shape[0] // 8), interpolation=cv2.INTER_NEAREST),
                     np.ones((5, 5), np.uint8)).astype(np.float32)
    win = cv2.resize(cv2.GaussianBlur(win, (0, 0), 2), (bright.shape[1], bright.shape[0]), interpolation=cv2.INTER_LINEAR)
    win = np.clip(geometry(win), 0, 1)
    out = np.empty_like(img)
    for y0 in range(0, img.shape[0], strip):
        lab = cv2.cvtColor(np.clip(img[y0:y0 + strip], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        C = np.hypot(a, b)
        hue = np.degrees(np.arctan2(b, a))
        cool = (np.clip((L - 62) / 10, 0, 1) * np.clip((26 - C) / 6, 0, 1)
                * np.clip(1 - np.abs(hue + 85) / 55, 0, 1) * (1 - win[y0:y0 + strip]))
        f = 1 - 0.9 * cool
        lab[..., 1] = a * f
        lab[..., 2] = b * f
        out[y0:y0 + strip] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    return out


def _highlight_detail(img, amount=1.4, strip=512):
    """White doors, trim and panelled walls flatten out near white: bring back their shading
    (panel mouldings, wall grooves) with local contrast in the bright tones only. The base is
    edge-aware, so dark objects on white walls (faucets, hinges, towel rings) get no halo."""
    h, w = img.shape[:2]
    Lfull = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)[..., 0]
    small = cv2.resize(Lfull, (w // 4, h // 4), interpolation=cv2.INTER_AREA) / 100.0
    base = _gf(small.astype(np.float32), small.astype(np.float32), 6, 0.0015) * 100.0
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
    # no boost next to strong edges (dark fixtures against white)
    ge = cv2.resize(cv2.dilate((np.abs(small - cv2.GaussianBlur(small, (0, 0), 2)) > 0.06).astype(np.uint8),
                               np.ones((5, 5), np.uint8)).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    del small
    out = np.empty_like(img)
    for y0 in range(0, h, strip):
        lab = cv2.cvtColor(np.clip(img[y0:y0 + strip], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        L = lab[..., 0]
        d = np.clip(cv2.GaussianBlur(L, (0, 0), 2.5) - base[y0:y0 + strip], -8, 8)
        wgt = np.clip((base[y0:y0 + strip] - 65) / 10, 0, 1) * (1 - ge[y0:y0 + strip])
        lab[..., 0] = np.clip(L + amount * d * wgt, 0, 100)
        out[y0:y0 + strip] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    del Lfull, base, ge
    return out


def _floor_clarity(img, amount=0.9, strip=512):
    """Crisp wood grain on floors and furniture (mid/dark tones): local contrast at grain
    scale, so floors read sharp instead of smeared by soft reflections."""
    h, w = img.shape[:2]
    out = np.empty_like(img)
    pad = 32
    sig = max(2.0, w / 6000 * 8)
    for y0 in range(0, h, strip):
        a, b = max(0, y0 - pad), min(h, y0 + strip + pad)
        lab = cv2.cvtColor(np.clip(img[a:b], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        L = lab[..., 0]
        base = cv2.GaussianBlur(L, (0, 0), sig)
        d = np.clip(L - base, -6, 6)
        wgt = np.clip((base - 18) / 10, 0, 1) * np.clip((72 - base) / 10, 0, 1)
        # glossy reflections on dark floors: soften the brightest streaks toward the floor tone
        flo = cv2.GaussianBlur(L, (0, 0), sig * 12)
        over = np.maximum(cv2.GaussianBlur(base, (0, 0), sig) - (flo + 8), 0) * np.clip((72 - flo) / 15, 0, 1)
        L = L - 0.8 * over
        lab[..., 0] = np.clip(L + amount * d * wgt, 0, 100)
        res = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
        out[y0:y0 + strip] = res[y0 - a:y0 - a + min(strip, h - y0)]
    return out


def _radiance_detail(lin):
    """Shading of bright surfaces (door panels, mouldings, wall grooves) measured in the true
    scene light, before any tone step squeezes the highlights. Half resolution, log scale."""
    h, w = lin.shape[:2]
    Y = cv2.resize(lin.mean(2), (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
    lg = np.log(np.maximum(Y, 1e-5))
    base = cv2.GaussianBlur(lg, (0, 0), max(3.0, w / 2 / 400))
    d = np.clip(lg - base, -0.25, 0.25)
    d = cv2.GaussianBlur(d, (0, 0), 1.0)  # panel/groove scale, not noise
    return d.astype(np.float32)


def _apply_highlight_detail(img, geometry, d, strength=22.0, strip=512):
    """Put that shading back into the bright, washed-out areas of the finished photo."""
    h, w = img.shape[:2]
    dd = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    dd = geometry(dd)
    out = np.empty_like(img)
    for y0 in range(0, h, strip):
        lab = cv2.cvtColor(np.clip(img[y0:y0 + strip], 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        L = lab[..., 0]
        Lb = cv2.GaussianBlur(L, (0, 0), 4)
        # only where the photo is bright and has lost its own shading
        own = np.abs(L - Lb)
        wgt = np.clip((Lb - 76) / 6, 0, 1) * np.clip((3.0 - own) / 2.0, 0, 1)
        lab[..., 0] = np.clip(L + strength * dd[y0:y0 + strip] * wgt, 0, 100)
        out[y0:y0 + strip] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    return out


def _wood(img, chroma=0.85, dl=9.0, lift=24.0, strip=512):
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
    # floors: soften big reflection streaks (large-scale light/dark) by 40%, grain untouched
    mf = np.clip(1 - np.abs(hue - 70) / 35, 0, 1) * np.clip((C - 8) / 5, 0, 1) * np.clip((L - 20) / 10, 0, 1) * np.clip((70 - L) / 10, 0, 1)
    mf = cv2.GaussianBlur(mf, (0, 0), 4)
    Llo = cv2.GaussianBlur(L, (0, 0), 6)
    Lref = cv2.GaussianBlur(L * mf, (0, 0), 60) / np.maximum(cv2.GaussianBlur(mf, (0, 0), 60), 1e-3)
    fl = np.clip((mf - 0.3) / 0.4, 0, 1)
    # glare = brighter AND washed-out (less coloured) than the rest of the floor; an evenly light
    # oak floor is left alone
    Clo = cv2.GaussianBlur(C, (0, 0), 6)
    Cref = cv2.GaussianBlur(C * mf, (0, 0), 60) / np.maximum(cv2.GaussianBlur(mf, (0, 0), 60), 1e-3)
    washed = np.clip((Cref - Clo - 0.5) / 3, 0, 1)
    glare = np.maximum(Llo - Lref, 0) * washed
    dL = cv2.GaussianBlur(m * dL, (0, 0), 3) - 0.6 * glare * fl
    # glare washes the wood colour out: give it back where we darken
    fc_glare = 1 + np.clip(glare / 15, 0, 1) * 0.35 * fl
    fc = cv2.GaussianBlur((1 + (chroma - 1) * m) * fc_glare, (0, 0), 3)
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
    hl_detail = None
    if len(frames) > 1:
        frames = _align(frames)
        # windows = blown in a normal exposure too (with 4-5 brackets the brightest one also
        # blows out floor reflections, which are not windows)
        bright = frames[1 if len(frames) >= 4 else 0].max(2)
        dark_lin = _srgb_lin(frames[-1]) / 1.6
        # editor-style blend: brightest photo as the base, darker photos only where it is blown
        # (soft masks, same exposure) -> no halos around doors, no dark-bracket colour on floors
        lin, _ = _blend_render(frames)
        hl_detail = _radiance_detail(lin)
        fused = _to_display(lin)
        del lin
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
        try:
            img = _window_tint(img, geometry, bright)
        except Exception:
            pass
        try:
            img = _flare_tint(img, geometry, bright)
        except Exception:
            pass
        del bright, dark_lin
        gc.collect()
    img = _wood(img)
    img = _floor_reflections(img)
    img = _highlight_detail(img)
    img = _floor_clarity(img)
    if hl_detail is not None:
        try:
            img = _apply_highlight_detail(img, geometry, hl_detail)
        except Exception:
            pass
        del hl_detail
    img = _soft_contrast(img)
    try:
        img = _deflare(img)
    except Exception:
        pass
    img = _white_surfaces(img)
    img = _wall_marks(img)
    before = cv2.resize(geometry(mid), (preview_width, int(preview_width * h / w)), interpolation=cv2.INTER_AREA)
    del mid
    out = (img * 255 + 0.5).astype(np.uint8)
    after = cv2.resize(out, (preview_width, int(preview_width * out.shape[0] / out.shape[1])),
                       interpolation=cv2.INTER_AREA)
    return out, (before * 255 + 0.5).astype(np.uint8), after
