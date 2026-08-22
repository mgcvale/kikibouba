"""Offline companion to index.html's visualisation panels.

Rerun this whenever the model is retrained or re-exported:

    python tools/make_vis_data.py --generator ~/Downloads/curr_generator.py

It does three things:

1. Adds the intermediate tensors the page wants to draw (`hardtanh*`,
   `max_pool2d_2`, `cat`) to `model_web.onnx`'s output list. Pure graph
   surgery -- no weights change and `logit` stays bit-identical.
2. Runs a batch of freshly generated synthetic shapes through the model to
   build the latent-space reference cloud and the 2-D projection basis.
3. Harvests the top-activating input patches for every last-layer channel
   into a sprite sheet, so the page can show what each filter looks for.

Everything lands in `vis_data.json` + `patches.png`, which index.html fetches.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

# --- must stay in sync with index.html ---
MODEL_SIZE = 192
INK_THRESHOLD = 250
TARGET_FILL = 0.55
MIN_COMPONENT_FRACTION = 0.05

# Tensors to expose, with the shapes they take at MODEL_SIZE. Names come from
# the dynamo exporter; if a retrain renames them, fix them here and in the
# LAYERS table in index.html.
EXPOSE = [
    ("hardtanh", [8, MODEL_SIZE, MODEL_SIZE]),
    ("hardtanh_1", [16, MODEL_SIZE // 2, MODEL_SIZE // 2]),
    ("hardtanh_2", [32, MODEL_SIZE // 4, MODEL_SIZE // 4]),
    ("max_pool2d_2", [32, MODEL_SIZE // 8, MODEL_SIZE // 8]),
    ("cat", [64]),
]

# Receptive field of one `hardtanh_2` unit, in input pixels, and the stride of
# that grid. Three 3x3 same-padded convs with two 2x2 pools between them:
# rf 3 -> 4 -> 8 -> 10 -> 18, stride 1 -> 2 -> 2 -> 4 -> 4, and the centre of
# unit j lands at 4j + 1.5.
RF_SIZE = 18
RF_STRIDE = 4
RF_OFFSET = 1.5
PATCH = 24          # crop a little wider than the RF, for context
PATCHES_PER_CHANNEL = 6


def expose_intermediates(path):
    m = onnx.load(path)
    g = m.graph
    produced = {o for n in g.node for o in n.output}
    have = {o.name for o in g.output}
    added = []
    for name, shape in EXPOSE:
        if name in have:
            continue
        if name not in produced:
            sys.exit(f"tensor {name!r} is not produced by this graph -- did the "
                     f"architecture change? nodes produce: {sorted(produced)}")
        g.output.append(helper.make_tensor_value_info(
            name, TensorProto.FLOAT, ["batch"] + shape))
        added.append(name)
    if added:
        onnx.checker.check_model(m)
        onnx.save_model(m, path, save_as_external_data=False)
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    return added, init


def load_generator(path):
    spec = importlib.util.spec_from_file_location("curr_generator", os.path.expanduser(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preprocess(img):
    """Crop-and-rescale exactly as index.html's toModelTensor does.

    The reference cloud is only comparable to a visitor's drawing if both
    reach the model through the same normalisation, so this mirrors the JS:
    threshold to ink, drop specks, take the bounding box, and rescale it to
    TARGET_FILL of a square frame.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ink = lum < INK_THRESHOLD
    if not ink.any():
        return None

    # Speck filter, same intent as inkClusters: an isolated dot must not be
    # allowed to blow the bounding box up and shrink the real shape.
    n, labels, stats, _ = __import__("cv2").connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    areas = stats[1:, 4]
    if len(areas) == 0:
        return None
    keep = 1 + np.flatnonzero(areas >= areas.max() * MIN_COMPONENT_FRACTION)
    mask = np.isin(labels, keep)
    ys, xs = np.nonzero(mask)
    minX, maxX, minY, maxY = xs.min(), xs.max(), ys.min(), ys.max()

    side = max(maxX - minX + 1, maxY - minY + 1) / TARGET_FILL
    cx, cy = (minX + maxX) / 2, (minY + maxY) / 2
    box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    # White fill outside the source, matching the canvas's transparent-reads-white.
    crop = img.convert("L").transform(
        (MODEL_SIZE, MODEL_SIZE), Image.EXTENT, box,
        resample=Image.BILINEAR, fillcolor=255)
    return np.asarray(crop, dtype=np.float32) / 255.0


def build(args):
    added, init = expose_intermediates(args.model)
    print(f"model: exposed {added or '(already present)'}")

    w = init["classifier_head.1.weight"].reshape(-1).astype(np.float64)
    bias = float(init["classifier_head.1.bias"].reshape(-1)[0])
    conv1 = init["backbone.0.block.0.weight"]           # [8,1,3,3]
    n_feat = conv1.shape[0]
    n_last = len(w) // 2
    w_gap, w_gmp = w[:n_last], w[n_last:]

    gen = load_generator(args.generator)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    random.seed(args.seed)
    np.random.seed(args.seed)

    embeds, labels, logits = [], [], []
    frames = []                       # kept for patch harvesting
    acts = []                         # hardtanh_2 per sample

    per_class = args.samples // 2
    for is_kiki in (False, True):
        made = 0
        while made < per_class:
            # "plain" is the domain that looks like the app: white ground,
            # near-black stroke, no fill. Mixing the textured domains in would
            # spread the cloud by paper texture rather than by shape, which is
            # the opposite of the point being made.
            img = gen.generate_sample(is_kiki=is_kiki, variance_level=1.5, domain="plain")
            x = preprocess(img)
            if x is None:
                continue
            out = sess.run(["cat", "logit", "hardtanh_2"],
                           {"input": x[None, None].astype(np.float32)})
            embeds.append(out[0][0])
            logits.append(float(out[1].ravel()[0]))
            labels.append(1 if is_kiki else 0)
            acts.append(out[2][0])
            frames.append((x * 255).astype(np.uint8))
            made += 1
        print(f"embedded {per_class} {'kiki' if is_kiki else 'bouba'} samples")

    E = np.array(embeds, dtype=np.float64)
    y = np.array(labels)
    mu = E.mean(0)
    C = E - mu

    # --- 2-D basis ---------------------------------------------------------
    # x is the decision direction itself: the classifier reads the embedding
    # through exactly one vector, w, so projecting onto w/|w| gives an axis on
    # which position is (up to the bias) the logit. That makes the horizontal
    # position of a point *mean* something a visitor can check against the
    # verdict, which no t-SNE axis ever does.
    #
    # y is the leading direction of what is left once w is removed -- the
    # biggest source of variation the classifier ignores. Together they give a
    # stable, out-of-sample-friendly projection: a new drawing is one dot
    # product away from a position, so a visitor's shape lands on the plot
    # instantly. t-SNE has no such mapping (it would need refitting per point)
    # and its axes carry no units, so it is the wrong tool for a live exhibit.
    u1 = w / np.linalg.norm(w)
    R = C - np.outer(C @ u1, u1)
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    u2 = Vt[0]
    u2 -= (u2 @ u1) * u1
    u2 /= np.linalg.norm(u2)

    P = np.stack([C @ u1, C @ u2], 1)
    # Report the separation actually achieved, so a bad retrain is obvious.
    sep = abs(P[y == 1, 0].mean() - P[y == 0, 0].mean()) / (P[:, 0].std() + 1e-9)
    print(f"latent: axis-1 class separation {sep:.2f} sd, "
          f"acc {( (np.array(logits) > 0) == (y == 1) ).mean():.3f}")

    # --- per-channel behaviour --------------------------------------------
    # Three separate numbers, because conflating them is easy and wrong:
    #
    #   firesOn   which class actually excites the channel, measured from the
    #             activations alone with no reference to the weights. This is
    #             what the top-activating patches illustrate.
    #   push      which way firing moves the logit -- purely the sign of the
    #             channel's weights. Activations are non-negative (the block
    #             ends in a clip), so a negative weight means "firing is
    #             evidence *against* kiki".
    #   strength  how much the channel actually shifts the logit between the
    #             classes. Ranks the channels and exposes dead ones: a large
    #             weight on a filter that never fires decides nothing.
    #
    # Only `strength` is a "does this channel matter" measure. It is positive
    # for every informative channel by construction -- a channel that fires on
    # boubas and subtracts scores exactly as positive as one that fires on
    # kikis and adds -- so it says nothing on its own about which class the
    # channel is *for*. That question is firesOn x push.
    A = np.array([a.reshape(n_last, -1) for a in acts])       # [N, C, HW]
    gap, gmp = A.mean(2), A.max(2)
    contrib = gap * w_gap + gmp * w_gmp                        # [N, C]
    strength = contrib[y == 1].mean(0) - contrib[y == 0].mean(0)

    peak_k, peak_b = gmp[y == 1].mean(0), gmp[y == 0].mean(0)
    fires_on = (peak_k - peak_b) / (gmp.mean(0) + 1e-9)
    alive = gmp.mean(0) > 0.01 * gmp.mean()
    n_bouba_det = int(((w_gmp < 0) & (fires_on < 0) & alive).sum())
    print(f"channels: {int(alive.sum())}/{n_last} alive, "
          f"{int((w_gmp[alive] > 0).sum())} push kiki when firing, "
          f"{int((w_gmp[alive] < 0).sum())} push bouba; "
          f"{n_bouba_det} are genuine bouba detectors "
          f"(fire on boubas AND subtract)")

    # --- top-activating patches -------------------------------------------
    # Classic feature-visualisation: for each channel, the input crops that
    # made it fire hardest. Restricted to one patch per source image so a
    # single lucky shape cannot fill a whole row.
    sheet = np.full((n_last * PATCH, PATCHES_PER_CHANNEL * PATCH), 255, np.uint8)
    peaks = np.array([a.reshape(n_last, -1).max(1) for a in acts])   # [N, C]
    half = PATCH // 2
    for c in range(n_last):
        order = np.argsort(-peaks[:, c])[:PATCHES_PER_CHANNEL]
        for k, idx in enumerate(order):
            a = acts[idx][c]
            i, j = np.unravel_index(np.argmax(a), a.shape)
            cy = int(round(RF_STRIDE * i + RF_OFFSET))
            cx = int(round(RF_STRIDE * j + RF_OFFSET))
            f = np.full((PATCH, PATCH), 255, np.uint8)
            y0, x0 = cy - half, cx - half
            sy0, sx0 = max(0, y0), max(0, x0)
            sy1, sx1 = min(MODEL_SIZE, y0 + PATCH), min(MODEL_SIZE, x0 + PATCH)
            f[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frames[idx][sy0:sy1, sx0:sx1]
            sheet[c * PATCH:(c + 1) * PATCH, k * PATCH:(k + 1) * PATCH] = f
    Image.fromarray(sheet).save(args.patches)
    print(f"patches: {args.patches} ({sheet.shape[1]}x{sheet.shape[0]})")

    data = {
        "modelSize": MODEL_SIZE,
        "targetFill": TARGET_FILL,
        "classifier": {"w": w.tolist(), "b": bias, "nLast": n_last},
        "latent": {
            "mu": mu.tolist(),
            "u1": u1.tolist(),
            "u2": u2.tolist(),
            # Rounded hard -- the cloud is decoration at plot scale and this
            # keeps the JSON a fifth of the size.
            "points": [[round(float(a), 3), round(float(b), 3), int(l)]
                       for (a, b), l in zip(P, y)],
            "separation": round(float(sep), 3),
        },
        "channels": {
            "strength": [round(float(v), 5) for v in strength],
            "firesOn": [round(float(v), 4) for v in fires_on],
            "alive": [bool(v) for v in alive],
            "wGap": [round(float(v), 5) for v in w_gap],
            "wGmp": [round(float(v), 5) for v in w_gmp],
            "patchSize": PATCH,
            "patchesPerChannel": PATCHES_PER_CHANNEL,
            "rfSize": RF_SIZE,
            "rfStride": RF_STRIDE,
            "rfOffset": RF_OFFSET,
        },
        "conv1": {
            "n": int(n_feat),
            "k": int(conv1.shape[-1]),
            "weights": [[round(float(v), 5) for v in f.reshape(-1)] for f in conv1],
        },
        # Per-connection summaries for the network diagram. A connection
        # between two channels is a whole 3x3 kernel, not a scalar, so it gets
        # two numbers: `norm` (how strongly the pair is coupled at all, drawn
        # as line width) and `sum` (the kernel's net gain, whose sign picks the
        # line colour). Indexed [out * nIn + in].
        "convs": [
            {
                "nOut": int(W.shape[0]),
                "nIn": int(W.shape[1]),
                "norm": [round(float(v), 4) for v in
                         np.linalg.norm(W.reshape(W.shape[0], W.shape[1], -1), axis=2).reshape(-1)],
                "sum": [round(float(v), 4) for v in
                        W.reshape(W.shape[0], W.shape[1], -1).sum(2).reshape(-1)],
            }
            for W in (init["backbone.0.block.0.weight"],
                      init["backbone.1.block.0.weight"],
                      init["backbone.2.block.0.weight"])
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model_web.onnx")
    p.add_argument("--generator", default="~/Downloads/curr_generator.py")
    p.add_argument("--samples", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="vis_data.json")
    p.add_argument("--patches", default="patches.png")
    build(p.parse_args())
