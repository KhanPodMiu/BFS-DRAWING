"""Click a colored point on screen -> BFS the same-color region -> drag the mouse over it."""
import math
import re
import sys
sys.setrecursionlimit(100_000)  # strokes() DFS recurses once per centerline cell
import time
from collections import deque

import pyautogui
from PIL import Image, ImageChops, ImageFilter, ImageStat

TOL = 45            # max per-channel RGB difference from the clicked color; 50+ leaks black ink into dark window borders
INK = 12            # min per-channel difference from the local background to count as a stroke
STEP = 2            # grid step in screen points (smaller = denser stroke, slower)
# shortcut: the annotated image must be a mid gray between these, the app UI darker (dark theme) or
# brighter (white toolbars); widen the band for very dark boards or blown-out white ones
CANVAS_MIN, CANVAS_MAX = 50, 240
LOW = 0.6           # ink this fraction of a box's cut still counts when it hangs onto full-strength ink
NORM = 6            # a word is resampled so its strokes are about this many cells thick before tracing
SPEED = 240         # pen speed in screen points per second; apps round off the corners of fast strokes

pyautogui.FAILSAFE = True  # slam the mouse into a screen corner to abort
pyautogui.PAUSE = 0.003  # delay after each press / release / pyautogui move; the drag itself is paced by SPEED

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = DIRS + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def background(img, radius):
    """Local background estimate: a blur much wider than a stroke washes the ink out."""
    return img.convert("RGB").filter(ImageFilter.BoxBlur(radius))


def is_ink(c, b):
    return max(abs(x - y) for x, y in zip(c[:3], b[:3])) >= INK


def bfs(img, start, tol=TOL, step=1, bg=None):
    """Grid cells (i, j) -> pixel start + (i, j) * step that are ink (stand out from the local
    background) and close to the start color. The ink test stops leaks into a background whose
    color is near the stroke's (dim photos, dark window borders)."""
    px = img.load()
    bgpx = (bg or background(img, 8 * step)).load()
    w, h = img.size
    sx, sy = start
    base = px[sx, sy][:3]

    def ok(i, j):
        x, y = sx + i * step, sy + j * step
        return (0 <= x < w and 0 <= y < h and is_ink(px[x, y], bgpx[x, y])
                and all(abs(a - b) <= tol for a, b in zip(px[x, y][:3], base)))

    seen = {(0, 0)}
    q = deque([(0, 0)])
    while q:
        i, j = q.popleft()
        for di, dj in DIRS:
            n = (i + di, j + dj)
            if n not in seen and ok(*n):
                seen.add(n)
                q.append(n)
    return seen


def thin(cells):
    """Zhang-Suen thinning: shrink the filled stroke down to its 1-cell-wide centerline."""
    s = set(cells)

    def ring(i, j):  # P2..P9, clockwise from north (y grows downward)
        return [(i, j - 1) in s, (i + 1, j - 1) in s, (i + 1, j) in s, (i + 1, j + 1) in s,
                (i, j + 1) in s, (i - 1, j + 1) in s, (i - 1, j) in s, (i - 1, j - 1) in s]

    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            rm = []
            for c in s:
                p = ring(*c)
                if not 2 <= sum(p) <= 6 or sum(not p[k] and p[(k + 1) % 8] for k in range(8)) != 1:
                    continue
                p2, _, p4, _, p6, _, p8, _ = p
                if (p2 and p4 and p6 or p4 and p6 and p8) if phase == 0 else (p2 and p4 and p8 or p2 and p6 and p8):
                    continue
                rm.append(c)
            if rm:
                s.difference_update(rm)
                changed = True
    return s


def radius(cells):
    """Per cell: steps to the nearest edge of the ink (a 4-neighbour distance transform)."""
    rad = {c: 0 for c in cells if any((c[0] + di, c[1] + dj) not in cells for di, dj in DIRS)}
    q = deque(rad)
    while q:
        c = q.popleft()
        for di, dj in DIRS:
            n = (c[0] + di, c[1] + dj)
            if n in cells and n not in rad:
                rad[n] = rad[c] + 1
                q.append(n)
    return rad


def extend_tips(path, cells, back=4):
    """Thinning eats about a stroke radius off every free end, so legs and tails come out short.
    Walk each tip (both ends and every turnaround) on along its own direction until leaving the ink.
    A tip that touches another part of the path is no free end: the pen came round a loop (an o, the
    bowl of an R) and stopped one cell short of where it started, so there the loop is closed instead."""
    on_path = set(path)

    def ext(tip, frm):
        dx, dy = tip[0] - frm[0], tip[1] - frm[1]
        n = math.hypot(dx, dy)
        out, t = [], 1
        while n and (round(tip[0] + dx / n * t), round(tip[1] + dy / n * t)) in cells:
            out.append((tip[0] + dx / n * t, tip[1] + dy / n * t))
            t += 1
        return out

    def closes(tip, nb):
        for dx, dy in DIRS8:
            q = (tip[0] + dx, tip[1] + dy)
            if q in on_path and q != nb and max(abs(q[0] - nb[0]), abs(q[1] - nb[1])) > 1:
                return q

    if len(path) < 2:
        return path
    b = min(back, len(path) - 1)
    out = [] if closes(path[0], path[1]) else ext(path[0], path[b])[::-1]
    for k, c in enumerate(path):
        out.append(c)
        if 0 < k < len(path) - 1 and path[k - 1] == path[k + 1]:
            q = closes(c, path[k - 1])
            e = [q] if q else ext(c, path[max(0, k - b)])
            out += e + e[-2::-1] + ([c] if q else [])  # out to the real tip (or across the gap) and back
    q = closes(path[-1], path[-2])
    return out + ([q] if q else ext(path[-1], path[-1 - b]))


def strokes(skel, cells):
    """One continuous pen path over the centerline, the way a hand writes it: from the left end along
    the longest route, dipping into side branches (the middle leg of an m) by going down and back up.
    Returned as a one-item list of lines."""
    skel = set(skel) or {min(cells)}  # thinning erases a 2x2 dot (an i dot, a period) completely

    def nbrs(c):
        i, j = c
        out = []
        for di, dj in DIRS8:
            n = (i + di, j + dj)
            # skip a diagonal that just cuts the corner of an L-turn, else it becomes a tiny extra stroke
            if n in skel and not (di and dj and ((i + di, j) in skel or (i, j + dj) in skel)):
                out.append(n)
        return out

    # prune thinning spurs. A twig no longer than the stroke is thick where it branches off is noise;
    # anything longer is a real branch (a fat, short m leg stays even though it is only a few cells)
    rad = radius(cells)
    for end in [c for c in skel if len(nbrs(c)) == 1]:
        twig, seen_twig, cur = [], set(), end
        while True:
            twig.append(cur)
            seen_twig.add(cur)
            nxt = [n for n in nbrs(cur) if n not in seen_twig]
            if len(nxt) != 1:
                break
            cur = nxt[0]
        if len(nxt) >= 2 and len(twig) - 1 <= rad.get(twig[-1], 0) + 1:
            skel.difference_update(twig[:-1])

    def bfs_tree(src):
        par, q = {src: None}, deque([src])
        while q:
            c = q.popleft()
            for n in nbrs(c):
                if n not in par:
                    par[n] = c
                    q.append(n)
        return c, par  # c = farthest cell from src

    a, _ = bfs_tree(next(iter(skel)))
    b, _ = bfs_tree(a)
    start, end = min(a, b), max(a, b)  # the two ends of the longest route; write from the left one
    _, par = bfs_tree(start)
    main, c = set(), end
    while c is not None:
        main.add(c)
        c = par[c]

    # DFS that always walks back after a branch (so the pen never jumps), taking the onward route last;
    # everything after the final new cell is pure walking back, so it is cut off.
    path, seen, last_new = [], set(), 0

    def go(c):
        nonlocal last_new
        path.append(c)
        seen.add(c)
        last_new = len(path)
        for n in sorted(nbrs(c), key=lambda n: n in main):
            if n not in seen:
                go(n)
                path.append(c)

    go(start)
    return [path[:last_new]]


def smooth(line, r=3):
    """Moving average over up to 2r+1 points to iron out the grid staircase. The ends and every
    turnaround (down a side branch and back) stay pinned, else averaging would shorten them."""
    pins = sorted({0, len(line) - 1, *(k for k in range(1, len(line) - 1) if line[k - 1] == line[k + 1])})
    out = list(line)
    for a, b in zip(pins, pins[1:]):
        for k in range(a + 1, b):
            w = line[max(a, k - r):min(b, k + r) + 1]
            out[k] = (sum(p[0] for p in w) / len(w), sum(p[1] for p in w) / len(w))
    return out


def components(cells, dirs=DIRS):
    """Split a set of grid cells into its connected blobs (lists of cells). Empties the set."""
    while cells:
        comp = [cells.pop()]
        for x, y in comp:  # comp grows while we walk it
            for dx, dy in dirs:
                n = (x + dx, y + dy)
                if n in cells:
                    cells.remove(n)
                    comp.append(n)
        yield comp


def strength(img, bg):
    """Per pixel, how far img is from bg (an image, or one color), as an L image. Measured in grayscale:
    video stores color at a fraction of the resolution, so the color difference of red ink is a blocky
    smear around the stroke while its brightness difference is as sharp as black ink's.
    shortcut: ink exactly as bright as the board is invisible here, add a color term if that ever shows up"""
    if not isinstance(bg, Image.Image):
        bg = Image.new("RGB", img.size, bg)
    return ImageChops.difference(img.convert("L"), bg.convert("L"))


def otsu(hist):
    """The cut that best splits a strength histogram into background and ink (never below INK)."""
    n, total, n0, s0, best, cut = sum(hist), sum(v * c for v, c in enumerate(hist)), 0, 0, -1, INK
    for v, c in enumerate(hist):
        n0, s0 = n0 + c, s0 + v * c
        if 0 < n0 < n and (var := n0 * (n - n0) * (s0 / n0 - (total - s0) / (n - n0)) ** 2) > best:
            best, cut = var, max(INK, v)
    return cut


def find_canvas(shot, k=8):
    """Pixel box of the image being annotated: the largest blob that is neither dark UI nor white UI
    (buttons, labels and thumbnails are far smaller). None if the screen has no such blob."""
    # sampled, not averaged: averaging turns every white-on-dark UI edge into a mid-gray seam that joins the image
    small = shot.convert("L").resize((shot.width // k, shot.height // k), 0)
    px = small.load()
    best = max(components({(x, y) for x in range(small.width) for y in range(small.height)
                           if CANVAS_MIN < px[x, y] < CANVAS_MAX}), key=len, default=None)
    if not best:
        return None
    xs, ys = [x for x, _ in best], [y for _, y in best]
    return (min(xs) * k, min(ys) * k, min(shot.width, (max(xs) + 1) * k), min(shot.height, (max(ys) + 1) * k))


def ink_boxes(img):
    """Pixel boxes around every ink blob in img, writing or not: brackets, arrows, lone digits and
    whatever else OCR does not read as text."""
    k = max(1, img.width // 600)  # coarse is enough to find blobs, the tracing itself looks again at full detail
    small = img.reduce(k)
    # background = a median far wider than any stroke, which removes the ink outright. A blur would only
    # smear it into a halo that then counts as ink itself
    s = strength(small, small.reduce(4).filter(ImageFilter.MedianFilter(9)).resize(small.size, Image.BILINEAR))
    # a fixed cut, not Otsu: one toolbar or one bright pen stroke in view would drag a global cut far above pale ink
    px, out = s.load(), []
    for comp in components({(x, y) for x in range(s.width) for y in range(s.height) if px[x, y] > INK}, DIRS8):
        xs, ys = [x for x, _ in comp], [y for _, y in comp]
        # specks are noise; a blob as wide or tall as the image is its frame, not ink
        if len(comp) < 3 or max(xs) - min(xs) > 0.7 * s.width or max(ys) - min(ys) > 0.7 * s.height:
            continue
        # padded so the box holds more background than ink even around a single straight bar
        out.append((max(0, (min(xs) - 4) * k), max(0, (min(ys) - 4) * k),
                    min(img.width, (max(xs) + 5) * k), min(img.height, (max(ys) + 5) * k)))
    return out


def scan(shot):
    """-> (canvas box or None, clickable boxes) in screenshot pixels. Boxes = the words OCR finds in
    the annotated image, plus every other ink blob there that no word box already covers."""
    canvas = find_canvas(shot)
    if canvas is None:
        return None, []
    crop = shot.crop(canvas)
    # OCR only the image itself, so the app's own labels and buttons are never candidates
    words = ocr_words(crop)
    rest = [b for b in ink_boxes(crop)
            if not any(w[0] <= (b[0] + b[2]) / 2 < w[2] and w[1] <= (b[1] + b[3]) / 2 < w[3] for w in words)]
    boxes = [(b[0] + canvas[0], b[1] + canvas[1], b[2] + canvas[0], b[3] + canvas[1]) for b in words + rest]
    # toolbars floating over the image (zoom %, pen picker) sit on white or black, not on the board
    return canvas, [b for b in boxes if b[2] > b[0] and b[3] > b[1] and CANVAS_MIN < sum(box_bg(shot, b)) / 3 < CANVAS_MAX]


def ocr_words(img):
    """[(x0, y0, x1, y1) in img pixels] for every word macOS Vision can read in img. Only where the
    writing is matters, what it says is dropped."""
    import io
    import Vision

    buf = io.BytesIO()
    img.save(buf, "PNG")
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(buf.getvalue(), None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setUsesLanguageCorrection_(False)  # handwriting + math: "correcting" to dictionary words only hurts
    handler.performRequests_error_([req], None)
    out = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)[0]
        for m in re.finditer(r"\S+", cand.string()):
            rect, _ = cand.boundingBoxForRange_error_((m.start(), m.end() - m.start()), None)
            if rect is None:
                continue
            bb = rect.boundingBox()  # normalized, origin bottom-left
            out.append((round(bb.origin.x * img.width), round((1 - bb.origin.y - bb.size.height) * img.height),
                        round((bb.origin.x + bb.size.width) * img.width), round((1 - bb.origin.y) * img.height)))
    return out


def to_lines(cells, start_px, step_px, scale):
    """bfs() cells -> pen paths in screen points."""
    return [[((start_px[0] + i * step_px) / scale, (start_px[1] + j * step_px) / scale)
             for i, j in smooth(extend_tips(l, cells))] for l in strokes(thin(cells), cells)]


def box_bg(shot, box):
    """Background color of a pixel box: the per-channel median (ink covers far less than half of a word's box)."""
    return tuple(ImageStat.Stat(shot.crop(box)).median)


def word_lines(shot, box, scale):
    """Pen paths for every ink blob lying mostly inside the pixel box (one OCR word or one blob), left to right.
    Unlike bfs() the ink test adapts to the box: the cut between ink and background is picked per box,
    and the box is resampled so strokes are NORM cells thick whatever the zoom. A cell is then ink only
    if it is also darker than its surroundings two strokes around, which keeps the hole of an o and the
    gaps between m legs open even when blur has turned them gray (else the letter traces as one bar)."""
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return []
    m = (y1 - y0) // 2  # blobs may run out of the box by this much (descenders, ascenders)
    ex = (max(0, x0 - m), max(0, y0 - m), min(shot.width, x1 + m), min(shot.height, y1 + m))
    s = strength(shot.crop(ex), box_bg(shot, box))
    inner = s.crop((x0 - ex[0], y0 - ex[1], x1 - ex[0], y1 - ex[1]))
    cut = otsu(inner.histogram())

    # stroke thickness = 2 * area / outline length, measured on the rough ink mask
    rough = inner.point(lambda v: 255 if v > cut else 0)
    area = rough.histogram()[255]
    if not area:
        return []
    thick = 2 * area / max(1, area - rough.filter(ImageFilter.MinFilter(3)).histogram()[255])
    f = min(3, max(0.25, NORM / thick))
    s = s.resize((max(1, round(s.width * f)), max(1, round(s.height * f))), Image.BICUBIC).filter(ImageFilter.GaussianBlur(1))
    px, around = s.load(), s.filter(ImageFilter.BoxBlur(NORM)).load()
    # somewhat paler than the cut also counts, but only in blobs that reach full strength somewhere (below):
    # that keeps the pale joins between m legs and the fading end of an underline. Never below INK, where
    # video noise lives (it turned faint red writing into scribble)
    low = max(INK, LOW * cut)
    ink = {(i, j) for i in range(s.width) for j in range(s.height) if px[i, j] > low and px[i, j] >= 0.9 * around[i, j]}

    bx0, by0, bx1, by1 = [(v - o) * f for v, o in zip(box, ex[:2] * 2)]
    blobs = []
    for comp in components(ink):
        # specks are noise; a blob mostly outside the box is a neighbour poking in (a g tail from the line above)
        if (len(comp) >= NORM and any(px[c] > cut for c in comp)
                and 2 * sum(bx0 <= a < bx1 and by0 <= b < by1 for a, b in comp) > len(comp)):
            blobs.append((min(a for a, _ in comp), set(comp)))
    return [l for _, cells in sorted(blobs, key=lambda t: t[0])
            for l in to_lines(cells, (ex[0] + 0.5 / f, ex[1] + 0.5 / f), 1 / f, scale)]


def lines_at(shot, bg, x, y, step_px, scale):
    """Pen paths for the single stroke under screen point (x, y); [] if there is no ink there."""
    # thin strokes: the click often lands on a faint edge, so start from the strongest ink nearby
    cx, cy, r = round(x * scale), round(y * scale), 2 * step_px
    start_px = max(((px, py) for px in range(max(0, cx - r), min(shot.width, cx + r + 1))
                    for py in range(max(0, cy - r), min(shot.height, cy + r + 1))),
                   key=lambda p: max(abs(a - b) for a, b in zip(shot.getpixel(p), bg.getpixel(p))))
    color = shot.getpixel(start_px)
    if not is_ink(color, bg.getpixel(start_px)):
        print(f"Màu {color} lẫn vào nền xung quanh, hãy click đúng vào nét chữ.")
        return []
    return to_lines(bfs(shot, start_px, step=step_px, bg=bg), start_px, step_px, scale)


def draw(lines, quit_, paused):
    """Drag the mouse along the pen paths. Returns False if a was pressed mid-drawing."""
    import Quartz

    if not lines:
        return True
    DOWN, DRAG, UP = Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseDragged, Quartz.kCGEventLeftMouseUp

    def post(kind, px, py, pause=pyautogui.PAUSE):
        # Our own events instead of pyautogui's: its moveTo sends "mouse moved" even with the button
        # down (apps see no drag), and synthetic presses default to click count 0 and pressure 0,
        # which browsers / ink tools may ignore.
        e = Quartz.CGEventCreateMouseEvent(None, kind, (px, py), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventSetIntegerValueField(e, Quartz.kCGMouseEventClickState, 1)
        Quartz.CGEventSetDoubleValueField(e, Quartz.kCGMouseEventPressure, 0.0 if kind == UP else 1.0)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)
        time.sleep(pause)

    try:
        # The picking click was swallowed, so the app under the stroke may still be in the background,
        # and macOS spends the first press on a background window just bringing it forward.
        # Click once on the stroke start to focus it (at worst a dot that the stroke covers anyway).
        pyautogui.moveTo(*lines[0][0])
        post(DOWN, *lines[0][0])
        post(UP, *lines[0][0])
        time.sleep(0.3)
        for line in lines:
            pyautogui.moveTo(*line[0])
            post(DOWN, *line[0])
            time.sleep(0.03)  # let the app register the press before the drag starts
            for prev, p in zip(line, line[1:]):
                if quit_.is_set():
                    return False
                if paused.is_set():
                    # lift the pen while paused so moving the mouse paints nothing, then put it back down here
                    post(UP, *prev)
                    while paused.is_set() and not quit_.is_set():
                        time.sleep(0.05)
                    if quit_.is_set():
                        return False
                    pyautogui.moveTo(*prev)
                    post(DOWN, *prev)
                    time.sleep(0.03)
                pyautogui.failSafeCheck()
                post(DRAG, *p, math.dist(prev, p) / SPEED)
            post(UP, *line[-1])
    finally:
        pyautogui.mouseUp()
    print("Xong.")
    return True


class Overlay:
    """Click-through transparent window over the whole screen that outlines the OCR'd words."""

    def __init__(self):
        import AppKit
        import Quartz
        self.AppKit, self.Quartz = AppKit, Quartz
        AppKit.NSApplication.sharedApplication().setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        frame = AppKit.NSScreen.screens()[0].frame()
        self.h = frame.size.height
        self.win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False)
        self.win.setOpaque_(False)
        self.win.setBackgroundColor_(AppKit.NSColor.clearColor())
        self.win.setIgnoresMouseEvents_(True)
        self.win.setLevel_(AppKit.NSScreenSaverWindowLevel)
        self.win.contentView().setWantsLayer_(True)
        self.win.orderFrontRegardless()

    def show(self, boxes):
        """boxes: [(x0, y0, x1, y1) in screen points, top-left origin]; [] clears the screen."""
        Quartz, root = self.Quartz, self.win.contentView().layer()
        root.setSublayers_(None)
        green = Quartz.CGColorCreateGenericRGB(0.1, 1.0, 0.3, 1.0)
        for x0, y0, x1, y1 in boxes:
            box = Quartz.CALayer.layer()
            box.setFrame_(((x0, self.h - y1), (x1 - x0, y1 - y0)))  # layers count y from the bottom
            box.setBorderWidth_(1)
            box.setBorderColor_(green)
            root.addSublayer_(box)
        self.pump(0.05)

    def pump(self, seconds):
        """Let AppKit put our changes on screen (we never enter its main loop)."""
        self.AppKit.NSRunLoop.currentRunLoop().runUntilDate_(self.AppKit.NSDate.dateWithTimeIntervalSinceNow_(seconds))


def main():
    """Loop: OCR the annotated image -> outline the words -> click a word and it gets traced whole
    (a click on ink outside every box traces just that stroke). Pressing a quits, m pauses / resumes, also mid-drawing."""
    import threading
    import Quartz
    from pynput import keyboard, mouse

    armed, picked, quit_, rescan, paused = (threading.Event() for _ in range(5))
    pos, canvas_pt = [], []

    def intercept(event_type, event):
        # While armed, swallow the picking click so it never reaches the app (a pen tool would paint a
        # dot there). Clicks outside the image (Save & Next, Undo...) go through to the app and trigger
        # a rescan, as do all clicks while we are not armed, including our own synthetic presses.
        if not armed.is_set():
            return event
        if event_type == Quartz.kCGEventLeftMouseDown:
            p = Quartz.CGEventGetLocation(event)
            if not (canvas_pt[0] <= p.x < canvas_pt[2] and canvas_pt[1] <= p.y < canvas_pt[3]):
                return event
            pos[:] = [(int(p.x), int(p.y))]
            return None
        if event_type == Quartz.kCGEventLeftMouseUp:
            if pos:
                armed.clear()
                picked.set()
                return None
            rescan.set()
        return event

    def on_key(k):
        c = getattr(k, "char", None)
        if c in ("a", "A"):
            quit_.set()
        elif c in ("m", "M"):
            paused.clear() if paused.is_set() else paused.set()
        elif c in ("r", "R"):
            rescan.set()

    # Both started once and never stopped: pynput's stop() leaves the event tap installed but
    # unserviced, which swallowed our mouse-downs. Their daemon threads die with the script.
    mouse.Listener(darwin_intercept=intercept).start()
    keyboard.Listener(on_press=on_key).start()
    overlay = Overlay()

    print("Phím a: thoát (hoặc đưa chuột vào góc màn hình). Phím m: tạm dừng / chạy tiếp. Phím r: quét lại (sau khi zoom / kéo ảnh).")
    while True:
        overlay.show([])  # our boxes must not end up in the screenshot
        overlay.pump(0.1)
        shot = pyautogui.screenshot().convert("RGB")
        scale = shot.width / pyautogui.size().width  # Retina: screenshot pixels = 2x screen points
        step_px = max(1, round(STEP * scale))
        canvas, words = scan(shot)
        if canvas is None:
            print("Không thấy vùng ảnh nào trên màn hình. Mở ảnh lên rồi bấm r.")
            canvas = (0, 0, 0, 0)
        canvas_pt[:] = [v / scale for v in canvas]
        overlay.show([canvas_pt] + [[v / scale for v in b] for b in words])
        print(f"\nTìm thấy {len(words)} chỗ có nét.")
        print("Click vào một khung để đồ các nét trong đó, hoặc click thẳng vào một nét nằm ngoài khung...")

        pos.clear()
        picked.clear()
        rescan.clear()
        armed.set()
        while not picked.is_set() and not rescan.is_set():
            overlay.pump(0.1)
            if quit_.is_set():
                armed.clear()
                print("Đã thoát (a).")
                return
        armed.clear()
        if not picked.is_set():
            overlay.pump(0.7)  # let the app finish reacting to the click (loading the next image)
            continue
        overlay.show([])  # get the boxes out of the way of the drawing
        x, y = pos[0]
        hit = [b for b in words if b[0] <= x * scale < b[2] and b[1] <= y * scale < b[3]]
        if hit:
            box = min(hit, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))  # overlapping boxes: the tightest
            lines = word_lines(shot, box, scale)
            print(f"Đồ trong khung: {len(lines)} nét.")
        else:
            lines = lines_at(shot, background(shot, 8 * step_px), x, y, step_px, scale)
            print(f"Đồ nét tại {(x, y)}: {len(lines)} nét.")
        if not draw(lines, quit_, paused):
            print("Đã dừng giữa chừng và thoát (a).")
            return
        overlay.pump(0.3)  # let the app paint the new strokes before the next scan


def self_test():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (100, 100), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((10, 10, 15, 80), fill="black")   # L shape
    d.rectangle((10, 75, 70, 80), fill="black")
    d.rectangle((50, 10, 60, 20), fill="black")   # separate blob, must not be reached
    cells = bfs(img, (12, 12), step=2)
    assert all(img.getpixel((12 + i * 2, 12 + j * 2)) == (0, 0, 0) for i, j in cells)
    assert all(12 + i * 2 < 50 or 12 + j * 2 > 20 for i, j in cells), "leaked into separate blob"
    assert (0, 34) in cells and (29, 34) in cells, "missed part of the L"
    skel = thin(cells)
    assert skel <= cells and len(skel) < len(cells) / 2, "thinning did not shrink the stroke"
    lines = strokes(skel, cells)
    assert len(lines) == 1, f"an L should be one stroke, got {len(lines)}"
    line = lines[0]
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(line, line[1:]))
    assert len(set(line)) == len(line), "stroke revisits a cell"
    stair = [(k // 2 + k % 2, k // 2) for k in range(20)]  # 1-cell staircase along a diagonal
    sm = smooth(stair)
    assert sm[0] == stair[0] and sm[-1] == stair[-1]
    # raw stair swings x-y between 0 and 1 around the ideal line x-y=0.5
    assert all(abs(px - py - 0.5) <= 0.15 for px, py in sm[3:-3]), "staircase not ironed out"

    # dim chalkboard: faint stroke only ~30 darker than a gray-green board whose color is within TOL of it
    dim = Image.new("RGB", (200, 100), (85, 90, 80))
    ImageDraw.Draw(dim).line((20, 50, 180, 50), fill=(55, 60, 52), width=5)
    cells = bfs(dim, (100, 50), step=2)
    xs = [i for i, _ in cells]
    ys = [j for _, j in cells]
    assert (max(xs) - min(xs)) * 2 >= 150, "dim stroke cut short"
    assert (max(ys) - min(ys)) * 2 <= 6, "leaked into the board"

    # handwritten-style m: three legs joined by two humps must come out as ONE continuous pen path
    m = Image.new("RGB", (140, 110), "white")
    d = ImageDraw.Draw(m)
    d.line((20, 35, 20, 95), fill="black", width=8)
    d.arc((20, 15, 70, 65), 180, 360, fill="black", width=8)
    d.line((66, 40, 66, 95), fill="black", width=8)
    d.arc((66, 15, 116, 65), 180, 360, fill="black", width=8)
    d.line((112, 40, 112, 95), fill="black", width=8)
    cells = bfs(m, (20, 90), step=2)
    (path,) = strokes(thin(cells), cells)
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(path, path[1:])), "pen jumps"
    to_px = [(20 + i * 2, 90 + j * 2) for i, j in path]
    assert to_px[0][0] < 30 and to_px[-1][0] > 100, "should write left leg -> right leg"
    assert any(path[k - 1] == path[k + 1] for k in range(1, len(path) - 1)), "middle leg not retraced"
    sm = smooth(path)
    tip = max(range(len(path)), key=lambda k: path[k][1] if 50 < 20 + path[k][0] * 2 < 80 else -99)
    assert sm[tip] == path[tip], "turnaround at the middle leg's foot must stay pinned"

    # fat stroke: thinning shortens the ends and the short fat middle leg; extend_tips must win them back
    fat = Image.new("RGB", (220, 120), "white")
    d = ImageDraw.Draw(fat)
    d.line((20, 40, 200, 40), fill="black", width=18)
    d.line((110, 40, 110, 75), fill="black", width=22)   # short fat leg, ends at y=75
    cells = bfs(fat, (60, 40), step=2)
    (path,) = strokes(thin(cells), cells)
    full = [(60 + i * 2, 40 + j * 2) for i, j in extend_tips(path, cells)]
    assert min(x for x, _ in full) <= 22 and max(x for x, _ in full) >= 198, "line ends not reached"
    assert max(y for _, y in full) >= 72, "short fat leg lost or cut short"

    # dark app UI with a bright button and a big image: the image is the canvas
    ui = Image.new("RGB", (800, 600), (30, 30, 30))
    d = ImageDraw.Draw(ui)
    d.rectangle((40, 100, 599, 499), fill=(120, 135, 125))
    d.rectangle((650, 100, 750, 140), fill=(0, 110, 190))
    d.rectangle((0, 0, 799, 99), fill="white")  # white toolbar touching the image must not join it
    box = find_canvas(ui)
    assert all(abs(a - b) <= 8 for a, b in zip(box, (40, 100, 600, 500))), box
    assert find_canvas(Image.new("RGB", (80, 80), (30, 30, 30))) is None

    # a word = two separate letters inside the box; a stroke outside the box must be left alone
    w = Image.new("RGB", (300, 120), "white")
    d = ImageDraw.Draw(w)
    d.line((30, 30, 30, 90), fill="black", width=6)
    d.line((80, 30, 80, 90), fill="black", width=6)
    d.line((200, 30, 200, 90), fill="black", width=6)
    wl = word_lines(w, (10, 10, 120, 110), 1)
    assert len(wl) == 2, f"expected the 2 letters in the box, got {len(wl)}"
    assert wl[0][0][0] < 50 < wl[1][0][0] < 120, "letters should come left to right"
    assert all(max(y for _, y in l) - min(y for _, y in l) >= 50 for l in wl), "letter cut short"
    # thick soft-edged stroke (zoomed-in blurry frame): must stay ONE path covering its whole length
    soft = Image.new("RGB", (400, 160), (120, 135, 125))
    ImageDraw.Draw(soft).line((40, 80, 360, 80), fill=(50, 60, 55), width=40)
    (l,) = word_lines(soft.filter(ImageFilter.GaussianBlur(10)), (10, 10, 390, 150), 1)
    assert l[-1][0] - l[0][0] >= 300 and all(abs(y - 80) <= 10 for _, y in l), "soft thick stroke shredded"

    # blurry o: the gray hole must stay open, so the path goes around it instead of straight through
    o = Image.new("RGB", (200, 200), (120, 135, 125))
    ImageDraw.Draw(o).ellipse((70, 70, 130, 130), outline=(50, 60, 55), width=14)
    (l,) = word_lines(o.filter(ImageFilter.GaussianBlur(5)), (40, 40, 160, 160), 1)
    assert all(math.hypot(x - 100, y - 100) >= 12 for x, y in l), "hole of the o got filled"
    assert math.dist(l[0], l[-1]) <= 4, "the o should end where it started, with no tail shooting off"

    # m whose joins are much paler than its legs: still one blob, the pale parts are kept
    pm = Image.new("RGB", (220, 140), (120, 135, 125))
    d = ImageDraw.Draw(pm)
    for x in (40, 100, 160):
        d.line((x, 40, x, 110), fill=(45, 55, 50), width=7)
    d.line((40, 40, 160, 40), fill=(95, 108, 100), width=7)
    assert len(word_lines(pm.filter(ImageFilter.GaussianBlur(1.5)), (15, 15, 205, 125), 1)) == 1, "pale joins dropped"

    # non-text ink (a bracket-like bar and a ring) gets its own boxes
    nb = Image.new("RGB", (600, 300), (120, 135, 125))
    d = ImageDraw.Draw(nb)
    d.line((100, 60, 100, 240), fill=(45, 55, 50), width=6)
    d.ellipse((350, 100, 430, 180), outline=(45, 55, 50), width=6)
    bx = ink_boxes(nb)
    assert len(bx) == 2 and all(any(b[0] <= x < b[2] and b[1] <= y < b[3] for b in bx) for x, y in ((100, 150), (390, 100))), bx
    assert len(word_lines(nb, min(bx), 1)) == 1, "the bar inside its own box should trace as one stroke"
    print("self-test ok")


if __name__ == "__main__":
    self_test() if "--test" in sys.argv else main()
