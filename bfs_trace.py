"""Click a colored point on screen -> BFS the same-color region -> drag the mouse over it."""
import math
import sys
sys.setrecursionlimit(100_000)  # strokes() DFS recurses once per centerline cell
import time
from collections import deque

import pyautogui
from PIL import ImageFilter

TOL = 45            # max per-channel RGB difference from the clicked color; 50+ leaks black ink into dark window borders
INK = 12            # min per-channel difference from the local background to count as a stroke
STEP = 2            # grid step in screen points (smaller = denser stroke, slower)

pyautogui.FAILSAFE = True  # slam the mouse into a screen corner to abort
pyautogui.PAUSE = 0.003  # delay per drag point; higher = slower, steadier drawing

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


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
    Walk each tip (both ends and every turnaround) on along its own direction until leaving the ink."""
    def ext(tip, frm):
        dx, dy = tip[0] - frm[0], tip[1] - frm[1]
        n = math.hypot(dx, dy)
        out, t = [], 1
        while n and (round(tip[0] + dx / n * t), round(tip[1] + dy / n * t)) in cells:
            out.append((tip[0] + dx / n * t, tip[1] + dy / n * t))
            t += 1
        return out

    if len(path) < 2:
        return path
    b = min(back, len(path) - 1)
    out = ext(path[0], path[b])[::-1]
    for k, c in enumerate(path):
        out.append(c)
        if 0 < k < len(path) - 1 and path[k - 1] == path[k + 1]:
            e = ext(c, path[max(0, k - b)])
            out += e + e[-2::-1]  # out to the real tip and back
    return out + ext(path[-1], path[-1 - b])


def strokes(skel, cells):
    """One continuous pen path over the centerline, the way a hand writes it: from the left end along
    the longest route, dipping into side branches (the middle leg of an m) by going down and back up.
    Returned as a one-item list of lines."""
    skel = set(skel)

    def nbrs(c):
        i, j = c
        out = []
        for di, dj in DIRS + ((1, 1), (1, -1), (-1, 1), (-1, -1)):
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


def to_svg(lines, width=3):
    xs = [x for l in lines for x, _ in l]
    ys = [y for l in lines for _, y in l]
    d = " ".join("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in l) for l in lines)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min(xs) - 5:.0f} {min(ys) - 5:.0f} '
            f'{max(xs) - min(xs) + 10:.0f} {max(ys) - min(ys) + 10:.0f}">'
            f'<path d="{d}" fill="none" stroke="black" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>\n')


def wait_click():
    # Swallow the picking click so it never reaches the app (a pen tool would paint a dot there
    # and the screenshot would then read the pen color instead of the stroke). macOS only.
    import threading
    import Quartz
    from pynput import mouse
    pos, picked = [], threading.Event()

    def intercept(event_type, event):
        if picked.is_set():
            return event  # after the pick, let everything through, including our own synthetic presses
        if event_type == Quartz.kCGEventLeftMouseDown:
            p = Quartz.CGEventGetLocation(event)
            pos.append((int(p.x), int(p.y)))
            return None
        if event_type == Quartz.kCGEventLeftMouseUp and pos:
            picked.set()
            return None
        return event

    # Not stopped on purpose: pynput's stop() leaves the event tap installed but unserviced, which kept
    # swallowing our mouse-downs (cursor moved, nothing drew). Its daemon thread dies with the script.
    mouse.Listener(darwin_intercept=intercept).start()
    picked.wait()
    return pos[0]


def main():
    print("Click vào 1 điểm màu trên màn hình... (đưa chuột vào góc màn hình để dừng khẩn cấp)")
    x, y = wait_click()
    shot = pyautogui.screenshot()
    scale = shot.width / pyautogui.size().width  # Retina: screenshot pixels = 2x screen points
    step_px = max(1, round(STEP * scale))
    shot = shot.convert("RGB")
    bg = background(shot, 8 * step_px)
    # thin strokes: the click often lands on a faint edge, so start from the strongest ink nearby
    cx, cy, r = round(x * scale), round(y * scale), 2 * step_px
    start_px = max(((px, py) for px in range(max(0, cx - r), min(shot.width, cx + r + 1))
                    for py in range(max(0, cy - r), min(shot.height, cy + r + 1))),
                   key=lambda p: max(abs(a - b) for a, b in zip(shot.getpixel(p), bg.getpixel(p))))
    x, y = start_px[0] / scale, start_px[1] / scale

    color = shot.getpixel(start_px)
    if not is_ink(color, bg.getpixel(start_px)):
        print(f"Màu {color} lẫn vào nền xung quanh, hãy click đúng vào nét chữ.")
        return
    cells = bfs(shot, start_px, step=step_px, bg=bg)
    lines = [[(x + i * step_px / scale, y + j * step_px / scale) for i, j in smooth(extend_tips(l, cells))]
             for l in strokes(thin(cells), cells)]
    print(f"Màu {color} tại {(x, y)}: {len(cells)} điểm -> {len(lines)} nét. Bấm Esc để dừng.")

    import Quartz
    from pynput import keyboard

    DOWN, DRAG, UP = Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseDragged, Quartz.kCGEventLeftMouseUp

    def post(kind, px, py):
        # Our own events instead of pyautogui's: its moveTo sends "mouse moved" even with the button
        # down (apps see no drag), and synthetic presses default to click count 0 and pressure 0,
        # which browsers / ink tools may ignore.
        e = Quartz.CGEventCreateMouseEvent(None, kind, (px, py), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventSetIntegerValueField(e, Quartz.kCGMouseEventClickState, 1)
        Quartz.CGEventSetDoubleValueField(e, Quartz.kCGMouseEventPressure, 0.0 if kind == UP else 1.0)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)
        time.sleep(pyautogui.PAUSE)

    stop = []
    keys = keyboard.Listener(on_press=lambda k: stop.append(1) if k == keyboard.Key.esc else None)
    keys.start()
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
            for p in line[1:]:
                if stop:
                    print("Đã dừng (Esc).")
                    return
                pyautogui.failSafeCheck()
                post(DRAG, *p)
            post(UP, *line[-1])
    finally:
        pyautogui.mouseUp()
        keys.stop()
    name = time.strftime("trace_%H%M%S.svg")
    with open(name, "w") as f:
        f.write(to_svg(lines))
    print(f"Xong. Đã lưu {name}")


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
    assert to_svg([[(0, 0), (1, 2)]]).count("<path") == 1

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
    print("self-test ok")


if __name__ == "__main__":
    self_test() if "--test" in sys.argv else main()
