"""
Reusable comparison montage: tile labeled images into a grid at reduced size.
Usage: python montage.py <out.png> <max_w> <img1> <label1> <img2> <label2> ...
"""
import sys
import cv2
import numpy as np


def load_label(path, label, cell_w):
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((10, cell_w, 3), np.uint8)
    h, w = img.shape[:2]
    sf = cell_w / w
    img = cv2.resize(img, (cell_w, int(h * sf)), interpolation=cv2.INTER_AREA)
    bar = np.zeros((34, cell_w, 3), np.uint8)
    cv2.putText(bar, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return np.vstack([bar, img])


def main():
    out = sys.argv[1]
    max_w = int(sys.argv[2])
    rest = sys.argv[3:]
    pairs = [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]

    n = len(pairs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    cell_w = max_w // cols

    cells = [load_label(p, l, cell_w) for p, l in pairs]
    max_h = max(c.shape[0] for c in cells)
    cells = [np.vstack([c, np.zeros((max_h - c.shape[0], cell_w, 3), np.uint8)])
             if c.shape[0] < max_h else c for c in cells]

    grid_rows = []
    for r in range(rows):
        row_cells = cells[r * cols:(r + 1) * cols]
        while len(row_cells) < cols:
            row_cells.append(np.zeros((max_h, cell_w, 3), np.uint8))
        grid_rows.append(np.hstack(row_cells))
    grid = np.vstack(grid_rows)
    cv2.imwrite(out, grid)
    print(f'Montage saved: {out} ({grid.shape[1]}x{grid.shape[0]})')


if __name__ == '__main__':
    main()
