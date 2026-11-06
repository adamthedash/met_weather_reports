/** Box geometry: hit-testing, view fitting, and image-pixel/normalised conversion. */

import type { Box, BoxN } from "./types";

/** A point in image pixels. */
export type Pt = { x: number; y: number };

/** A resize handle of a box (corner or edge midpoint). */
export type Handle = { id: string; x: number; y: number };

/** Clamp `v` to the range [lo, hi]. */
export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/**
 * Integer magnification that fits the image into the available area, so
 * pixels stay crisp; falls back to fractional scaling only for images that
 * would overflow at 1:1 (full-page scans need zoom below 1×).
 */
export function fitScale(
  img: { width: number; height: number }, availW: number, availH: number,
): number {
  const fit = Math.min(availW / img.width, availH / img.height);
  return fit >= 1 ? Math.floor(fit) : fit;
}

/** The eight resize-handle positions (corners and edge midpoints) of box `b`. */
export function handlePoints(b: Box): Handle[] {
  const x1 = b.x, y1 = b.y, x2 = b.x + b.w, y2 = b.y + b.h;
  const xm = b.x + b.w / 2, ym = b.y + b.h / 2;
  return [
    { id: "nw", x: x1, y: y1 }, { id: "n", x: xm, y: y1 }, { id: "ne", x: x2, y: y1 },
    { id: "e", x: x2, y: ym }, { id: "se", x: x2, y: y2 }, { id: "s", x: xm, y: y2 },
    { id: "sw", x: x1, y: y2 }, { id: "w", x: x1, y: ym },
  ];
}

/** Resize handle of box `selected` at image point (px, py), or null. */
export function hitHandle(
  boxes: Box[], selected: number, px: number, py: number, zoom: number,
): string | null {
  if (selected < 0 || selected >= boxes.length) return null;
  const r = 7 / zoom;
  for (const h of handlePoints(boxes[selected])) {
    if (Math.abs(px - h.x) <= r && Math.abs(py - h.y) <= r) return h.id;
  }
  return null;
}

/** Topmost box at image point (px, py), or -1 when the point hits none. */
export function hitBox(boxes: Box[], px: number, py: number, zoom: number): number {
  const g = 4 / zoom;
  for (let i = boxes.length - 1; i >= 0; i--) {
    const b = boxes[i];
    if (px >= b.x - g && px <= b.x + b.w + g && py >= b.y - g && py <= b.y + b.h + g) {
      return i;
    }
  }
  return -1;
}

/** Normalise an image-pixel box to YOLO wire format (6 decimals). */
export function toNormalized(b: Box, iw: number, ih: number): BoxN {
  return {
    cls: b.cls,
    cx: +((b.x + b.w / 2) / iw).toFixed(6),
    cy: +((b.y + b.h / 2) / ih).toFixed(6),
    w: +(b.w / iw).toFixed(6),
    h: +(b.h / ih).toFixed(6),
  };
}

/** Denormalise a YOLO wire box to image pixels. */
export function fromNormalized(b: BoxN, iw: number, ih: number): Box {
  return {
    cls: b.cls,
    x: (b.cx - b.w / 2) * iw,
    y: (b.cy - b.h / 2) * ih,
    w: b.w * iw,
    h: b.h * ih,
  };
}
