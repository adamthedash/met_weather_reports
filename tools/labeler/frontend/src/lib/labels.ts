/**
 * Box-label chip tunables, drawing, and whole-image reading composition.
 *
 * The label text sizes are constants so they can be tweaked without touching
 * the renderer: change one and rebuild the frontend. Chip padding scales with
 * the text size, so one knob keeps the chip proportions. Sizes are screen CSS
 * px and do not scale with the crop's zoom.
 */
import { classGlyph } from "./palette";
import type { Box } from "./types";

/** Class-label chip text size — the bold chip on every box. */
export const LABEL_TEXT_PX = 22;

/** Confidence-chip text size — the separate prediction confidence chip. */
export const CONF_TEXT_PX = 16;

/** Whole-image reading text size — the review sidebar's big reading string. */
export const READING_TEXT_PX = 60;

/** Horizontal chip padding as a fraction of the text size. */
const PAD_X = 0.4;

/** Vertical chip padding as a fraction of the text size. */
const PAD_Y = 0.18;

/** Chip metrics: text size/weight and optional padding overrides (px). */
export type ChipMetrics = {
  /** Chip text size (CSS px). */
  textPx: number;
  /** Bold text (class chip) vs regular (confidence chip, badges). */
  bold?: boolean;
  /** Horizontal padding; defaults to a fraction of `textPx`. */
  padX?: number;
  /** Vertical padding; defaults to a fraction of `textPx`. */
  padY?: number;
};

/** Chip look: metrics plus fill and text colour. */
export type ChipStyle = ChipMetrics & {
  /** Chip fill. Opaque when the text colour is contrast-picked against it. */
  bg: string;
  /** Chip text colour. */
  fg: string;
};

/** Canvas font string for a chip's metrics. */
function fontOf(m: ChipMetrics): string {
  return `${m.bold ? "bold " : ""}${m.textPx}px system-ui, sans-serif`;
}

/** Measured chip size (px) of `text` (text plus padding). */
export function chipSize(
  ctx: CanvasRenderingContext2D, text: string, m: ChipMetrics,
): { w: number; h: number } {
  ctx.font = fontOf(m);
  return {
    w: Math.round(ctx.measureText(text).width + 2 * (m.padX ?? PAD_X * m.textPx)),
    h: Math.round(m.textPx + 2 * (m.padY ?? PAD_Y * m.textPx)),
  };
}

/**
 * Draw `text` as a chip whose top-left corner is at (x, y): a filled rect
 * with the text vertically centred and padded. Leaves no lasting canvas
 * state behind.
 */
export function drawChip(
  ctx: CanvasRenderingContext2D, text: string, s: ChipStyle,
  x: number, y: number,
): void {
  ctx.save();
  const { w, h } = chipSize(ctx, text, s);
  const padX = s.padX ?? PAD_X * s.textPx;
  ctx.fillStyle = s.bg;
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = s.fg;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + padX, y + h / 2);
  ctx.restore();
}

/**
 * Whole-image predicted reading of a box set: each box's class glyph (via
 * `classGlyph`, so "dot" renders "." and "dash" renders "-") concatenated in
 * left-to-right box-centre order with no separator, e.g. boxes 0, dot, 3, 2
 * read "0.32". Equal centres keep their input order (stable sort); every box
 * is included regardless of confidence. Returns "" for no boxes.
 */
export function readingOf(boxes: Box[], classes: string[]): string {
  return [...boxes]
    .sort((a, b) => a.x + a.w / 2 - (b.x + b.w / 2))
    .map((b) => {
      const name = classes[b.cls] ?? String(b.cls);
      return classGlyph(name) ?? name;
    })
    .join("");
}
