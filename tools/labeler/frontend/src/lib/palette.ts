/** Class palette: deterministic colours per class index.
 *
 * Class presentation (display glyph, keyboard shortcut) is not derived here:
 * it is dataset configuration served by `/api/classes` (the dataset's
 * classes.yaml). */

/** Deterministic hue of a class index (the varying part of its colour). */
function classHue(cls: number): number {
  return Math.round((cls * 137.508) % 360);
}

/** Deterministic colour for a class index. */
export function classColor(cls: number, alpha = 1): string {
  return `hsla(${classHue(cls)}, 65%, 52%, ${alpha})`;
}

/** Relative luminance (WCAG) of an sRGB colour with channels in 0-1. */
function luminance(r: number, g: number, b: number): number {
  const lin = (c: number): number =>
    c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

/** WCAG contrast ratio between two relative luminances. */
function contrast(a: number, b: number): number {
  const hi = Math.max(a, b), lo = Math.min(a, b);
  return (hi + 0.05) / (lo + 0.05);
}

/**
 * Text colour to draw on top of a class colour: whichever of near-black and
 * white has the higher WCAG contrast ratio against `classColor(cls)` (alpha
 * 1). Class hues differ in luminance, so some classes get dark text and some
 * light.
 */
export function classTextColor(cls: number): string {
  const h = classHue(cls) / 360;
  const s = 0.65, l = 0.52;
  // HSL -> sRGB channels (0-1)
  const f = (n: number): number => {
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  const bg = luminance(f(0), f(8), f(4));
  const dark = 0x11 / 255;
  return contrast(bg, luminance(1, 1, 1)) >=
      contrast(bg, luminance(dark, dark, dark))
    ? "#fff"
    : "#111";
}
