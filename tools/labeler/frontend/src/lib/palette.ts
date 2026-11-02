/** Class palette: deterministic colours and keyboard shortcuts. */

/** Deterministic colour for a class index. */
export function classColor(cls: number, alpha = 1): string {
  const h = Math.round((cls * 137.508) % 360);
  return `hsla(${h}, 65%, 52%, ${alpha})`;
}

/** Keyboard shortcut character for a class name, or null when it has none. */
export function shortcutFor(name: string): string | null {
  if (name === "dot") return ".";
  if (name === "dash") return "-";
  return name.length === 1 ? name : null;
}

/** Map every class shortcut to its class index (first class wins a key). */
export function buildShortcutMap(classes: string[]): Record<string, number> {
  const map: Record<string, number> = {};
  classes.forEach((name, i) => {
    const s = shortcutFor(name);
    if (s !== null && !(s in map)) map[s] = i;
  });
  return map;
}
