<script lang="ts">
  /**
   * Batch grid tile: read-only display of one queued crop with its predicted
   * boxes and whole-image predicted reading, plus the tile's state chrome.
   *
   * The canvas mirrors the review workspace's rendering at tile scale: the
   * crop with its predicted boxes (class glyph on a class-coloured chip,
   * tile-size constants in `labels.ts`) and the reading on a dark chip at the
   * image's top-centre. State chrome is DOM, not drawn: the `PREDICTED EMPTY`
   * badge, the pending mark label (`REJECTED` / `NEEDS MANUAL`, with the
   * tile's border coloured to match), a malformed prediction's parse message,
   * and an image-load-failure note. There is no caption strip by design.
   *
   * The whole tile is one button: a plain click toggles the reject mark, a
   * shift-click toggles the needs-manual mark (the parent owns the marks and
   * decides what a toggle means via `ontoggle`).
   */
  import { loadImage, api } from "./api";
  import { clamp, fromNormalized } from "./boxes";
  import {
    TILE_LABEL_TEXT_PX, TILE_READING_TEXT_PX, chipSize, drawChip, readingOf,
  } from "./labels";
  import { classColor, classTextColor } from "./palette";
  import type { BatchEntry, ClassInfo, TileMark } from "./types";

  /** Largest tile footprint the crop is fitted into (CSS px). */
  const MAX_W = 200;
  const MAX_H = 130;

  let {
    entry,
    classes,
    mark,
    ontoggle,
  }: {
    entry: BatchEntry;
    /** Class presentation (index = class id): chips draw `glyph`. */
    classes: ClassInfo[];
    /** The tile's pending mark ("" = accepted on commit). */
    mark: TileMark;
    /** Report a click: true for shift-click (flag), false for plain (reject). */
    ontoggle: (manual: boolean) => void;
  } = $props();

  let cv: HTMLCanvasElement | undefined = $state(undefined);
  let img = $state<HTMLImageElement | null>(null);
  let failed = $state(false);

  // the grid keys tiles by path, so each tile loads its image once
  $effect(() => {
    const path = entry.path;
    img = null;
    failed = false;
    loadImage(api.imageURL(path)).then(
      (im) => {
        img = im;
      },
      () => {
        failed = true;
      },
    );
  });

  /** Draw the crop, its predicted boxes, and its reading (tile-scale chips). */
  function render(): void {
    const c = cv;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const im = img;
    if (!im) {
      c.width = 0;
      c.height = 0;
      return;
    }
    const z = Math.min(MAX_W / im.width, MAX_H / im.height);
    const W = im.width * z, H = im.height * z;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(W * dpr);
    c.height = Math.round(H * dpr);
    c.style.width = W + "px";
    c.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(im, 0, 0, W, H);

    const pred = entry.boxes;
    if (pred === null) return; // malformed: nothing parseable to draw
    const boxes = pred.map((b) => fromNormalized(b, im.width, im.height));
    for (const b of boxes) {
      const color = classColor(b.cls);
      const x = b.x * z, y = b.y * z, w = b.w * z, h = b.h * z;
      ctx.fillStyle = classColor(b.cls, 0.12);
      ctx.fillRect(x, y, w, h);
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      const glyph = classes[b.cls]?.glyph ?? String(b.cls);
      const chip = {
        textPx: TILE_LABEL_TEXT_PX,
        bold: true,
        bg: color,
        fg: classTextColor(b.cls),
      };
      const size = chipSize(ctx, glyph, chip);
      const cx = clamp(x, 0, Math.max(0, W - size.w));
      const cy = y >= size.h ? y - size.h : y + h;
      drawChip(ctx, glyph, chip, cx, clamp(cy, 0, Math.max(0, H - size.h)));
    }

    // whole-image reading on a dark chip at the tile's top-centre ("—"
    // placeholder for a predicted background, shrunk to fit narrow crops)
    const text = readingOf(boxes, classes) || "—";
    let style = {
      textPx: TILE_READING_TEXT_PX,
      mono: true,
      bg: "rgba(0,0,0,0.72)",
      fg: entry.empty ? "#aaa" : "#fff",
    };
    let size = chipSize(ctx, text, style);
    const maxW = W - 8;
    if (size.w > maxW && maxW > 0) {
      style = {
        ...style,
        textPx: Math.max(8, Math.round(style.textPx * (maxW / size.w))),
      };
      size = chipSize(ctx, text, style);
    }
    drawChip(ctx, text, style, clamp(Math.round((W - size.w) / 2), 0, W), 4);
  }

  // re-render on any state the drawing depends on (boxes reads are deep-tracked)
  $effect(() => {
    render();
  });
</script>

<button
  class="tile"
  class:reject={mark === "reject"}
  class:manual={mark === "manual"}
  title="click: reject · shift-click: flag for manual adjustment"
  onclick={(e) => ontoggle(e.shiftKey)}
>
  <canvas bind:this={cv}></canvas>
  {#if entry.empty}
    <span class="chip empty">PREDICTED EMPTY</span>
  {/if}
  {#if mark === "reject"}
    <span class="chip reject-mark">REJECTED</span>
  {:else if mark === "manual"}
    <span class="chip manual-mark">NEEDS MANUAL</span>
  {/if}
  {#if failed}
    <span class="failed">image failed</span>
  {/if}
  {#if entry.error}
    <span class="error">{entry.error}</span>
  {/if}
</button>

<style>
  .tile {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 180px;
    min-height: 110px;
    padding: 0;
    border: 2px solid var(--edge);
    border-radius: 6px;
    background: #101216;
    overflow: hidden;
    cursor: pointer;
    font: inherit;
  }
  .tile:hover {
    border-color: var(--dim);
  }
  .tile.reject {
    border-color: var(--error);
  }
  .tile.manual {
    border-color: var(--accent);
  }
  canvas {
    display: block;
    image-rendering: pixelated;
  }
  .chip {
    position: absolute;
    top: 4px;
    padding: 1px 5px;
    border-radius: 4px;
    font: 10px ui-monospace, monospace;
    background: rgba(0, 0, 0, 0.72);
    color: #fff;
    pointer-events: none;
  }
  .chip.empty {
    left: 4px;
  }
  .chip.reject-mark {
    right: 4px;
    background: var(--error);
    font-weight: 600;
  }
  .chip.manual-mark {
    right: 4px;
    background: var(--accent);
    font-weight: 600;
  }
  .failed {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--dim);
    font-size: 12px;
  }
  .error {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 3px 5px;
    background: rgba(0, 0, 0, 0.82);
    color: var(--error);
    font-size: 10px;
    text-align: left;
    word-break: break-word;
    pointer-events: none;
  }
</style>
