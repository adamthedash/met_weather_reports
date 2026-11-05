<script lang="ts">
  /**
   * Batch grid tile: read-only display of one queued crop with its predicted
   * boxes and whole-image predicted reading, plus the tile's state chrome.
   *
   * The canvas mirrors the review workspace's rendering at tile scale (the
   * fit box scales with the `scale` prop while chip text sizes stay fixed
   * screen px — the constants in `labels.ts`): the crop with its predicted
   * boxes — class glyph on a class-coloured chip plus the box's confidence on
   * a small neutral chip right-aligned at its top-right — and the reading on
   * a dark chip at the image's top-centre, which reads "EMPTY" for a
   * predicted background (the tile's only empty label). State chrome is DOM,
   * not drawn: the pending mark label (`REJECTED` / `NEEDS MANUAL`, with the
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
    TILE_CONF_TEXT_PX, TILE_LABEL_TEXT_PX, TILE_READING_TEXT_PX, chipSize,
    drawChip, readingOf,
  } from "./labels";
  import { classColor, classTextColor } from "./palette";
  import type { BatchEntry, ClassInfo, TileMark } from "./types";

  /** Largest tile footprint the crop is fitted into at scale 1 (CSS px). */
  const MAX_W = 200;
  const MAX_H = 130;

  let {
    entry,
    classes,
    mark,
    scale = 1,
    ontoggle,
  }: {
    entry: BatchEntry;
    /** Class presentation (index = class id): chips draw `glyph`. */
    classes: ClassInfo[];
    /** The tile's pending mark ("" = accepted on commit). */
    mark: TileMark;
    /** Tile size scale (the workspace's image-size ± controls). */
    scale?: number;
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
    const z = Math.min((MAX_W * scale) / im.width, (MAX_H * scale) / im.height);
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
    const boxes = pred.map((b) => ({
      ...fromNormalized(b, im.width, im.height),
      conf: b.conf,
    }));
    for (const b of boxes) {
      const color = classColor(b.cls);
      const x = b.x * z, y = b.y * z, w = b.w * z, h = b.h * z;
      ctx.fillStyle = classColor(b.cls, 0.12);
      ctx.fillRect(x, y, w, h);
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      // class glyph on a class-coloured chip, the box's confidence on a small
      // neutral chip right-aligned at its top-right (review's prominent
      // layout in miniature): one row above the box, flipped below it when
      // there is no room, both clamped into the tile
      const glyph = classes[b.cls]?.glyph ?? String(b.cls);
      const clsChip = {
        textPx: TILE_LABEL_TEXT_PX,
        bold: true,
        bg: color,
        fg: classTextColor(b.cls),
      };
      const confChip = {
        textPx: TILE_CONF_TEXT_PX,
        bg: "rgba(0,0,0,0.72)",
        fg: "#fff",
      };
      const confText = b.conf.toFixed(2);
      const lw = chipSize(ctx, glyph, clsChip);
      const cw = chipSize(ctx, confText, confChip);
      const band = Math.max(lw.h, cw.h);
      const rowTop = y >= band ? y - band : y + h;
      drawChip(
        ctx, glyph, clsChip,
        clamp(x, 0, Math.max(0, W - lw.w)),
        clamp(rowTop + (band - lw.h) / 2, 0, Math.max(0, H - lw.h)),
      );
      drawChip(
        ctx, confText, confChip,
        clamp(x + w - cw.w, 0, Math.max(0, W - cw.w)),
        clamp(rowTop + (band - cw.h) / 2, 0, Math.max(0, H - cw.h)),
      );
    }

    // the whole-image reading on a dark chip at the tile's top-centre; a
    // predicted background's chip reads "EMPTY" (it has no reading) as the
    // tile's only empty label, and narrow crops shrink the chip to fit
    const reading = readingOf(boxes, classes);
    const text = entry.empty ? "EMPTY" : reading || "—";
    let style = {
      textPx: TILE_READING_TEXT_PX,
      mono: true,
      bg: "rgba(0,0,0,0.72)",
      fg: entry.empty || reading ? "#fff" : "#aaa",
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
  style:min-width="{Math.round(180 * scale)}px"
  style:min-height="{Math.round(110 * scale)}px"
  title="click: reject · shift-click: flag for manual adjustment"
  onclick={(e) => ontoggle(e.shiftKey)}
>
  <canvas bind:this={cv}></canvas>
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
