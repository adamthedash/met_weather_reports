<script lang="ts">
  /**
   * Shared canvas box editor: renders a crop with its boxes and runs the
   * draw/move/resize/delete gestures of the annotate workflow.
   *
   * State lives in the parent workspace (via `bind:boxes` / `bind:selected`);
   * the editor owns only the in-progress gesture and reports committed edits
   * with their pre-mutation snapshot so the parent can drive undo/autosave.
   * In `readonly` mode (used by the review workspace) gestures are disabled
   * and the canvas is display-only. Boxes with `source: "pred"` are stroked
   * dashed so the audit workspace can overlay predictions on annotations.
   * Box labels render in one of two styles: `labelStyle: "plain"` keeps
   * annotate's compact single tag; `"prominent"` (review/audit) draws the
   * class glyph on a large class-coloured chip with contrast-picked text plus
   * the confidence on a separate small chip at the box's top-right (text
   * sizes are tweakable constants in `labels.ts`).
   */
  import {
    clamp, handlePoints, hitBox, hitHandle, type Pt,
  } from "./boxes";
  import {
    CONF_TEXT_PX, LABEL_TEXT_PX, READING_TEXT_PX, chipSize, drawChip,
  } from "./labels";
  import { classColor, classTextColor } from "./palette";
  import type { Box, ClassInfo } from "./types";

  type OpBase = { start: Pt; moved: boolean };
  type Op =
    | ({ type: "draw"; cur: Pt } & OpBase)
    | ({ type: "move"; idx: number; b0: Box; snapshot: Box[] } & OpBase)
    | ({ type: "resize"; id: string; idx: number; b0: Box; snapshot: Box[] } & OpBase);

  let {
    image,
    boxes = $bindable([]),
    selected = $bindable(-1),
    zoom,
    classes,
    currentClass = null,
    readonly = false,
    labelStyle = "plain",
    badge = "",
    reading = undefined,
    onedit = () => {},
    onstatus = () => {},
  }: {
    image: HTMLImageElement | null;
    boxes: Box[];
    selected?: number;
    zoom: number;
    /** Class presentation (index = class id): chips draw `glyph`. */
    classes: ClassInfo[];
    currentClass?: number | null;
    readonly?: boolean;
    /** Box-label rendering: "plain" keeps annotate's compact single tag;
     *  "prominent" (review/audit) draws the class on a large class-coloured
     *  chip and the confidence on a separate small chip at the box's
     *  top-right. */
    labelStyle?: "plain" | "prominent";
    /** Fixed-size badge text drawn in the image's top-left corner (e.g.
     *  "EMPTY", "PREDICTED EMPTY"); hidden when empty. */
    badge?: string;
    /** Whole-image reading drawn on a dark chip at the image's top-centre
     *  (review). Omitted draws nothing (annotate/audit); empty draws a dimmed
     *  "—" placeholder; any other string is the reading itself. */
    reading?: string;
    onedit?: (snapshot: Box[]) => void;
    onstatus?: (msg: string, kind?: "" | "ok" | "error") => void;
  } = $props();

  let cv: HTMLCanvasElement;
  let op = $state<Op | null>(null);

  /** Copy of the current boxes, suitable for the undo stack. */
  function snapshot(): Box[] {
    return boxes.map((b) => ({ ...b }));
  }

  /** Image-space coordinates of a mouse event over the canvas. */
  function pos(e: MouseEvent): Pt {
    const r = cv.getBoundingClientRect();
    return { x: (e.clientX - r.left) / zoom, y: (e.clientY - r.top) / zoom };
  }

  /** Abort the in-progress gesture (restoring its pre-drag boxes), or deselect. */
  export function cancelOp(): void {
    if (!op) {
      selected = -1;
      return;
    }
    if (op.type !== "draw") boxes = op.snapshot;
    op = null;
  }

  /**
   * Start a resize, move, or draw gesture from the mouse position.
   *
   * A plain press only edits the selected box (move its interior, resize via a
   * handle); anywhere else — including inside unselected boxes — draws a new
   * box. A shift+press is the select-and-move gesture: it selects the topmost
   * box under the cursor and starts moving it (handles never resize with
   * shift). Drawing requires a current class.
   */
  function onMouseDown(e: MouseEvent): void {
    if (readonly || e.button !== 0 || !image) return;
    e.preventDefault();
    const p = pos(e);
    const iw = image.width, ih = image.height;
    if (p.x < 0 || p.y < 0 || p.x > iw || p.y > ih) return;

    const idx = hitBox(boxes, p.x, p.y, zoom);
    if (e.shiftKey) {
      if (idx >= 0) {
        if (selected !== idx) selected = idx;
        op = {
          type: "move", idx, b0: { ...boxes[idx] }, start: p,
          snapshot: snapshot(), moved: false,
        };
        return;
      }
      // shift on empty space falls through to the plain behaviour below
    } else {
      const handle = hitHandle(boxes, selected, p.x, p.y, zoom);
      if (handle !== null) {
        op = {
          type: "resize", id: handle, idx: selected,
          b0: { ...boxes[selected] }, start: p,
          snapshot: snapshot(), moved: false,
        };
        return;
      }
      if (idx >= 0 && idx === selected) {
        op = {
          type: "move", idx, b0: { ...boxes[idx] }, start: p,
          snapshot: snapshot(), moved: false,
        };
        return;
      }
    }

    // a press outside the selected box deselects and draws a new box
    selected = -1;
    if (currentClass === null) {
      onstatus("pick a class first: press its key or click a chip", "error");
      return;
    }
    op = { type: "draw", start: p, cur: p, moved: false };
  }

  /** Advance the in-progress draw/move/resize to the current mouse position. */
  function onMouseMove(e: MouseEvent): void {
    const o = op;
    if (!o || !image) return;
    e.preventDefault();
    const p = pos(e);
    const iw = image.width, ih = image.height;
    if (
      Math.abs(p.x - o.start.x) * zoom > 2 ||
      Math.abs(p.y - o.start.y) * zoom > 2
    ) {
      o.moved = true;
    }

    if (o.type === "draw") {
      o.cur = { x: clamp(p.x, 0, iw), y: clamp(p.y, 0, ih) };
    } else if (o.type === "move") {
      const b = boxes[o.idx];
      b.x = clamp(o.b0.x + (p.x - o.start.x), 0, iw - b.w);
      b.y = clamp(o.b0.y + (p.y - o.start.y), 0, ih - b.h);
    } else if (o.type === "resize") {
      const b = boxes[o.idx];
      let x1 = o.b0.x, y1 = o.b0.y, x2 = o.b0.x + o.b0.w, y2 = o.b0.y + o.b0.h;
      if (o.id.includes("w")) x1 = clamp(p.x, 0, x2 - 1);
      if (o.id.includes("e")) x2 = clamp(p.x, x1 + 1, iw);
      if (o.id.includes("n")) y1 = clamp(p.y, 0, y2 - 1);
      if (o.id.includes("s")) y2 = clamp(p.y, y1 + 1, ih);
      b.x = x1;
      b.y = y1;
      b.w = x2 - x1;
      b.h = y2 - y1;
    }
  }

  /**
   * Finish the gesture, committing newly drawn or moved boxes.
   *
   * A drawn box under 3 screen-px is an accidental click and is discarded; a
   * move/resize only commits (and pushes its pre-drag snapshot) when the box
   * actually moved.
   */
  function onMouseUp(): void {
    const o = op;
    op = null;
    if (!o) return;

    if (o.type === "draw") {
      const a = o.start, b = o.cur;
      const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
      const w = Math.abs(b.x - a.x), h = Math.abs(b.y - a.y);
      if (w * zoom >= 3 && h * zoom >= 3 && currentClass !== null) {
        onedit(snapshot());
        // new boxes are never auto-selected; shift-click to edit this one
        boxes.push({ cls: currentClass, x, y, w, h });
      }
    } else if (o.moved) {
      onedit(o.snapshot);
    }
  }

  /** Right-click deletes the box under the cursor. */
  function onContext(e: MouseEvent): void {
    e.preventDefault();
    if (readonly || !image) return;
    const p = pos(e);
    const idx = hitBox(boxes, p.x, p.y, zoom);
    if (idx >= 0) {
      onedit(snapshot());
      boxes.splice(idx, 1);
      if (selected === idx) selected = -1;
      else if (selected > idx) selected--;
    }
  }

  /**
   * Prominent label chips of one box (review/audit). The class name goes on
   * a chip in the class colour with contrast-picked text; a prediction box
   * additionally gets its bare confidence on a separate neutral chip,
   * right-aligned on the same row. The row sits above the box and flips
   * below it when there is no room; a box too narrow for both chips wraps
   * the confidence chip to a second row beside the class chip. Chips are
   * clamped into the image. All geometry arguments are screen px.
   */
  function drawBoxChips(
    ctx: CanvasRenderingContext2D,
    b: Box,
    glyph: string,
    x: number,
    y: number,
    w: number,
    h: number,
    iw: number,
    ih: number,
  ): void {
    const GAP = 2; // px between the two chips and between wrapped rows
    const clsChip = {
      textPx: LABEL_TEXT_PX,
      bold: true,
      bg: classColor(b.cls),
      fg: classTextColor(b.cls),
    };
    const confText = b.conf !== undefined ? b.conf.toFixed(2) : null;
    const confChip = { textPx: CONF_TEXT_PX, bg: "rgba(0,0,0,0.72)", fg: "#fff" };

    const lw = chipSize(ctx, glyph, clsChip);
    const cw = confText !== null ? chipSize(ctx, confText, confChip) : null;
    const cwW = cw?.w ?? 0;
    const cwH = cw?.h ?? 0;

    const wrap = cw !== null && lw.w + cwW + GAP > w;
    const band = wrap ? lw.h + cwH + GAP : Math.max(lw.h, cwH);
    const above = y >= band;

    let clsY: number;
    let confY: number;
    if (wrap) {
      // class chip next to the box, confidence chip beyond it
      clsY = above ? y - lw.h : y + h;
      confY = above ? clsY - GAP - cwH : clsY + lw.h + GAP;
    } else {
      // one shared row, both chips centred in it
      const rowTop = above ? y - band : y + h;
      clsY = rowTop + (band - lw.h) / 2;
      confY = rowTop + (band - cwH) / 2;
    }
    clsY = clamp(clsY, 0, Math.max(0, ih - lw.h));

    drawChip(ctx, glyph, clsChip, clamp(x, 0, Math.max(0, iw - lw.w)), clsY);
    if (confText !== null && cw) {
      drawChip(
        ctx, confText, confChip,
        clamp(x + w - cwW, 0, Math.max(0, iw - cwW)),
        clamp(confY, 0, Math.max(0, ih - cwH)),
      );
    }
  }

  /** Draw the image and its boxes (label chips, handles, badge, reading). */
  function render(): void {
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const img = image;
    if (!img) {
      cv.width = 0;
      cv.height = 0;
      return;
    }
    const z = zoom;
    const W = img.width * z, H = img.height * z;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    cv.style.width = W + "px";
    cv.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(img, 0, 0, W, H);

    boxes.forEach((b, i) => {
      const sel = i === selected;
      const color = classColor(b.cls);
      const x = b.x * z, y = b.y * z, w = b.w * z, h = b.h * z;
      ctx.fillStyle = classColor(b.cls, sel ? 0.28 : 0.12);
      ctx.fillRect(x, y, w, h);
      ctx.lineWidth = sel ? 3 : 2;
      ctx.strokeStyle = color;
      if (b.source === "pred") ctx.setLineDash([5, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      const info = classes[b.cls];
      const glyph = info ? info.glyph : String(b.cls);
      if (labelStyle === "prominent") {
        drawBoxChips(ctx, b, glyph, x, y, w, h, W, H);
      } else {
        // annotate's compact tag: class and confidence in one small chip
        const tag = b.conf !== undefined ? `${glyph} ${b.conf.toFixed(2)}` : glyph;
        ctx.font = "12px system-ui, sans-serif";
        const tw = ctx.measureText(tag).width;
        const ty = y >= 16 ? y - 15 : y + h;
        ctx.fillStyle = "rgba(0,0,0,0.72)";
        ctx.fillRect(x, ty, tw + 8, 15);
        ctx.fillStyle = "#fff";
        ctx.fillText(tag, x + 4, ty + 11.5);
      }

      if (sel) {
        ctx.fillStyle = color;
        for (const hp of handlePoints(b)) {
          ctx.fillRect(hp.x * z - 3.5, hp.y * z - 3.5, 7, 7);
        }
      }
    });

    // fixed-size badge in the image's top-left corner ("EMPTY" for crops that
    // are done but hold no boxes, "PREDICTED EMPTY" for predicted background);
    // drawn after the boxes so it stays on top. Keeps its historical 12px
    // sizing (explicit padding): it is not a box label.
    if (badge) {
      drawChip(ctx, badge, {
        textPx: 12, bg: "rgba(0,0,0,0.72)", fg: "#fff",
        padX: 4, padY: 1.5,
      }, 6, 6);
    }

    // whole-image reading (review): a large monospace chip centred at the
    // image's top edge — this area is normally free of predicted boxes.
    // Omitted draws nothing; empty text draws a dimmed placeholder. Fixed
    // screen px like the badge (does not scale with zoom), on top of all.
    if (reading !== undefined) {
      const text = reading || "—";
      let style = {
        textPx: READING_TEXT_PX, mono: true,
        bg: "rgba(0,0,0,0.72)", fg: reading ? "#fff" : "#aaa",
      };
      let size = chipSize(ctx, text, style);
      const maxW = W - 12; // 6px margins, matching the badge inset
      if (size.w > maxW && maxW > 0) {
        // shrink-to-fit narrow crops (padding scales with the text size)
        style = {
          ...style,
          textPx: Math.max(10, Math.round(style.textPx * (maxW / size.w))),
        };
        size = chipSize(ctx, text, style);
      }
      drawChip(ctx, text, style, clamp(Math.round((W - size.w) / 2), 0, W), 6);
    }
  }

  // re-render on any state the drawing depends on (boxes reads are deep-tracked)
  $effect(() => {
    render();
  });
</script>

<svelte:window onmousemove={onMouseMove} onmouseup={onMouseUp} />

<canvas
  bind:this={cv}
  class:editing={!readonly}
  onmousedown={onMouseDown}
  oncontextmenu={onContext}
></canvas>

<style>
  canvas {
    image-rendering: pixelated;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.55);
    display: block;
  }
  canvas.editing {
    cursor: crosshair;
  }
</style>
