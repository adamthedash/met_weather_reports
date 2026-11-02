<script lang="ts">
  /**
   * Review workspace: the accept/reject workflow of tools/review.md.
   *
   * Ports the behaviour of tools/review.html: one keystroke per crop (y/n),
   * decision-free browsing, fit-to-window display, and the outcome messages.
   * Decisions are destructive and immediate (no undo) — accept writes the
   * label before deleting the prediction, reject only discards the prediction.
   * The shared BoxEditor is used read-only: predictions are display-only here;
   * wrong ones are fixed in the annotate stage after rejecting.
   */
  import { onMount } from "svelte";

  import BoxEditor from "../lib/BoxEditor.svelte";
  import { api, loadImage } from "../lib/api";
  import { fromNormalized } from "../lib/boxes";
  import { classColor } from "../lib/palette";
  import type { Box, PredBoxN, PredEntry } from "../lib/types";

  let classes = $state<string[]>([]);
  let items = $state<PredEntry[]>([]);
  let index = $state(0);
  let reviewed = $state(0);
  let skipped = $state(0); // stale prediction files the queue ignores
  let img = $state<HTMLImageElement | null>(null);
  let boxes = $state<Box[]>([]);
  let selected = $state(-1); // read-only view: never leaves -1
  let loadSeq = 0;
  let loading = $state(false); // a display load is in flight
  let busy = $state(false); // a decision request is in flight
  let status = $state<{ msg: string; kind: "" | "ok" | "error" }>({
    msg: "",
    kind: "",
  });

  let wrapEl = $state<HTMLDivElement | undefined>(undefined);
  let wrapSize = $state({ w: 0, h: 0 });

  const curItem = $derived(
    index >= 0 && index < items.length ? items[index] : null,
  );
  /** Badge for a predicted background crop (malformed files get no badge). */
  const badge = $derived(
    curItem && !curItem.error && boxes.length === 0 ? "PREDICTED EMPTY" : "",
  );
  const emptyNote = $derived(
    !curItem
      ? ""
      : curItem.error
        ? `malformed prediction: ${curItem.error} — accept is disabled; reject discards the file`
        : boxes.length === 0
          ? "predicted empty — accept marks this crop empty/ignored"
          : "",
  );

  function errMsg(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
  }

  function setStatus(msg: string, kind: "" | "ok" | "error" = ""): void {
    status = { msg, kind };
  }

  /**
   * Integer magnification that fits the crop into the canvas area, so pixels
   * stay crisp; falls back to fractional scaling only for crops that would
   * overflow at 1:1.
   */
  function fitScale(img: HTMLImageElement, availW: number, availH: number): number {
    const fit = Math.min(availW / img.width, availH / img.height);
    return fit >= 1 ? Math.floor(fit) : fit;
  }

  const zoom = $derived(
    img && wrapSize.w > 0 ? fitScale(img, wrapSize.w, wrapSize.h) : 1,
  );

  /** Track the canvas area size (the crop is fitted to it on every change). */
  function syncWrap(): void {
    if (!wrapEl) return;
    const cs = getComputedStyle(wrapEl);
    wrapSize = {
      w: wrapEl.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight),
      h: wrapEl.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom),
    };
  }

  // -- data io ----------------------------------------------------------
  /**
   * Fetch classes and the review queue, then show the first undecided
   * prediction. `anchor` (a crop path) keeps the displayed prediction put when
   * the queue is re-synced from disk.
   */
  async function loadList(anchor: string | null = null): Promise<void> {
    const [cls, listing] = await Promise.all([api.classes(), api.reviewList()]);
    classes = cls.classes;
    items = listing.predictions;
    skipped = listing.total - listing.remaining;
    const i = anchor ? items.findIndex((e) => e.path === anchor) : -1;
    index = i >= 0 ? i : Math.min(index, items.length - 1);
    if (index < 0) index = 0;
    await loadCurrent();
  }

  /** Load the image and prediction of the current item; false when loading fails. */
  async function loadCurrent(): Promise<boolean> {
    const e = curItem;
    const seq = ++loadSeq;
    loading = true;
    boxes = [];
    img = null;
    if (!e) {
      loading = false;
      setStatus(
        reviewed
          ? "all reviewed — nothing left"
          : "nothing to review — run `just bootstrap <model>` first",
      );
      return true;
    }
    let ok = true;
    try {
      const [pred, im] = await Promise.all([
        // a malformed file is already described by e.error; don't refetch it
        e.error
          ? Promise.resolve({ path: e.path, boxes: [] as PredBoxN[] })
          : api.prediction(e.path),
        loadImage(api.imageURL(e.path)),
      ]);
      if (seq !== loadSeq) return false; // superseded by a newer load
      img = im;
      boxes = pred.boxes.map((b) => ({
        ...fromNormalized(b, im.width, im.height),
        conf: b.conf,
      }));
      setStatus("");
    } catch (err) {
      if (seq !== loadSeq) return false; // superseded by a newer load
      setStatus("load failed: " + errMsg(err), "error");
      ok = false;
    }
    loading = false;
    return ok;
  }

  /** Send an accept/reject decision for the current prediction and advance. */
  async function decide(accept: boolean): Promise<void> {
    const e = curItem;
    if (!e || busy || loading) return;
    busy = true;
    try {
      const res = await api.decide(e.path, accept);
      items.splice(index, 1);
      reviewed++;
      if (index >= items.length) index = Math.max(0, items.length - 1);
      let msg: string;
      if (!accept) {
        msg = "rejected: prediction discarded";
      } else if (res.already_annotated) {
        msg = "crop already annotated — stale prediction dropped";
      } else if (res.saved > 0) {
        msg = `accepted: ${res.saved} box${res.saved === 1 ? "" : "es"} saved to labels/ ✓`;
      } else {
        msg = "accepted: marked empty/ignored ✓";
      }
      busy = false;
      if (await loadCurrent()) setStatus(msg, "ok");
    } catch (err) {
      busy = false;
      // the queue mirrors the files on disk, so re-syncing reconciles items
      // that were decided elsewhere or disappeared underneath us
      let msg = "decision failed: " + errMsg(err);
      try {
        await loadList(e.path);
      } catch (err2) {
        msg += " (queue reload failed: " + errMsg(err2) + ")";
      }
      setStatus(msg, "error");
    }
  }

  /** Move to the next/previous queued prediction without deciding. */
  async function step(delta: number): Promise<void> {
    if (busy) return;
    const j = index + delta;
    if (j < 0 || j >= items.length) {
      setStatus(delta > 0 ? "last prediction" : "first prediction");
      return;
    }
    index = j;
    await loadCurrent();
  }

  // -- keys -------------------------------------------------------------
  /** Global shortcuts: accept, reject, and decision-free browsing. */
  function onKey(e: KeyboardEvent): void {
    const t = e.target as HTMLElement | null;
    if (
      t &&
      (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")
    ) {
      return;
    }
    if (e.ctrlKey || e.metaKey) return;

    switch (e.key) {
      case "y":
      case "Y":
        e.preventDefault();
        if (!e.repeat) void decide(true); // held keys must not burn through the queue
        break;
      case "n":
      case "N":
        e.preventDefault();
        if (!e.repeat) void decide(false);
        break;
      case "ArrowRight":
      case "d":
        e.preventDefault();
        void step(1);
        break;
      case "ArrowLeft":
      case "a":
        e.preventDefault();
        void step(-1);
        break;
    }
  }

  onMount(() => {
    syncWrap();
    void loadList().catch((err) =>
      setStatus("startup failed: " + errMsg(err), "error"),
    );
  });
</script>

<svelte:window onkeydown={onKey} onresize={syncWrap} />

<div class="workspace">
  <header>
    <div class="row">
      <strong>review</strong>
      <span class="filename">{curItem ? curItem.path : "—"}</span>
      <span class="spacer"></span>
      <span class="progress">
        reviewed {reviewed} · remaining {items.length}{skipped
          ? ` · ${skipped} stale ignored`
          : ""}
      </span>
    </div>
    <div class="row">
      <button onclick={() => step(-1)} disabled={busy || index <= 0}>← prev</button>
      <button onclick={() => step(1)} disabled={busy || index >= items.length - 1}>
        next →
      </button>
      <button
        class="reject"
        onclick={() => void decide(false)}
        disabled={busy || loading || !curItem}
      >
        reject <kbd>n</kbd>
      </button>
      <button
        class="accept"
        onclick={() => void decide(true)}
        disabled={busy || loading || !curItem || !!curItem.error || !img}
      >
        accept <kbd>y</kbd>
      </button>
      <span class="spacer"></span>
      <span class="status" class:ok={status.kind === "ok"} class:error={status.kind === "error"}>
        {status.msg}
      </span>
    </div>
  </header>

  <main>
    <div class="canvas-wrap" bind:this={wrapEl}>
      <BoxEditor image={img} bind:boxes bind:selected {zoom} {classes} {badge} readonly />
    </div>
    <aside>
      <h3>prediction</h3>
      <ul class="boxlist">
        {#each boxes as b, i (i)}
          <li>
            <span class="swatch" style={`background: ${classColor(b.cls)}`}></span>
            <span>{classes[b.cls] ?? String(b.cls)}</span>
            <small>conf {b.conf?.toFixed(4)}</small>
          </li>
        {/each}
      </ul>
      <div class="empty-note">{emptyNote}</div>
      <h3>keys</h3>
      <ul class="help">
        <li><kbd>y</kbd><span>accept: write label, drop prediction</span></li>
        <li><kbd>n</kbd><span>reject: discard prediction</span></li>
        <li><kbd>← →</kbd><span>prev / next (no decision)</span></li>
      </ul>
    </aside>
  </main>
</div>

<style>
  .workspace {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  header {
    background: var(--panel);
    border-bottom: 1px solid var(--edge);
    padding: 6px 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .row strong {
    font-size: 14px;
  }
  .filename {
    color: var(--dim);
    font-family: ui-monospace, monospace;
  }
  .spacer {
    flex: 1;
  }
  .status {
    color: var(--dim);
  }
  .status.ok {
    color: var(--ok);
  }
  .status.error {
    color: var(--error);
  }

  button {
    background: #2a2e37;
    color: var(--text);
    border: 1px solid var(--edge);
    border-radius: 5px;
    padding: 4px 9px;
    font: inherit;
    cursor: pointer;
  }
  button:hover:not(:disabled) {
    background: #343945;
  }
  button:disabled {
    opacity: 0.45;
    cursor: default;
  }
  button.accept {
    border-color: var(--ok);
    color: var(--ok);
  }
  button.reject {
    border-color: var(--error);
    color: var(--error);
  }
  kbd {
    background: #2a2e37;
    border: 1px solid var(--edge);
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 0 5px;
    font: 11px ui-monospace, monospace;
    color: var(--dim);
  }

  main {
    flex: 1;
    display: flex;
    min-height: 0;
  }
  .canvas-wrap {
    flex: 1;
    overflow: auto;
    padding: 18px;
    display: flex;
    align-items: flex-start;
    justify-content: flex-start;
    background:
      repeating-conic-gradient(#191b20 0% 25%, #16181d 0% 50%) 0 / 24px 24px;
  }
  aside {
    width: 270px;
    border-left: 1px solid var(--edge);
    background: var(--panel);
    overflow-y: auto;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  aside h3 {
    margin: 6px 0 2px;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--dim);
  }
  .boxlist {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .boxlist li {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 3px 7px;
    border-radius: 5px;
  }
  .boxlist .swatch {
    width: 10px;
    height: 10px;
    border-radius: 3px;
    flex: none;
  }
  .boxlist small {
    color: var(--dim);
    margin-left: auto;
    font-family: ui-monospace, monospace;
  }
  .empty-note {
    color: var(--dim);
    font-style: italic;
  }
  .help {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
    color: var(--dim);
  }
  .help li {
    display: flex;
    gap: 8px;
  }
  .help kbd {
    flex: none;
    min-width: 58px;
    text-align: center;
  }
</style>
