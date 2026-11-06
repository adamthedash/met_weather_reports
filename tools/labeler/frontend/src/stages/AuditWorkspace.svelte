<script lang="ts">
  /**
   * Audit workspace: model-vs-label disagreement review over annotated crops.
   *
   * The crop renders twice side by side: annotations on the left, model
   * predictions on the right, each badged with its side (and its empty
   * state). Every disagreement is listed in plain words. One keystroke per
   * crop: `l` keeps the existing label, `p` takes the model's prediction,
   * `x` discards both (the crop returns to the annotate queue). Every decision
   * deletes the crop's audit prediction file, so the queue empties as the
   * user works and a re-run of `scripts/audit_annotations.py` is a fresh
   * round. Decisions are destructive and immediate (no undo) — a wrongly
   * discarded label is re-annotated in the annotate stage. The shared
   * BoxEditor is used read-only: box fix-ups happen in the annotate stage.
   */
  import { onMount } from "svelte";

  import BoxEditor from "../lib/BoxEditor.svelte";
  import { api, loadImage } from "../lib/api";
  import { fitScale, fromNormalized } from "../lib/boxes";
  import { classColor } from "../lib/palette";
  import type {
    AuditAction, AuditEntry, AuditIssue, Box, ClassInfo, CompareResult,
  } from "../lib/types";

  let classes = $state<ClassInfo[]>([]);
  let items = $state<AuditEntry[]>([]);
  let compared = $state(0);
  let index = $state(0);
  let decided = $state(0);
  let img = $state<HTMLImageElement | null>(null);
  let labelBoxes = $state<Box[]>([]);
  let predBoxes = $state<Box[]>([]);
  let issues = $state<AuditIssue[]>([]);
  let loadSeq = 0;
  let loading = $state(false); // a display load is in flight
  let busy = $state(false); // a decision request is in flight
  let status = $state<{ msg: string; kind: "" | "ok" | "error" }>({
    msg: "",
    kind: "",
  });

  let wrapEl = $state<HTMLDivElement | undefined>(undefined);
  let wrapSize = $state({ w: 0, h: 0 });

  /** Gap between the two panes; keep in sync with the .canvas-wrap style. */
  const PANE_GAP = 16;

  const curItem = $derived(
    index >= 0 && index < items.length ? items[index] : null,
  );
  /** Pane badges: which side the pane shows, with its empty state. */
  const labelBadge = $derived(
    curItem ? (curItem.label_boxes === 0 ? "LABELS (EMPTY)" : "LABELS") : "",
  );
  const predBadge = $derived(
    curItem
      ? curItem.pred_boxes === 0
        ? "PREDICTIONS (EMPTY)"
        : "PREDICTIONS"
      : "",
  );
  const emptyNote = $derived(
    !curItem
      ? ""
      : curItem.error
        ? `malformed file: ${curItem.error} — the broken side can not be accepted; discard both is always available`
        : labelBoxes.length === 0 && predBoxes.length > 0
          ? "label empty — keep confirms this crop as empty/ignored"
          : predBoxes.length === 0 && labelBoxes.length > 0
            ? "predicted empty — take prediction marks this crop empty/ignored"
            : "",
  );

  function errMsg(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
  }

  function setStatus(msg: string, kind: "" | "ok" | "error" = ""): void {
    status = { msg, kind };
  }

  /** Class display name, falling back to the bare class id. */
  function clsName(cls: number): string {
    return classes[cls]?.name ?? String(cls);
  }

  /** Plain-words description of one disagreement (indices match the lists). */
  function issueText(is: AuditIssue): string {
    const lb = is.label !== null ? labelBoxes[is.label] : undefined;
    const pb = is.pred !== null ? predBoxes[is.pred] : undefined;
    switch (is.kind) {
      case "unmatched_label":
        return `label ${is.label} (${clsName(lb?.cls ?? -1)}) has no prediction`;
      case "unmatched_pred":
        return `pred ${is.pred} (${clsName(pb?.cls ?? -1)}, conf ${pb?.conf?.toFixed(2)}) not labeled`;
      case "class_mismatch":
        return `label ${is.label} ↔ pred ${is.pred}: class ${clsName(lb?.cls ?? -1)} vs ${clsName(pb?.cls ?? -1)}`;
      case "weak_overlap":
        return `label ${is.label} ↔ pred ${is.pred}: weak overlap (IoU ${(is.iou ?? 0).toFixed(2)})`;
    }
  }

  /**
   * Fit magnification (from the shared box geometry helpers): integer while
   * the crop is magnified so pixels stay crisp, fractional only for crops
   * that would overflow at 1:1 (full-page scans).
   */
  const zoom = $derived(
    img && wrapSize.w > 0
      ? fitScale(img, Math.max((wrapSize.w - PANE_GAP) / 2, 1), wrapSize.h)
      : 1,
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
   * Fetch classes and the audit queue, then show the first undecided crop.
   * `anchor` (a crop path) keeps the displayed crop put when the queue is
   * re-synced from disk.
   */
  async function loadList(anchor: string | null = null): Promise<void> {
    const [cls, listing] = await Promise.all([api.classes(), api.auditList()]);
    classes = cls.classes;
    items = listing.audits;
    compared = listing.total;
    const i = anchor ? items.findIndex((e) => e.path === anchor) : -1;
    index = i >= 0 ? i : Math.min(index, items.length - 1);
    if (index < 0) index = 0;
    await loadCurrent();
  }

  /** Load the crop and both box sets of the current item; false when loading fails. */
  async function loadCurrent(): Promise<boolean> {
    const e = curItem;
    const seq = ++loadSeq;
    loading = true;
    labelBoxes = [];
    predBoxes = [];
    issues = [];
    img = null;
    if (!e) {
      loading = false;
      setStatus(
        decided
          ? "all reviewed — nothing left"
          : "nothing to audit — run `just audit <model>` first",
      );
      return true;
    }
    let ok = true;
    try {
      const [cmp, im] = await Promise.all([
        // a malformed file is already described by e.error; don't refetch it
        e.error
          ? Promise.resolve({
              path: e.path, labels: [], predictions: [], issues: [],
            } as CompareResult)
          : api.auditCompare(e.path),
        loadImage(api.imageURL(e.path)),
      ]);
      if (seq !== loadSeq) return false; // superseded by a newer load
      img = im;
      labelBoxes = cmp.labels.map((b): Box => ({
        ...fromNormalized(b, im.width, im.height),
        source: "label",
      }));
      predBoxes = cmp.predictions.map((b): Box => ({
        ...fromNormalized(b, im.width, im.height),
        conf: b.conf,
        source: "pred",
      }));
      issues = cmp.issues;
      setStatus("");
    } catch (err) {
      if (seq !== loadSeq) return false; // superseded by a newer load
      setStatus("load failed: " + errMsg(err), "error");
      ok = false;
    }
    loading = false;
    return ok;
  }

  /** Send a keep/take/discard decision for the current crop and advance. */
  async function decide(action: AuditAction): Promise<void> {
    const e = curItem;
    if (!e || busy || loading) return;
    busy = true;
    try {
      const res = await api.auditDecision(e.path, action);
      items.splice(index, 1);
      decided++;
      if (index >= items.length) index = Math.max(0, items.length - 1);
      let msg: string;
      if (action === "keep_label") {
        msg = "kept the existing label — model prediction discarded";
      } else if (action === "take_prediction") {
        msg = res.saved > 0
          ? `took the prediction: ${res.saved} box${res.saved === 1 ? "" : "es"} saved to labels/ ✓`
          : "took the prediction: marked empty/ignored ✓";
      } else {
        msg = "discarded both — crop back in the annotate queue";
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

  /** Move to the next/previous queued crop without deciding. */
  async function step(delta: number): Promise<void> {
    if (busy) return;
    const j = index + delta;
    if (j < 0 || j >= items.length) {
      setStatus(delta > 0 ? "last crop" : "first crop");
      return;
    }
    index = j;
    await loadCurrent();
  }

  // -- keys -------------------------------------------------------------
  /** Global shortcuts: the three decisions and decision-free browsing. */
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
      case "l":
      case "L":
        e.preventDefault();
        if (!e.repeat) void decide("keep_label"); // held keys must not burn through the queue
        break;
      case "p":
      case "P":
        e.preventDefault();
        if (!e.repeat) void decide("take_prediction");
        break;
      case "x":
      case "X":
        e.preventDefault();
        if (!e.repeat) void decide("discard_both");
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
      <strong>audit</strong>
      <span class="filename">{curItem ? curItem.path : "—"}</span>
      <span class="spacer"></span>
      <span class="progress">
        reviewed {decided} · to review {items.length} · {compared} compared
      </span>
    </div>
    <div class="row">
      <button onclick={() => step(-1)} disabled={busy || index <= 0}>← prev</button>
      <button onclick={() => step(1)} disabled={busy || index >= items.length - 1}>
        next →
      </button>
      <button
        class="reject"
        onclick={() => void decide("discard_both")}
        disabled={busy || loading || !curItem}
      >
        discard both <kbd>x</kbd>
      </button>
      <button
        class="take"
        onclick={() => void decide("take_prediction")}
        disabled={busy || loading || !curItem || curItem.pred_boxes === null || !img}
      >
        take prediction <kbd>p</kbd>
      </button>
      <button
        class="accept"
        onclick={() => void decide("keep_label")}
        disabled={busy || loading || !curItem || curItem.label_boxes === null}
      >
        keep label <kbd>l</kbd>
      </button>
      <span class="spacer"></span>
      <span class="status" class:ok={status.kind === "ok"} class:error={status.kind === "error"}>
        {status.msg}
      </span>
    </div>
  </header>

  <main>
    <div class="canvas-wrap" bind:this={wrapEl}>
      <BoxEditor
        image={img}
        boxes={labelBoxes}
        {zoom}
        {classes}
        badge={labelBadge}
        labelStyle="prominent"
        readonly
      />
      <BoxEditor
        image={img}
        boxes={predBoxes}
        {zoom}
        {classes}
        badge={predBadge}
        labelStyle="prominent"
        readonly
      />
    </div>
    <aside>
      <h3>annotations</h3>
      <ul class="boxlist">
        {#each labelBoxes as b, i (i)}
          <li>
            <span class="swatch" style={`background: ${classColor(b.cls)}`}></span>
            <span>{classes[b.cls]?.name ?? String(b.cls)}</span>
            <small>#{i}</small>
          </li>
        {/each}
      </ul>
      <h3>predictions</h3>
      <ul class="boxlist pred">
        {#each predBoxes as b, i (i)}
          <li>
            <span class="swatch" style={`background: ${classColor(b.cls)}`}></span>
            <span>{classes[b.cls]?.name ?? String(b.cls)}</span>
            <small>conf {b.conf?.toFixed(2)} #{i}</small>
          </li>
        {/each}
      </ul>
      <h3>issues</h3>
      <ul class="issues">
        {#each issues as is, i (i)}
          <li>{issueText(is)}</li>
        {/each}
      </ul>
      <div class="empty-note">{emptyNote}</div>
      <h3>keys</h3>
      <ul class="help">
        <li><kbd>l</kbd><span>keep the existing label</span></li>
        <li><kbd>p</kbd><span>take the model's prediction</span></li>
        <li><kbd>x</kbd><span>discard both</span></li>
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
  button.take {
    border-color: var(--accent);
    color: var(--accent);
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
    gap: 16px;
    align-items: flex-start;
    justify-content: flex-start;
    background:
      repeating-conic-gradient(#191b20 0% 25%, #16181d 0% 50%) 0 / 24px 24px;
  }
  aside {
    width: 300px;
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
  .boxlist.pred li {
    border: 1px dashed var(--edge);
  }
  .issues {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .issues li {
    color: var(--error);
    font-size: 12px;
    padding: 2px 4px;
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
