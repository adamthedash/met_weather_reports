<script lang="ts">
  /**
   * Review workspace: the accept/reject/flag workflow over bootstrap predictions.
   *
   * One keystroke per crop (y/n/m), decision-free browsing, fit-to-window
   * display, explicit outcome messages, and neighbour prefetch so advancing
   * after a decision paints immediately. Decisions are destructive and
   * immediate (no undo) — accept writes the label before deleting the
   * prediction, reject only discards the prediction, and needs-adjustment
   * discards the prediction and flags the crop for the manual queue (annotate
   * serves flagged crops first). The shared BoxEditor is used read-only:
   * predictions are display-only here; wrong ones are fixed in the annotate
   * stage after rejecting.
   *
   * The queue's presentation order comes from a selectable ranking (rare
   * labels first, ...), persisted at `localStorage` key
   * `labeler.review.ranking`; every ranking serves the same queue with the
   * same decisions and only changes which crop comes next.
   *
   * The crop's whole-image predicted reading (the boxes' class glyphs
   * concatenated left-to-right, e.g. "0.32") is drawn in large text on a
   * dark chip at the image's top-centre, so a crop can be judged without
   * reading each box.
   */
  import { onMount } from "svelte";

  import BoxEditor from "../lib/BoxEditor.svelte";
  import { api, loadImage } from "../lib/api";
  import { fromNormalized } from "../lib/boxes";
  import { readingOf } from "../lib/labels";
  import { classColor } from "../lib/palette";
  import type {
    Box, LabelError, PredBoxN, PredEntry, RankingSummary,
  } from "../lib/types";

  /** localStorage key of the persisted queue ranking choice. */
  const RANK_KEY = "labeler.review.ranking";

  let classes = $state<string[]>([]);
  let items = $state<PredEntry[]>([]);
  let index = $state(0);
  let reviewed = $state(0);
  let skipped = $state(0); // stale prediction files the queue ignores
  let labelErrors = $state<LabelError[]>([]); // label files a ranking skipped
  let rankings = $state<RankingSummary[]>([]);
  let rankingId = $state("natural"); // current queue ranking
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
  /**
   * Whole-image predicted reading: the boxes' class glyphs concatenated in
   * left-to-right box-centre order (e.g. "0.32"), so the reviewer compares
   * one big string instead of reading every box. Empty for no boxes.
   */
  const reading = $derived(readingOf(boxes, classes));
  const emptyNote = $derived(
    !curItem
      ? ""
      : curItem.error
        ? `malformed prediction: ${curItem.error} — accept is disabled; reject discards the file, or press m to flag it for manual adjustment`
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
  /** One fetched queue entry: decoded crop image + raw prediction boxes. */
  type Fetched = { im: HTMLImageElement; pred: PredBoxN[] };

  /** Prefetched entries keyed by crop path; kept to the current neighbourhood. */
  const prefetch = new Map<string, Fetched>();

  /** Fetch one entry's crop image and prediction boxes in parallel. */
  async function fetchEntry(e: PredEntry): Promise<Fetched> {
    const [pred, im] = await Promise.all([
      // a malformed file is already described by e.error; don't refetch it
      e.error
        ? Promise.resolve([] as PredBoxN[])
        : api.prediction(e.path).then((r) => r.boxes),
      loadImage(api.imageURL(e.path)),
    ]);
    return { im, pred };
  }

  /**
   * Prefetch the crops around the current one so step/decide paint the next
   * crop immediately. Best-effort: a failed fetch stays uncached and is
   * reported by the real load instead.
   */
  async function prefetchNeighbours(): Promise<void> {
    const near = [index + 1, index - 1]
      .map((j) => (j >= 0 && j < items.length ? items[j] : null))
      .filter((e): e is PredEntry => !!e && !prefetch.has(e.path));
    await Promise.all(
      near.map(async (e) => {
        try {
          prefetch.set(e.path, await fetchEntry(e));
        } catch {
          // leave uncached so the real load reports the error
        }
      }),
    );
    // bound the cache to the current neighbourhood
    for (const key of prefetch.keys()) {
      if (prefetch.size <= 4) break;
      prefetch.delete(key);
    }
  }

  /**
   * Fetch classes and the review queue (in the current ranking's order), then
   * show the first undecided prediction. `anchor` (a crop path) keeps the
   * displayed prediction put when the queue is re-synced from disk.
   */
  async function loadList(anchor: string | null = null): Promise<void> {
    const [cls, listing] = await Promise.all([
      api.classes(),
      api.reviewList(rankingId),
    ]);
    classes = cls.classes;
    items = listing.predictions;
    prefetch.clear(); // the queue was rebuilt from disk; cached neighbours may be stale
    skipped = listing.total - listing.remaining;
    labelErrors = listing.label_errors;
    rankingId = listing.ranking;
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
      const { im, pred } = prefetch.get(e.path) ?? (await fetchEntry(e));
      if (seq !== loadSeq) return false; // superseded by a newer load
      img = im;
      boxes = pred.map((b) => ({
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
    void prefetchNeighbours();
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

  /**
   * Flag the current crop for manual adjustment and advance: the prediction
   * is discarded (without parsing, so malformed files can be flagged too) and
   * the crop joins the manual queue that annotate serves first.
   */
  async function markManual(): Promise<void> {
    const e = curItem;
    if (!e || busy || loading) return;
    busy = true;
    try {
      const res = await api.markManual(e.path);
      items.splice(index, 1);
      reviewed++;
      if (index >= items.length) index = Math.max(0, items.length - 1);
      const msg = res.already_annotated
        ? "crop already annotated — stale prediction dropped"
        : "flagged for manual adjustment: prediction discarded ✓";
      busy = false;
      if (await loadCurrent()) setStatus(msg, "ok");
    } catch (err) {
      busy = false;
      // same re-sync as decide(): the queue mirrors the files on disk
      let msg = "flag failed: " + errMsg(err);
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

  /** Switch the queue ranking: persist the choice and reload from the top. */
  async function onRankingChange(e: Event): Promise<void> {
    const id = (e.currentTarget as HTMLSelectElement).value;
    if (id === rankingId) return;
    rankingId = id;
    localStorage.setItem(RANK_KEY, id);
    index = 0; // a new order: start from its first entry
    try {
      await loadList();
    } catch (err) {
      setStatus("reload failed: " + errMsg(err), "error");
    }
  }

  // -- keys -------------------------------------------------------------
  /** Global shortcuts: accept, reject, flag for manual, and browsing. */
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
      case "m":
      case "M":
        e.preventDefault();
        if (!e.repeat) void markManual();
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
    rankingId = localStorage.getItem(RANK_KEY) || "natural";
    void (async () => {
      try {
        rankings = (await api.reviewRankings()).rankings;
        if (!rankings.some((r) => r.id === rankingId)) {
          rankingId = "natural"; // stale stored choice
        }
        await loadList();
      } catch (err) {
        setStatus("startup failed: " + errMsg(err), "error");
      }
    })();
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
          : ""}{labelErrors.length
          ? ` · ${labelErrors.length} malformed label file${
              labelErrors.length === 1 ? "" : "s"
            } skipped in rarity counts`
          : ""}
      </span>
    </div>
    <div class="row">
      <label class="rank">
        rank
        <select
          value={rankingId}
          disabled={busy}
          onchange={(e) => void onRankingChange(e)}
        >
          {#each rankings as r (r.id)}
            <option value={r.id}>{r.title}</option>
          {/each}
        </select>
      </label>
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
        class="manual"
        onclick={() => void markManual()}
        disabled={busy || loading || !curItem}
      >
        needs adjustment <kbd>m</kbd>
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
      <BoxEditor
        image={img}
        bind:boxes
        bind:selected
        {zoom}
        {classes}
        {badge}
        {reading}
        labelStyle="prominent"
        readonly
      />
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
      {#if curItem?.rank_note}
        <div class="rank-note">why here: {curItem.rank_note}</div>
      {/if}
      <div class="empty-note">{emptyNote}</div>
      <h3>keys</h3>
      <ul class="help">
        <li><kbd>y</kbd><span>accept: write label, drop prediction</span></li>
        <li><kbd>n</kbd><span>reject: discard prediction</span></li>
        <li><kbd>m</kbd><span>needs adjustment: discard prediction, flag for manual queue</span></li>
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
  button.manual {
    border-color: var(--accent);
    color: var(--accent);
  }
  .rank {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--dim);
    font-size: 12px;
  }
  select {
    background: #2a2e37;
    color: var(--text);
    border: 1px solid var(--edge);
    border-radius: 5px;
    padding: 3px 6px;
    font: inherit;
    cursor: pointer;
  }
  select:disabled {
    opacity: 0.45;
    cursor: default;
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
  .rank-note {
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
