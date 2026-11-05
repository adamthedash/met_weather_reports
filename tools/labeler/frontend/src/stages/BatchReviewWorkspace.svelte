<script lang="ts">
  /**
   * Batch review workspace: the bulk accept/reject/flag workflow over
   * bootstrap predictions.
   *
   * The composed queue (high-confidence-first with empty predictions
   * sprinkled in at 25%) arrives whole from one `GET` and the first `size`
   * entries form the displayed batch, so the batch-size slider (1-32,
   * persisted at `localStorage` key `labeler.batch.size`) only changes how
   * many tiles are shown — moving it never reshuffles the grid. Tiles are
   * marked in place (plain click rejects, shift-click flags for manual
   * adjustment; the parent owns the marks) and one commit button applies the
   * whole batch: unmarked tiles are accepted as labels, marked tiles get
   * their action. Nothing is written while marking; decisions go through the
   * review stage's decision functions, so they are destructive and immediate
   * exactly like review's (accept writes the label before deleting the
   * prediction, existing labels are never overwritten).
   *
   * The commit is one request with a per-crop result, so a stale or broken
   * crop never blocks the rest; the queue is then re-synced from disk (the
   * compose reshuffles the remaining empties). Marks belong to the displayed
   * grid — nothing is written while marking — so a mark whose tile leaves
   * the grid is discarded.
   */
  import { onMount } from "svelte";

  import Tile from "../lib/Tile.svelte";
  import { api } from "../lib/api";
  import type {
    BatchDecision, BatchEntry, ClassInfo, TileMark,
  } from "../lib/types";

  /** localStorage key of the persisted batch size. */
  const SIZE_KEY = "labeler.batch.size";
  /** The batch-size slider's span and its default. */
  const MIN_SIZE = 1;
  const MAX_SIZE = 32;
  const DEFAULT_SIZE = 8;

  let classes = $state<ClassInfo[]>([]);
  let entries = $state<BatchEntry[]>([]);
  let size = $state(DEFAULT_SIZE);
  let marks = $state(new Map<string, TileMark>());
  let busy = $state(false); // a commit request is in flight
  let reviewed = $state(0); // crops decided this session
  let skipped = $state(0); // stale prediction files the queue ignores
  let status = $state<{ msg: string; kind: "" | "ok" | "error" }>({
    msg: "",
    kind: "",
  });

  /** The displayed batch: the first `size` entries of the composed queue. */
  const shown = $derived(entries.slice(0, size));
  /** Tiles accepted on commit: unmarked and not malformed. */
  const acceptCount = $derived(
    shown.filter((e) => !e.error && (marks.get(e.path) ?? "") === "").length,
  );
  const rejectCount = $derived(
    shown.filter((e) => marks.get(e.path) === "reject").length,
  );
  const manualCount = $derived(
    shown.filter((e) => marks.get(e.path) === "manual").length,
  );
  const commitLabel = $derived(
    [
      acceptCount ? `accept ${acceptCount}` : "",
      rejectCount ? `reject ${rejectCount}` : "",
      manualCount ? `flag ${manualCount}` : "",
    ]
      .filter(Boolean)
      .join(" · ") || "nothing to commit",
  );
  const canCommit = $derived(!busy && acceptCount + rejectCount + manualCount > 0);

  function errMsg(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
  }

  function setStatus(msg: string, kind: "" | "ok" | "error" = ""): void {
    status = { msg, kind };
  }

  /**
   * Toggle a tile's mark: a plain click marks rejected, a shift-click marks
   * needs manual adjustment; clicking the marked tile again clears the mark
   * (back to accept-pending), and clicking with the other modifier switches
   * the mark kind. A tile carries at most one mark.
   */
  function toggleMark(path: string, manual: boolean): void {
    if (busy) return;
    const kind: TileMark = manual ? "manual" : "reject";
    const cur = marks.get(path) ?? "";
    const next: TileMark = cur === kind ? "" : kind;
    if (next === "") marks.delete(path);
    else marks.set(path, next);
  }

  /**
   * Discard marks outside the displayed grid: marks are pending batch state
   * (nothing is written while marking), so a mark whose tile left the grid —
   * truncated by the batch-size slider, or gone from the re-synced queue —
   * is dropped.
   */
  function pruneMarks(): void {
    const visible = new Set(shown.map((e) => e.path));
    for (const path of [...marks.keys()]) {
      if (!visible.has(path)) marks.delete(path);
    }
  }

  /**
   * Re-sync the composed queue from disk. Marks are pruned to the new grid,
   * so crops decided elsewhere lose their marks with their tiles.
   */
  async function loadQueue(): Promise<void> {
    const listing = await api.batchList();
    entries = listing.predictions;
    skipped = listing.total - listing.remaining;
    pruneMarks();
  }

  /**
   * Commit the displayed batch: accept every unmarked tile, apply the marks
   * to the rest. Unmarked malformed tiles are skipped and reported (accept
   * refuses malformed predictions), and per-crop failures are reported while
   * the rest of the batch goes through.
   */
  async function commit(): Promise<void> {
    if (!canCommit) return;
    const decisions: BatchDecision[] = [];
    let malformed = 0;
    for (const e of shown) {
      const mark = marks.get(e.path) ?? "";
      if (mark === "reject") decisions.push({ path: e.path, action: "reject" });
      else if (mark === "manual") decisions.push({ path: e.path, action: "manual" });
      else if (e.error) malformed += 1; // unmarked malformed: skipped
      else decisions.push({ path: e.path, action: "accept" });
    }
    busy = true;
    try {
      const { results } = await api.batchDecision(decisions);
      let accepted = 0, boxes = 0, rejected = 0, flagged = 0, stale = 0;
      const failed: string[] = [];
      for (const r of results) {
        if (!r.ok) {
          failed.push(`${r.path}: ${r.error}`);
          continue; // keep the mark while the tile stays displayed
        }
        marks.delete(r.path);
        if (r.already_annotated) stale += 1;
        if (r.action === "accept") {
          accepted += 1;
          boxes += r.saved;
        } else if (r.action === "reject") rejected += 1;
        else flagged += 1;
      }
      reviewed += results.length - failed.length;
      busy = false;
      await loadQueue();
      const parts: string[] = [];
      if (accepted) {
        parts.push(
          `accepted ${accepted} (${boxes} box${boxes === 1 ? "" : "es"}) ✓`,
        );
      }
      if (rejected) parts.push(`rejected ${rejected}`);
      if (flagged) parts.push(`flagged ${flagged}`);
      if (stale) {
        parts.push(
          `${stale} already annotated — stale prediction${stale === 1 ? "" : "s"} dropped`,
        );
      }
      if (malformed) parts.push(`${malformed} malformed skipped`);
      if (failed.length) {
        parts.push(`${failed.length} failed: ${failed[0]}`);
      }
      setStatus(parts.join(" · "), failed.length ? "error" : "ok");
    } catch (err) {
      busy = false;
      // the queue mirrors the files on disk, so re-syncing reconciles items
      // that were decided elsewhere or disappeared underneath us
      let msg = "commit failed: " + errMsg(err);
      try {
        await loadQueue();
      } catch (err2) {
        msg += " (queue reload failed: " + errMsg(err2) + ")";
      }
      setStatus(msg, "error");
    }
  }

  /** Keep the slider's value and persist the choice. */
  function onSizeInput(e: Event): void {
    const n = Number((e.currentTarget as HTMLInputElement).value);
    size = Math.min(MAX_SIZE, Math.max(MIN_SIZE, Math.round(n)));
    localStorage.setItem(SIZE_KEY, String(size));
    pruneMarks(); // truncating the batch discards the marks it drops
  }

  onMount(() => {
    const stored = parseInt(localStorage.getItem(SIZE_KEY) ?? "", 10);
    if (Number.isFinite(stored)) {
      size = Math.min(MAX_SIZE, Math.max(MIN_SIZE, stored));
    }
    void (async () => {
      try {
        classes = (await api.classes()).classes;
        await loadQueue();
      } catch (err) {
        setStatus("startup failed: " + errMsg(err), "error");
      }
    })();
  });
</script>

<div class="workspace">
  <header>
    <div class="row">
      <strong>batch review</strong>
      <span class="spacer"></span>
      <span class="progress">
        reviewed {reviewed} · remaining {entries.length}{skipped
          ? ` · ${skipped} stale ignored`
          : ""}
      </span>
    </div>
    <div class="row">
      <label class="batch-size">
        batch
        <input
          type="range"
          min={MIN_SIZE}
          max={MAX_SIZE}
          step="1"
          value={size}
          oninput={onSizeInput}
        />
        <span class="count">{size}</span>
      </label>
      <button class="accept" onclick={() => void commit()} disabled={!canCommit}>
        {commitLabel}
      </button>
      <span class="spacer"></span>
      <span class="hint">click: reject · shift-click: flag for manual</span>
      <span
        class="status"
        class:ok={status.kind === "ok"}
        class:error={status.kind === "error"}
      >
        {status.msg}
      </span>
    </div>
  </header>

  <main>
    {#if shown.length > 0}
      <div class="grid">
        {#each shown as e (e.path)}
          <Tile
            entry={e}
            {classes}
            mark={marks.get(e.path) ?? ""}
            ontoggle={(manual) => toggleMark(e.path, manual)}
          />
        {/each}
      </div>
    {:else}
      <p class="empty">
        {reviewed
          ? "all reviewed — nothing left"
          : "nothing to review — run `just bootstrap <model>` first"}
      </p>
    {/if}
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
  .spacer {
    flex: 1;
  }
  .progress {
    color: var(--dim);
  }
  .hint {
    color: var(--dim);
    font-size: 11px;
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

  .batch-size {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--dim);
    font-size: 12px;
  }
  .batch-size input {
    width: 140px;
    accent-color: var(--accent);
  }
  .batch-size .count {
    min-width: 20px;
    text-align: right;
    font-family: ui-monospace, monospace;
    color: var(--text);
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

  main {
    flex: 1;
    overflow: auto;
    padding: 14px;
    background:
      repeating-conic-gradient(#191b20 0% 25%, #16181d 0% 50%) 0 / 24px 24px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 12px;
    justify-items: center;
  }
  .empty {
    color: var(--dim);
    font-style: italic;
    text-align: center;
    margin-top: 40px;
  }
</style>
