<script lang="ts">
  /**
   * Summary workspace: read-only overview of the dataset as it stands.
   *
   * One fetch on load (plus a manual refresh) of `/api/summary/overview`
   * renders the labeling state without touching anything: headline progress
   * totals, the labeled boxes per class (every class shown, 0-counts
   * included) alongside the crops marked empty, and one row per subject
   * (source page) with the full per-class breakdown. Malformed label files
   * and label files mirroring no crop image are listed as problems instead
   * of being counted, so nothing corrupt silently skews the numbers.
   */
  import { onMount } from "svelte";

  import { api } from "../lib/api";
  import { classColor, classTextColor } from "../lib/palette";
  import type { ClassInfo, SummaryOverview } from "../lib/types";

  let data = $state<SummaryOverview | null>(null);
  /** Class presentation for the chips (same index order as `data.classes`). */
  let infos = $state<ClassInfo[]>([]);
  let loading = $state(false);
  let status = $state<{ msg: string; kind: "" | "ok" | "error" }>({
    msg: "",
    kind: "",
  });

  /**
   * Largest class count; the distribution bars scale against it (1 avoids a
   * divide-by-zero when nothing is labeled yet).
   */
  const maxClass = $derived(data ? Math.max(1, ...data.class_counts) : 1);

  function errMsg(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
  }

  /** Fetch the overview; a failed load keeps the previous one and reports. */
  async function load(): Promise<void> {
    if (loading) return;
    loading = true;
    try {
      const [classes, overview] = await Promise.all([
        api.classes(),
        api.summaryOverview(),
      ]);
      infos = classes.classes;
      data = overview;
      status = { msg: "", kind: "" };
    } catch (err) {
      status = { msg: "load failed: " + errMsg(err), kind: "error" };
    }
    loading = false;
  }

  onMount(() => void load());
</script>

<div class="workspace">
  <header>
    <div class="row">
      <strong>summary</strong>
      <span class="dim">{data ? `${data.subjects.length} subjects` : ""}</span>
      <span class="spacer"></span>
      <button onclick={() => void load()} disabled={loading}>refresh</button>
      <span
        class="status"
        class:ok={status.kind === "ok"}
        class:error={status.kind === "error"}
      >
        {status.msg}
      </span>
    </div>
  </header>

  {#if data}
    <main>
      <section>
        <h3>progress</h3>
        <div class="totals">
          <div class="stat"><b>{data.totals.crops}</b><span>crops</span></div>
          <div class="stat"><b>{data.totals.annotated}</b><span>annotated</span></div>
          <div class="stat"><b>{data.totals.empty}</b><span>of which empty</span></div>
          <div class="stat"><b>{data.totals.unlabeled}</b><span>unlabeled</span></div>
          <div class="stat"><b>{data.totals.boxes}</b><span>labeled boxes</span></div>
        </div>
      </section>

      <section>
        <h3>
          class distribution
          <small>labeled boxes per class · the empty row counts crops</small>
        </h3>
        <ul class="dist">
          {#each data.class_counts as count, cls (cls)}
            <li>
              <span
                class="chip"
                style={`background: ${classColor(cls)}; color: ${classTextColor(cls)}`}
              >
                {infos[cls]?.glyph ?? data.classes[cls]}
              </span>
              <span class="name">{data.classes[cls]}</span>
              <span class="bar">
                <i
                  style={`width: ${(count / maxClass) * 100}%; background: ${classColor(cls)}`}
                ></i>
              </span>
              <span class="count">{count}</span>
            </li>
          {/each}
          <li class="empty">
            <span class="chip ghost">∅</span>
            <span class="name">empty</span>
            <span class="bar"></span>
            <span class="count">{data.totals.empty} <small>crops</small></span>
          </li>
        </ul>
      </section>

      <section>
        <h3>
          per-subject distribution
          <small>one row per source page, natural order</small>
        </h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th class="left">subject</th>
                <th>crops</th>
                <th>annotated</th>
                <th>empty</th>
                <th>unlabeled</th>
                <th>boxes</th>
                {#each data.classes as name, cls (cls)}
                  <th class="cls" title={name}>
                    <span
                      class="chip"
                      style={`background: ${classColor(cls)}; color: ${classTextColor(cls)}`}
                    >
                      {infos[cls]?.glyph ?? name}
                    </span>
                  </th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each data.subjects as s (s.id)}
                <tr>
                  <td class="left mono">{s.id}</td>
                  <td>{s.crops}</td>
                  <td>{s.annotated}</td>
                  <td>{s.empty}</td>
                  <td>{s.unlabeled}</td>
                  <td>{s.boxes}</td>
                  {#each s.class_counts as n, cls (cls)}
                    <td class:zero={n === 0}>{n}</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
            <tfoot>
              <tr>
                <td class="left">total</td>
                <td>{data.totals.crops}</td>
                <td>{data.totals.annotated}</td>
                <td>{data.totals.empty}</td>
                <td>{data.totals.unlabeled}</td>
                <td>{data.totals.boxes}</td>
                {#each data.class_counts as n, cls (cls)}
                  <td class:zero={n === 0}>{n}</td>
                {/each}
              </tr>
            </tfoot>
          </table>
        </div>
        {#if data.subjects.length === 0}
          <p class="note">no crops under the images root yet</p>
        {/if}
      </section>

      {#if data.label_errors.length > 0 || data.orphan_labels.length > 0}
        <section>
          <h3>problems</h3>
          <ul class="problems">
            {#each data.label_errors as e (e.path)}
              <li><code>{e.path}</code> malformed label file: {e.error}</li>
            {/each}
            {#each data.orphan_labels as path (path)}
              <li><code>{path}</code> label file with no matching crop image</li>
            {/each}
          </ul>
        </section>
      {/if}
    </main>
  {:else if status.kind === "error"}
    <main><p class="note">{status.msg}</p></main>
  {:else}
    <main><p class="note">loading…</p></main>
  {/if}
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
  .dim {
    color: var(--dim);
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

  main {
    flex: 1;
    overflow-y: auto;
    padding: 14px 18px;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }
  section h3 {
    margin: 0 0 6px;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--dim);
    font-weight: 600;
  }
  section h3 small {
    text-transform: none;
    letter-spacing: normal;
    font-size: 11px;
    font-weight: 400;
    margin-left: 8px;
  }

  .totals {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  .stat {
    background: var(--panel);
    border: 1px solid var(--edge);
    border-radius: 6px;
    padding: 8px 14px;
    display: flex;
    flex-direction: column;
    min-width: 96px;
  }
  .stat b {
    font-size: 18px;
    font-variant-numeric: tabular-nums;
  }
  .stat span {
    color: var(--dim);
    font-size: 11px;
  }

  ul.dist {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
    max-width: 42rem;
  }
  ul.dist li {
    display: grid;
    grid-template-columns: 20px 5rem 1fr 7rem;
    align-items: center;
    gap: 8px;
    padding: 2px 4px;
  }
  .chip {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    flex: none;
  }
  .chip.ghost {
    background: #2a2e37;
    color: var(--dim);
  }
  .bar {
    height: 8px;
    background: #22262e;
    border-radius: 4px;
    overflow: hidden;
  }
  .bar i {
    display: block;
    height: 100%;
    border-radius: 4px;
  }
  .count {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .count small {
    color: var(--dim);
  }
  ul.dist li.empty .name,
  ul.dist li.empty .count {
    color: var(--dim);
  }

  .table-wrap {
    overflow-x: auto;
    border: 1px solid var(--edge);
    border-radius: 6px;
    background: var(--panel);
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-variant-numeric: tabular-nums;
  }
  th,
  td {
    padding: 4px 8px;
    text-align: right;
    border-bottom: 1px solid var(--edge);
    white-space: nowrap;
  }
  th.left,
  td.left {
    text-align: left;
  }
  th {
    color: var(--dim);
    font-size: 11px;
    font-weight: 500;
  }
  th.cls {
    padding: 3px 5px;
    text-align: center;
  }
  th.cls .chip {
    width: 18px;
    height: 18px;
    font-size: 11px;
  }
  tbody tr:hover {
    background: #22262e;
  }
  tbody tr:last-child td {
    border-bottom: none;
  }
  td.zero {
    color: var(--dim);
  }
  tfoot td {
    font-weight: 600;
    border-bottom: none;
    border-top: 1px solid var(--edge);
  }
  .mono {
    font-family: ui-monospace, monospace;
  }

  .note {
    color: var(--dim);
  }
  ul.problems {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
    color: var(--error);
    max-width: 60rem;
  }
  ul.problems code {
    background: #2a2e37;
    border-radius: 4px;
    padding: 0 5px;
    color: var(--text);
  }
</style>
