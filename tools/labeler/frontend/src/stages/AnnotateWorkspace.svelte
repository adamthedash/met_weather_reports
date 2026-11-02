<script lang="ts">
  /**
   * Annotate workspace: the manual bounding-box workflow of tools/annotate.md.
   *
   * Ports the behaviour of tools/annotate.html: image-pixel editing with a
   * sticky current class, 600 ms debounced autosave, 100-deep undo, view-mode
   * filtering of prev/next (all → annotated → non-annotated), and localStorage
   * resume. Boxes live here and are edited through the shared BoxEditor.
   */
  import { onMount } from "svelte";

  import BoxEditor from "../lib/BoxEditor.svelte";
  import { api, loadImage } from "../lib/api";
  import { fromNormalized, toNormalized } from "../lib/boxes";
  import { buildShortcutMap, classColor, shortcutFor } from "../lib/palette";
  import type { Box, FilterMode, ImageEntry } from "../lib/types";

  const FILTER_CYCLE: Record<FilterMode, FilterMode> = {
    all: "annotated",
    annotated: "non-annotated",
    "non-annotated": "all",
  };

  /** Initial view mode from localStorage; unknown values degrade to "all". */
  function loadFilterMode(): FilterMode {
    const raw = localStorage.getItem("labeler.annotate.filterMode");
    return raw === "all" || raw === "annotated" || raw === "non-annotated"
      ? raw
      : "all";
  }

  let classes = $state<string[]>([]);
  let shortcutMap: Record<string, number> = {};
  let images = $state<ImageEntry[]>([]);
  let filterMode = $state<FilterMode>(loadFilterMode());
  let currentPath = $state<string | null>(null);
  let loadSeq = 0;
  let boxes = $state<Box[]>([]);
  let selected = $state(-1);
  let currentClass = $state<number | null>(null);
  let undoStack: Box[][] = [];
  let zoom = $state(8);
  let zoomInit = localStorage.getItem("labeler.annotate.zoom") !== null;
  let img = $state<HTMLImageElement | null>(null);
  let dirty = $state(false);
  let saveTimer = $state<ReturnType<typeof setTimeout> | undefined>(undefined);
  let status = $state<{ msg: string; kind: "" | "ok" | "error" }>({
    msg: "",
    kind: "",
  });
  let editor = $state<{ cancelOp(): void } | undefined>(undefined);

  if (zoomInit) zoom = Number(localStorage.getItem("labeler.annotate.zoom")) || 8;

  // -- derived view state -----------------------------------------------
  const curEntry = $derived(
    images.find((e) => e.path === currentPath) ?? null,
  );
  const curIndex = $derived(curEntry ? images.indexOf(curEntry) : -1);
  const doneCount = $derived(images.filter((e) => e.done).length);
  /** The crop on display is marked empty: done on disk, zero boxes now. */
  const emptyMarked = $derived(!!curEntry?.done && boxes.length === 0);

  /** Whether listing entry `e` passes the current view mode filter. */
  function visible(e: ImageEntry): boolean {
    if (filterMode === "annotated") return e.done;
    if (filterMode === "non-annotated") return !e.done;
    return true;
  }

  const canPrev = $derived(
    curIndex >= 0 && images.slice(0, curIndex).some(visible),
  );
  const canNext = $derived(
    curIndex >= 0 && images.slice(curIndex + 1).some(visible),
  );

  function errMsg(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
  }

  function setStatus(msg: string, kind: "" | "ok" | "error" = ""): void {
    status = { msg, kind };
  }

  // -- data io ----------------------------------------------------------
  /** Fetch classes and the crop listing, then restore the last-viewed crop. */
  async function loadList(): Promise<void> {
    const [cls, listing] = await Promise.all([api.classes(), api.list()]);
    classes = cls.classes;
    shortcutMap = buildShortcutMap(classes);
    images = listing.images;
    const last = localStorage.getItem("labeler.annotate.lastPath");
    const restored =
      images.find((e) => e.path === last) ?? images.find(visible) ?? null;
    currentPath = restored ? restored.path : null;
    await loadCurrent();
  }

  /** Load the image and labels of the current crop, ignoring superseded loads. */
  async function loadCurrent(): Promise<void> {
    const e = curEntry;
    const seq = ++loadSeq;
    boxes = [];
    selected = -1;
    undoStack = [];
    dirty = false;
    if (!e) {
      img = null;
      setStatus(
        images.length
          ? "nothing left in this view — press f to change the view mode"
          : "no images found",
      );
      return;
    }
    localStorage.setItem("labeler.annotate.lastPath", e.path);
    try {
      const [labels, im] = await Promise.all([
        api.labels(e.path),
        loadImage(api.imageURL(e.path)),
      ]);
      if (seq !== loadSeq) return; // superseded by a newer load
      img = im;
      boxes = labels.boxes.map((b) => fromNormalized(b, im.width, im.height));
      if (!zoomInit && im.height > 0) {
        zoom = Math.min(16, Math.max(2, Math.floor(420 / im.height)));
        zoomInit = true;
      }
      setStatus("");
    } catch (err) {
      if (seq !== loadSeq) return; // superseded by a newer load
      img = null;
      setStatus("load failed: " + errMsg(err), "error");
    }
  }

  /** Save the current boxes as YOLO labels; false when saving fails. */
  async function save(): Promise<boolean> {
    if (saveTimer !== undefined) clearTimeout(saveTimer);
    saveTimer = undefined;
    const e = curEntry;
    if (!e || !img) return true;
    const payload = boxes.map((b) => toNormalized(b, img!.width, img!.height));
    try {
      await api.saveLabels(e.path, payload);
      dirty = false;
      e.done = true;
      e.has_labels = payload.length > 0;
      setStatus(
        `saved ${payload.length} box${payload.length === 1 ? "" : "es"} ✓`,
        "ok",
      );
      return true;
    } catch (err) {
      setStatus("save failed: " + errMsg(err), "error");
      return false;
    }
  }

  /** Save now when a change is pending; resolves to whether the state is clean. */
  async function flushSave(): Promise<boolean> {
    return dirty || saveTimer ? await save() : true;
  }

  /** Debounce an autosave 600 ms after the last edit. */
  function scheduleSave(): void {
    if (saveTimer !== undefined) clearTimeout(saveTimer);
    saveTimer = setTimeout(save, 600);
  }

  /** Record an editor edit: pre-mutation snapshot to undo, then autosave. */
  function onEdit(snapshot: Box[]): void {
    undoStack.push(snapshot);
    if (undoStack.length > 100) undoStack.shift();
    dirty = true;
    scheduleSave();
  }

  /** Copy of the current boxes, suitable for the undo stack. */
  function pushUndo(): void {
    undoStack.push(boxes.map((b) => ({ ...b })));
    if (undoStack.length > 100) undoStack.shift();
  }

  /** Restore the most recent undo snapshot and schedule a save. */
  function undo(): void {
    if (undoStack.length === 0) {
      setStatus("nothing to undo");
      return;
    }
    boxes = undoStack.pop()!;
    selected = -1;
    dirty = true;
    scheduleSave();
  }

  // -- navigation -------------------------------------------------------
  /**
   * Move to the next/previous visible crop after flushing any pending save.
   *
   * Step from the current crop's place in the full listing, skipping crops the
   * current view mode hides. Anchoring on the full list (not a filtered view)
   * means a just-saved crop is still findable, so the very next undone crop is
   * opened — never skipped — and a save can never detach the displayed
   * image/annotations from its file.
   */
  async function navigate(delta: number): Promise<void> {
    if (!(await flushSave())) return;
    const all = images;
    const cur = curEntry;
    let j = cur ? all.indexOf(cur) + delta : delta > 0 ? 0 : all.length - 1;
    while (j >= 0 && j < all.length && !visible(all[j])) j += delta;
    if (j < 0 || j >= all.length) {
      setStatus(
        cur
          ? delta > 0
            ? "last crop"
            : "first crop"
          : "nothing left in this view — press f to change the view mode",
      );
      return;
    }
    currentPath = all[j].path;
    await loadCurrent();
  }

  /** Discard the crop's boxes (marking it empty/ignored) and advance. */
  async function markEmpty(): Promise<void> {
    const e = curEntry;
    if (!e) return;
    if (
      boxes.length > 0 &&
      !window.confirm(
        `Discard ${boxes.length} box(es) and mark this crop empty/ignored?`,
      )
    ) {
      return;
    }
    boxes = [];
    selected = -1;
    dirty = true;
    if (await save()) await navigate(1);
  }

  /**
   * Cycle the view mode all → annotated → non-annotated (persisted); the
   * displayed crop stays put, the mode only restricts prev/next.
   */
  async function cycleFilter(): Promise<void> {
    if (!(await flushSave())) return;
    filterMode = FILTER_CYCLE[filterMode];
    localStorage.setItem("labeler.annotate.filterMode", filterMode);
    if (!curEntry) {
      // nothing on display (empty view) — jump to the first visible crop
      const first = images.find(visible) ?? null;
      currentPath = first ? first.path : null;
      await loadCurrent();
    }
    // else: the crop on display stays put; the filter only restricts prev/next
  }

  // -- editing ----------------------------------------------------------
  /** Delete box `i`, keeping the selection index consistent. */
  function deleteBox(i: number): void {
    if (i < 0 || i >= boxes.length) return;
    pushUndo();
    boxes.splice(i, 1);
    if (selected === i) selected = -1;
    else if (selected > i) selected--;
    dirty = true;
    scheduleSave();
  }

  /** Delete the selected box, if any. */
  function deleteSelected(): void {
    if (selected < 0) {
      setStatus("no box selected");
      return;
    }
    deleteBox(selected);
  }

  /** Make class `cls` current and apply it to the selected box. */
  function assignClass(cls: number): void {
    currentClass = cls;
    const sel = selected >= 0 ? boxes[selected] : null;
    if (sel && sel.cls !== cls) {
      pushUndo();
      sel.cls = cls;
      dirty = true;
      scheduleSave();
      setStatus(`box set to "${classes[cls]}"`);
    }
  }

  /** Make the class bound to `key` current and apply it to the selected box. */
  function assignClassByShortcut(key: string): void {
    if (!(key in shortcutMap)) {
      setStatus(`no class bound to "${key}"`, "error");
      return;
    }
    assignClass(shortcutMap[key]);
  }

  /** Set the zoom level (clamped and persisted). */
  function setZoom(z: number): void {
    zoom = Math.min(24, Math.max(2, z));
    zoomInit = true;
    localStorage.setItem("labeler.annotate.zoom", String(zoom));
  }

  // -- keys -------------------------------------------------------------
  /** Global shortcuts: save, undo, class keys, navigation, zoom. */
  function onKey(e: KeyboardEvent): void {
    const t = e.target as HTMLElement | null;
    if (
      t &&
      (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")
    ) {
      return;
    }

    if (e.ctrlKey || e.metaKey) {
      if (e.key === "s") {
        e.preventDefault();
        void save();
      } else if (e.key === "z") {
        e.preventDefault();
        undo();
      }
      return;
    }

    if ((e.key >= "0" && e.key <= "9") || e.key === "." || e.key === "-") {
      e.preventDefault();
      assignClassByShortcut(e.key);
      return;
    }

    switch (e.key) {
      case "ArrowRight":
      case "d":
        e.preventDefault();
        void navigate(1);
        break;
      case "ArrowLeft":
      case "a":
        e.preventDefault();
        void navigate(-1);
        break;
      case "Enter":
        e.preventDefault();
        void save().then((ok) => {
          if (ok) void navigate(1);
        });
        break;
      case "Backspace":
      case "Delete":
        e.preventDefault();
        deleteSelected();
        break;
      case "z":
        undo();
        break;
      case "x":
        void markEmpty();
        break;
      case "f":
        void cycleFilter();
        break;
      case "[":
        setZoom(zoom - 1);
        break;
      case "]":
        setZoom(zoom + 1);
        break;
      case "Escape":
        editor?.cancelOp();
        break;
    }
  }

  /** Warn before leaving the page with unsaved changes. */
  function onBeforeUnload(e: BeforeUnloadEvent): void {
    if (dirty || saveTimer) {
      e.preventDefault();
      e.returnValue = "";
    }
  }

  onMount(() => {
    void loadList().catch((err) =>
      setStatus("startup failed: " + errMsg(err), "error"),
    );
  });
</script>

<svelte:window onkeydown={onKey} onbeforeunload={onBeforeUnload} />

<div class="workspace">
  <header>
    <div class="row">
      <strong>annotate</strong>
      <span class="filename">{curEntry ? curEntry.path : "—"}</span>
      <span class="spacer"></span>
      <span class="progress">
        {curIndex >= 0 ? curIndex + 1 : 0} / {images.length}
        · done {doneCount} / {images.length}
      </span>
      <span class="dirty-flag">{dirty || saveTimer ? "● unsaved" : ""}</span>
    </div>
    <div class="row">
      <button onclick={() => navigate(-1)} disabled={!canPrev}>← prev</button>
      <button onclick={() => navigate(1)} disabled={!canNext}>next →</button>
      <button
        onclick={() =>
          void save().then((ok) => {
            if (ok) void navigate(1);
          })}
        disabled={!curEntry}
      >
        save &amp; next <kbd>⏎</kbd>
      </button>
      <button onclick={() => void markEmpty()} disabled={!curEntry}>
        mark empty <kbd>x</kbd>
      </button>
      <button onclick={undo}>undo <kbd>z</kbd></button>
      <button onclick={deleteSelected} disabled={selected < 0}>
        delete box <kbd>⌫</kbd>
      </button>
      <button class:on={filterMode !== "all"} onclick={() => void cycleFilter()}>
        view: {filterMode} <kbd>f</kbd>
      </button>
      <button onclick={() => setZoom(zoom - 1)}>− <kbd>[</kbd></button>
      <span class="zoom-label">{img ? `${zoom}× · ${img.width}×${img.height} px` : ""}</span>
      <button onclick={() => setZoom(zoom + 1)}>+ <kbd>]</kbd></button>
      <span class="spacer"></span>
      <span class="status" class:ok={status.kind === "ok"} class:error={status.kind === "error"}>
        {status.msg}
      </span>
    </div>
    <div class="row palette">
      {#each classes as name, i (i)}
        <button
          class="chip"
          class:on={currentClass === i}
          style={`--chip: ${classColor(i)}`}
          onclick={() => assignClass(i)}
        >
          <span class="sw" style={`background: ${classColor(i)}`}></span>
          {name}
          {#if shortcutFor(name)}<kbd>{shortcutFor(name)}</kbd>{/if}
        </button>
      {/each}
    </div>
  </header>

  <main>
    <div class="canvas-wrap">
      <BoxEditor
        bind:this={editor}
        bind:boxes
        bind:selected
        image={img}
        {zoom}
        {classes}
        {currentClass}
        badge={emptyMarked ? "EMPTY" : ""}
        onedit={onEdit}
        onstatus={setStatus}
      />
    </div>
    <aside>
      <h3>boxes</h3>
      <ul class="boxlist">
        {#each boxes as b, i (i)}
          <li class:sel={i === selected}>
            <button class="row" onclick={() => (selected = i)}>
              <span class="swatch" style={`background: ${classColor(b.cls)}`}></span>
              <span>{classes[b.cls] ?? String(b.cls)}</span>
              <small>
                {Math.round(b.x)},{Math.round(b.y)} · {Math.round(b.w)}×{Math.round(b.h)}
              </small>
            </button>
            <button class="del" title="delete box" onclick={() => deleteBox(i)}>×</button>
          </li>
        {/each}
      </ul>
      <div class="empty-note">
        {boxes.length ? "" : "no boxes yet — drag on the image to draw one"}
      </div>
      <h3>keys</h3>
      <ul class="help">
        <li><kbd>drag</kbd><span>draw box (needs a current class)</span></li>
        <li><kbd>shift-click</kbd><span>select box</span></li>
        <li><kbd>shift-drag</kbd><span>select &amp; move box</span></li>
        <li><kbd>drag box</kbd><span>move (selected box only)</span></li>
        <li><kbd>drag handle</kbd><span>resize selected box</span></li>
        <li><kbd>right-click</kbd><span>delete box</span></li>
        <li><kbd>0-9 . -</kbd><span>class (selected box + current)</span></li>
        <li><kbd>← → / a d</kbd><span>prev / next crop</span></li>
        <li><kbd>⏎</kbd><span>save &amp; next</span></li>
        <li><kbd>x</kbd><span>mark empty / ignored</span></li>
        <li><kbd>z</kbd><span>undo</span></li>
        <li><kbd>esc</kbd><span>deselect / cancel drag</span></li>
        <li><kbd>ctrl+s</kbd><span>save now</span></li>
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
  .progress {
    color: var(--text);
  }
  .dirty-flag {
    color: var(--accent);
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
  .zoom-label {
    color: var(--dim);
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
  button.on {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
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

  .palette {
    gap: 5px;
  }
  .chip {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 2px 8px;
    border-radius: 12px;
    border: 1px solid var(--edge);
    background: #22262e;
    cursor: pointer;
  }
  .chip.on {
    border-color: var(--chip);
    background: #2b303a;
  }
  .chip .sw {
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }
  .chip kbd {
    padding: 0 4px;
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
    border: 1px solid transparent;
    border-radius: 5px;
    cursor: pointer;
  }
  .boxlist li:hover {
    background: #262a33;
  }
  .boxlist li.sel {
    border-color: var(--accent);
    background: #262a33;
  }
  .boxlist .row {
    display: flex;
    align-items: center;
    gap: 7px;
    flex: 1;
    min-width: 0;
    background: none;
    border: none;
    padding: 0;
    text-align: left;
    color: var(--text);
  }
  .boxlist .row:hover {
    background: none;
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
    white-space: nowrap;
  }
  .boxlist .del {
    padding: 0 6px;
    line-height: 18px;
    border: none;
    background: none;
    color: var(--dim);
  }
  .boxlist .del:hover {
    color: var(--error);
    background: none;
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
