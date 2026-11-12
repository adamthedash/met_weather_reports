# Rewrite the labeler as a Rust workspace (Axum + Leptos), alongside the current tool, then cut over

## Summary

Add a Rust implementation of the labeler under `tools/labeler-rs` as a Cargo workspace: a pure `labeler-core` crate for dataset/workflow logic, an Axum/Tokio `labeler-server` that embeds the built UI, and a Leptos CSR/WASM `labeler-ui`. It reproduces today's workflows, on-disk formats and HTTP contract so it can run beside the Python/Svelte tool and be verified for parity; `just` and the docs cut over once all five stages match, with deletion of the Python implementation left to a follow-up change.

## Motivation

The labeler currently needs two runtimes to operate (the pixi-managed Python environment and Node/pnpm for the UI build) and has no automated tests, while its behaviour is defined only by prose in `tools/labeler/README.md`. A Rust implementation delivers the same tool as a single compiled binary with the UI embedded, makes the dataset model and API contract typed and compile-checked, separates blocking filesystem scans from the async request path explicitly, and gives the workflows their first executable specification in the form of core unit tests plus a parity harness. The annotator keeps the same five workflows and keyboard-driven UI; maintainers get one language and one build toolchain for the tool; the pipeline is unaffected because the on-disk contracts do not change.

## Current State

### Backend (Python, FastAPI + uvicorn)

- `tools/labeler/app.py:create_app` builds the ASGI app: core routes (`/api/health`, `/api/stages`, `/api/classes`, `/api/image`), per-stage routers mounted at `/api/<stage.id>`, a `no_store` middleware setting `Cache-Control: no-store`, exception handlers turning `ApiError`, `ValueError`, `HTTPException` and unhandled exceptions into `{"error": ...}`, and the `spa` fallback that serves `frontend/dist` (503 with a build hint when missing, 404 for missing asset-like paths and unknown `/api/*`).
- `tools/labeler/main.py:main` parses the CLI flags (`--dataset`, `--images`, `--labels`, `--manual`, `--predictions`, `--audit-predictions`, `--data-yaml`, `--class-map`, `--host`, `--port`, `--dist`), loads the dataset strictly, prints the startup summary and runs uvicorn.
- `tools/labeler/config.py:Config.dataset`, `default_dataset`, `dataset_paths`, `DATASET_ENV` derive the standard layout from `--dataset` / `$LABELER_DATASET` (default `datasets/digits`), with each root individually overridable.
- `tools/labeler/dataset.py:Dataset` is the domain model: `list_images`, `image_path`, `label_path_for`, `manual_path_for`, `is_manual`, `mark_manual`, `clear_manual`, `read_labels`, `read_label_file`, `validate_boxes`, `write_labels`, `prediction_path_for`, `audit_prediction_path_for`, `read_prediction`, `read_audit_prediction`, `read_prediction_file`, `queue_and_stale`, `label_class_counts`, `audit_queue`, `summary`. Module helpers: `natural_key`, `safe_path`, `load_classes`, `load_class_map`, `_StrictLoader` (PyYAML with duplicate-key rejection), and the constants `IMAGE_SUFFIXES`, `MAX_BOXES`, `RESERVED_KEYS`.
- `tools/labeler/compare.py:compare_boxes` and `iou` do greedy IoU pairing with `MIN_IOU = 0.5`.
- `tools/labeler/rankings.py:RANKINGS` registers `natural_order`, `confidence_order`, `rare_labels_order`, `fill_gaps_order`; `find_ranking` validates ids; `summaries` feeds the UI selector.
- `tools/labeler/stages/base.py:Stage` is the extension seam; `tools/labeler/stages/__init__.py:build_stages` registers annotate, review, batch-review, audit and summary. `review.py:decide`/`mark_manual` are reused by `batch_review.py:apply_decision`.
- `tools/labeler/web.py:read_json_body` (1 MB cap, empty body as `{}`) and `require_arg`; `tools/labeler/errors.py:ApiError` carries an HTTP status plus message.
- `safe_path` currently rejects empty, absolute and backslash paths and `..`, and resolves symlinks so a request cannot escape a configured root. **This is deliberately not carried over** (see Proposed Behavior).

### Frontend (Svelte 5 + Vite + TypeScript)

- `tools/labeler/frontend/src/App.svelte` probes `/api/health` for the dataset id, loads `/api/stages`, and routes between workspaces.
- `src/lib/BoxEditor.svelte` is the imperative canvas engine (draw/move/resize/delete, read-only mode, zoom, chips, badges, reading); `src/lib/Tile.svelte` is the batch grid tile.
- `src/lib/labels.ts` holds the tunable text sizes and `chipSize`, `drawChip`, `readingOf`; `src/lib/boxes.ts` holds `fitScale`, `handlePoints`, `hitHandle`, `hitBox`, `toNormalized`, `fromNormalized`; `src/lib/palette.ts` holds `classColor`/`classTextColor`; `src/lib/state.ts` namespaces persisted state by dataset id; `src/lib/api.ts` and `types.ts` are the typed client and wire types; `src/stages/registry.ts` maps stage ids to workspaces (annotate, review, batch-review, audit, summary).
- Build and run: `package.json` pins Svelte 5, Vite 7, TypeScript 5.9; `vite.config.ts` proxies `/api` to port 8125; the `justfile` recipes `labeler` and `labeler-build` drive it.

### Contracts that must be preserved

- **On-disk layout and formats** (`tools/labeler/README.md`, "Data layout"): YOLO labels `<cls> <cx> <cy> <w> <h>` normalised, 6 decimals; predictions add `<conf>`; mirrored sidecar trees under `labels/`, `manual/`, `predictions/`, `audit_predictions/`; empty label file = done/empty; manual marker + no label = manual queue; prediction + no label + no marker = review queue. `scripts/prepare_dataset.py`, `bootstrap_annotations.py`, `audit_annotations.py` and `recombine_labels.py` depend on these formats; none of them call the labeler's HTTP API.
- **Decision ordering** (review/audit documents in `tools/labeler/README.md`): accept writes the label before deleting the prediction; flag writes the manual marker before deleting the prediction; audit `take_prediction` writes before deleting; `discard_both` deletes the label before the prediction. Existing labels are never overwritten.
- **Strictness**: malformed label/prediction lines are reported with their file and line, never skipped; `data.yaml`/`classes.yaml` are validated strictly at startup, including duplicate keys, unknown fields, reserved keyboard shortcuts and class-name matching.
- **HTTP contract** (`tools/labeler/README.md`, "API contract"): routes, JSON field names, `{"error": ...}` bodies, `Cache-Control: no-store`, and the 400/404/413/500 statuses.
- **UI behaviour** (`tools/labeler/README.md`, "Workflows"): sticky class, 600 ms autosave and navigation flush, `beforeunload`, zoom/fit and persisted view state, undo depth 100, the manual-first annotate queue, view-mode filters, review/audit chips and reading, ranking selector, batch tile scale and size, 25% empty sprinkle, per-crop batch results, side-by-side audit, summary panels.

There are no tests for the labeler today (no `def test_` under `tools/`), so the current behaviour is specified only by the README and code.

## Proposed Behavior

### Deliverable and runtime

- A new workspace `tools/labeler-rs` with three crates: `labeler-core` (no HTTP or UI dependencies), `labeler-server` (binary), `labeler-ui` (WASM app). A committed `Cargo.lock`, `rust-toolchain.toml` pinning stable (MSRV at least 1.89, required by `serde-saphyr`) plus the `wasm32-unknown-unknown` target, and a committed UI build pipeline via `trunk`.
- `labeler-server` accepts the same flags with the same defaults and env fallback as `tools/labeler/main.py:main` (`LABELER_DATASET`, default `datasets/digits`, host `127.0.0.1`). `--port` follows the Python tool's contract: omitted, it binds the first free port on the host scanning upward from 8125 (so several instances can run at once); given, it pins that exact port and fails clearly when it is in use. `--dist` remains available to serve a UI build from disk; without it, the release binary serves the embedded `labeler-ui/dist`.
- Startup is strict: a missing images root, unreadable/invalid `data.yaml`, or invalid/mismatched `classes.yaml` aborts with a clear message and a non-zero exit, before binding the port. The same startup summary is printed (dataset, images, labels, classes, keys, done, manual, queue/stale, audit compared/disagreements, stages, asset source).
- One process, no browser-side build step at runtime; `Ctrl+C` stops cleanly.

### Workflow parity

All five stages behave as documented in `tools/labeler/README.md` ("Workflows"), including their keyboard maps, badges, queue ordering, decision semantics and persisted state:

- **Annotate** serves the manual queue first, then the backlog in natural order; sticky current class; save/mark-empty/undo/filter/zoom/fit; autosave and flush; manual flag cleared by a label write.
- **Review** serves the queue under a selected ranking (natural, high-confidence, rare-labels, fill-gaps) with `rank_note` and `label_errors`; accept/reject/flag have the same destructive, ordered file effects; malformed predictions are visible and can only be rejected, flagged or skipped.
- **Batch review** serves the confidence order with predicted backgrounds sprinkled at 25% and malformed entries last, inlined boxes, batch size 1–32, tile scale, marks and one commit with per-crop results.
- **Audit** queues crops whose label and audit prediction disagree per `compare_boxes`; side-by-side rendering; keep/take/discard with the same file ordering; broken sides cannot be taken/kept.
- **Summary** rescans on request and reports totals, class distribution, per-subject distribution, malformed labels and orphan labels.

### HTTP contract

Routes, request/response shapes, `{"error": ...}` bodies, `Cache-Control: no-store` and status codes stay identical so the two servers are interchangeable during the parity period. This includes `GET /api/docs` and `GET /api/openapi.json`. Malformed payload handling may be stricter than Python's implicit coercions (`int("1")`, `float("1.5")`): the Rust API requires JSON numbers and integer class ids, which the UI already sends and no script relies on. Match the labeler's own parse-error messages exactly (for example `"<file> line <n>: expected 5 fields, got <k>"`); YAML parser wording may differ as long as the file and problem are named.

### Path handling

Paths from query parameters and JSON bodies are treated as trusted relative paths and joined directly under the configured root. There is no traversal, absolute-path or symlink rejection: `labeler-core` exposes one `sidecar(rel)`/`join(root, rel)` helper and the `Dataset` accessors use it. This is an intentional simplification for a localhost-only, single-user tool whose dataset and clients are trusted; it is documented in the tool's "Notes & limitations". The on-disk mirroring (`<stem>.txt` in the same subdirectories) is unchanged.

### UI parity

The Leptos app keeps the same top bar, stage navigation, `/` picker, deep links (`/annotate`, `/review`, `/batch-review`, `/audit`, `/summary`), visual design (hand-rolled CSS, class palette, chip sizes and reading composition) and keyboard shortcuts. Persisted keys keep their names and dataset-id namespacing so state carries across the cutover on the same origin and port.

### Parity and cutover

1. Run the Rust server beside the Python server on different ports and dataset copies.
2. Run the existing Svelte build against the Rust server as an HTTP-contract smoke test.
3. Compare deterministic responses and the file tree after a scripted decision sequence against fixtures; compare batch-review as a multiset and per-entry classification, not order, because the sprinkle is random.
4. Cut over the `justfile` and READMEs once all five stages and the acceptance checklist pass, renaming the Python recipes to `labeler-py*` for the transition.
5. Delete the Python/Svelte implementation in a follow-up change.

### Failure handling

- Missing prediction/label/image on a decision → 404 with the existing message format.
- Malformed label/prediction files → 400 for reads, reported per entry for queue listings, never silently skipped.
- Body over 1,000,000 bytes → 413; invalid JSON or a non-object body → 400; unknown ranking → 400 listing valid ids; unknown `/api/*` route → 404 JSON.
- A failing crop in a batch commit reports `error` in its result and never aborts the other decisions.
- Unhandled failures → 500 `{"error": ...}` with the connection kept open.
- Scans do not block the async runtime; the UI never sees a stale response (`no-store` on every response, including images and errors).

## Implementation Notes

### Workspace layout

```
tools/labeler-rs/
├── Cargo.toml                     # workspace: labeler-core, labeler-server, labeler-ui, xtask
├── Cargo.lock                     # committed
├── rust-toolchain.toml            # stable + wasm32-unknown-unknown
└── crates/
    ├── labeler-core/              # pure domain logic, native tests
    │   └── src/{lib,config,yaml,dataset,compare,rankings,queue,summary}.rs
    │   └── src/stages/{annotate,review,batch_review,audit,summary}.rs
    ├── labeler-server/            # Axum binary
    │   └── src/{main,app,state,error,assets}.rs + src/routes/*.rs
    └── labeler-ui/                # Leptos CSR
        ├── index.html, Trunk.toml
        └── src/{main,app,api,state,geometry,labels,palette}.rs + src/components/*.rs + src/stages/*.rs
```

`labeler-core` mirrors the Python modules one-for-one: `yaml.rs` ports `load_classes`/`load_class_map` (including `RESERVED_KEYS`, duplicate-key and unknown-field rejection) using `serde-saphyr`; `dataset.rs` ports `Dataset` and `natural_key` and replaces `safe_path` with a plain join/sidecar helper; `compare.rs` ports `iou`/`compare_boxes`/`MIN_IOU`; `rankings.rs` ports the four rankings and `find_ranking`; `queue.rs` ports `queue_and_stale`/`label_class_counts`/`audit_queue`; `summary.rs` ports `Dataset.summary`. Stage decision functions live in `core/src/stages/*` so both the HTTP handlers and tests call the same code; this is the analogue of `stages/review.py:decide` being reused by `batch_review.py`.

`labeler-server` maps the Python app: `app.rs:create_app` for routers, middleware and the SPA/404 fallback; `error.rs:ApiError` with `IntoResponse`; `routes/core.rs`, `routes/annotate.rs`, `routes/review.rs`, `routes/batch_review.rs`, `routes/audit.rs`, `routes/summary.rs` as thin handlers; `assets.rs` for embedded/on-disk assets with the same 404/503 rules as `app.py:spa`.

`labeler-ui` maps the Svelte tree: `geometry.rs` ← `boxes.ts`, `labels.rs` ← `labels.ts`, `palette.rs` ← `palette.ts`, `state.rs` ← `state.ts`, `api.rs` ← `api.ts` + `types.ts`, `components/box_editor.rs` ← `BoxEditor.svelte`, `components/tile.rs` ← `Tile.svelte`, `stages/*` ← the five workspaces, `stages/registry.rs` ← `registry.ts`.

### Library choices and rationale

Versions below are the current stable releases checked while writing this plan (November 2026).

| Concern | Current implementation | Rust choice | Why |
|---|---|---|---|
| HTTP server/runtime | FastAPI + uvicorn | **axum 0.8 + tokio 1.54** | Typed extractors, `Router` nesting that maps directly to the stage seam, Tower middleware, and `tower-http` for static assets and headers. Actix-web is the only serious alternative; its separate middleware/runtime conventions buy nothing for a localhost tool. |
| JSON model | Raw dicts, manually assembled | **serde 1 + serde_json 1**, `serde_path_to_error` | Typed request/response structs replace `dict[str, object]`; field names and shapes are compile-checked; path-aware 400s for bad bodies. |
| YAML config | PyYAML + custom `_StrictLoader` | **serde-saphyr 2.1** (`default-features = false, features = ["deserialize"]`) | Rejects duplicate mapping keys by default, deserializes directly into typed structs with span-rich errors, no intermediate value tree, WASM-compatible. Built on `granit-parser` (a saphyr fork); not the `saphyr` crate itself. See the migration notes below. |
| CLI/config | argparse | **clap 4.6** (derive, `env`) | Reproduces `--dataset`/`$LABELER_DATASET` fallback and per-root overrides with generated help. |
| Errors | `ApiError` + FastAPI handlers | **thiserror 2 + `IntoResponse`** | Typed variants carrying status and message; `?` ergonomics; one place emits `{"error": ...}`. |
| Logging | uvicorn defaults | **tracing + tracing-subscriber** | Structured logs with `RUST_LOG`; low overhead; startup errors go to stderr. |
| Static UI | `FileResponse` + SPA fallback | **rust-embed 8** | One self-contained binary; debug builds read `labeler-ui/dist` from disk for fast UI iteration; `--dist` keeps the old on-disk story. `tower-http::ServeDir` remains the alternative if embedding is dropped. |
| MIME | `mimetypes` | **mime_guess 2** | Same guessing (including `.webp`, `.bmp`, `.wasm`) without OS tables. |
| Path handling | `safe_path` + `resolve`/`is_relative_to` | **plain `root.join(rel)`** | Per scope, the dataset and clients are trusted: no traversal or symlink checks. Centralise the join and sidecar mirroring so the roots remain the only source of paths. |
| Tree walking | `Path.rglob` | **walkdir 2.5** inside `spawn_blocking` | Streaming traversal without a full-tree allocation, and an explicit blocking boundary that keeps Tokio responsive. `ignore` is rejected because gitignore semantics would skip generated datasets. |
| Natural ordering | `natural_key` | **hand-rolled `NaturalKey` comparator** | Must match exactly (lowercase non-digit runs, numeric digit runs, stable ties). A crate such as `natord` risks subtly different ordering. |
| Parallel scans | none (GIL-bound) | **rayon 1.12** (optional, behind a feature) | `summary` and `label_class_counts` are per-file independent; add only after profiling. |
| Randomness | `random` module | **rand 0.10**, `&mut impl Rng` passed into `sprinkle` | Deterministic tests and a stable 25% sprinkle without a global mutable RNG. |
| API docs | FastAPI `/api/docs` | **utoipa 6 + utoipa-scalar 0.4** | Keeps a browsable reference and `/api/openapi.json` without bundling Swagger JS. |
| UI framework | Svelte 5 + Vite 7 + TypeScript | **Leptos 0.8 (CSR) + trunk 0.22** | Signals map closely to Svelte 5 runes; fine-grained updates avoid VDOM churn during canvas gestures; `trunk` replaces Vite and removes Node; SSR/server functions remain possible later. Dioxus 0.7 is credible but brings a VDOM and desktop/mobile machinery this tool does not need; Yew and Sycamore are older or smaller-ecosystem choices. |
| HTTP client | `fetch` helper | **gloo-net 0.7** | Thin fetch wrapper with typed JSON; much smaller than `reqwest` compiled to WASM. |
| Local persistence | `localStorage` | **gloo-storage 0.4** | Typed `LocalStorage` access; preserves key names and dataset namespacing. |
| Timers | `setTimeout` | **gloo-timers 0.4** (`futures`) | 600 ms autosave debounce and keyboard repeat without dragging in a runtime. |
| Canvas/browser APIs | Canvas 2D + DOM | **web-sys 0.3 + js-sys + wasm-bindgen 0.2** | The editor is already imperative; Leptos provides refs/events, web-sys provides Canvas 2D, text metrics and image loading. No rendering or geometry crate is needed. |
| Routing | `pushState` + path parsing | **leptos_router 0.8** | Same paths and deep links, with back/forward handled. |
| Testing | none | **rstest, insta, tempfile; axum-test 21, tower `ServiceExt`, http-body-util; assert_cmd, predicates; wasm-bindgen-test** | Fixture datasets and snapshot responses for core logic, in-process route tests, startup/CLI tests, and native-testable UI geometry/label modules. |

Rejected specifically: `serde_yaml` (archived), `saphyr` as the primary config parser (good parser, but its typed Serde integration is what we want and `serde-saphyr` provides it; duplicate handling and config validation still need custom code), `actix-web` (separate ecosystem, no benefit here), `reqwest`-wasm (heavier client), `natord` (ordering fidelity risk), `image`/`fast_image_resize` (the backend never decodes images), `cap-std`/path hardening (dropped per scope).

### YAML with `serde-saphyr`: extra work and what to skip

`serde-saphyr` gives us the two things the current strict loader needs for free: duplicate mapping keys are rejected by default (`DuplicateKeyPolicy::Error`), and deserialization is directly into Rust types with line/column spans in the error. It has no generic `Yaml` tree, so the dynamic validation in `load_classes`/`load_class_map` must be expressed as typed models plus a small custom layer.

**Extra work**

1. **Custom scalar and key types.** The config files accept scalars that Serde does not model directly:
   - `NameIndex` for `data.yaml`'s `names:` keys: YAML integer or numeric string → `i64`; reject booleans, floats, null, sequences/maps.
   - `ScalarText` for glyphs, shortcuts and class names: YAML string or integer → `String`; reject booleans, floats, null, and empty results.
   - `ClassName` for `classes.yaml`'s top-level keys: integer or string → `String` (real datasets may key the class named `"0"` as `0:`).
   - `KeySpec` for `key:`: a required field that accepts YAML `null` (click-only) or a scalar, so a missing `key` stays an error while `key: null` stays valid.
   These are small visitors (~150 lines including tests) but they are the bulk of the migration work.
2. **Manual semantic checks stay manual**: non-empty `names`, sorted indices, exact class-name-set matching, reserved annotate keys, duplicate key bindings, empty glyph/key rejection, and the `key`-must-be-present rule. (~80 lines.)
3. **Error framing.** `serde_saphyr::Error` is span-rich but worded differently from PyYAML; wrap it with the file path and keep our own semantic messages. Do not chase byte-identical YAML syntax errors.
4. **Dialect differences.** `serde-saphyr` follows YAML 1.2 semantics (with deliberate PyYAML compatibility in places), while PyYAML is 1.1: `yes`/`no`/`on`/`off` are strings rather than booleans, leading-zero numbers are not octal, and merge keys/anchors follow the crate's policies. Some odd configs will therefore be accepted or rejected differently.
5. **Null handling.** A null into a plain `String` is `Error::NullIntoString`; the typed model above handles `key: null` deliberately and lets `glyph: null` fail with a clear wrapped message.
6. **MSRV/dependency lineage.** Requires Rust 1.89+ and pulls `granit-parser`; it is not the `saphyr` crate and shares no code with the archived `serde_yaml`. Both are acceptable, but the toolchain pin must reflect the MSRV.
7. **No untyped tree.** The Python code parses to dicts and inspects them at runtime; the Rust version must decide every accepted shape in the type model. This is more upfront work, but it is also what makes the strictness testable.

**What we can skip**

1. **Byte-identical PyYAML error text** for syntax errors — use `serde-saphyr`'s span-rich message wrapped with the file path.
2. **YAML 1.1 compatibility shims** (`yes`/`on` booleans, octal leading zeros). Accept YAML 1.2 semantics; document the difference.
3. **Merge keys, anchors/tags, `!include`, properties, comments, and validation integrations** — unused by these two files. Disable the optional features (`default-features = false, features = ["deserialize"]`) and leave merge-key handling at the crate default; no compatibility work.
4. **A generic `serde_json::Value` config path** — typed structs are stricter and simpler to test; `serde-saphyr` supports untyped reads, but we do not need them.
5. **Exact unknown-field wording** in `classes.yaml` — `#[serde(deny_unknown_fields)]` gives a clear serde error; no need to reproduce the sorted field list.
6. **`data.yaml` unknown-field rejection** — the current loader ignores keys other than `names:`; keep that leniency.
7. **Pathological converted-duplicate checks** (`names: {0: a, "0": b}`, or `classes.yaml` keys `0` and `"0"`). `serde-saphyr`'s duplicate policy catches same-spelling duplicates; skip the int/string-equivalent collapse check and document the edge case.
8. **Class-index contiguity validation** — never implemented today, and remains out of scope.
9. **Path safety** — removed entirely per your feedback; no `safe_path`, no canonicalisation, no traversal tests.

**Typed model sketch**

```rust
#[derive(Deserialize)]
struct DataYaml {
    names: Option<BTreeMap<NameIndex, ScalarText>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassEntry {
    glyph: ScalarText,   // required
    key: KeySpec,        // required; YAML null -> click-only
}
```

`load_classes` sorts `NameIndex`, rejects an empty/absent `names`, and converts values to `String`; `load_class_map` deserializes `BTreeMap<ClassName, ClassEntry>`, compares the key set to the class names exactly, then validates reserved keys and duplicate bindings. Any `serde_saphyr::Error` is wrapped as `"<path>: invalid YAML: <message>"`.

### Why a static (CSR) UI

- **Partly carry-over:** the current UI is a Vite build served as static files by FastAPI; porting it as a Leptos CSR app is the lowest-risk parity path and keeps the dev loop familiar.
- **Independently justified:** the tool is localhost-only and single-user, with no SEO, shared-URL or network first-paint requirement; each stage fetches JSON and paints a canvas, so server-rendered HTML would only be a shell. A static bundle embeds cleanly in the server binary, avoids `cargo-leptos`' SSR-plus-hydration build and its client/server state boundary, keeps the JSON API independently testable, and lets the existing Svelte build validate the Rust server during parity. Everything is bundled locally; there are no CDN dependencies.
- **Costs:** a few MB of WASM and a client boot, negligible on localhost or through an SSH tunnel; `wasm-opt` and optional `tower-http` compression can shrink the bundle if it ever matters.
- **Not implied:** embedding assets is a packaging choice, not the definition of "static"; the UI can still be served from `--dist` during development.

### Backend design notes

- `AppState` holds `Arc<Dataset>` (immutable after startup) and an `Assets` enum (embedded or `--dist` directory); no locks are needed. `Dataset` is `Clone`/`Send`/`Sync`, so scans can run on the blocking pool.
- Decision routes use `tokio::fs` and preserve the operation order above; scan routes (`/list`, `/overview`) call `spawn_blocking` around `walkdir`-based core functions. This is a deliberate improvement over FastAPI handlers doing synchronous file I/O on the event loop.
- Stage seam: a `Stage` trait with `id`, `title`, `description`, and `router(&self) -> Router<AppState>`, registered in `stages::registry()`. Ranking seam: a `Ranking` trait with `summary()` and `order(&Dataset, Vec<QueueEntry>) -> RankingResult`, registered in `RANKINGS: &[&dyn Ranking]`. Both mirror `stages/base.py:Stage` and `rankings.py:Ranking`; the compile-time registry replaces the dynamic import list.
- Queue entries become typed structs (`QueueEntry { path, boxes: Option<usize>, empty, error: Option<String>, rank_note: Option<String> }`) instead of JSON-shaped dicts; the wire representation is produced by serde.
- `validate_boxes` and `read_prediction_file` keep the same limits and checks (`MAX_BOXES = 64`, class range, finite values, `0..=1` for predictions, clamping in `validate_boxes`). Use `f64::total_cmp` where Python sorts floats.
- Label writing uses `format!("{cls} {cx:.6} {cy:.6} {w:.6} {h:.6}")` plus a trailing newline only when non-empty, matching `Dataset.write_labels` byte-for-byte; verify rounding against Python with fixture values.
- `natural_key` is implemented as an `Ord` wrapper over path segments; unit tests cover `10_2` before `10_10`, mixed case, and ties.
- `load_classes` keeps the current semantics: sort the int-convertible `names:` keys and take the values; no contiguity check in this change.
- `sprinkle` takes an RNG so tests can be deterministic; production uses `rand::rng()`. The 25% target, empty pool random order and malformed-last rule are unchanged.
- `ApiJson<T>` maps extractor rejections (invalid JSON, non-object, body limit) to `ApiError`; `require_arg` keeps the `missing query parameter 'path'` message.
- `/api/docs` is served by `utoipa-scalar` from the `utoipa` spec; annotate the request/response structs used by the handlers.
- `.gitignore` gains `tools/labeler-rs/target`; `**/dist` already covers the UI build output.

### Frontend design notes

- `api.rs` mirrors `api.ts` endpoint-for-endpoint; wire structs mirror `types.ts`, with serde renames for `cls`, `already_annotated`, `rank_note`, `label_errors`, `orphan_labels` and the other snake_case fields.
- `BoxEditor` keeps its design: working boxes in image pixels, an `Op` enum for Draw/Move/Resize, parent-owned `boxes`/`selected` signals, an `on_edit`/`on_status` callback pair, read-only mode and a `label_style` switch for review/audit chips.
- Geometry and label modules stay free of web-sys where possible so they compile and test natively; only the canvas wrapper needs the browser.
- `app.rs` fetches `/api/health` before mounting workspaces (so dataset-namespaced storage keys are correct on first use), then renders the top bar, picker or routed workspace from `/api/stages`.
- Keyboard shortcuts are bound with Leptos window listeners (`window_event_listener` plus `on_cleanup`) and mirror the exact key map in the README; `beforeunload` uses `web_sys` and is registered/removed in an effect.
- The router uses `leptos_router`; `stages/registry.rs` can be a `match` on the stage id returning the component rather than a heterogeneous map, since Rust component types differ.
- `trunk serve` proxies `/api` to 8125 (equivalent to `vite.config.ts`); `trunk build --release` produces the dist that `rust-embed` consumes. Commit `Cargo.lock` and pin the toolchain so `wasm-bindgen` and `trunk` stay reproducible.

### Testing and parity harness

- Core tests use `tempfile` fixture datasets and `rstest` cases for parsing, clamping, natural ordering, IoU pairing, rankings and summary; `insta` snapshots the JSON-shaped outputs.
- YAML tests cover: duplicate keys (rejected by default), int and numeric-string `names:` keys, integer class names/glyphs, `key: null` vs missing `key`, unknown fields, empty strings, booleans/floats in scalar positions, and class-set mismatches. Include a small fixture that demonstrates the YAML 1.1 vs 1.2 difference so the accepted behaviour is explicit.
- Server tests build the `Router` in-process and hit it with `tower::ServiceExt::oneshot`/`axum-test` and `http-body-util`, covering routes, error bodies, status codes and `no-store`.
- CLI/startup tests use `assert_cmd` + `predicates` for flag parsing, env fallback and strict dataset failures.
- `xtask parity` (or an equivalent `just labeler-rs-parity` recipe) copies a fixture dataset twice, starts `pixi run python -m tools.labeler` and the Rust server on distinct ports, replays a scripted request sequence, compares deterministic responses, runs a scripted decision sequence, and diffs the resulting file trees. This harness is the executable definition of the migration and should be kept as a regression check until the Python tool is removed.
- UI logic (geometry, reading composition, palette) is tested natively; browser-specific code gets `wasm-bindgen-test` where practical and a manual acceptance checklist otherwise.

### Trade-offs against the current stack

FastAPI gives automatic OpenAPI and very terse handlers; Rust needs more boilerplate (serde structs, custom YAML visitors, utoipa attributes, explicit extractors) and two build artifacts (WASM + native). Compile times are longer than Python startup, and contributors need Rust plus `trunk`. In exchange, the dataset/API model is typed, blocking scans are explicit, the tool ships as one binary, and the behaviour gains tests it has never had. The riskiest parity area is YAML strictness: the typed model is more verbose than PyYAML's dict walks, and the YAML 1.2 dialect differs from PyYAML 1.1 in edge cases, so the config fixtures should cover those explicitly. Float formatting and labeler-generated error-message wording are the other parity-sensitive areas.

## Impact / Compatibility

- **Pipeline scripts**: unaffected. `scripts/prepare_dataset.py`, `bootstrap_annotations.py`, `audit_annotations.py` and `recombine_labels.py` share only the on-disk formats and semantics, which are preserved byte-for-byte.
- **HTTP API**: only the labeler UI consumes it today; no script calls it. It is kept identical during the transition so the Svelte build can be pointed at the Rust server, then the Rust UI becomes the sole client.
- **Path handling**: the rewrite drops `safe_path`. A caller that can reach the bound address can supply an absolute or `..`-containing path and make the server read (and, on write routes, create) files outside the dataset roots. This is accepted because the tool is localhost-only, unauthenticated by design, and the UI sends only paths it received from listings; it is recorded in the tool's "Notes & limitations". Reintroducing a join guard later would be a small, local change.
- **Toolchain**: adds Rust stable (MSRV 1.89), the `wasm32-unknown-unknown` target and `trunk`. The Python/pixi environment remains required for the rest of the pipeline. Node/pnpm is no longer needed for the labeler after cutover.
- **Build and run**: the release build is `trunk build` then `cargo build --release`, producing one binary with the UI embedded; debug builds read the UI from disk. `just` gains `labeler-rs`, `labeler-rs-build`, `labeler-rs-api`, `labeler-rs-ui`, `labeler-rs-test` and `labeler-rs-parity`; at cutover the existing `labeler*` recipes point at Rust and the Python ones are renamed `labeler-py*`.
- **Local state**: same origin, port and storage key names means zoom/crop/filter/ranking/batch-size state survives the cutover.
- **During the transition**: both implementations coexist under `tools/labeler` and `tools/labeler-rs`; documentation is updated in both places with a note that the Rust tool is the target implementation.

## Out of Scope

- Deleting the Python backend and Svelte frontend; that is the follow-up change after cutover.
- New stages, rankings, workflows or behaviour changes; the port replicates the documented behaviour.
- Changes to dataset format, sidecar semantics, class validation (including enforcing contiguous class ids) or the training pipeline.
- Path traversal/symlink hardening and re-introducing `safe_path` (dropped by design); the dataset and clients are trusted.
- YAML 1.1 compatibility shims, merge-key/tag/include support, or PyYAML-identical error text.
- Authentication, TLS, remote/multi-user hosting; the tool remains localhost-only.
- Desktop/mobile packaging (Tauri or similar), server-side rendering, or Leptos server functions.
- Backend image decoding, resizing, thumbnails or format conversion; images continue to be streamed as files.
- Performance tuning beyond the explicit blocking boundary and optional rayon; micro-benchmarks are deferred.

## Open Questions

None.
