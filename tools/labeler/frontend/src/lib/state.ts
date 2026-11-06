/**
 * Per-dataset localStorage key namespacing.
 *
 * The labeler serves one dataset at a time, but the browser origin (and so
 * the localStorage) is shared between datasets. Dataset-shaped view state —
 * the annotate zoom, the last-viewed crop, the batch tile scale — is keyed by
 * the mounted dataset's id so switching datasets never carries one dataset's
 * state into another. Generic preferences (queue ranking, filter mode, batch
 * size) stay shared under their plain keys.
 *
 * The id comes from `GET /api/health`; the app fetches it before mounting
 * any workspace (see App.svelte), so keys are correct by first use.
 */

/** The mounted dataset's id, "" until the health probe reports it. */
let id = "";

/** Record the mounted dataset's id (called once from the app's health probe). */
export function setDatasetId(datasetId: string): void {
  id = datasetId;
}

/** localStorage key of persisted state `name`, namespaced by the dataset id. */
export function stateKey(name: string): string {
  return id ? `${name}:${id}` : name;
}
