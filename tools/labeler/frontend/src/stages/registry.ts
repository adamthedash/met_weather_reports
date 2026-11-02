/**
 * Frontend stage registry: maps a stage id (same id as the backend registry in
 * tools/labeler/stages/base.py) to its workspace component and keymap.
 *
 * The annotate and review workspaces land with the shared engine; a new stage
 * adds one entry here and one backend stage module.
 */
export type StageWorkspace = {
  id: string;
  /** Svelte component rendering the stage's workspace. */
  component: unknown;
};

export const stages: StageWorkspace[] = [];
