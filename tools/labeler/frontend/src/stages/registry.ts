/**
 * Frontend stage registry: maps a stage id (same id as the backend registry in
 * tools/labeler/stages/) to its workspace component. A new stage adds one
 * entry here and one backend stage module; the shared engine stays untouched.
 */
import type { Component } from "svelte";

import AnnotateWorkspace from "./AnnotateWorkspace.svelte";

export const workspaces: Record<string, Component> = {
  annotate: AnnotateWorkspace,
};
