/** Typed fetch helpers for the labeler JSON API. */

import type {
  AuditAction, AuditDecisionResult, AuditList, BoxN, CompareResult,
  DecisionResult, Listing, PredBoxN, ReviewList,
} from "./types";

/** Shape every API error response has (see tools/labeler/app.py). */
export type ApiError = { error: string };

/** Static description of a workflow stage (see tools/labeler/stages/base.py). */
export type StageSummary = { id: string; title: string; description: string };

/** GET `url` and parse the JSON body, throwing the API's error message on failure. */
export async function fetchJSON<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as Partial<ApiError>;
      if (body.error) message = body.error;
    } catch {
      // non-JSON error body: keep the status text
    }
    throw new Error(message);
  }
  return (await res.json()) as T;
}

/** Load an image URL into an HTMLImageElement, as a promise. */
export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("failed to load image"));
    img.src = url;
  });
}

/** Save result of POST /api/annotate/labels. */
export type SaveResult = { ok: boolean; saved: number; has_labels: boolean };

export const api = {
  /** Every registered workflow stage. */
  stages: () => fetchJSON<{ stages: StageSummary[] }>("/api/stages"),
  /** The dataset's class names. */
  classes: () => fetchJSON<{ classes: string[] }>("/api/classes"),
  /** URL of a crop image (for <img src> / loadImage). */
  imageURL: (path: string) => `/api/image?path=${encodeURIComponent(path)}`,
  /** The annotate queue listing. */
  list: () => fetchJSON<Listing>("/api/annotate/list"),
  /** The crop's current boxes (normalised), empty when unlabeled. */
  labels: (path: string) =>
    fetchJSON<{ path: string; boxes: BoxN[] }>(
      `/api/annotate/labels?path=${encodeURIComponent(path)}`,
    ),
  /** Save the crop's boxes; an empty list marks the crop empty. */
  saveLabels: (path: string, boxes: BoxN[]) =>
    fetchJSON<SaveResult>("/api/annotate/labels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, boxes }),
    }),

  // -- review stage -----------------------------------------------------
  /** The review queue: undecided predictions + the stale count. */
  reviewList: () => fetchJSON<ReviewList>("/api/review/list"),
  /** The prediction boxes (normalised, with confidence) for one crop. */
  prediction: (path: string) =>
    fetchJSON<{ path: string; boxes: PredBoxN[] }>(
      `/api/review/prediction?path=${encodeURIComponent(path)}`,
    ),
  /** Apply an accept/reject decision for one prediction. */
  decide: (path: string, accept: boolean) =>
    fetchJSON<DecisionResult>("/api/review/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, accept }),
    }),

  // -- audit stage ------------------------------------------------------
  /** The audit queue: disagreeing crops + the compared count. */
  auditList: () => fetchJSON<AuditList>("/api/audit/list"),
  /** Both box sets and the disagreement reasons for one crop. */
  auditCompare: (path: string) =>
    fetchJSON<CompareResult>(
      `/api/audit/compare?path=${encodeURIComponent(path)}`,
    ),
  /** Apply a keep-label / take-prediction / discard-both decision. */
  auditDecision: (path: string, action: AuditAction) =>
    fetchJSON<AuditDecisionResult>("/api/audit/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, action }),
    }),
};
