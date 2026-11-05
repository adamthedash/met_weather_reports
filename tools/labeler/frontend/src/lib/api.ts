/** Typed fetch helpers for the labeler JSON API. */

import type {
  AuditAction, AuditDecisionResult, AuditList, BatchCommitResult,
  BatchDecision, BatchList, BoxN, ClassInfo, CompareResult, DecisionResult,
  Listing, ManualResult, PredBoxN, RankingSummary, ReviewList, SummaryOverview,
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
  /** The dataset's classes and their presentation (names from data.yaml,
   *  display glyphs and keyboard shortcuts from classes.yaml). */
  classes: () => fetchJSON<{ classes: ClassInfo[] }>("/api/classes"),
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
  /** Queue rankings that order the review queue (drives the UI selector). */
  reviewRankings: () =>
    fetchJSON<{ rankings: RankingSummary[] }>("/api/review/rankings"),
  /** The review queue in `ranking`'s order: undecided predictions + the stale count. */
  reviewList: (ranking?: string) =>
    fetchJSON<ReviewList>(
      `/api/review/list${ranking ? `?ranking=${encodeURIComponent(ranking)}` : ""}`,
    ),
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
  /** Flag a prediction's crop as needing manual adjustment: discard the
   *  prediction and add the crop to the manual queue (annotate serves it
   *  first). */
  markManual: (path: string) =>
    fetchJSON<ManualResult>("/api/review/manual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }),

  // -- batch review stage -----------------------------------------------
  /** The batch-review queue: the review queue high-confidence-first with
   *  empty predictions sprinkled in, boxes inlined per entry. */
  batchList: () => fetchJSON<BatchList>("/api/batch-review/list"),
  /** Commit a displayed batch: accept/reject/manual per crop, one result
   *  per decision (a failing crop never blocks the rest). */
  batchDecision: (decisions: BatchDecision[]) =>
    fetchJSON<BatchCommitResult>("/api/batch-review/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decisions }),
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

  // -- summary stage ----------------------------------------------------
  /** The dataset's aggregate state: totals, class and per-subject distributions. */
  summaryOverview: () => fetchJSON<SummaryOverview>("/api/summary/overview"),
};
