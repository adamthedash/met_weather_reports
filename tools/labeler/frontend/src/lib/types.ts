/** Shared types for the labeler UI. */

/** Which side of the audit comparison a box came from (drives rendering). */
export type BoxSource = "label" | "pred";

/** A bounding box in image pixels (the editor's working unit).
 *  `conf` is display-only and carried by review predictions. */
export type Box = {
  cls: number;
  x: number;
  y: number;
  w: number;
  h: number;
  conf?: number;
  /** Audit overlay side: predictions render dashed, annotations solid. */
  source?: BoxSource;
};

/** A YOLO bounding box, normalised 0-1 (the wire/disk unit). */
export type BoxN = { cls: number; cx: number; cy: number; w: number; h: number };

/** A predicted YOLO box, normalised 0-1, with its confidence score. */
export type PredBoxN = BoxN & { conf: number };

/** One queue entry: a crop image and its labeling status. */
export type ImageEntry = { path: string; done: boolean; has_labels: boolean };

/** The annotate queue listing. */
export type Listing = { images: ImageEntry[]; total: number; done: number };

/** Navigation filter over the queue (prev/next predicate, not a view). */
export type FilterMode = "all" | "annotated" | "non-annotated";

/** One review-queue entry (mirrors the server's queue_and_stale entries):
 *  `boxes` counts predicted boxes (null for a malformed file), `error` carries
 *  the parse message of a malformed prediction file. */
export type PredEntry = {
  path: string;
  boxes: number | null;
  empty: boolean;
  error: string | null;
};

/** The review queue listing. */
export type ReviewList = {
  predictions: PredEntry[];
  total: number;
  remaining: number;
};

/** Result of a review decision (POST /api/review/decision). */
export type DecisionResult = {
  ok: boolean;
  accepted: boolean;
  saved: number;
  already_annotated: boolean;
  remaining: number;
};

/** One audit-queue entry: an annotated crop whose model prediction disagrees.
 *  `label_boxes` / `pred_boxes` count each side's boxes (null for a malformed
 *  file — the side that failed is the one with the null count), `issues`
 *  counts the disagreements (null when a side is malformed), and `error`
 *  carries the parse message of a malformed file. */
export type AuditEntry = {
  path: string;
  label_boxes: number | null;
  pred_boxes: number | null;
  issues: number | null;
  error: string | null;
};

/** The audit queue listing; `total` counts every compared crop. */
export type AuditList = {
  audits: AuditEntry[];
  total: number;
  remaining: number;
};

/** One model-vs-label disagreement of a crop.
 *  `label` / `pred` index the compare endpoint's box lists (null for the
 *  side an issue does not involve); `iou` is set for matched pairs only. */
export type AuditIssue = {
  kind: "unmatched_label" | "unmatched_pred" | "class_mismatch" | "weak_overlap";
  label: number | null;
  pred: number | null;
  iou: number | null;
};

/** Result of GET /api/audit/compare: both box sets and their disagreements. */
export type CompareResult = {
  path: string;
  labels: BoxN[];
  predictions: PredBoxN[];
  issues: AuditIssue[];
};

/** A three-way audit decision (POST /api/audit/decision). */
export type AuditAction = "keep_label" | "take_prediction" | "discard_both";

/** Result of an audit decision. */
export type AuditDecisionResult = {
  ok: boolean;
  action: AuditAction;
  saved: number;
  remaining: number;
};
