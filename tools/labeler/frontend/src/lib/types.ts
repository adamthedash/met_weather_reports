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

/** One class's presentation (from the dataset's classes.yaml via
 *  GET /api/classes): the display glyph drawn on chips and in readings, and
 *  the annotate-stage keyboard shortcut; `key` is null for a click-only
 *  class. Index in the `/api/classes` list = class id. */
export type ClassInfo = { name: string; glyph: string; key: string | null };

/** One queue entry: a crop image and its labeling status. `manual` flags the
 *  manual queue (marked in review as needing manual adjustment, not yet
 *  labeled); the listing serves manual entries first. */
export type ImageEntry = {
  path: string;
  done: boolean;
  has_labels: boolean;
  manual: boolean;
};

/** The annotate queue listing; `manual` counts the flagged crops. */
export type Listing = {
  images: ImageEntry[];
  total: number;
  done: number;
  manual: number;
};

/** Navigation filter over the queue (prev/next predicate, not a view).
 *  `manual` restricts prev/next to the manual queue (flagged in review as
 *  needing manual adjustment, not yet labeled). */
export type FilterMode = "all" | "annotated" | "non-annotated" | "manual";

/** One review-queue entry (mirrors the server's queue_and_stale entries):
 *  `boxes` counts predicted boxes (null for a malformed file), `error` carries
 *  the parse message of a malformed prediction file, and `rank_note` is the
 *  ranking's one-line explanation of the entry's placement (null when the
 *  ranking has nothing to say). */
export type PredEntry = {
  path: string;
  boxes: number | null;
  empty: boolean;
  error: string | null;
  rank_note: string | null;
};

/** One label file a ranking could not parse while scoring (skipped, reported). */
export type LabelError = { path: string; error: string };

/** The review queue listing: the queue in the echoed `ranking`'s order. */
export type ReviewList = {
  predictions: PredEntry[];
  total: number;
  remaining: number;
  ranking: string;
  label_errors: LabelError[];
};

/** Static description of a queue ranking (see tools/labeler/rankings.py). */
export type RankingSummary = { id: string; title: string; description: string };

/** Result of a review decision (POST /api/review/decision). */
export type DecisionResult = {
  ok: boolean;
  accepted: boolean;
  saved: number;
  already_annotated: boolean;
};

/** Result of a needs-manual-adjustment flag (POST /api/review/manual):
 *  `marked` is false only for the already-annotated outcome — the stale
 *  prediction was dropped and no marker written. */
export type ManualResult = {
  ok: boolean;
  marked: boolean;
  already_annotated: boolean;
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
};

/** Dataset-wide crop and box totals of the summary overview. */
export type SummaryTotals = {
  crops: number;
  annotated: number;
  empty: number;
  unlabeled: number;
  boxes: number;
};

/** One summary subject row: one source page's crop counts and its labeled
 *  box counts per class (`class_counts` parallels the overview's `classes`). */
export type SummarySubject = SummaryTotals & {
  id: string;
  class_counts: number[];
};

/** Result of GET /api/summary/overview: the dataset's aggregate labeling
 *  state. `class_counts` holds one box count per class (0 for never-labeled
 *  classes); `label_errors` are malformed label files and `orphan_labels`
 *  are label files mirroring no crop image — both excluded from the counts. */
export type SummaryOverview = {
  classes: string[];
  totals: SummaryTotals;
  class_counts: number[];
  subjects: SummarySubject[];
  label_errors: LabelError[];
  orphan_labels: string[];
};
