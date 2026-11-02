/** Shared types for the labeler UI. */

/** A bounding box in image pixels (the editor's working unit).
 *  `conf` is display-only and carried by review predictions. */
export type Box = {
  cls: number;
  x: number;
  y: number;
  w: number;
  h: number;
  conf?: number;
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
