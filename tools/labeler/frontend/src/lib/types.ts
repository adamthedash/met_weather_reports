/** Shared types for the labeler UI. */

/** A bounding box in image pixels (the editor's working unit). */
export type Box = { cls: number; x: number; y: number; w: number; h: number };

/** A YOLO bounding box, normalised 0-1 (the wire/disk unit). */
export type BoxN = { cls: number; cx: number; cy: number; w: number; h: number };

/** One queue entry: a crop image and its labeling status. */
export type ImageEntry = { path: string; done: boolean; has_labels: boolean };

/** The annotate queue listing. */
export type Listing = { images: ImageEntry[]; total: number; done: number };

/** Navigation filter over the queue (prev/next predicate, not a view). */
export type FilterMode = "all" | "annotated" | "non-annotated";
