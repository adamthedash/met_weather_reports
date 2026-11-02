/** Typed fetch helpers for the labeler JSON API. */

/** Shape every API error response has (see tools/labeler/app.py). */
export type ApiError = { error: string };

/** Static description of a workflow stage (see tools/labeler/stages/base.py). */
export type StageSummary = { id: string; title: string; description: string };

/** GET `url` and parse the JSON body, throwing the API's error message on failure. */
export async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
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
