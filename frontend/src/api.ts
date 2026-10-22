export type Mode = "observed" | "synthetic";

export class ApiError extends Error {
  status: number;
  code?: string;
  constructor(status: number, message: string, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  let data: any = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!r.ok) {
    const d = data?.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x: any) => x.msg).join("; ") : r.statusText;
    throw new ApiError(r.status, msg, data?.error);
  }
  return data as T;
}

export const api = {
  get: <T = any>(p: string) => req<T>("GET", p),
  post: <T = any>(p: string, b?: unknown) => req<T>("POST", p, b ?? {}),
};

export async function waitJob(id: string, onProgress?: (j: any) => void): Promise<any> {
  for (;;) {
    const j = await api.get(`/api/jobs/${id}`);
    onProgress?.(j);
    if (j.status === "succeeded") return j;
    if (j.status === "failed" || j.status === "interrupted") throw new Error(j.error || j.status);
    await new Promise((r) => setTimeout(r, 1500));
  }
}

export const pct = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined || Number.isNaN(x) ? "–" : `${(x * 100).toFixed(d)}%`;
export const num = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined || Number.isNaN(x) ? "–" : x.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
export const money = (x: number | null | undefined) =>
  x === null || x === undefined ? "–" : x.toLocaleString(undefined, { maximumFractionDigits: 0 });
export const signCls = (x: number | null | undefined) => (x == null ? "" : x > 0 ? "pos" : x < 0 ? "neg" : "");
