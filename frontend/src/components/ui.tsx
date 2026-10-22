import { useMemo, useState, type ReactNode } from "react";
import type { Mode } from "../api";

export function DataBadge({ kind }: { kind: "observed" | "prediction" | "simulated" }) {
  const label = { observed: "● OBSERVED", prediction: "◇ MODEL PREDICTION", simulated: "▲ SIMULATED" }[kind];
  return <span className={`badge ${kind}`} title={{
    observed: "Market data from a real provider",
    prediction: "Model estimate - uncertain, not a guarantee",
    simulated: "Simulated / backtested / synthetic - not real outcomes",
  }[kind]}>{label}</span>;
}

export function ModeBadge({ mode }: { mode: Mode | string }) {
  return String(mode).toLowerCase() === "synthetic" ? <DataBadge kind="simulated" /> : <DataBadge kind="observed" />;
}

export function SimBanner({ mode }: { mode: Mode }) {
  if (mode !== "synthetic") return null;
  return (
    <div className="banner sim" role="alert">
      <strong>▲ SYNTHETIC DATA MODE.</strong> Every price, forecast and backtest on this screen comes from a simulated market generated
      for testing the platform. None of it describes real securities.
    </div>
  );
}

export function ErrorBox({ error, hint }: { error: unknown; hint?: ReactNode }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return <div className="banner err"><strong>Unavailable:</strong> {msg}{hint && <div style={{ marginTop: 6 }}>{hint}</div>}</div>;
}

export function Stat({ label, value, sub, cls }: { label: string; value: ReactNode; sub?: ReactNode; cls?: string }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className={`value ${cls || ""}`}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Action({ a }: { a: string }) {
  return <span className={`action ${a}`}>{a === "INSUFFICIENT_EVIDENCE" ? "INSUFFICIENT EVIDENCE" : a}</span>;
}

export type Col<T> = { key: string; label: string; render?: (r: T) => ReactNode; sort?: (r: T) => number | string; left?: boolean };

export function Table<T extends Record<string, any>>({ rows, cols, onRow, initialSort, maxHeight }: {
  rows: T[]; cols: Col<T>[]; onRow?: (r: T) => void; initialSort?: string; maxHeight?: number;
}) {
  const [sortKey, setSortKey] = useState<string | undefined>(initialSort);
  const [asc, setAsc] = useState(false);
  const sorted = useMemo(() => {
    if (!sortKey) return rows;
    const c = cols.find((x) => x.key === sortKey);
    const f = c?.sort || ((r: T) => r[sortKey]);
    return [...rows].sort((a, b) => {
      const va = f(a), vb = f(b);
      if (va == null) return 1;
      if (vb == null) return -1;
      return (va < vb ? -1 : va > vb ? 1 : 0) * (asc ? 1 : -1);
    });
  }, [rows, cols, sortKey, asc]);
  return (
    <div className="table-wrap" style={maxHeight ? { maxHeight } : undefined}>
      <table>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c.key} className={c.left ? "l" : ""} onClick={() => { if (sortKey === c.key) setAsc(!asc); else { setSortKey(c.key); setAsc(false); } }}>
                {c.label}{sortKey === c.key ? (asc ? " ▲" : " ▼") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={i} className={onRow ? "clickable" : ""} onClick={() => onRow?.(r)}>
              {cols.map((c) => <td key={c.key} className={c.left ? "l" : ""}>{c.render ? c.render(r) : String(r[c.key] ?? "–")}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      {!rows.length && <div className="muted" style={{ padding: 10 }}>No rows.</div>}
    </div>
  );
}

export function Tabs({ tabs, value, onChange }: { tabs: string[]; value: string; onChange: (t: string) => void }) {
  return <div className="tabs">{tabs.map((t) => <button key={t} className={t === value ? "active" : ""} onClick={() => onChange(t)}>{t}</button>)}</div>;
}

export function Progress({ job }: { job: any }) {
  if (!job) return null;
  return (
    <div className="row" style={{ fontSize: 12 }}>
      <div className="progress"><div style={{ width: `${Math.round((job.progress || 0) * 100)}%` }} /></div>
      <span className="muted">{job.status} {job.message}</span>
    </div>
  );
}
