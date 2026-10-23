import { useEffect, useState } from "react";
import { api, pct } from "../api";
import type { Ctx } from "../App";
import { Plot, cssVar, seriesColor } from "../components/Plot";
import { DataBadge, ErrorBox, ModeBadge, SimBanner, Stat, Table } from "../components/ui";

export default function ModelsPage({ ctx }: { ctx: Ctx }) {
  const [d, setD] = useState<any>(null);
  const [err, setErr] = useState<unknown>(null);
  const [sel, setSel] = useState<any>(null);
  const [msg, setMsg] = useState<any>(null);
  const load = () => api.get(`/api/models?mode=${ctx.mode}`).then(setD).catch(setErr);
  useEffect(() => { setD(null); setSel(null); load(); }, [ctx.mode]);

  const promote = async (v: string) => { try { setMsg(await api.post(`/api/models/${v}/promote`, { mode: ctx.mode })); load(); ctx.refreshHealth(); } catch (e) { setErr(e); } };
  const rollback = async (h: number) => { if (!window.confirm(`Roll back the champion for horizon ${h}?`)) return; try { setMsg(await api.post(`/api/models/rollback`, { mode: ctx.mode, horizon: h })); load(); ctx.refreshHealth(); } catch (e) { setErr(e); } };

  if (err) return <div><h1>Model Intelligence</h1><ErrorBox error={err} /></div>;
  if (!d) return <div className="muted">Loading…</div>;
  const lp = d.live_performance || {};
  return (
    <div>
      <div className="page-head"><h1>Model Intelligence</h1><ModeBadge mode={ctx.mode} /></div>
      <SimBanner mode={ctx.mode} />
      <div className="grid g4">
        <Stat label="Champions" value={Object.keys(d.champions).length ? Object.entries(d.champions).map(([h]) => `${h}d`).join(" · ") : "none"} sub={Object.values(d.champions).join(", ")} />
        <Stat label="Logged predictions" value={lp.n_logged ?? 0} sub={`${lp.n_resolved ?? 0} resolved`} />
        <Stat label="Live rank IC (post-deployment)" value={lp.rank_ic_mean != null ? lp.rank_ic_mean.toFixed(3) : "–"} sub={lp.note || ""} />
        <Stat label="Live 80% interval coverage" value={pct(lp.coverage80, 0)} sub={lp.brier != null ? `Brier ${lp.brier.toFixed(4)}` : ""} />
      </div>
      {msg && <pre className="mono" style={{ marginTop: 14 }}>{JSON.stringify(msg, null, 1).slice(0, 2500)}</pre>}
      <div className="card" style={{ marginTop: 14 }}>
        <h3>Model versions <span className="muted" style={{ fontWeight: 400 }}>click a row for gates & quality</span></h3>
        <Table rows={d.versions} onRow={setSel} cols={[
          { key: "version", label: "Version", left: true, render: (v) => <>{v.version}{Object.values(d.champions).includes(v.version) ? <strong className="pos"> ★ champion</strong> : null}</> },
          { key: "horizon", label: "Horizon" }, { key: "ensemble_skill_t", label: "OOS IC t", render: (v) => v.ensemble_skill_t?.toFixed(2) },
          { key: "n_eval_periods", label: "Eval periods" }, { key: "oos_start", label: "OOS from" }, { key: "oos_end", label: "OOS to" },
          { key: "gates", label: "Gates", render: (v) => v.gates?.passed ? <span className="pos">✓ pass</span> : <span className="neg">✖ fail</span>, sort: (v) => (v.gates?.passed ? 1 : 0) },
        ]} />
      </div>
      {sel && <div className="card" style={{ marginTop: 14 }}>
        <h3>{sel.version}
          {!Object.values(d.champions).includes(sel.version) && <button className="btn" onClick={() => promote(sel.version)}>Challenge & promote (gates enforced)</button>}
          {d.champions[String(sel.horizon)] && <button className="btn" onClick={() => rollback(sel.horizon)}>Rollback h={sel.horizon}</button>}
        </h3>
        <div className="grid g2">
          <div><strong>Promotion gates</strong>
            <Table rows={Object.entries(sel.gates.checks).map(([k, c]: any) => ({ k, ...c }))} cols={[{ key: "k", label: "Gate", left: true },
              { key: "pass", label: "Result", render: (c) => c.pass ? <span className="pos">✓ PASS</span> : <span className="neg">✖ FAIL</span> },
              { key: "value", label: "Value", render: (c) => typeof c.value === "number" ? c.value.toFixed(4) : String(c.value) },
              { key: "threshold", label: "Requirement", render: (c) => c.band ? `${c.band[0]}–${c.band[1]}` : c.climatology != null ? `≤ ${(c.climatology + 0.005).toFixed(4)}`
                : c.min_psr != null ? `net IR > ${c.threshold} and P(edge>0) ≥ ${c.min_psr} (now ${c.psr_excess?.toFixed(2)})` : c.threshold }]} /></div>
          <div><strong>Ensemble weights</strong><pre className="mono">{JSON.stringify(sel.ensemble_weights, null, 1)}</pre></div>
        </div>
        <Table rows={sel.forecast_quality} cols={[{ key: "model", label: "Model", left: true }, { key: "rank_ic_mean", label: "Rank IC", render: (r) => r.rank_ic_mean?.toFixed(4) ?? "–" },
          { key: "rank_ic_t", label: "IC t", render: (r) => r.rank_ic_t?.toFixed(2) }, { key: "interval80_coverage", label: "80% coverage", render: (r) => pct(r.interval80_coverage, 1) },
          { key: "brier", label: "Brier", render: (r) => r.brier?.toFixed(4) }, { key: "mae", label: "MAE", render: (r) => pct(r.mae, 2) }]} />
      </div>}
      <div className="grid g2" style={{ marginTop: 14 }}>
        <div className="card"><h3>Rolling 63-day OOS rank IC ({d.monitored_is_champion ? "champion" : "latest version, research only"}) <DataBadge kind="prediction" /></h3>
          {d.rolling_ic ? <Plot theme={ctx.theme} data={[{ x: d.rolling_ic.dates, y: d.rolling_ic.values, type: "scatter", mode: "lines", line: { width: 2, color: seriesColor(0) }, name: "rolling IC" }]}
            layout={{ showlegend: false, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: cssVar("--axis"), width: 1 } }] }} /> : <div className="muted">No trained model yet.</div>}
        </div>
        <div className="card"><h3>Feature drift (PSI vs training, last 126 days)</h3>
          <Table rows={d.drift || []} maxHeight={320} cols={[{ key: "feature", label: "Feature", left: true }, { key: "kind", label: "Kind", left: true },
            { key: "psi", label: "PSI", render: (r) => r.psi?.toFixed(3) ?? "–" },
            { key: "status", label: "Status", render: (r) => r.status === "alert" ? <span className="neg">✖ alert</span> : r.status === "warn" ? <span>⚠ warn</span> : <span className="pos">✓ ok</span> }]} />
          <div className="muted" style={{ fontSize: 12 }}>market_state drift usually reflects a regime change rather than a broken input.</div>
        </div>
      </div>
      <div className="card" style={{ marginTop: 14 }}><h3>Registry history</h3>
        <Table rows={[...d.history].reverse()} maxHeight={260} cols={[{ key: "ts", label: "Time", left: true }, { key: "event", label: "Event", left: true }, { key: "version", label: "Version", left: true }, { key: "horizon", label: "h" }]} /></div>
      {lp.realized_by_action && <div className="card" style={{ marginTop: 14 }}><h3>Historical decision quality (realized excess by action)</h3>
        <Table rows={lp.realized_by_action} cols={[{ key: "action", label: "Action", left: true }, { key: "mean", label: "Mean realized excess", render: (r) => pct(r.mean) }, { key: "count", label: "N" }]} /></div>}
    </div>
  );
}
