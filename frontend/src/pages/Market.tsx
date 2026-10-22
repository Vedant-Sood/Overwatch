import { useEffect, useMemo, useState } from "react";
import { api, pct } from "../api";
import type { Ctx } from "../App";
import { Plot, cssVar, seriesColor } from "../components/Plot";
import { ErrorBox, ModeBadge, SimBanner, Stat, Table } from "../components/ui";

export default function MarketPage({ ctx }: { ctx: Ctx }) {
  const [d, setD] = useState<any>(null);
  const [err, setErr] = useState<unknown>(null);
  useEffect(() => { setD(null); setErr(null); api.get(`/api/market/overview?mode=${ctx.mode}`).then(setD).catch(setErr); }, [ctx.mode]);

  const sectors = useMemo(() => {
    if (!d) return [];
    const g: Record<string, number[]> = {};
    d.assets.filter((a: any) => a.ret_20d != null && a.sector && a.sector !== "Index").forEach((a: any) => (g[a.sector] ||= []).push(a.ret_20d));
    return Object.entries(g).map(([s, v]) => ({ sector: s, ret: v.reduce((x, y) => x + y, 0) / v.length, n: v.length })).sort((a, b) => b.ret - a.ret);
  }, [d]);

  if (err) return <div><h1>Market Intelligence</h1><ErrorBox error={err} /></div>;
  if (!d) return <div className="muted">Loading…</div>;
  const r = d.regime || {};
  return (
    <div>
      <div className="page-head"><h1>Market Intelligence</h1><span className="row"><ModeBadge mode={d.data_mode} /><span className="muted">as of {d.as_of}</span></span></div>
      <SimBanner mode={ctx.mode} />
      <div className="grid g4">
        <Stat label={`Regime (${d.benchmark})`} value={r.label?.replace("_", " · ") ?? "–"} sub="trend vs 200d MA × vol vs history" />
        <Stat label="Benchmark 20d vol (ann.)" value={pct(r.vol_20d_ann, 1)} sub={`percentile ${pct(r.vol_percentile, 0)}`} />
        <Stat label="P(high-vol regime)" value={pct(r.markov_p_high_vol, 0)} sub="Markov switching, filtered" />
        <Stat label="Drawdown from peak" value={pct(r.drawdown, 1)} cls={r.drawdown < -0.1 ? "neg" : ""} />
        <Stat label="Avg pairwise correlation" value={r.avg_corr_60d != null ? r.avg_corr_60d.toFixed(2) : "–"}
          sub={r.correlation_breakdown ? "⚠ top decile - diversification weak" : `percentile ${pct(r.avg_corr_percentile, 0)}`} />
        <Stat label="Cross-sectional regime" value={r.momentum_or_reversal ?? "–"} sub={r.xs_momentum_ic_120d != null ? `20d IC ${r.xs_momentum_ic_120d.toFixed(3)}` : ""} />
        <Stat label="Liquidity stress" value={r.liquidity_stress ? "⚠ elevated" : r.liquidity_stress_percentile != null ? "normal" : "–"} sub={r.liquidity_stress_percentile != null ? `percentile ${pct(r.liquidity_stress_percentile, 0)}` : ""} />
        <Stat label="Vol change-points (1y, CUSUM)" value={(r.vol_changepoints_recent || []).length} sub={(r.vol_changepoints_recent || []).slice(-2).join(", ")} />
      </div>
      <div className="grid g2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Benchmark volatility & regime <ModeBadge mode={d.data_mode} /></h3>
          {r.history && <Plot theme={ctx.theme} height={220} data={[
            { x: r.history.dates, y: r.history.vol20, type: "scatter", mode: "lines", name: "20d realized vol", line: { width: 2, color: seriesColor(0) }, hovertemplate: "%{y:.1%}<extra></extra>" },
          ]} layout={{ yaxis: { tickformat: ".0%", title: { text: "20d realized vol" } }, showlegend: false }} />}
          {r.markov_history && <Plot theme={ctx.theme} height={160} data={[
            { x: r.markov_history.dates, y: r.markov_history.p_high_vol, type: "scatter", mode: "lines", fill: "tozeroy", name: "P(high-vol regime)",
              line: { width: 2, color: seriesColor(6) }, fillcolor: cssVar("--series-7") + "22", hovertemplate: "%{y:.0%}<extra></extra>" },
          ]} layout={{ yaxis: { tickformat: ".0%", range: [0, 1], title: { text: "P(high-vol)" } }, showlegend: false, margin: { l: 56, r: 16, t: 6, b: 30 } }} />}
        </div>
        <div className="card">
          <h3>Sector performance, 20 days (equal-weight average of covered assets) <ModeBadge mode={d.data_mode} /></h3>
          <Plot theme={ctx.theme} height={Math.max(220, sectors.length * 34 + 60)} data={[{ type: "bar", orientation: "h", y: sectors.map((s) => s.sector), x: sectors.map((s) => s.ret),
            marker: { color: sectors.map((s) => (s.ret >= 0 ? cssVar("--series-1") : cssVar("--series-8"))) },
            text: sectors.map((s) => `${(s.ret * 100).toFixed(1)}%`), textposition: sectors.map((s) => (s.ret >= 0 ? "outside" : "inside")), insidetextanchor: "start", cliponaxis: false, hovertemplate: "%{y}: %{x:.2%}<extra></extra>" }]}
            layout={{ margin: { l: 160, r: 60, t: 10, b: 30 }, xaxis: { tickformat: ".0%" }, yaxis: { autorange: "reversed" }, hovermode: "closest", showlegend: false }} />
        </div>
      </div>
      {d.macro && Object.keys(d.macro).length > 0 && (
        <div className="grid g4" style={{ marginTop: 14 }}>
          {Object.entries(d.macro).filter(([, v]: any) => v && v.value !== undefined).map(([k, v]: any) =>
            <Stat key={k} label={k.replaceAll("_", " ")} value={v.value.toFixed(2)} sub={`${v.date} · 10y pct ${pct(v.percentile_10y, 0)} · FRED`} />)}
        </div>
      )}
      {d.macro_series && Object.keys(d.macro_series).length > 0 && (
        <div className="card" style={{ marginTop: 14 }}><h3>Rates & credit (FRED) <ModeBadge mode="observed" /></h3>
          <Plot theme={ctx.theme} data={Object.entries(d.macro_series).filter(([k]) => k !== "VIXCLS").map(([k, v]: any, i) =>
            ({ x: v.dates, y: v.values, type: "scatter", mode: "lines", name: k, line: { width: 2, color: seriesColor(i) } }))}
            layout={{ yaxis: { title: { text: "percent" } } }} /></div>
      )}
      <div className="card" style={{ marginTop: 14 }}>
        <h3>Global markets & assets <ModeBadge mode={d.data_mode} /></h3>
        <Table rows={d.assets} initialSort="ret_20d" onRow={(row) => ctx.go("research", row.symbol)} cols={[
          { key: "symbol", label: "Symbol", left: true }, { key: "sector", label: "Sector / class", left: true, render: (a) => a.sector || a.asset_class },
          { key: "last_date", label: "Last date" }, { key: "last", label: "Last", render: (a) => a.last?.toFixed(2) },
          ...["ret_1d", "ret_5d", "ret_20d", "ret_ytd", "ret_252d"].map((k) => ({ key: k, label: k.replace("ret_", ""), render: (a: any) => <span className={a[k] > 0 ? "pos" : a[k] < 0 ? "neg" : ""}>{pct(a[k], 1)}</span> })),
          { key: "vol_20d", label: "Vol 20d", render: (a) => pct(a.vol_20d, 1) },
        ]} />
      </div>
    </div>
  );
}
