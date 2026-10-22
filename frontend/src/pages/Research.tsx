import { useEffect, useState } from "react";
import { api, num, pct, signCls } from "../api";
import type { Ctx } from "../App";
import { Plot, cssVar, seriesColor } from "../components/Plot";
import { Action, DataBadge, ErrorBox, ModeBadge, SimBanner, Stat, Table } from "../components/ui";
import { Thesis } from "./Scanner";

export default function ResearchPage({ ctx }: { ctx: Ctx }) {
  const [syms, setSyms] = useState<any[]>([]);
  const [sym, setSym] = useState<string>(ctx.param || "");
  const [d, setD] = useState<any>(null);
  const [err, setErr] = useState<unknown>(null);
  useEffect(() => { api.get(`/api/symbols?mode=${ctx.mode}`).then(setSyms).catch(() => setSyms([])); }, [ctx.mode]);
  useEffect(() => { if (ctx.param) setSym(ctx.param); }, [ctx.param]);
  useEffect(() => {
    if (!sym) return;
    setD(null); setErr(null);
    api.get(`/api/research/${encodeURIComponent(sym)}?mode=${ctx.mode}`).then(setD).catch(setErr);
  }, [sym, ctx.mode]);

  const fc = d?.forecasts?.[String(ctx.health?.primary_horizon || 20)];
  const lastDate = d?.prices?.dates?.at(-1);
  const lastPx = d?.prices?.adj_close?.at(-1);

  // forecast cone: future business days from last date, using the primary-horizon calibrated quantiles of total return
  let cone: any[] = [];
  if (fc && !fc.error && lastDate) {
    const h = Number(ctx.health?.primary_horizon || 20);
    const days: string[] = [];
    const t = new Date(lastDate);
    while (days.length < h) { t.setDate(t.getDate() + 1); if (t.getDay() % 6 !== 0) days.push(t.toISOString().slice(0, 10)); }
    const bench = d.decision?.expected_gross != null ? d.decision.expected_gross - d.decision.expected_excess : 0;
    const path = (q: number) => [lastPx, ...days.map((_, i) => lastPx * (1 + (bench + q) * ((i + 1) / h)))];
    const x = [lastDate, ...days];
    cone = [
      { x, y: path(fc.q90), type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x, y: path(fc.q10), type: "scatter", mode: "lines", fill: "tonexty", fillcolor: cssVar("--prediction") + "33", line: { width: 0 }, name: "80% forecast interval", hoverinfo: "skip" },
      { x, y: path(fc.q50), type: "scatter", mode: "lines", line: { width: 2, dash: "dash", color: cssVar("--prediction") }, name: "median forecast (model)" },
    ];
  }

  return (
    <div>
      <div className="page-head">
        <h1>Stock Research {d && <span className="secondary">· {d.symbol} {d.info?.name ? `· ${d.info.name}` : ""}</span>}</h1>
        <label className="f">Symbol
          <select value={sym} onChange={(e) => ctx.go("research", e.target.value)}>
            <option value="">choose…</option>
            {syms.map((s) => <option key={s.symbol} value={s.symbol}>{s.symbol}{s.name ? ` – ${s.name}` : ""}</option>)}
          </select>
        </label>
      </div>
      <SimBanner mode={ctx.mode} />
      <ErrorBox error={err} />
      {!sym && <div className="muted">Choose a symbol.</div>}
      {d && <>
        <div className="grid g4">
          <Stat label="Last close" value={num(d.prices.close.at(-1))} sub={`${d.last_date} · ${d.source} · ${d.adjustment}`} />
          <Stat label="Sector / class" value={d.info?.sector || "–"} sub={`${d.info?.asset_class || ""} · ${d.info?.market || ""} · ${d.info?.currency || ""}`} />
          <Stat label="Benchmark" value={d.benchmark} />
          <Stat label="Decision (committee input)" value={d.decision ? <Action a={d.decision.action} /> : "–"} sub={d.decision ? `size ${pct(d.decision.suggested_weight, 1)}` : d.decision_error} />
        </div>
        <div className="card" style={{ marginTop: 14 }}>
          <h3>Price (adjusted) <ModeBadge mode={ctx.mode} /> with forecast cone <DataBadge kind="prediction" /></h3>
          <Plot theme={ctx.theme} height={380} data={[
            { x: d.prices.dates, y: d.prices.adj_close, type: "scatter", mode: "lines", name: `${d.symbol} adj. close`, line: { width: 2, color: seriesColor(0) } },
            ...cone,
          ]} layout={{ yaxis: { title: { text: "price" }, autorange: true }, xaxis: { range: [d.prices.dates.at(-Math.min(504, d.prices.dates.length)), cone.length ? cone[0].x.at(-1) : lastDate] } }} />
        </div>
        <div className="grid g2" style={{ marginTop: 14 }}>
          <div className="card">
            <h3>Forecasts by horizon <DataBadge kind="prediction" /></h3>
            <Table rows={Object.entries(d.forecasts).map(([h, f]: any) => ({ h: +h, ...f }))} cols={[
              { key: "h", label: "Horizon (d)", left: true },
              { key: "mean", label: "E[excess]", render: (f) => f.error ? <span className="neg">{f.error}</span> : <span className={signCls(f.mean)}>{pct(f.mean)}</span> },
              { key: "q10", label: "80% interval", render: (f) => `${pct(f.q10, 1)} … ${pct(f.q90, 1)}` },
              { key: "p_pos", label: "P(outperform)", render: (f) => pct(f.p_pos, 0) },
              { key: "skill_t", label: "Model skill t", render: (f) => f.skill_t?.toFixed(2) },
            ]} />
            {fc?.model_means && <>
              <h3 style={{ marginTop: 12 }}>Model disagreement ({ctx.health?.primary_horizon}d)</h3>
              <Plot theme={ctx.theme} height={200} data={[{ type: "bar", x: Object.keys(fc.model_means), y: Object.values(fc.model_means), marker: { color: seriesColor(6) }, hovertemplate: "%{x}: %{y:.2%}<extra></extra>" }]}
                layout={{ yaxis: { tickformat: ".1%" }, hovermode: "closest", margin: { l: 56, r: 10, t: 6, b: 60 } }} />
            </>}
            {fc?.top_drivers && <div className="muted" style={{ fontSize: 12 }}>Largest LightGBM feature contributions: {fc.top_drivers.map((x: any) => `${x[0]} ${x[1] > 0 ? "+" : ""}${x[1].toFixed(4)}`).join(", ")}</div>}
          </div>
          <div className="card">
            <h3>Fundamental valuation <ModeBadge mode={ctx.mode} /> · scenarios <DataBadge kind="prediction" /></h3>
            {!d.valuation?.available ? <div className="muted">{d.valuation?.reason}</div> : <>
              <div className="grid g3">
                <Stat label="P/E" value={num(d.valuation.pe, 1)} /><Stat label="FCF yield" value={pct(d.valuation.fcf_yield)} />
                <Stat label="Revenue growth YoY" value={pct(d.valuation.revenue_growth_yoy, 1)} /><Stat label="ROIC" value={pct(d.valuation.roic, 1)} />
                <Stat label="Operating margin" value={pct(d.valuation.operating_margin, 1)} /><Stat label="Debt / equity" value={num(d.valuation.debt_to_equity, 2)} />
              </div>
              <Table rows={Object.entries(d.valuation.scenarios).map(([k, v]: any) => ({ k, ...v }))} cols={[
                { key: "k", label: "DCF scenario", left: true }, { key: "growth", label: "Growth", render: (r) => pct(r.growth, 1) },
                { key: "discount_rate", label: "Discount", render: (r) => pct(r.discount_rate, 1) },
                { key: "value_per_share", label: "Value/share", render: (r) => num(r.value_per_share) },
                { key: "implied_return", label: "Implied return", render: (r) => r.note || <span className={signCls(r.implied_return)}>{pct(r.implied_return, 0)}</span> },
              ]} />
              <div className="muted" style={{ fontSize: 12 }}>Quality score {num(d.valuation.quality_score?.score, 0)}/100 · based on SEC filing available {d.valuation.filing_based_on} · period end {d.valuation.latest_period_end}</div>
            </>}
          </div>
        </div>
        {d.decision && <div className="card" style={{ marginTop: 14 }}><h3>Investment thesis <DataBadge kind="prediction" /></h3><Thesis t={d.decision.thesis} /></div>}
        {d.features && <div className="card" style={{ marginTop: 14 }}><h3>Latest features (point-in-time) <ModeBadge mode={ctx.mode} /></h3>
          <div className="grid g3">{Object.entries(d.features).map(([g, f]: any) => <div key={g}><strong>{g}</strong>
            <table><tbody>{Object.entries(f).map(([k, v]: any) => <tr key={k}><td>{k}</td><td>{v == null ? "–" : Math.abs(v) < 10 ? v.toFixed(4) : num(v, 1)}</td></tr>)}</tbody></table></div>)}</div>
        </div>}
        <div className="card" style={{ marginTop: 14 }}>
          <h3>Recent news headlines <ModeBadge mode={ctx.mode} /> <span className="muted" style={{ fontWeight: 400 }}>
            {d.news?.items?.length ? `${d.news.n_30d} in 30 days · mean headline sentiment ${d.news.mean_sentiment_30d != null ? d.news.mean_sentiment_30d.toFixed(2) : "–"} (lexicon, weak signal, not a model input)` : ""}</span></h3>
          {!d.news?.items?.length ? <div className="muted">{d.news?.note || "No headlines."}</div> :
            <Table rows={d.news.items} maxHeight={300} cols={[
              { key: "published_at", label: "Published (ET)", left: true }, { key: "source_domain", label: "Source", left: true },
              { key: "credibility", label: "Credibility", left: true },
              { key: "sentiment", label: "Sentiment", render: (n) => <span className={signCls(n.sentiment)}>{n.sentiment?.toFixed(2)}</span> },
              { key: "title", label: "Headline", left: true, render: (n) => <a href={n.url} target="_blank" rel="noreferrer" style={{ whiteSpace: "normal" }}>{n.title}</a> },
            ]} />}
        </div>
        <div className="card" style={{ marginTop: 14 }}>
          <h3>Filings & events (SEC EDGAR) <ModeBadge mode={ctx.mode} /></h3>
          {d.events?.note && <div className="muted">{d.events.note}</div>}
          <Table rows={d.events?.events || []} cols={[
            { key: "accepted_at", label: "Accepted (ET)", left: true }, { key: "form", label: "Form", left: true }, { key: "event_type", label: "Type", left: true },
            { key: "importance", label: "Importance", render: (e) => num(e.importance, 2) },
            { key: "mean_excess", label: "Hist. 5d excess", render: (e) => e.reliable ? `${pct(e.mean_excess)} (n=${e.n})` : "too few" },
            { key: "stale", label: "Fresh", render: (e) => e.stale ? "stale" : "✓" },
            { key: "url", label: "Source", left: true, render: (e) => <a href={e.url} target="_blank" rel="noreferrer">filing</a> },
          ]} maxHeight={300} />
        </div>
      </>}
    </div>
  );
}
