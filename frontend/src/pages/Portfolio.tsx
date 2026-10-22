import { useEffect, useState } from "react";
import { api, money, pct, signCls } from "../api";
import type { Ctx } from "../App";
import { Plot, cssVar, seriesColor } from "../components/Plot";
import { DataBadge, ErrorBox, ModeBadge, SimBanner, Stat, Table } from "../components/ui";

export default function PortfolioPage({ ctx }: { ctx: Ctx }) {
  const [accounts, setAccounts] = useState<any[]>([]);
  const [form, setForm] = useState({ profile: "balanced", method: "mean_variance", account_id: "", capital: "", years: "", goal: "", monthly: "0", holdings: "" });
  const [p, setP] = useState<any>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [exec, setExec] = useState<any>(null);
  useEffect(() => { api.get(`/api/paper/accounts?mode=${ctx.mode}`).then(setAccounts).catch(() => setAccounts([])); setP(null); }, [ctx.mode]);
  const set = (k: string) => (e: any) => setForm({ ...form, [k]: e.target.value });

  const propose = async () => {
    setErr(null); setBusy(true); setExec(null);
    try {
      const holdings: Record<string, number> = {};
      form.holdings.split(/[\n,]+/).map((x) => x.trim()).filter(Boolean).forEach((x) => { const [s, v] = x.split(/[\s:=]+/); if (s && v) holdings[s.toUpperCase()] = +v; });
      setP(await api.post("/api/portfolio/propose", {
        mode: ctx.mode, profile: form.profile, method: form.method, account_id: form.account_id || null,
        capital: form.account_id ? null : form.capital ? +form.capital : null, holdings: form.account_id ? null : holdings,
        years: form.years ? +form.years : null, goal: form.goal ? +form.goal : null, monthly_contribution: +form.monthly || 0,
      }));
    } catch (e) { setErr(e); } finally { setBusy(false); }
  };
  const execute = async () => {
    if (!p || !form.account_id) return;
    if (!window.confirm(`Execute ${p.trades.length} SIMULATED trades in paper account ${form.account_id}? No real money is involved.`)) return;
    try { setExec(await api.post(`/api/paper/accounts/${form.account_id}/rebalance`, { mode: ctx.mode, trades: p.trades, confirm: true })); } catch (e) { setErr(e); }
  };

  const w = p ? Object.entries(p.weights as Record<string, number>).sort((a, b) => b[1] - a[1]) : [];
  return (
    <div>
      <div className="page-head"><h1>Portfolio Command Center</h1><ModeBadge mode={ctx.mode} /></div>
      <SimBanner mode={ctx.mode} />
      <div className="card">
        <h3>Proposal inputs</h3>
        <div className="row" style={{ alignItems: "end" }}>
          <label className="f">Profile<select value={form.profile} onChange={set("profile")}>{(ctx.health?.profiles || ["conservative", "balanced", "aggressive", "experimental"]).map((x: string) => <option key={x}>{x}</option>)}</select></label>
          <label className="f">Method<select value={form.method} onChange={set("method")}>{["mean_variance", "robust", "black_litterman", "min_cvar", "risk_parity", "hrp"].map((x) => <option key={x}>{x}</option>)}</select></label>
          <label className="f">Paper account<select value={form.account_id} onChange={set("account_id")}><option value="">— none (enter capital) —</option>{accounts.map((a) => <option key={a.account_id} value={a.account_id}>{a.name} ({a.account_id})</option>)}</select></label>
          {!form.account_id && <label className="f">Capital (required)<input type="number" value={form.capital} onChange={set("capital")} placeholder="e.g. 100000" /></label>}
          <label className="f">Goal amount (optional)<input type="number" value={form.goal} onChange={set("goal")} /></label>
          <label className="f">Years (for projection)<input type="number" value={form.years} onChange={set("years")} /></label>
          <label className="f">Monthly contribution<input type="number" value={form.monthly} onChange={set("monthly")} /></label>
          <button className="btn primary" disabled={busy} onClick={propose}>{busy ? "Optimizing…" : "Propose allocation"}</button>
        </div>
        {!form.account_id && <label className="f" style={{ marginTop: 8 }}>Existing holdings (symbol amount, one per line - optional)<textarea value={form.holdings} onChange={set("holdings")} placeholder={"SPY 40000\nAAPL 5000"} /></label>}
        <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>BigBoss never assumes your capital, goals, or holdings - only what you enter is used.</div>
      </div>
      <ErrorBox error={err} />
      {p && <>
        {p.status === "REJECTED_BY_RISK_GATE" && <div className="banner err" style={{ marginTop: 14 }}><strong>✖ Rejected by the independent risk gate:</strong> {p.risk_check.violations.join("; ")}</div>}
        {p.notice && <div className="banner warn" style={{ marginTop: 14 }}>{p.notice}</div>}
        {p.notes?.map((n: string) => <div key={n} className="banner warn" style={{ marginTop: 14 }}>{n}</div>)}
        <div className="grid g4" style={{ marginTop: 14 }}>
          <Stat label="Expected active return (horizon)" value={pct(p.expected_active_return_h)} sub={`${p.horizon}d, vs ${p.benchmark} · model ${p.model_version}`} />
          <Stat label="Expected tracking error (horizon)" value={pct(p.expected_tracking_error_h)} />
          <Stat label="Ex-ante volatility (ann.)" value={pct(p.risk_check.ex_ante_ann_vol, 1)} sub={`limit ${pct(p.profile_params.max_vol, 0)}`} />
          <Stat label="Risk gate" value={p.risk_check.passed ? "✓ PASSED" : "✖ FAILED"} cls={p.risk_check.passed ? "pos" : "neg"} />
          <Stat label="1-day VaR / ES (95%, hist.)" value={`${pct(p.var_es_1d.var)} / ${pct(p.var_es_1d.es)}`} />
          <Stat label={`${p.horizon}-day VaR / ES (95%)`} value={`${pct(p.var_es_h.var)} / ${pct(p.var_es_h.es)}`} />
          <Stat label="Estimated rebalance cost" value={money(p.total_est_cost)} sub={`${p.trades.length} trades`} />
          <Stat label="Cash" value={pct(p.cash_weight, 1)} />
        </div>
        <div className="grid g2" style={{ marginTop: 14 }}>
          <div className="card"><h3>Target weights <DataBadge kind="prediction" /></h3>
            <Plot theme={ctx.theme} height={Math.max(220, w.length * 26)} data={[{ type: "bar", orientation: "h", y: w.map((x) => x[0]), x: w.map((x) => x[1]), marker: { color: seriesColor(0) },
              text: w.map((x) => pct(x[1], 1)), textposition: "outside", cliponaxis: false, hovertemplate: "%{y}: %{x:.2%}<extra></extra>" }]}
              layout={{ yaxis: { autorange: "reversed" }, xaxis: { tickformat: ".0%" }, margin: { l: 90, r: 50, t: 6, b: 30 }, hovermode: "closest", showlegend: false }} /></div>
          <div className="card"><h3>Risk contributions (share of variance)</h3>
            <Plot theme={ctx.theme} height={Math.max(220, w.length * 26)} data={[{ type: "bar", orientation: "h", y: Object.keys(p.risk_contributions), x: Object.values(p.risk_contributions), marker: { color: seriesColor(1) }, hovertemplate: "%{y}: %{x:.1%}<extra></extra>" }]}
              layout={{ yaxis: { autorange: "reversed" }, xaxis: { tickformat: ".0%" }, margin: { l: 90, r: 20, t: 6, b: 30 }, hovermode: "closest", showlegend: false }} /></div>
        </div>
        <div className="grid g2" style={{ marginTop: 14 }}>
          <div className="card"><h3>Rebalancing trades (estimated costs)</h3>
            <Table rows={p.trades} cols={[{ key: "symbol", label: "Symbol", left: true }, { key: "side", label: "Side", left: true }, { key: "notional", label: "Notional", render: (t) => money(t.notional) },
              { key: "current_weight", label: "From", render: (t) => pct(t.current_weight, 1) }, { key: "target_weight", label: "To", render: (t) => pct(t.target_weight, 1) },
              { key: "est_cost_bps", label: "Cost bps", render: (t) => t.est_cost_bps.toFixed(1) }, { key: "est_cost", label: "Cost", render: (t) => money(t.est_cost) }]} />
            {form.account_id && <button className="btn primary" style={{ marginTop: 8 }} disabled={!p.risk_check.passed} onClick={execute}>Execute in paper account (simulated)…</button>}
            {exec && <div style={{ marginTop: 8 }}>{exec.results.map((r: any, i: number) => <div key={i} className={r.status === "FILLED" ? "pos" : "neg"}>{r.symbol} {r.side} {r.status} {r.reason || ""}</div>)}</div>}
          </div>
          <div className="card"><h3>Historical stress scenarios <DataBadge kind="simulated" /></h3>
            <Table rows={p.stress} cols={[{ key: "scenario", label: "Scenario", left: true }, { key: "start", label: "From" }, { key: "end", label: "To" },
              { key: "portfolio_return", label: "Return", render: (s) => <span className={signCls(s.portfolio_return)}>{pct(s.portfolio_return, 1)}</span> },
              { key: "weight_covered", label: "Covered (of invested)", render: (s) => pct(s.weight_covered, 0) }]} />
            <div className="muted" style={{ fontSize: 12 }}>Current weights applied to each asset's realized return in the window; assets without history in a window are not covered.</div>
          </div>
        </div>
        {p.projection && <div className="card" style={{ marginTop: 14 }}>
          <h3>Goal projection - {p.projection.years}y, {p.projection.n_paths.toLocaleString()} paths <DataBadge kind="simulated" /></h3>
          <div className="grid g4" style={{ marginBottom: 8 }}>
            {p.projection.prob_reach_goal != null && <Stat label={`P(reach ${money(p.projection.goal)})`} value={pct(p.projection.prob_reach_goal, 0)} />}
            <Stat label="Median terminal wealth" value={money(p.projection.terminal_percentiles.p50)} sub={`5th–95th: ${money(p.projection.terminal_percentiles.p5)} – ${money(p.projection.terminal_percentiles.p95)}`} />
            <Stat label="P(below contributions)" value={pct(p.projection.prob_loss, 0)} />
            <Stat label="Worst-5% max drawdown" value={pct(p.projection.max_drawdown_percentiles.p95_worst, 0)} />
          </div>
          <Plot theme={ctx.theme} data={[
            { x: p.projection.fan.day, y: p.projection.fan.p95, type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
            { x: p.projection.fan.day, y: p.projection.fan.p5, type: "scatter", mode: "lines", fill: "tonexty", fillcolor: cssVar("--series-1") + "22", line: { width: 0 }, name: "5th–95th percentile" },
            { x: p.projection.fan.day, y: p.projection.fan.p75, type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
            { x: p.projection.fan.day, y: p.projection.fan.p25, type: "scatter", mode: "lines", fill: "tonexty", fillcolor: cssVar("--series-1") + "44", line: { width: 0 }, name: "25th–75th percentile" },
            { x: p.projection.fan.day, y: p.projection.fan.p50, type: "scatter", mode: "lines", line: { width: 2, color: seriesColor(0) }, name: "median" },
          ]} layout={{ xaxis: { title: { text: "trading days" } }, yaxis: { title: { text: "wealth" } } }} />
          <div className="muted" style={{ fontSize: 12 }}>{p.projection.method}. {p.projection.caveat}</div>
        </div>}
      </>}
    </div>
  );
}
