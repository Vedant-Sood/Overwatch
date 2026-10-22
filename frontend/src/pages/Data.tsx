import { useEffect, useState } from "react";
import { api, waitJob } from "../api";
import type { Ctx } from "../App";
import { ErrorBox, ModeBadge, Progress, SimBanner, Table } from "../components/ui";

export default function DataPage({ ctx }: { ctx: Ctx }) {
  const [st, setSt] = useState<any>(null);
  const [err, setErr] = useState<unknown>(null);
  const [job, setJob] = useState<any>(null);
  const [groups, setGroups] = useState<string[]>(["cross_asset"]);
  const [withNews, setWithNews] = useState(false);
  const [symbols, setSymbols] = useState("");
  const [horizon, setHorizon] = useState(20);
  const load = () => api.get(`/api/data/status?mode=${ctx.mode}`).then(setSt).catch(setErr);
  useEffect(() => { setSt(null); load(); }, [ctx.mode]);

  const run = async (path: string, body: any) => {
    setErr(null);
    try {
      const { job_id } = await api.post(path, body);
      await waitJob(job_id, setJob);
      load();
      ctx.refreshHealth();
    } catch (e) { setErr(e); }
  };
  const allGroups = ["cross_asset", "us_etfs", "us_equities", "india_equities", "india_etfs", "indices"];

  return (
    <div>
      <div className="page-head"><h1>Data & Setup</h1><ModeBadge mode={ctx.mode} /></div>
      <SimBanner mode={ctx.mode} />
      <ErrorBox error={err} />
      <div className="grid g2">
        {ctx.mode === "observed" ? (
          <div className="card">
            <h3>1 · Ingest market data</h3>
            <p className="secondary" style={{ marginTop: 0 }}>Prices from Stooq / Yahoo / Alpha Vantage (in that order), macro from FRED & World Bank,
              fundamentals and filings from SEC EDGAR. <strong>cross_asset</strong> needs no API key: index, rates and FX proxies derived from FRED.
              Stock-level prices need a free key (MASSIVE_API_KEY, FMP_API_KEY or ALPHAVANTAGE_API_KEY) unless Stooq/Yahoo are reachable. Failures are logged; missing data is never filled with synthetic values.</p>
            <div className="row">{allGroups.map((g) => (
              <label key={g} className="row" style={{ gap: 4 }}>
                <input type="checkbox" checked={groups.includes(g)} onChange={(e) => setGroups(e.target.checked ? [...groups, g] : groups.filter((x) => x !== g))} />{g}
              </label>))}
            </div>
            <label className="f" style={{ marginTop: 8 }}>Or specific symbols (space separated)
              <input value={symbols} onChange={(e) => setSymbols(e.target.value)} placeholder="AAPL MSFT RELIANCE.NS" />
            </label>
            <div className="row" style={{ marginTop: 10 }}>
              <button className="btn primary" disabled={job?.status === "running"} onClick={() => run("/api/jobs/ingest", symbols.trim() ? { symbols: symbols.trim().split(/\s+/), news: withNews } : { groups, news: withNews })}>Run ingestion</button>
              <label className="row" style={{ gap: 4 }}><input type="checkbox" checked={withNews} onChange={(e) => setWithNews(e.target.checked)} />headlines (GDELT, ~5 s per symbol)</label>
            </div>
          </div>
        ) : (
          <div className="card">
            <h3>1 · Generate the synthetic test market</h3>
            <p className="secondary" style={{ marginTop: 0 }}>A labelled simulated market (regime-switching factor model with a weak planted signal) for exercising the
              full pipeline offline. It is stored separately and never mixed with observed data.</p>
            <button className="btn primary" onClick={() => run("/api/jobs/demo-data", {})}>Generate synthetic market</button>
          </div>
        )}
        <div className="card">
          <h3>2 · Train forecasting models</h3>
          <p className="secondary" style={{ marginTop: 0 }}>Walk-forward training with purging, embargo and a locked final holdout. A version becomes champion only if it passes the
            predetermined gates (skill, calibration, sample size).</p>
          <div className="row">
            <label className="f">Horizon (trading days)
              <select value={horizon} onChange={(e) => setHorizon(+e.target.value)}>{(ctx.health?.horizons || [1, 5, 20, 60, 252]).map((h: number) => <option key={h}>{h}</option>)}</select>
            </label>
            <button className="btn primary" style={{ alignSelf: "end" }} onClick={() => run("/api/jobs/train", { mode: ctx.mode, horizon })}>Train</button>
            <button className="btn" style={{ alignSelf: "end" }} onClick={() => run("/api/jobs/experiment", { mode: ctx.mode, horizon })}>Run backtest experiment</button>
          </div>
          <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>Champions: {JSON.stringify(ctx.health?.champions?.[ctx.mode] || {})}</div>
        </div>
      </div>
      {job && <div className="card" style={{ marginTop: 14 }}><h3>Job {job.id}</h3><Progress job={job} />
        {job.status === "succeeded" && job.result && <pre className="mono">{JSON.stringify(job.result, null, 1).slice(0, 3000)}</pre>}</div>}

      {st && (
        <div className="grid" style={{ marginTop: 14 }}>
          <div className="card">
            <h3>Price coverage & freshness <ModeBadge mode={ctx.mode} /></h3>
            <Table rows={st.prices} initialSort="symbol" cols={[
              { key: "symbol", label: "Symbol", left: true },
              { key: "first_date", label: "First" }, { key: "last_date", label: "Last" },
              { key: "n_rows", label: "Rows" }, { key: "source", label: "Source" }, { key: "adjustment", label: "Adjustment" },
              { key: "age_days", label: "Age (d)" },
              { key: "stale", label: "Freshness", render: (r) => r.stale ? <span className="neg">⚠ STALE</span> : <span className="pos">✓ fresh</span> },
            ]} />
          </div>
          <div className="grid g2">
            <div className="card"><h3>Validation issues (latest run)</h3>
              <Table rows={st.validation_issues} cols={[
                { key: "symbol", label: "Symbol", left: true },
                { key: "severity", label: "Severity", render: (r) => <span className={r.severity === "ERROR" ? "neg" : ""}>{r.severity === "ERROR" ? "✖ " : "⚠ "}{r.severity}</span> },
                { key: "check_name", label: "Check" }, { key: "detail", label: "Detail", left: true },
              ]} maxHeight={320} /></div>
            <div className="card"><h3>Macro & fundamentals</h3>
              <Table rows={st.macro} cols={[{ key: "series_id", label: "Series", left: true }, { key: "last_date", label: "Last" }, { key: "n", label: "Obs" }, { key: "source", label: "Source" }]} maxHeight={200} />
              <Table rows={st.fundamentals} cols={[{ key: "symbol", label: "Symbol", left: true }, { key: "metrics", label: "Metrics" }, { key: "last_filing", label: "Last filing" }]} maxHeight={200} />
              <div className="muted">Filing events stored: {st.n_events} · news headlines stored: {st.n_news}</div>
            </div>
          </div>
          <div className="card"><h3>Ingestion log</h3>
            <Table rows={st.ingestion_log} cols={[
              { key: "ts", label: "Time", left: true }, { key: "provider", label: "Provider", left: true }, { key: "item", label: "Item", left: true },
              { key: "status", label: "Status", render: (r) => <span className={r.status === "OK" ? "pos" : "neg"}>{r.status}</span> },
              { key: "rows", label: "Rows" }, { key: "message", label: "Message", left: true, render: (r) => <span style={{ whiteSpace: "normal" }}>{r.message}</span> },
            ]} maxHeight={320} /></div>
        </div>
      )}
    </div>
  );
}
