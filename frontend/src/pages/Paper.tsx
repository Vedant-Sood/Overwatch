import { useEffect, useState } from "react";
import { api, money, num, pct, signCls } from "../api";
import type { Ctx } from "../App";
import { Plot, seriesColor } from "../components/Plot";
import { DataBadge, ErrorBox, ModeBadge, SimBanner, Stat, Table } from "../components/ui";

export default function PaperPage({ ctx }: { ctx: Ctx }) {
  const [accounts, setAccounts] = useState<any[]>([]);
  const [aid, setAid] = useState("");
  const [a, setA] = useState<any>(null);
  const [err, setErr] = useState<unknown>(null);
  const [msg, setMsg] = useState<any>(null);
  const [newAcc, setNewAcc] = useState({ name: "", cash: "100000" });
  const [order, setOrder] = useState({ symbol: "", side: "BUY", quantity: "", order_type: "MARKET", limit_price: "" });
  const [audit, setAudit] = useState<any[] | null>(null);
  const loadAccounts = () => api.get(`/api/paper/accounts?mode=${ctx.mode}`).then((l) => { setAccounts(l); if (!aid && l.length) setAid(l[0].account_id); }).catch(setErr);
  const loadAccount = () => aid && api.get(`/api/paper/accounts/${aid}?mode=${ctx.mode}`).then(setA).catch(setErr);
  useEffect(() => { setAid(""); setA(null); loadAccounts(); }, [ctx.mode]);
  useEffect(() => { setAudit(null); loadAccount(); }, [aid]);

  const create = async () => {
    try { const r = await api.post("/api/paper/accounts", { mode: ctx.mode, name: newAcc.name, initial_cash: +newAcc.cash }); await loadAccounts(); setAid(r.account_id); } catch (e) { setErr(e); }
  };
  const submit = async () => {
    setErr(null);
    try {
      const r = await api.post(`/api/paper/accounts/${aid}/orders`, { mode: ctx.mode, symbol: order.symbol.toUpperCase(), side: order.side, quantity: +order.quantity, order_type: order.order_type, limit_price: order.limit_price ? +order.limit_price : null });
      setMsg(r); loadAccount();
    } catch (e) { setErr(e); }
  };
  const kill = async (on: boolean) => {
    if (on && !window.confirm("Engage the kill switch? All pending orders are cancelled and new orders are blocked.")) return;
    await api.post(`/api/paper/accounts/${aid}/kill-switch`, { mode: ctx.mode, on }); loadAccount();
  };
  const s = a?.summary;
  return (
    <div>
      <div className="page-head"><h1>Paper Trading Terminal</h1><span className="row"><DataBadge kind="simulated" />{ctx.mode === "observed" && <ModeBadge mode={ctx.mode} />}</span></div>
      <SimBanner mode={ctx.mode} />
      <div className="banner">Virtual brokerage. Orders fill at the latest stored close plus estimated spread, slippage, impact and commission. <strong>No real money and no broker connection.</strong></div>
      <ErrorBox error={err} />
      <div className="card"><div className="row" style={{ alignItems: "end" }}>
        <label className="f">Account<select value={aid} onChange={(e) => setAid(e.target.value)}><option value="">—</option>{accounts.map((x) => <option key={x.account_id} value={x.account_id}>{x.name} ({x.account_id})</option>)}</select></label>
        <span className="muted">or create:</span>
        <label className="f">Name<input value={newAcc.name} onChange={(e) => setNewAcc({ ...newAcc, name: e.target.value })} /></label>
        <label className="f">Virtual cash<input type="number" value={newAcc.cash} onChange={(e) => setNewAcc({ ...newAcc, cash: e.target.value })} /></label>
        <button className="btn" disabled={!newAcc.name} onClick={create}>Create paper account</button>
      </div></div>
      {s && <>
        {s.kill_switch ? <div className="banner err" style={{ marginTop: 14 }}><strong>✖ KILL SWITCH ENGAGED</strong> - new orders are blocked. <button className="btn" onClick={() => kill(false)}>Release</button></div> : null}
        <div className="grid g4" style={{ marginTop: 14 }}>
          <Stat label="Equity (simulated)" value={money(s.equity)} sub={`${s.base_currency} · start ${money(s.initial_cash)}`} />
          <Stat label="Total return" value={pct(s.total_return)} cls={signCls(s.total_return)} />
          <Stat label="Cash" value={money(s.cash)} />
          <Stat label="Realized / unrealized P&L" value={`${money(s.realized_pnl)} / ${money(s.unrealized_pnl)}`} sub={`commissions ${money(s.commissions_paid)}`} />
        </div>
        <div className="grid g2" style={{ marginTop: 14 }}>
          <div className="card"><h3>New simulated order</h3>
            <div className="row" style={{ alignItems: "end" }}>
              <label className="f">Symbol<input value={order.symbol} onChange={(e) => setOrder({ ...order, symbol: e.target.value })} style={{ width: 110 }} /></label>
              <label className="f">Side<select value={order.side} onChange={(e) => setOrder({ ...order, side: e.target.value })}><option>BUY</option><option>SELL</option></select></label>
              <label className="f">Quantity<input type="number" value={order.quantity} onChange={(e) => setOrder({ ...order, quantity: e.target.value })} style={{ width: 100 }} /></label>
              <label className="f">Type<select value={order.order_type} onChange={(e) => setOrder({ ...order, order_type: e.target.value })}><option>MARKET</option><option>LIMIT</option></select></label>
              {order.order_type === "LIMIT" && <label className="f">Limit<input type="number" value={order.limit_price} onChange={(e) => setOrder({ ...order, limit_price: e.target.value })} style={{ width: 90 }} /></label>}
              <button className="btn primary" disabled={!order.symbol || !order.quantity || !!s.kill_switch} onClick={submit}>Submit paper order</button>
              <button className="btn danger" onClick={() => kill(true)} disabled={!!s.kill_switch}>Kill switch</button>
            </div>
            {msg && <div className={`banner ${msg.status === "FILLED" ? "ok" : "err"}`} style={{ marginTop: 10 }}>
              {msg.status === "FILLED" ? `FILLED ${msg.side} ${msg.quantity} ${msg.symbol} @ ${num(msg.fill_price, 4)} (ref close ${num(msg.ref_price, 2)} on ${msg.price_date}, commission ${num(msg.commission, 2)})` : `REJECTED: ${msg.reason}`}</div>}
          </div>
          <div className="card"><h3>Equity history</h3>
            {a.equity_history.length < 2 ? <div className="muted">Equity is marked once per day with activity; the chart appears after two days of history.</div> :
              <Plot theme={ctx.theme} height={200} data={[{ x: a.equity_history.map((x: any) => x.date), y: a.equity_history.map((x: any) => x.equity), type: "scatter", mode: "lines+markers", line: { width: 2, color: seriesColor(0) }, marker: { size: 8 } }]} layout={{ showlegend: false, xaxis: { type: "date", tickformat: "%Y-%m-%d" } }} />}
            <div className="muted" style={{ fontSize: 12 }}>One mark per day with activity.</div>
          </div>
        </div>
        <div className="card" style={{ marginTop: 14 }}><h3>Positions</h3>
          <Table rows={a.positions} cols={[{ key: "symbol", label: "Symbol", left: true }, { key: "quantity", label: "Qty", render: (p) => num(p.quantity, 4) }, { key: "avg_cost", label: "Avg cost", render: (p) => num(p.avg_cost, 4) },
            { key: "price", label: "Last close", render: (p) => `${num(p.price)} (${p.price_date})` }, { key: "market_value", label: "Value", render: (p) => money(p.market_value) },
            { key: "unrealized_pnl", label: "Unrealized", render: (p) => <span className={signCls(p.unrealized_pnl)}>{money(p.unrealized_pnl)}</span> }, { key: "realized_pnl", label: "Realized", render: (p) => money(p.realized_pnl) }]} /></div>
        <div className="grid g2" style={{ marginTop: 14 }}>
          <div className="card"><h3>Orders</h3><Table rows={a.orders} maxHeight={300} cols={[{ key: "created_at", label: "Time", left: true, render: (o) => o.created_at.slice(0, 19) }, { key: "symbol", label: "Symbol", left: true }, { key: "side", label: "Side" }, { key: "quantity", label: "Qty", render: (o) => num(o.quantity, 4) },
            { key: "status", label: "Status", render: (o) => <span className={o.status === "FILLED" ? "pos" : "neg"}>{o.status}</span> }, { key: "strategy", label: "Strategy", left: true }, { key: "reason", label: "Reason", left: true }]} /></div>
          <div className="card"><h3>Strategy attribution</h3><Table rows={a.attribution} cols={[{ key: "strategy", label: "Strategy", left: true }, { key: "total_pnl", label: "P&L", render: (r) => <span className={signCls(r.total_pnl)}>{money(r.total_pnl)}</span> }, { key: "commissions", label: "Commissions", render: (r) => num(r.commissions) }, { key: "n_fills", label: "Fills" }]} />
            <button className="btn" style={{ marginTop: 10 }} onClick={() => api.get(`/api/paper/accounts/${aid}/audit?mode=${ctx.mode}`).then(setAudit)}>Show audit log</button>
            {audit && <pre className="mono">{audit.map((x) => `${x.ts} ${x.actor} ${x.action} ${x.detail}`).join("\n")}</pre>}</div>
        </div>
      </>}
    </div>
  );
}
