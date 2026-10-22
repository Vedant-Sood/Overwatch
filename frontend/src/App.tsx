import { useEffect, useState } from "react";
import { api, type Mode } from "./api";
import DataPage from "./pages/Data";
import MarketPage from "./pages/Market";
import ScannerPage from "./pages/Scanner";
import ResearchPage from "./pages/Research";
import PortfolioPage from "./pages/Portfolio";
import StrategyLab from "./pages/StrategyLab";
import ConsolePage from "./pages/Console";
import PaperPage from "./pages/Paper";
import ModelsPage from "./pages/Models";

export type Ctx = {
  mode: Mode;
  theme: string;
  health: any;
  go: (page: string, param?: string) => void;
  param?: string;
  refreshHealth: () => void;
};

const PAGES: [string, string][] = [
  ["market", "Market Intelligence"],
  ["scanner", "AI Opportunity Scanner"],
  ["research", "Stock Research"],
  ["portfolio", "Portfolio Command Center"],
  ["lab", "Strategy Laboratory"],
  ["console", "AI Research Console"],
  ["paper", "Paper Trading Terminal"],
  ["models", "Model Intelligence"],
  ["data", "Data & Setup"],
];

function readHash(): [string, string | undefined] {
  const [p, q] = window.location.hash.replace(/^#\/?/, "").split("/");
  return [p || "market", q ? decodeURIComponent(q) : undefined];
}

export default function App() {
  const [[page, param], setRoute] = useState(readHash());
  const [mode, setMode] = useState<Mode>(() => (localStorageGet("bb-mode") as Mode) || "observed");
  const [theme, setTheme] = useState<string>(() => localStorageGet("bb-theme") || "system");
  const [health, setHealth] = useState<any>(null);

  const refreshHealth = () => api.get("/api/health").then(setHealth).catch(() => setHealth({ error: true }));
  useEffect(() => { refreshHealth(); }, []);
  useEffect(() => {
    const f = () => setRoute(readHash());
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  useEffect(() => {
    if (theme === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    localStorageSet("bb-theme", theme);
  }, [theme]);
  useEffect(() => localStorageSet("bb-mode", mode), [mode]);

  const go = (p: string, q?: string) => { window.location.hash = `/${p}${q ? "/" + encodeURIComponent(q) : ""}`; };
  const effectiveTheme = theme === "system" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : theme;
  const ctx: Ctx = { mode, theme: effectiveTheme, health, go, param, refreshHealth };
  const noData = health && health.modes && !health.modes[mode];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">BIGBOSS</div>
        <div className="brand-sub">Investment research terminal</div>
        <nav className="nav">
          {PAGES.map(([k, label]) => <button key={k} className={page === k ? "active" : ""} onClick={() => go(k)}>{label}</button>)}
        </nav>
        <div className="sidebar-foot">
          <label className="f">Data mode
            <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              <option value="observed">Observed (real providers)</option>
              <option value="synthetic">Synthetic (test/demo)</option>
            </select>
          </label>
          <label className="f">Theme
            <select value={theme} onChange={(e) => setTheme(e.target.value)}>
              <option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option>
            </select>
          </label>
          <div>Live trading: <strong>{health?.live_trading_enabled ? "ENABLED" : "disabled"}</strong></div>
          <div>LLM: {health?.llm?.available ? health.llm.provider : "none (deterministic reports)"}</div>
          <div>Research & paper trading only. Not investment advice; no outcome is guaranteed.</div>
        </div>
      </aside>
      <main className="main">
        {health?.error && <div className="banner err">Backend unreachable. Start it with <code>bigboss serve</code>.</div>}
        {noData && page !== "data" && (
          <div className="banner warn">The <strong>{mode}</strong> store has no price data yet. <a href="#/data">Open Data & Setup</a> to ingest data
            {mode === "synthetic" ? " or generate the synthetic test market" : ""}.</div>
        )}
        {page === "market" && <MarketPage ctx={ctx} />}
        {page === "scanner" && <ScannerPage ctx={ctx} />}
        {page === "research" && <ResearchPage ctx={ctx} />}
        {page === "portfolio" && <PortfolioPage ctx={ctx} />}
        {page === "lab" && <StrategyLab ctx={ctx} />}
        {page === "console" && <ConsolePage ctx={ctx} />}
        {page === "paper" && <PaperPage ctx={ctx} />}
        {page === "models" && <ModelsPage ctx={ctx} />}
        {page === "data" && <DataPage ctx={ctx} />}
      </main>
    </div>
  );
}

function localStorageGet(k: string): string | null {
  try { return localStorage.getItem(k); } catch { return null; }
}
function localStorageSet(k: string, v: string) {
  try { localStorage.setItem(k, v); } catch { /* storage unavailable */ }
}
