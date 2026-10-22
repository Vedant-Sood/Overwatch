import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export const seriesColor = (i: number) => cssVar(`--series-${(i % 8) + 1}`);

/** Plotly wrapper that reads chart chrome from the current theme's CSS tokens. */
export function Plot({ data, layout, height = 320, theme }: { data: any[]; layout?: any; height?: number; theme: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const ink = cssVar("--text-secondary");
    const grid = cssVar("--grid");
    const base: any = {
      height,
      margin: { l: 56, r: 16, t: 16, b: 40 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 12, color: ink },
      xaxis: { gridcolor: grid, linecolor: cssVar("--axis"), zeroline: false },
      yaxis: { gridcolor: grid, linecolor: cssVar("--axis"), zerolinecolor: cssVar("--axis") },
      legend: { orientation: "h", y: -0.18, font: { color: ink } },
      hovermode: "x unified",
      hoverlabel: { bgcolor: cssVar("--surface-1"), bordercolor: cssVar("--axis"), font: { color: cssVar("--text-primary") } },
    };
    const merged = { ...base, ...layout, xaxis: { ...base.xaxis, ...(layout?.xaxis || {}) }, yaxis: { ...base.yaxis, ...(layout?.yaxis || {}) } };
    Plotly.react(ref.current, data, merged, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] });
  }, [data, layout, height, theme]);
  useEffect(() => {
    const el = ref.current;
    return () => { if (el) Plotly.purge(el); };
  }, []);
  return <div ref={ref} style={{ width: "100%" }} />;
}
