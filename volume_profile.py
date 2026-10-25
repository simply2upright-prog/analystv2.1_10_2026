# volume_profile.py — Volumen-Zonen: wo wurde zu welchem Kurs wirklich gehandelt?
#
# Anders als ein Histogramm der Schlusskurse (altes "POC"-Diagramm) wird hier das VOLUMEN jedes Tages gleichmäßig über
# die Spanne Tief..Hoch dieses Tages verteilt. Daraus:
#   POC           = Kurs mit dem meisten Umsatz im Betrachtungszeitraum
#   Wertbereich   = Preisband, in dem 70 % des Umsatzes lag (die "übliche Handelsspanne")
#   HVN / LVN     = Volumen-Hügel (viel gehandelt, oft Halt/Widerstand) und -Täler (wenig gehandelt, Kurs läuft schneller)
#   VWAP          = volumengewichteter Durchschnittskurs des Zeitraums
# Grenze: Aus Tagesdaten ist die Verteilung innerhalb eines Tages geschätzt. Ob Kurse an Volumen-Zonen
# tatsächlich häufiger halten, ist für diese Strategie NICHT backgetestet - das Bild ist Orientierung, kein Signal.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks


def compute_profile(df: pd.DataFrame, lookback: int = 252, bins: int = 60, va_share: float = 0.70) -> dict | None:
    if df is None or not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        return None
    d = df.tail(lookback)
    ok = np.isfinite(d["High"]) & np.isfinite(d["Low"]) & np.isfinite(d["Close"]) & np.isfinite(d["Volume"]) & (d["Volume"] > 0)
    d = d[ok]
    if len(d) < 30:
        return None
    lo, hi = float(d["Low"].min()), float(d["High"].max())
    if not hi > lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    L, H, V = d["Low"].to_numpy(float), d["High"].to_numpy(float), d["Volume"].to_numpy(float)
    C = d["Close"].to_numpy(float)
    overlap = np.clip(np.minimum(H[:, None], edges[None, 1:]) - np.maximum(L[:, None], edges[None, :-1]), 0, None)
    rng = H - L
    w = np.where(rng[:, None] > 0, overlap / np.where(rng > 0, rng, 1.0)[:, None], 0.0)
    flat = np.flatnonzero(rng <= 0)                     # Tage ohne Spanne: alles in den Bin des Schlusskurses
    if len(flat):
        w[flat, np.clip(np.searchsorted(edges, C[flat], side="right") - 1, 0, bins - 1)] = 1.0
    vol = (w * V[:, None]).sum(axis=0)
    total = float(vol.sum())
    if total <= 0:
        return None
    centers = (edges[:-1] + edges[1:]) / 2

    poc_i = int(np.argmax(vol))
    lo_i = hi_i = poc_i
    acc = float(vol[poc_i])
    while acc < va_share * total and (lo_i > 0 or hi_i < bins - 1):
        below = vol[lo_i - 1] if lo_i > 0 else -1.0
        above = vol[hi_i + 1] if hi_i < bins - 1 else -1.0
        if above >= below:
            hi_i += 1
            acc += float(vol[hi_i])
        else:
            lo_i -= 1
            acc += float(vol[lo_i])

    smooth = np.convolve(vol, np.array([1, 2, 1]) / 4.0, mode="same")
    mean_s = float(smooth.mean())
    pk, _ = find_peaks(smooth, height=1.15 * mean_s, distance=3)
    vl, _ = find_peaks(-smooth, distance=3)
    vl = [i for i in vl if smooth[i] < 0.6 * mean_s]

    tp = ((d["High"] + d["Low"] + d["Close"]) / 3).to_numpy(float)
    vwap = float((tp * V).sum() / V.sum())
    vstd = float(np.sqrt(((tp - vwap) ** 2 * V).sum() / V.sum()))
    return {
        "edges": edges, "centers": centers, "vol": vol, "smooth": smooth, "total": total, "step": float(edges[1] - edges[0]),
        "poc": float(centers[poc_i]), "poc_i": poc_i, "va_lo_i": lo_i, "va_hi_i": hi_i,
        "va_lo": float(edges[lo_i]), "va_hi": float(edges[hi_i + 1]), "va_share": float(acc / total),
        "hvn": [float(centers[i]) for i in pk], "hvn_i": [int(i) for i in pk],
        "lvn": [float(centers[i]) for i in vl], "vwap": vwap, "vwap_std": vstd,
        "lo": lo, "hi": hi, "lookback": lookback, "n_days": int(len(d)),
    }


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def describe_profile(p: dict, price: float, cur: str = "") -> list[str]:
    """Kurze deutsche Einordnung des aktuellen Kurses im Volumenbild."""
    out = []
    c = f" {cur}" if cur else ""
    if p["va_lo"] <= price <= p["va_hi"]:
        out.append(f"Kurs liegt **im Wertbereich** {_fmt(p['va_lo'])}–{_fmt(p['va_hi'])}{c} — hier lag {p['va_share']*100:.0f} % "
                   f"des Umsatzes der letzten {p['n_days']} Handelstage (übliche Handelsspanne).")
    elif price < p["va_lo"]:
        out.append(f"Kurs liegt **unter dem Wertbereich** ({(p['va_lo']/price-1)*100:.1f} % bis zur Unterkante {_fmt(p['va_lo'])}{c}) "
                   f"— außerhalb der üblichen Spanne; Rückkehr in den Bereich ist ein mögliches, kein sicheres Szenario.")
    else:
        out.append(f"Kurs liegt **über dem Wertbereich** ({(price/p['va_hi']-1)*100:.1f} % über der Oberkante {_fmt(p['va_hi'])}{c}).")
    out.append(f"**POC** (Kurs mit dem meisten Umsatz): {_fmt(p['poc'])}{c} ({(p['poc']/price-1)*100:+.1f} % vom Kurs) · "
               f"**VWAP**: {_fmt(p['vwap'])}{c} ({(p['vwap']/price-1)*100:+.1f} %).")
    smax = float(p["smooth"].max())
    below = [(h, i) for h, i in zip(p["hvn"], p["hvn_i"]) if h < price * 0.995]
    above = [(h, i) for h, i in zip(p["hvn"], p["hvn_i"]) if h > price * 1.005]
    if below:
        h, i = max(below)
        out.append(f"Nächste **Volumen-Zone darunter**: {_fmt(h)}{c} ({(h/price-1)*100:+.1f} %), Gewicht {p['smooth'][i]/smax*100:.0f} % des stärksten Hügels.")
    else:
        out.append("Darunter gibt es keine ausgeprägte Volumen-Zone — wenig Handelshistorie, die als Halt dienen könnte.")
    if above:
        h, i = min(above)
        out.append(f"Nächste **Volumen-Zone darüber**: {_fmt(h)}{c} ({(h/price-1)*100:+.1f} %), Gewicht {p['smooth'][i]/smax*100:.0f} % des stärksten Hügels.")
    else:
        out.append("Darüber gibt es keine ausgeprägte Volumen-Zone.")
    near = (p["centers"] >= price * 0.97) & (p["centers"] <= price * 1.03)
    if near.any():
        share = float(p["vol"][near].sum() / p["total"])
        expected = float(near.sum() / len(p["centers"]))
        ratio = share / expected if expected > 0 else 1.0
        if ratio < 0.6:
            out.append(f"Im Umfeld ±3 % um den Kurs wurde **wenig** gehandelt ({share*100:.0f} % des Umsatzes bei {expected*100:.0f} % der Preisstufen) "
                       f"— dünner Bereich, Kurs kann hier schneller laufen.")
        elif ratio > 1.4:
            out.append(f"Im Umfeld ±3 % um den Kurs wurde **viel** gehandelt ({share*100:.0f} % des Umsatzes bei {expected*100:.0f} % der Preisstufen) "
                       f"— dichter Bereich, tendenziell zäher.")
    return out


def build_profile_figure(df: pd.DataFrame, p: dict, cur: str = "", height: int = 620) -> go.Figure:
    """Links Kurs + gleitende Durchschnitte + Wertbereich, rechts das Volumenprofil auf derselben Preisachse."""
    n = p["lookback"]
    close = df["Close"]
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    d = df.tail(n)
    price = float(close.iloc[-1])

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.76, 0.24], horizontal_spacing=0.01)
    fig.add_hrect(y0=p["va_lo"], y1=p["va_hi"], fillcolor="rgba(59,130,246,0.18)", line_width=0, row=1, col=1,
                  annotation_text="Wertbereich (70 % Umsatz)", annotation_position="bottom right",
                  annotation_font=dict(color="#93c5fd", size=11))
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Kurs", line=dict(color="#facc15", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=sma50.loc[d.index], name="SMA50", line=dict(color="#38bdf8", width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=sma200.loc[d.index], name="SMA200", line=dict(color="#f97316", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=ema200.loc[d.index], name="EMA200", line=dict(color="#a78bfa", width=1.2, dash="dash"),
                             visible="legendonly"), row=1, col=1)
    if {"BB_Upper", "BB_Lower"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_Upper"], name="Bollinger(200) oben", line=dict(color="#00eeee", width=.8, dash="dot"),
                                 visible="legendonly"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_Lower"], name="Bollinger(200) unten", line=dict(color="#00eeee", width=.8, dash="dot"),
                                 visible="legendonly"), row=1, col=1)
    fig.add_hline(y=p["poc"], line_color="#22c55e", line_width=2, row=1, col=1,
                  annotation_text=f"POC {p['poc']:.2f}", annotation_position="top left", annotation_font=dict(color="#22c55e", size=11))
    fig.add_hline(y=p["vwap"], line_color="#c084fc", line_dash="dash", line_width=1.2, row=1, col=1,
                  annotation_text=f"VWAP {p['vwap']:.2f}", annotation_position="bottom left", annotation_font=dict(color="#c084fc", size=11))
    fig.add_hline(y=price, line_color="#facc15", line_dash="dot", line_width=1.2, row=1, col=1,
                  annotation_text=f"Kurs {price:.2f}", annotation_position="top right", annotation_font=dict(color="#facc15", size=11))

    colors = ["#3b82f6" if p["va_lo_i"] <= i <= p["va_hi_i"] else "#334155" for i in range(len(p["centers"]))]
    colors[p["poc_i"]] = "#22c55e"
    fig.add_trace(go.Bar(y=p["centers"], x=p["vol"], orientation="h", marker_color=colors, name="Volumen je Preisstufe",
                         width=p["step"] * 0.92, hovertemplate="%{y:.2f}: %{x:,.0f}<extra></extra>", showlegend=False), row=1, col=2)
    for h in p["hvn"]:
        fig.add_hline(y=h, line_color="rgba(148,163,184,0.45)", line_dash="dot", line_width=1, row=1, col=1)
    pad = (p["hi"] - p["lo"]) * 0.03
    fig.update_yaxes(range=[min(p["lo"], price) - pad, max(p["hi"], price) + pad], showgrid=True, gridcolor="#1e2235",
                     title_text=f"Preis ({cur})" if cur else None, row=1, col=1)
    fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=2)
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.06, x=0), bargap=0.05)
    return fig
