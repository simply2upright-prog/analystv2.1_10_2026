# options_data.py — Put/Call-Ratio und Optionsmarkt-Kennzahlen aus der aktuellen Yahoo-Optionskette
#
# WICHTIG: Yahoo liefert nur die AKTUELLE Kette (keine Historie). Die Kennzahlen sind deshalb Live-Information im
# Detail-Scanner und NICHT gegen die Strategie backgetestet. Wer sie prüfen will, speichert sie im Signal-Journal
# (flat_snapshot) und wertet nach einigen Monaten aus.
# Aussagekraft: nur bei liquiden Optionsmärkten (meist US-Aktien/große ETFs); Yahoo-Werte sind teils veraltet
# (Open Interest wird einmal täglich aktualisiert, Volumen ist außerhalb der Börsenzeit oft 0).

from datetime import date, datetime

import numpy as np
import pandas as pd


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("strike", "openInterest", "volume", "impliedVolatility"):
        d[c] = _num(d[c]) if c in d.columns else np.nan
    d = d[np.isfinite(d["strike"])]
    d["openInterest"] = d["openInterest"].fillna(0.0)
    d["volume"] = d["volume"].fillna(0.0)
    return d


def _max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    strikes = np.unique(np.concatenate([calls["strike"].to_numpy(float), puts["strike"].to_numpy(float)]))
    if len(strikes) == 0:
        return None
    cs, co = calls["strike"].to_numpy(float), calls["openInterest"].to_numpy(float)
    ps, po = puts["strike"].to_numpy(float), puts["openInterest"].to_numpy(float)
    pain = [(co * np.maximum(k - cs, 0)).sum() + (po * np.maximum(ps - k, 0)).sum() for k in strikes]
    return float(strikes[int(np.argmin(pain))])


def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, price: float) -> float | None:
    """Ø implizite Volatilität von Call und Put am Strike nahe dem Kurs (nur plausible Werte > 1 %)."""
    both = np.intersect1d(calls["strike"].to_numpy(float), puts["strike"].to_numpy(float))
    if len(both) == 0:
        both = np.unique(np.concatenate([calls["strike"].to_numpy(float), puts["strike"].to_numpy(float)]))
    if len(both) == 0:
        return None
    for k in both[np.argsort(np.abs(both - price))][:4]:           # die nächsten Strikes, bis einer brauchbare IV hat
        ivs = [float(x) for x in (calls.loc[calls["strike"] == k, "impliedVolatility"].tolist() +
                                  puts.loc[puts["strike"] == k, "impliedVolatility"].tolist()) if np.isfinite(x) and x > 0.01]
        if ivs:
            return float(np.mean(ivs))
    return None


def get_options_summary(ticker: str, price: float, max_expiries: int = 4, max_dte: int = 60,
                        today: date | None = None, tk=None) -> dict:
    """
    Fasst die nahen Verfallstermine zusammen (max. `max_expiries` innerhalb von `max_dte` Tagen).
    Gibt immer ein dict zurück: {"available": False, "reason": ...} oder die Kennzahlen.
    `tk` nur für Tests (Objekt mit .options und .option_chain()).
    """
    today = today or date.today()
    try:
        if tk is None:
            import yfinance as yf
            tk = yf.Ticker(ticker)
        exps = list(tk.options or [])
    except Exception as e:
        return {"available": False, "reason": f"Optionsdaten nicht abrufbar ({type(e).__name__})."}
    if not exps:
        return {"available": False, "reason": "Yahoo kennt für diesen Ticker keine Optionen (typisch für viele Nicht-US-Aktien)."}

    dated = []
    for e in exps:
        try:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        except Exception:
            continue
        if dte >= 1:
            dated.append((e, dte))
    sel = [x for x in dated if x[1] <= max_dte][:max_expiries] or dated[:2]
    if not sel:
        return {"available": False, "reason": "Keine zukünftigen Verfallstermine gefunden."}

    rows, all_c, all_p = [], [], []
    for e, dte in sel:
        try:
            ch = tk.option_chain(e)
            c, p = _clean(ch.calls), _clean(ch.puts)
        except Exception:
            continue
        if c.empty and p.empty:
            continue
        co, po = float(c["openInterest"].sum()), float(p["openInterest"].sum())
        cv, pv = float(c["volume"].sum()), float(p["volume"].sum())
        rows.append({"expiry": e, "dte": dte, "calls_oi": co, "puts_oi": po, "calls_vol": cv, "puts_vol": pv,
                     "pc_oi": po / co if co > 0 else None, "pc_vol": pv / cv if cv > 0 else None,
                     "atm_iv": _atm_iv(c, p, price), "max_pain": _max_pain(c, p)})
        all_c.append(c)
        all_p.append(p)
    if not rows:
        return {"available": False, "reason": "Optionsketten konnten nicht geladen werden."}

    calls, puts = pd.concat(all_c), pd.concat(all_p)
    co, po = float(calls["openInterest"].sum()), float(puts["openInterest"].sum())
    cv, pv = float(calls["volume"].sum()), float(puts["volume"].sum())

    # Strikes mit dem meisten Open Interest (aggregiert über die gewählten Verfälle) in ±30 % um den Kurs
    pw = puts[(puts["strike"] <= price) & (puts["strike"] >= price * 0.7)].groupby("strike")["openInterest"].sum()
    cw = calls[(calls["strike"] >= price) & (calls["strike"] <= price * 1.3)].groupby("strike")["openInterest"].sum()
    put_wall = {"strike": float(pw.idxmax()), "oi": float(pw.max())} if len(pw) and pw.max() > 0 else None
    call_wall = {"strike": float(cw.idxmax()), "oi": float(cw.max())} if len(cw) and cw.max() > 0 else None

    near = rows[0]
    ivs = [r["atm_iv"] for r in rows if r["atm_iv"]]
    em = near["atm_iv"] * np.sqrt(near["dte"] / 365.0) if near["atm_iv"] else None
    return {
        "available": True, "ticker": ticker, "price": float(price), "asof": today.isoformat(),
        "expiries": rows, "calls_oi": co, "puts_oi": po, "calls_vol": cv, "puts_vol": pv,
        "pc_oi": po / co if co > 0 else None, "pc_vol": pv / cv if cv > 0 else None,
        "atm_iv_near": near["atm_iv"], "atm_iv_far": ivs[-1] if len(ivs) >= 2 else None, "dte_near": near["dte"],
        "exp_move_pct": float(em * 100) if em else None,
        "exp_low": float(price * (1 - em)) if em else None, "exp_high": float(price * (1 + em)) if em else None,
        "put_wall": put_wall, "call_wall": call_wall,
        "max_pain": {"strike": near["max_pain"], "expiry": near["expiry"]} if near["max_pain"] else None,
        "thin": (co + po) < 1000,
    }


def describe_options(o: dict) -> list[str]:
    """Deutsche Einordnung der Kennzahlen. Bewusst vorsichtig formuliert (Stimmungsmaß, keine geprüfte Prognose)."""
    if not o.get("available"):
        return [o.get("reason", "Keine Optionsdaten.")]
    out = []
    if o["thin"]:
        out.append("⚠️ **Dünner Optionsmarkt** (weniger als 1.000 Kontrakte Open Interest in den nahen Verfällen) — die Kennzahlen sind kaum aussagekräftig.")
    pc = o["pc_oi"]
    if pc is not None:
        if pc < 0.6:
            t = "**Call-lastig** — deutlich mehr Calls als Puts offen (optimistische Positionierung oder Call-Spekulation)."
        elif pc <= 1.0:
            t = "**ausgewogen** bis leicht Call-lastig."
        elif pc <= 1.5:
            t = "**Put-lastig** — mehr Puts als Calls offen (Absicherung oder Pessimismus)."
        else:
            t = "**stark Put-lastig** — viele Puts offen (Absicherung/Pessimismus; manchmal auch ein Kontra-Indikator bei Übertreibung)."
        out.append(f"Put/Call (Open Interest) = **{pc:.2f}** → {t}")
    pv = o["pc_vol"]
    if pv is not None and pc is not None and abs(pv - pc) > 0.4:
        out.append(f"Put/Call nach **Tagesvolumen** ({pv:.2f}) weicht deutlich vom Open Interest ({pc:.2f}) ab — die aktuelle Aktivität ist anders gerichtet als die Bestände.")
    if o.get("exp_move_pct"):
        out.append(f"Der Optionsmarkt preist bis zum Verfall in {o['dte_near']} Tagen eine Bewegung von etwa **±{o['exp_move_pct']:.1f} %** ein "
                   f"(≈ {o['exp_low']:.2f} – {o['exp_high']:.2f}).")
    a, b = o.get("atm_iv_near"), o.get("atm_iv_far")
    if a and b and a > b * 1.10:
        out.append("Die kurzfristige Volatilität liegt **über** der längerfristigen (inverse Terminstruktur) — der Markt erwartet kurzfristig "
                   "mehr Bewegung, z. B. wegen Zahlen oder anderer Termine. Vorsicht bei Einstiegen vor dem Ereignis.")
    if o.get("put_wall"):
        w = o["put_wall"]
        out.append(f"Meiste **Put-Open-Interest** unter dem Kurs bei **{w['strike']:.2f}** ({w['oi']:,.0f} Kontrakte) — oft als Orientierung für eine Unterstützung genannt.".replace(",", "."))
    if o.get("call_wall"):
        w = o["call_wall"]
        out.append(f"Meiste **Call-Open-Interest** über dem Kurs bei **{w['strike']:.2f}** ({w['oi']:,.0f} Kontrakte) — oft als Orientierung für einen Widerstand genannt.".replace(",", "."))
    if o.get("max_pain"):
        out.append(f"„Max Pain“ des nächsten Verfalls ({o['max_pain']['expiry']}): **{o['max_pain']['strike']:.2f}** — der Kurs, bei dem Optionskäufer insgesamt am wenigsten bekommen. "
                   f"Kurz vor Verfall wird ein Anziehungseffekt diskutiert; die Evidenz ist schwach.")
    out.append("_Put/Call, Open-Interest-Zonen und Max Pain sind hier **nicht** gegen deine Strategie backgetestet (Yahoo liefert keine Options-Historie). "
               "Sie sind Zusatzinformation, kein Signal._")
    return out


def flat_snapshot(o: dict) -> dict:
    """Flache Felder für das Signal-Journal (spätere Auswertung)."""
    if not o or not o.get("available"):
        return {}
    return {"opt_pc_oi": o["pc_oi"], "opt_pc_vol": o["pc_vol"], "opt_iv_near": o["atm_iv_near"], "opt_iv_far": o["atm_iv_far"],
            "opt_dte": o["dte_near"], "opt_exp_move_pct": o["exp_move_pct"],
            "opt_put_wall": (o["put_wall"] or {}).get("strike"), "opt_call_wall": (o["call_wall"] or {}).get("strike"),
            "opt_oi_total": o["calls_oi"] + o["puts_oi"]}
