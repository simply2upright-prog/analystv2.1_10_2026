# ticker_search.py — v2
#   * Namen für alle Ticker (ticker_names.py) + optional automatisch erzeugte ticker_meta.json
#   * Tippfehler-Toleranz ("appel", "nvidea"), Zusammenschreibung ("pingan" = "Ping An"), "&"-Namen ("at&t", "p&g")
#   * ISIN- und WKN-Suche, Suche nach Thema/Sektor/Index ("rüstung", "uran", "defense", "dax")
#   * ADR -> Heimatbörse ("nsrgy" findet Nestlé in Vevey), Börse + Währung im Label
#   * Freie Yahoo-Eingabe als Fallback ("2318.HK", "RY.TO"), auch wenn der Ticker nicht in der Liste steht
#
# Öffentliche API unverändert: search_ticker(), label_to_ticker(), get_display_name(), TICKER_NAMES

import difflib
import json
import os
import re
import unicodedata
from functools import lru_cache

from database import TICKER_LISTS, ADR_ALIASES, get_currency, get_focus, get_group_label
from ticker_names import TICKER_NAMES  # noqa: F401  (re-export, bisher hier definiert)

SEP = " — "

# ── Börsen-Namen (Suffix nach dem Punkt = Yahoo-Börsenkennung) ───────────────────────────────
EXCHANGES = {
    ".DE": "Xetra", ".PA": "Paris", ".MI": "Mailand", ".MC": "Madrid", ".AS": "Amsterdam", ".L": "London",
    ".SW": "Zürich", ".ST": "Stockholm", ".CO": "Kopenhagen", ".OL": "Oslo", ".HE": "Helsinki", ".VI": "Wien",
    ".LS": "Lissabon", ".BR": "Brüssel", ".IS": "Istanbul", ".JO": "Johannesburg", ".SR": "Riad", ".T": "Tokio",
    ".HK": "Hongkong", ".KS": "Seoul", ".AX": "Sydney (ASX)", ".TO": "Toronto (TSX)", ".SA": "São Paulo", ".MX": "Mexiko",
}


def get_exchange(ticker: str) -> str:
    up = ticker.upper()
    for suf in sorted(EXCHANGES, key=len, reverse=True):
        if up.endswith(suf):
            return EXCHANGES[suf]
    return "USA"


# ── Normalisierung ───────────────────────────────────────────────────────────────────────────
_SPECIAL = str.maketrans({"ø": "o", "æ": "ae", "ł": "l", "ß": "ss", "đ": "d", "œ": "oe"})


def _norm(s: str) -> str:
    """Kleinschreibung, ohne Akzente/Umlaute, '&', Punkte und Apostrophe; sonstige Zeichen -> Leerzeichen."""
    s = unicodedata.normalize("NFKD", str(s).lower().translate(_SPECIAL))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[&.'’`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def _tokenize(s: str) -> list[str]:
    return _norm(s).split()


# ── ISIN / WKN ───────────────────────────────────────────────────────────────────────────────
def _isin_valid(isin: str) -> bool:
    if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", isin):
        return False
    digits = "".join(str(int(c, 36)) for c in isin)       # A=10 … Z=35
    total, dbl = 0, False
    for ch in reversed(digits):
        n = int(ch)
        if dbl:
            n *= 2
            n = n - 9 if n > 9 else n
        total += n
        dbl = not dbl
    return total % 10 == 0


# Bekannte Standardwerte (Prüfziffer wird beim Import geprüft; fehlerhafte Einträge werden verworfen).
# Alle weiteren ISIN kommen aus ticker_meta.json (build_ticker_meta.py).
_ISIN_SEED = {
    "DE0007164600": "SAP.DE", "DE0008404005": "ALV.DE", "DE0007236101": "SIE.DE", "DE0005557508": "DTE.DE",
    "DE0005190003": "BMW.DE", "DE0007664039": "VOW3.DE", "DE0007100000": "MBG.DE", "DE000BASF111": "BAS.DE",
    "DE000A1EWWW0": "ADS.DE", "DE0005140008": "DBK.DE", "DE0007030009": "RHM.DE", "DE0008430026": "MUV2.DE",
    "DE0006231004": "IFX.DE", "DE000BAY0017": "BAYN.DE", "DE000ENAG999": "EOAN.DE", "DE0007037129": "RWE.DE",
    "DE0005552004": "DHL.DE", "DE0006048432": "HEN3.DE", "DE000CBK1001": "CBK.DE", "NL0000235190": "AIR.PA",
    "US0378331005": "AAPL", "US5949181045": "MSFT", "US67066G1040": "NVDA", "US0231351067": "AMZN",
    "US02079K3059": "GOOGL", "US30303M1027": "META", "US88160R1014": "TSLA", "US1912161007": "KO",
    "US4781601046": "JNJ", "US7427181091": "PG", "US30231G1022": "XOM", "US46625H1005": "JPM",
    "CH0038863350": "NESN.SW", "CH0012005267": "NOVN.SW", "FR0000121014": "MC.PA", "NL0010273215": "ASML.AS",
    "DK0062498333": "NOVO-B.CO", "GB00BP6MXD84": "SHEL.L",
}

# ── Optionale, automatisch erzeugte Metadaten (Yahoo): Name, Sektor, Industrie, ISIN ───────────
_META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_meta.json")


@lru_cache(maxsize=1)
def _load_meta() -> dict:
    try:
        with open(_META_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tickers", data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_meta(ticker: str) -> dict:
    return _load_meta().get(ticker, {})


def get_sector(ticker: str) -> str:
    m = get_meta(ticker)
    return " · ".join(x for x in (m.get("sector"), m.get("industry")) if x)


# ── Themen-/Sektor-Synonyme (deutsche Labels werden automatisch aus database.GROUP_LABELS gelesen) ──
GROUP_SYNONYMS = {
    "defense": ["DEFENSE_AEROSPACE"], "defence": ["DEFENSE_AEROSPACE"], "verteidigung": ["DEFENSE_AEROSPACE"],
    "waffen": ["DEFENSE_AEROSPACE"], "ruestung": ["DEFENSE_AEROSPACE"], "aerospace": ["DEFENSE_AEROSPACE"],
    "oil": ["ENERGY_OIL_GAS", "SP500_ENERGY"], "oel": ["ENERGY_OIL_GAS"], "ol": ["ENERGY_OIL_GAS", "SP500_ENERGY"], "gas": ["ENERGY_OIL_GAS"],
    "erdoel": ["ENERGY_OIL_GAS"], "energy": ["ENERGY_OIL_GAS", "SP500_ENERGY", "EV_CLEAN_ENERGY"],
    "energie": ["ENERGY_OIL_GAS", "SP500_ENERGY", "EV_CLEAN_ENERGY", "UTILITIES_INFRASTRUCTURE"],
    "mining": ["METALS_MINING_DIVERSIFIED", "GOLD_SILVER", "URANIUM_NUCLEAR", "BATTERY_METALS_RARE_EARTHS"],
    "bergbau": ["METALS_MINING_DIVERSIFIED", "GOLD_SILVER", "URANIUM_NUCLEAR", "BATTERY_METALS_RARE_EARTHS"],
    "rohstoffe": ["METALS_MINING_DIVERSIFIED", "GOLD_SILVER", "URANIUM_NUCLEAR", "BATTERY_METALS_RARE_EARTHS", "ENERGY_OIL_GAS"],
    "metalle": ["METALS_MINING_DIVERSIFIED", "GOLD_SILVER", "BATTERY_METALS_RARE_EARTHS"],
    "gold": ["GOLD_SILVER"], "silver": ["GOLD_SILVER"], "silber": ["GOLD_SILVER"], "edelmetall": ["GOLD_SILVER"],
    "edelmetalle": ["GOLD_SILVER"], "uranium": ["URANIUM_NUCLEAR"], "uran": ["URANIUM_NUCLEAR"],
    "nuclear": ["URANIUM_NUCLEAR"], "atom": ["URANIUM_NUCLEAR"], "kernkraft": ["URANIUM_NUCLEAR"],
    "kernenergie": ["URANIUM_NUCLEAR"], "copper": ["METALS_MINING_DIVERSIFIED"], "kupfer": ["METALS_MINING_DIVERSIFIED"],
    "lithium": ["BATTERY_METALS_RARE_EARTHS"], "batterie": ["BATTERY_METALS_RARE_EARTHS"],
    "rare": ["BATTERY_METALS_RARE_EARTHS"], "seltene": ["BATTERY_METALS_RARE_EARTHS"],
    "chips": ["AI_SEMICONDUCTORS"], "halbleiter": ["AI_SEMICONDUCTORS"], "semiconductor": ["AI_SEMICONDUCTORS"],
    "semis": ["AI_SEMICONDUCTORS"], "ai": ["AI_SEMICONDUCTORS", "SOFTWARE_CLOUD_CYBER"], "ki": ["AI_SEMICONDUCTORS", "SOFTWARE_CLOUD_CYBER"],
    "software": ["SOFTWARE_CLOUD_CYBER"], "cloud": ["SOFTWARE_CLOUD_CYBER"], "cyber": ["SOFTWARE_CLOUD_CYBER"],
    "bank": ["FINANCIALS_PAYMENTS", "SP500_FINANCIALS"], "banken": ["FINANCIALS_PAYMENTS", "SP500_FINANCIALS"],
    "banks": ["FINANCIALS_PAYMENTS", "SP500_FINANCIALS"], "finanz": ["FINANCIALS_PAYMENTS", "SP500_FINANCIALS"],
    "versicherung": ["FINANCIALS_PAYMENTS", "SP500_FINANCIALS"], "insurance": ["FINANCIALS_PAYMENTS", "SP500_FINANCIALS"],
    "pharma": ["HEALTHCARE_PHARMA", "SP500_HEALTHCARE"], "healthcare": ["HEALTHCARE_PHARMA", "SP500_HEALTHCARE"],
    "gesundheit": ["HEALTHCARE_PHARMA", "SP500_HEALTHCARE"], "biotech": ["HEALTHCARE_PHARMA", "SP500_HEALTHCARE"],
    "immobilien": ["REAL_ESTATE_REITS_EUROPE", "SP500_REITS"], "reit": ["SP500_REITS", "REAL_ESTATE_REITS_EUROPE"],
    "reits": ["SP500_REITS", "REAL_ESTATE_REITS_EUROPE"], "realestate": ["REAL_ESTATE_REITS_EUROPE", "SP500_REITS"],
    "bau": ["CONSTRUCTION_INFRA_EUROPE"], "construction": ["CONSTRUCTION_INFRA_EUROPE"],
    "auto": ["AUTOS"], "autos": ["AUTOS"], "cars": ["AUTOS"], "ev": ["EV_CLEAN_ENERGY", "CHINA_EV_AUTO"],
    "telekom": ["TELECOM"], "telecom": ["TELECOM"], "mobilfunk": ["TELECOM"],
    "airline": ["TRAVEL_AIRLINES_LOGISTICS"], "airlines": ["TRAVEL_AIRLINES_LOGISTICS"], "reise": ["TRAVEL_AIRLINES_LOGISTICS"],
    "travel": ["TRAVEL_AIRLINES_LOGISTICS"], "logistik": ["TRAVEL_AIRLINES_LOGISTICS"], "shipping": ["TRAVEL_AIRLINES_LOGISTICS"],
    "krypto": ["CRYPTO_FINTECH"], "crypto": ["CRYPTO_FINTECH"], "bitcoin": ["CRYPTO_FINTECH"], "fintech": ["CRYPTO_FINTECH"],
    "versorger": ["UTILITIES_INFRASTRUCTURE", "SP500_UTILITIES"], "utilities": ["UTILITIES_INFRASTRUCTURE", "SP500_UTILITIES"],
    "medien": ["MEDIA_ENTERTAINMENT"], "media": ["MEDIA_ENTERTAINMENT"], "chemie": ["CHEMICALS_FERTILIZER"],
    "chemicals": ["CHEMICALS_FERTILIZER"], "duenger": ["CHEMICALS_FERTILIZER"], "luxus": ["LUXURY_FASHION"], "luxury": ["LUXURY_FASHION"],
    "tabak": ["GENUSS_TABAK_ALKOHOL_SUESSES"], "alkohol": ["GENUSS_TABAK_ALKOHOL_SUESSES"], "bier": ["GENUSS_TABAK_ALKOHOL_SUESSES"],
    "china": ["CHINA_TECH_INTERNET", "CHINA_FINANCE_ENERGY_TELECOM", "CHINA_EV_AUTO", "CHINA_CONSUMER_EDUCATION"],
    "korea": ["KOSPI_KOREA", "KOREA_ADR"], "japan": ["NIKKEI_SELECTION"], "kanada": ["CANADA_TSX"], "canada": ["CANADA_TSX"],
    "australien": ["AUSTRALIA_ASX"], "australia": ["AUSTRALIA_ASX"], "israel": ["ISRAEL_TECH_DEFENSE"],
    "quanten": ["SPECULATIVE_AI_QUANTUM"], "quantum": ["SPECULATIVE_AI_QUANTUM"],
    "sp500": [g for g in TICKER_LISTS if g.startswith("SP500")],
}


# ── Index (einmal aufbauen, pro Ticker-Liste zwischenspeichern) ─────────────────────────────────
class _Entry:
    __slots__ = ("ticker", "base", "raw", "aliases", "ngroups", "focus")


@lru_cache(maxsize=4)
def _build_index(all_tickers: tuple):
    meta = _load_meta()
    ngroups: dict[str, int] = {}
    for lst in TICKER_LISTS.values():
        for t in lst:
            ngroups[t] = ngroups.get(t, 0) + 1

    entries, isin_map, wkn_map = [], {}, {}
    for t in all_tickers:
        e = _Entry()
        e.ticker, e.raw = t, t.lower()
        e.base = t.split(".")[0].lower()
        names = list(TICKER_NAMES.get(t, []))
        m = meta.get(t, {})
        for extra in (m.get("name"), m.get("long_name")):
            if extra and extra not in names:
                names.append(extra)
        if not names:
            names = [t]
        e.aliases = []
        for a in names:
            n = _norm(a)
            if n:
                e.aliases.append((n, n.replace(" ", ""), n.split()))
        e.ngroups = ngroups.get(t, 0)
        e.focus = _tokenize(get_focus(t))
        entries.append(e)
        if m.get("isin"):
            isin_map[m["isin"].upper()] = t
    present = set(all_tickers)
    for isin, t in _ISIN_SEED.items():
        if _isin_valid(isin) and t in present:
            isin_map.setdefault(isin, t)
    for isin, t in isin_map.items():                 # WKN steckt bei deutschen ISIN in Stelle 6-11
        if isin.startswith("DE") and len(isin) == 12:
            wkn_map[isin[5:11]] = t

    group_index = []                                  # (Gruppenschlüssel, Tokens aus Schlüssel + deutschem Label)
    for g in TICKER_LISTS:
        toks = set(_tokenize(g.replace("_", " "))) | set(_tokenize(get_group_label(g)))
        group_index.append((g, toks))
    return entries, isin_map, wkn_map, group_index, present


def _matching_groups(q: str, qtoks: list[str], group_index) -> dict[str, int]:
    """Gruppe -> Basis-Score. Exaktes Synonym (z. B. 'uran', 'ki') stark, Treffer im Gruppen-Label mittel."""
    found: dict[str, int] = {}
    for key in (q, q.replace(" ", "")) + tuple(qtoks):
        for g in GROUP_SYNONYMS.get(key, []):
            if g in TICKER_LISTS:
                found[g] = 78
    long_toks = [qt for qt in qtoks if len(qt) >= 3]
    if long_toks:
        for g, toks in group_index:
            if g not in found and all(any(t.startswith(qt) for t in toks) for qt in long_toks):
                found[g] = 66
    return found


# ── Label ───────────────────────────────────────────────────────────────────────────────────
def _display_name(t: str) -> str:
    names = TICKER_NAMES.get(t)
    if names:
        return names[0]
    return get_meta(t).get("name") or t


def _label(t: str) -> str:
    focus = get_focus(t)
    extra = f" · {focus}" if focus else ""
    return f"{t}{SEP}{_display_name(t)} · {get_exchange(t)} {get_currency(t)}{extra}"


def label_to_ticker(label: str) -> str:
    return label.split(SEP)[0].strip()


def get_display_name(ticker: str) -> str:
    return _label(ticker)


# ── Suche ───────────────────────────────────────────────────────────────────────────────────
_TICKER_LIKE = re.compile(r"^[A-Za-z0-9\-\^=]{1,12}(\.[A-Za-z]{1,3})?$")


def search_ticker(query: str, all_tickers: list[str], max_results: int = 12) -> list[str]:
    qraw = (query or "").strip()
    if not qraw:
        return []
    entries, isin_map, wkn_map, group_index, present = _build_index(tuple(all_tickers))
    ql = qraw.lower()
    q = _norm(qraw)
    qc = q.replace(" ", "")
    qtoks = q.split()
    scores: dict[str, float] = {}

    def bump(t, s):
        if s > scores.get(t, 0):
            scores[t] = s

    # 1) ISIN / WKN
    up = qraw.upper().replace(" ", "")
    if up in isin_map:
        bump(isin_map[up], 100)
    if len(up) == 6 and up in wkn_map:
        bump(wkn_map[up], 98)

    # 2) ADR -> Heimatbörse
    for adr, home in ADR_ALIASES.items():
        if adr.lower() == ql and home in present:
            bump(home, 99)

    # 3) Ticker, Name, Zusammenschreibung
    for e in entries:
        s = 0.0
        if ql == e.raw:
            s = 100
        elif ql == e.base:
            s = 94
        elif e.raw.startswith(ql) or (len(ql) >= 2 and e.base.startswith(ql)):
            s = 86 - min(len(e.base) - len(ql), 6) * 0.5
        if q:
            for n, nc, toks in e.aliases:
                if n == q or nc == qc:
                    s = max(s, 92)
                elif n.startswith(q) or (len(qc) >= 3 and nc.startswith(qc)):
                    s = max(s, 82)
                elif all(any(t.startswith(qt) for t in toks) for qt in qtoks):
                    s = max(s, 74)
                elif len(qc) >= 4 and qc in nc:
                    s = max(s, 58)
        if q and e.focus and len(qc) >= 3 and all(any(t.startswith(qt) for t in e.focus) for qt in qtoks):
            s = max(s, 80)                         # Fokus-Hinweis (z. B. "Silber", "Kupfer", "Uran") trifft
        if s:
            bump(e.ticker, s + min(e.ngroups, 6) * 0.3)

    # 4) Thema / Sektor / Index
    if len(qc) >= 2:
        member_of: dict[str, list[int]] = {}
        for g, base in _matching_groups(q, qtoks, group_index).items():
            for t in TICKER_LISTS[g]:
                member_of.setdefault(t, []).append(base)
        ng = {e.ticker: e.ngroups for e in entries}
        for t, bases in member_of.items():
            if t in present:
                bump(t, max(bases) + min(len(bases) - 1, 3) + min(ng.get(t, 0), 6) * 0.3)

    # 5) Tippfehler (nur wenn nichts Gutes gefunden wurde)
    if len(qc) >= 4 and (not scores or max(scores.values()) < 70):
        for e in entries:
            best = 0.0
            for n, nc, toks in e.aliases:
                for cand in {nc, nc[:len(qc) + 1], *(t for t in toks if len(t) >= 4)}:
                    sm = difflib.SequenceMatcher(None, qc, cand)
                    if sm.real_quick_ratio() < 0.7 or sm.quick_ratio() < 0.7:
                        continue
                    best = max(best, sm.ratio())
            if len(e.base) >= 3:
                best = max(best, difflib.SequenceMatcher(None, ql, e.base).ratio())
            if best >= 0.78:
                bump(e.ticker, 40 + 30 * best)

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [_label(t) for t, _ in ranked[:max_results]]

    # 6) Freie Yahoo-Eingabe, wenn nichts Brauchbares gefunden wurde
    if (not ranked or ranked[0][1] < 60) and _TICKER_LIKE.match(qraw):
        free = qraw.upper()
        if free not in scores:
            labels.append(f"{free}{SEP}nicht in der Liste · direkt bei Yahoo abfragen ({get_exchange(free)})")
    return labels[:max_results]
