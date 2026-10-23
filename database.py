import json  # noqa: F401
# database.py — Globale Aktien-Datenbank v2.0
# Abdeckung: USA (S&P500, NASDAQ, Dow Jones, Russell) | UK (FTSE100)
# China (A/H-Shares, HK, ADR) | Südkorea (KOSPI) | Japan (Nikkei)
# Deutschland (DAX/MDAX/SDAX) | Frankreich (CAC40) | Spanien (IBEX35)
# Italien (FTSE MIB) | Niederlande (AEX) | Schweiz (SMI)
# Skandinavien | Sektoren: Energy, Mining, Healthcare, Financials,
# Tech, Consumer, REIT, Utilities, Sin-Stocks, Emerging Markets

TICKER_LISTS = {
    "SP500_TECH": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "QCOM", "TXN",
        "INTU", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "MCHP", "INTC", "IBM", "CSCO", "ACN", "FTNT", "PANW",
        "CRWD", "ZS", "DDOG", "MU", "SMCI", "VRT", "ANET", "NET", "OKTA", "SNOW", "MDB", "PLTR",
    ],
    "SP500_FINANCIALS": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA", "PYPL", "COF", "USB", "PNC",
        "TFC", "BNY", "STT", "FITB", "KEY", "RF", "HBAN", "AFL", "MET", "PRU", "ALL", "AIG", "CB", "PGR",
        "TRV", "CME", "ICE", "NDAQ", "SPGI", "MCO", "BRK-B", "AON", "MKTX", "TROW", "NTRS", "RJF", "AMP",
        "BRO", "CG", "BX", "LAZ", "EVR",
    ],
    "SP500_HEALTHCARE": [
        "JNJ", "LLY", "UNH", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "VRTX",
        "REGN", "BIIB", "SYK", "BSX", "MDT", "BDX", "EW", "ISRG", "ZBH", "HCA", "CI", "CVS", "MCK", "COR",
        "CAH", "IDXX", "IQV", "ILMN", "A", "WAT", "ZTS", "VTRS", "PRGO",
    ],
    "SP500_CONSUMER_STAPLES": [
        "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "PM", "MO", "KHC", "GIS", "CPB", "HRL", "CAG", "MKC", "CL",
        "CHD", "CLX", "EL", "KR", "DLTR", "DG", "SJM", "TSN", "KMB",
    ],
    "SP500_CONSUMER_DISC": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR", "HLT", "YUM", "CMG", "DPZ",
        "QSR", "F", "GM", "PHM", "DHI", "LEN", "NVR", "TOL", "ROST", "RL", "PVH", "EBAY", "ETSY", "RH",
        "WSM",
    ],
    "SP500_ENERGY": [
        "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "MPC", "PSX", "VLO", "DVN", "FANG", "APA", "OXY", "PBR",
        "BKR", "NOV", "CHRD", "MTDR", "PR", "SM",
    ],
    "SP500_INDUSTRIALS": [
        "GE", "HON", "UPS", "BA", "LMT", "RTX", "NOC", "GD", "CAT", "DE", "EMR", "ETN", "ITW", "PH", "IR",
        "ROK", "AME", "FTV", "XYL", "IEX", "ROP", "UNP", "CSX", "NSC", "FDX", "PCAR", "GWW", "WM", "RSG",
        "J", "ACM", "HUBB", "DCI", "NDSN", "GGG", "CTAS", "FAST", "CMI",
    ],
    "SP500_MATERIALS": [
        "LIN", "APD", "ECL", "SHW", "PPG", "NEM", "FCX", "NUE", "STLD", "AA", "CF", "MOS", "DOW", "LYB",
        "ALB", "MP", "CTVA", "FMC", "IFF", "AVY", "PKG", "BALL", "EMN", "WLK", "INGR",
    ],
    "SP500_UTILITIES": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG", "AWK", "WEC", "ES", "FE", "CNP", "AES", "NRG",
        "ETR", "EIX", "PPL", "DTE", "AEE", "CMS", "LNT", "ED", "XEL", "PEG", "EVRG", "ATO", "IDA", "NWN",
        "OGE", "BKH",
    ],
    "SP500_REITS": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "DLR", "VICI", "SPG", "AVB", "EQR", "MAA", "UDR", "CPT",
        "ESS", "WY", "ARE", "BXP", "KIM", "REG", "FRT", "HST", "VTR", "WELL", "IRM", "EXR", "DOC", "CUZ",
        "KRC", "SBAC", "EGP", "TRNO", "LXP", "REXR", "NNN", "WPC", "OHI", "IRT",
    ],
    "DOW_JONES_30": [
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW", "GS", "HD", "HON", "IBM",
        "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WMT",
    ],
    "NASDAQ_GROWTH_TECH": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "ADBE", "AMD", "QCOM",
        "TXN", "INTU", "AMAT", "ISRG", "BKNG", "VRTX", "PANW", "MU", "SNPS", "CDNS", "KLAC", "MELI", "PYPL",
        "SNOW", "PLTR", "SMCI", "VRT", "DECK", "SOFI", "AFRM", "COIN", "DKNG", "RBLX", "SNAP", "ZM", "DOCU",
        "OKTA", "NET", "MDB", "HOOD",
    ],
    "FTSE100": [
        "AZN.L", "HSBA.L", "ULVR.L", "GSK.L", "RIO.L", "BP.L", "SHEL.L", "BATS.L", "REL.L", "NG.L", "DGE.L",
        "IMB.L", "LSEG.L", "LLOY.L", "BARC.L", "NWG.L", "STAN.L", "PRU.L", "LGEN.L", "AV.L", "MNG.L", "RR.L",
        "BA.L", "MNDI.L", "ANTO.L", "AAL.L", "GLEN.L", "BHP.L", "SVT.L", "UU.L", "SSE.L", "BTRW.L", "PSN.L",
        "TW.L", "BKG.L", "SGRO.L", "BLND.L", "LAND.L", "WPP.L", "OCDO.L", "MKS.L", "NXT.L", "JD.L", "HBR.L",
        "IMI.L", "EXPN.L", "HLMA.L", "WEIR.L", "TSCO.L", "SBRY.L", "KGF.L", "AUTO.L", "III.L", "RKT.L",
    ],
    "DAX": [
        "ADS.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "CBK.DE", "CON.DE", "DTG.DE", "DBK.DE",
        "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "ENR.DE", "FRE.DE", "HEI.DE", "HEN3.DE", "HNR1.DE",
        "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "P911.DE", "PAH3.DE", "PUM.DE", "RHM.DE",
        "RWE.DE", "SAP.DE", "SRT3.DE", "SHL.DE", "SIE.DE", "SY1.DE", "TKA.DE", "VNA.DE", "VOW3.DE", "ZAL.DE",
        "AIR.PA",
    ],
    "MDAX": [
        "BC8.DE", "BNR.DE", "BOSS.DE", "DEQ.DE", "DUE.DE", "EVD.DE", "EVK.DE", "FPE.DE", "FRA.DE", "FME.DE",
        "GXI.DE", "HNR1.DE", "HOT.DE", "JUN3.DE", "KRN.DE", "LEG.DE", "NDX1.DE", "NEM.DE", "PUM.DE",
        "SDF.DE", "SIX2.DE", "SZG.DE", "TEG.DE", "TLX.DE", "VNA.DE", "WAF.DE", "WCH.DE",
    ],
    "SDAX": [
        "1U1.DE", "GBF.DE", "BVB.DE", "CEC.DE", "DWNI.DE", "FIE.DE", "FRA.DE", "FNTN.DE", "GFT.DE",
        "HHFA.DE", "HLAG.DE", "HDD.DE", "IVU.DE", "JEN.DE", "JST.DE", "KCO.DE", "KWS.DE", "MLP.DE", "PBB.DE",
        "PNE3.DE", "PSM.DE", "QIA.DE", "SDF.DE", "TEG.DE", "TTK.DE", "WUW.DE", "TUI1.DE", "LXS.DE",
    ],
    "CAC40": [
        "AI.PA", "AIR.PA", "ALO.PA", "CS.PA", "BN.PA", "DSY.PA", "ENGI.PA", "EL.PA", "RMS.PA", "KER.PA",
        "OR.PA", "LR.PA", "MC.PA", "ML.PA", "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA",
        "SAN.PA", "SU.PA", "GLE.PA", "STMPA.PA", "HO.PA", "TTE.PA", "CA.PA", "BNP.PA", "ACA.PA", "VIE.PA",
        "DG.PA", "WLN.PA", "EN.PA", "FR.PA", "LI.PA", "URW.PA", "MT.AS",
    ],
    "IBEX35": [
        "ACS.MC", "ACX.MC", "AMS.MC", "ANA.MC", "BBVA.MC", "BKT.MC", "CABK.MC", "CLNX.MC", "COL.MC",
        "ELE.MC", "ENG.MC", "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC", "IDR.MC", "ITX.MC", "MAP.MC", "MEL.MC",
        "MRL.MC", "NTGY.MC", "RED.MC", "REP.MC", "SAB.MC", "SAN.MC", "TEF.MC", "VIS.MC",
    ],
    "FTSE_MIB": [
        "A2A.MI", "AMP.MI", "BAMI.MI", "BZU.MI", "ENEL.MI", "ENI.MI", "G.MI", "HER.MI", "IG.MI", "INW.MI",
        "ISP.MI", "LDO.MI", "MB.MI", "MONC.MI", "PIRC.MI", "PRY.MI", "RACE.MI", "REC.MI", "SRG.MI", "TEN.MI",
        "TIT.MI", "TRN.MI", "UCG.MI", "UNI.MI",
    ],
    "AEX_NETHERLANDS": [
        "ADYEN.AS", "AD.AS", "AGN.AS", "AKZA.AS", "MT.AS", "ASML.AS", "HEIA.AS", "HEIO.AS", "IMCD.AS",
        "INGA.AS", "PHIA.AS", "PRX.AS", "RAND.AS", "NN.AS", "WKL.AS", "UMG.AS", "BESI.AS", "LIGHT.AS",
    ],
    "SMI_SWITZERLAND": [
        "NESN.SW", "NOVN.SW", "ABBN.SW", "ADEN.SW", "ALC.SW", "CFR.SW", "GEBN.SW", "GIVN.SW", "HOLN.SW",
        "KNIN.SW", "LONN.SW", "PGHN.SW", "SGSN.SW", "SIKA.SW", "SLHN.SW", "SREN.SW", "UBSG.SW", "ZURN.SW",
        "LISN.SW", "HBAN.SW",
    ],
    "NORDIC_OMXS": [
        "VOLV-B.ST", "ERIC-B.ST", "ASSA-B.ST", "SEB-A.ST", "SHB-A.ST", "SWED-A.ST", "ATCO-A.ST", "SAND.ST",
        "SKF-B.ST", "ALFA.ST", "INVE-B.ST", "ESSITY-B.ST", "NDA-SE.ST", "LOOMIS.ST", "HEXA-B.ST",
        "NIBE-B.ST", "HUFV-A.ST", "HUSQ-B.ST", "KINV-B.ST", "SINCH.ST", "NOVO-B.CO", "CARL-B.CO",
        "MAERSK-B.CO", "ORSTED.CO", "DEMANT.CO", "COLO-B.CO", "GN.CO", "ROCK-B.CO", "EQNR", "AKRBP.OL",
        "VAR.OL", "DNO.OL", "MOWI.OL", "TEL.OL", "ORK.OL", "SALM.OL", "YAR.OL", "KNEBV.HE", "NESTE.HE",
        "FORTUM.HE", "WRT1V.HE", "OUT1V.HE",
    ],
    "EURO_STOXX_50": [
        "AD.AS", "AI.PA", "AIR.PA", "ALV.DE", "ASML.AS", "CS.PA", "BAS.DE", "BAYN.DE", "BBVA.MC", "SAN.MC",
        "BMW.DE", "BNP.PA", "BN.PA", "DHL.DE", "DTE.DE", "ENEL.MI", "ENI.MI", "EL.PA", "IBE.MC", "ITX.MC",
        "IFX.DE", "INGA.AS", "ISP.MI", "KER.PA", "KNEBV.HE", "OR.PA", "MC.PA", "MBG.DE", "MUV2.DE", "RI.PA",
        "PRX.AS", "SAF.PA", "SGO.PA", "SAN.PA", "SAP.DE", "SU.PA", "SIE.DE", "TTE.PA", "DG.PA", "VOW3.DE",
        "VNA.DE", "WKL.AS", "RACE.MI", "UCG.MI", "ENGI.PA", "VIE.PA", "ADS.DE",
    ],
    "KOSPI_KOREA": [
        "005930.KS", "000660.KS", "005380.KS", "000270.KS", "005490.KS", "051910.KS", "006400.KS",
        "035420.KS", "035720.KS", "012330.KS", "066570.KS", "003550.KS", "028260.KS", "086790.KS",
        "105560.KS", "055550.KS", "032830.KS", "017670.KS", "030200.KS", "015760.KS", "096770.KS",
        "010950.KS", "034220.KS", "009150.KS", "011200.KS", "018260.KS", "032640.KS", "011070.KS",
        "042700.KS", "247540.KS", "086280.KS",
    ],
    "NIKKEI_SELECTION": [
        "7203.T", "6758.T", "9984.T", "6861.T", "8306.T", "9432.T", "6367.T", "6501.T", "4519.T", "9433.T",
        "8035.T", "6594.T", "4543.T", "7751.T", "7267.T", "6702.T", "8411.T", "7741.T", "4063.T", "9983.T",
        "2914.T", "7974.T", "4661.T", "6954.T", "6971.T", "5108.T", "8801.T", "3382.T", "2502.T", "4901.T",
    ],
    "CANADA_TSX": [
        "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "ENB.TO", "TRP.TO", "CNQ.TO", "SU.TO", "CP.TO", "CNR.TO",
        "ATD.TO", "NTR.TO",
    ],
    "AUSTRALIA_ASX": [
        "CBA.AX", "CSL.AX", "WBC.AX", "NAB.AX", "ANZ.AX", "FMG.AX", "WDS.AX", "MQG.AX", "NST.AX", "PLS.AX",
        "PDN.AX", "BOE.AX",
    ],
    "CHINA_TECH_INTERNET": [
        "BABA", "TCEHY", "JD", "BIDU", "PDD", "NTES", "BILI", "IQ", "VIPS", "MOMO", "YMM", "FINV", "ZH",
        "TIGR", "FUTU", "LKNCY", "TUYA", "KC", "9988.HK", "0700.HK", "9618.HK", "9888.HK", "9999.HK",
        "9626.HK", "0992.HK",
    ],
    "KOREA_ADR": [
        "KEP", "KT", "SKM", "PKX",
    ],
    "ENERGY_OIL_GAS": [
        "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "MPC", "PSX", "VLO", "DVN", "FANG", "APA", "OXY", "PBR",
        "BKR", "CHRD", "MTDR", "SM", "PR", "SHEL.L", "BP.L", "TTE.PA", "EQNR", "AKRBP.OL", "VAR.OL",
        "DNO.OL", "OMV.VI", "HBR.L", "ENI.MI", "REP.MC", "0386.HK", "0857.HK", "NESTE.HE", "2222.SR",
    ],
    "UTILITIES_INFRASTRUCTURE": [
        "VIE.PA", "ENGI.PA", "IBE.MC", "ENEL.MI", "EDP.LS", "ENG.MC", "NG.L", "UU.L", "RWE.DE", "EOAN.DE",
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "AWK", "SSE.L", "SVT.L", "A2A.MI", "SRG.MI", "IG.MI",
        "FORTUM.HE", "ORSTED.CO", "RED.MC", "TRN.MI",
    ],
    "EMERGING_MARKETS": [
        "PBR", "VALE", "ITUB", "BBD", "ABEV", "EMBJ", "INFY", "WIT", "HDB", "IBN", "SIFY", "NPN.JO",
        "FSR.JO", "SBK.JO", "TKFEN.IS", "BIMAS.IS", "EREGL.IS", "AMX", "FEMSAUBD.MX", "TLK", "2222.SR",
    ],
    "HEALTHCARE_PHARMA": [
        "JNJ", "PFE", "ABBV", "LLY", "MRK", "SYK", "UNH", "NOVN.SW", "SAN.PA", "BAYN.DE", "AZN.L", "GSK.L",
        "4519.T", "4543.T", "REC.MI", "GRF.MC", "RHHBY",
    ],
    "FINANCIALS_PAYMENTS": [
        "JPM", "BAC", "V", "MA", "AXP", "BLK", "MS", "GS", "HSBA.L", "STAN.L", "LLOY.L", "BARC.L", "BNP.PA",
        "SAN.MC", "BBVA.MC", "UCG.MI", "ISP.MI", "DBK.DE", "CBK.DE", "ADYEN.AS", "8306.T", "8411.T",
        "105560.KS", "055550.KS", "0939.HK", "1398.HK", "CATY", "EWBC", "FFIN", "FNB", "FRME", "HOMB",
        "INDB", "UBSI", "UCBI", "UMBF", "WSBC", "WSFS", "WTFC", "BANF", "CBC", "CHCO", "CLBK", "BKU", "BOH",
        "PB",
    ],
    "CONSUMER_STAPLES": [
        "PG", "KO", "PEP", "WMT", "COST", "CL", "KHC", "ULVR.L", "NESN.SW", "BN.PA", "DGE.L", "HEIA.AS",
        "CARL-B.CO", "NOVO-B.CO", "3382.T", "2502.T", "TSCO.L", "MKS.L",
    ],
    "CRYPTO_FINTECH": [
        "COIN", "MSTR", "RIOT", "MARA", "CLSK", "HUT", "CIFR", "WULF", "IREN", "HOOD", "SOFI", "AFRM",
        "UPST", "XYZ", "PYPL", "ADYEN.AS",
    ],
    "EV_CLEAN_ENERGY": [
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "BLNK", "CHPT", "EVGO", "FCEL", "PLUG", "BE", "ENPH",
        "SEDG", "ARRY", "NEE", "ORSTED.CO", "IBE.MC", "EOAN.DE", "RWE.DE", "NIBE-B.ST", "0175.HK", "9868.HK",
        "2015.HK", "066570.KS", "051910.KS", "006400.KS",
    ],
    "DEFENSE_AEROSPACE": [
        "LMT", "RTX", "NOC", "GD", "BA", "HII", "LHX", "TDG", "HEI", "RHM.DE", "MTX.DE", "SAF.PA", "HO.PA",
        "BA.L", "RR.L", "LDO.MI", "ESLT", "AIR.PA",
    ],
    "LUXURY_FASHION": [
        "MC.PA", "RMS.PA", "KER.PA", "RACE.MI", "MONC.MI", "OR.PA", "EL.PA", "CFR.SW", "LISN.SW", "ADS.DE",
        "PUM.DE", "NKE", "RL", "TPR", "CPRI", "EL",
    ],
    "GOLD_SILVER": [
        "NEM", "B", "AEM", "GFI", "AU", "KGC", "AGI", "HMY", "WPM", "FNV", "RGLD", "PAAS", "HL", "AG",
        "FRES.L", "NST.AX",
    ],
    "URANIUM_NUCLEAR": [
        "CCJ", "UEC", "UUUU", "DNN", "NXE", "LEU", "OKLO", "SMR", "NNE", "BWXT", "CEG", "PDN.AX", "BOE.AX",
    ],
    "BATTERY_METALS_RARE_EARTHS": [
        "ALB", "SQM", "MP", "LAC", "PLS.AX",
    ],
    "METALS_MINING_DIVERSIFIED": [
        "RIO.L", "VALE", "GLEN.L", "AAL.L", "FCX", "NUE", "STLD", "AA", "ANTO.L", "1088.HK", "005490.KS",
        "BHP.L", "SCCO", "TECK", "MT.AS",
    ],
    "INDUSTRIAL_CYCLICALS": [
        "CAT", "DE", "MMM", "GE", "HON", "UPS", "FDX", "NSC", "UNP", "EMR", "ETN", "ITW", "CNH",
    ],
    "CHEMICALS_FERTILIZER": [
        "DOW", "LYB", "CTVA", "CF", "MOS", "YAR.OL", "BAS.DE", "SY1.DE", "EVK.DE", "LXS.DE", "WCH.DE",
        "AKZA.AS", "GIVN.SW", "SIKA.SW", "AI.PA", "SOLB.BR",
    ],
    "AUTOS": [
        "F", "GM", "STLA", "TSLA", "HMC", "P911.DE", "PAH3.DE", "RNO.PA", "RACE.MI", "VOW3.DE", "BMW.DE",
        "MBG.DE", "DTG.DE", "CON.DE", "7203.T", "7267.T", "005380.KS", "000270.KS", "PIRC.MI",
    ],
    "GENUSS_TABAK_ALKOHOL_SUESSES": [
        "PM", "MO", "BATS.L", "IMB.L", "2914.T", "UVV", "RI.PA", "BF-B", "STZ", "BUD", "HEIA.AS", "HEIO.AS",
        "CARL-B.CO", "TAP", "MDLZ", "HSY", "NESN.SW", "LISN.SW", "MNST", "CELH", "KDP", "CPR.MI", "DGE.L",
    ],
    "MEDIA_ENTERTAINMENT": [
        "DIS", "UMG.AS", "NFLX", "WPP.L", "SPOT", "EA", "TTWO", "WBD",
    ],
    "CHINA_FINANCE_ENERGY_TELECOM": [
        "0939.HK", "1398.HK", "3988.HK", "0941.HK", "0728.HK", "0762.HK", "2628.HK", "0386.HK", "0857.HK",
        "1088.HK", "2318.HK",
    ],
    "CHINA_EV_AUTO": [
        "NIO", "XPEV", "LI", "0175.HK", "9868.HK", "2015.HK", "1211.HK",
    ],
    "CHINA_CONSUMER_EDUCATION": [
        "BEKE", "YUMC", "TAL", "EDU", "GOTU", "DAO",
    ],
    "ISRAEL_TECH_DEFENSE": [
        "CHKP", "NICE", "ESLT", "WIX", "MNDY", "TEVA",
    ],
    "REAL_ESTATE_REITS_EUROPE": [
        "VNA.DE", "LEG.DE", "TEG.DE", "DWNI.DE", "LI.PA", "URW.PA", "BLND.L", "SGRO.L", "LAND.L", "SBB-B.ST",
        "COL.MC", "MRL.MC",
    ],
    "CONSTRUCTION_INFRA_EUROPE": [
        "HOT.DE", "DG.PA", "SGO.PA", "EN.PA", "BTRW.L", "PSN.L", "TW.L", "BKG.L", "ACS.MC", "FER.MC",
        "SKA-B.ST", "CRH",
    ],
    "SPECULATIVE_AI_QUANTUM": [
        "IONQ", "RXRX", "SOUN", "BBAI", "RGTI", "QBTS",
    ],
    "AI_SEMICONDUCTORS": [
        "NVDA", "AMD", "AVGO", "QCOM", "TXN", "MU", "AMAT", "KLAC", "SNPS", "CDNS", "SMCI", "VRT", "IFX.DE",
        "STMPA.PA", "ASML.AS", "BESI.AS", "005930.KS", "000660.KS", "042700.KS", "6971.T", "TSM", "ARM",
        "MRVL", "ON", "INTC", "LRCX", "ANET", "DELL", "ASM.AS", "SOI.PA", "NXPI", "LOGI",
    ],
    "SOFTWARE_CLOUD_CYBER": [
        "ADBE", "INTU", "SNOW", "PLTR", "PANW", "NET", "MDB", "OKTA", "DOCU", "ZM", "CRM", "SAP.DE",
        "DSY.PA", "NEM.DE", "NOW", "WDAY", "SHOP", "CRWD", "DDOG",
    ],
    "TELECOM": [
        "DTE.DE", "ORA.PA", "TEF.MC", "TIT.MI", "TEL.OL", "VZ", "0941.HK", "0728.HK", "0762.HK", "KT", "SKM",
        "032640.KS", "1U1.DE", "FNTN.DE", "INW.MI", "T", "TMUS", "VOD.L", "TELIA.ST", "KPN.AS", "SCMN.SW",
        "CMCSA", "CHT", "TU", "BCE",
    ],
    "TRAVEL_AIRLINES_LOGISTICS": [
        "UPS", "FDX", "BKNG", "IAG.MC", "DHL.DE", "KNIN.SW", "TUI1.DE", "HHFA.DE", "MAERSK-B.CO", "LHA.DE",
        "AF.PA", "EZJ.L", "DSV.CO", "HLAG.DE", "ABNB", "UBER", "DAL", "UAL", "MAR", "HLT", "SNDR", "KNX",
    ],
    "CASINOS_GAMING": [
        "LVS", "MGM", "WYNN", "CZR", "PENN", "BYD", "1928.HK", "0027.HK", "0880.HK", "0200.HK", "2282.HK",
    ],
    "HOTELS_LODGING": [
        "MAR", "HLT", "HST", "CHH", "DRH", "SHO", "APLE", "RHP", "PK", "RLJ", "WH", "IHG.L",
    ],
    "CRUISE_LINES": [
        "CCL", "RCL", "NCLH",
    ],
    "LEASING_TRANSPORT_EQUIPMENT": [
        "AL", "AER", "FTAI", "GATX", "TRN", "R", "URI", "CAR",
    ],
    "SHIPPING_TANKERS_BULK": [
        "DHT", "INSW", "TNK", "TRMD", "STNG", "FRO", "GOGL", "GSL", "SBLK", "ESEA", "TK",
    ],
}


# ═══════════════════════════════════════════════════════════════
# WÄHRUNGS-MAPPING nach Ticker-Suffix
# ═══════════════════════════════════════════════════════════════
CURRENCY_MAP = {
    # Euro-Raum
    ".DE": "€",  ".PA": "€",  ".MC": "€",  ".MI": "€",
    ".AS": "€",  ".VI": "€",  ".LS": "€",  ".HE": "€",
    # Britisches Pfund
    ".L":  "£",
    # Schweizer Franken
    ".SW": "CHF",
    # Skandinavien
    ".OL": "NOK", ".ST": "SEK", ".CO": "DKK",
    # Asien
    ".T":  "¥",   ".HK": "HK$", ".KS": "₩",
    # Sonstige
    ".JO": "ZAR", ".SA": "BRL", ".MX": "MXN",
    ".IS": "TRY", ".SR": "SAR",
    ".TO": "C$",  ".AX": "A$",  ".BR": "€",
}


def get_currency(ticker: str) -> str:
    """Gibt das Währungssymbol für einen Ticker zurück."""
    upper = ticker.upper()
    for suffix, symbol in CURRENCY_MAP.items():
        if upper.endswith(suffix.upper()):
            return symbol
    return "$"  # Default: US-Dollar


def get_all_tickers() -> list:
    """Gibt alle einzigartigen Ticker in fester (deterministischer) Reihenfolge zurück."""
    seen   = set()
    result = []
    for tickers in TICKER_LISTS.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                result.append(t)
    return sorted(result)


def get_tickers_by_group(group: str) -> list:
    """Gibt die Ticker einer bestimmten Gruppe zurück."""
    return TICKER_LISTS.get(group, [])


def get_all_groups() -> list:
    """Gibt alle Gruppen-Namen zurück."""
    return list(TICKER_LISTS.keys())


def get_ticker_count() -> int:
    """Gibt die Anzahl einzigartiger Ticker zurück."""
    return len(get_all_tickers())


_TICKER_TO_GROUP = None

def get_group_for_ticker(ticker: str) -> str | None:
    """
    Gibt die (erste) Gruppe zurück, in der ein Ticker geführt wird.
    Wird für die Berechnung der relativen Stärke genutzt (Ticker vs.
    Durchschnitt seiner Peer-Gruppe im selben Scan-Lauf).
    """
    global _TICKER_TO_GROUP
    if _TICKER_TO_GROUP is None:
        _TICKER_TO_GROUP = {}
        for group, tickers in TICKER_LISTS.items():
            for t in tickers:
                _TICKER_TO_GROUP.setdefault(t, group)
    return _TICKER_TO_GROUP.get(ticker)


# ─────────────────────────────────────────────────────────────────────────────
# v5.1 — Bereinigung (Basis: Scan-Läufe 21./22.10.2026), Aliase, Fokus-Hinweise, Regionen
# ─────────────────────────────────────────────────────────────────────────────
# ADR/Zweitlisting -> Heimatbörse (ein Unternehmen = ein Ticker; die Suche findet das Alias trotzdem)
ADR_ALIASES = {
    "RIO": "RIO.L",
    "BHP": "BHP.L",
    "GLNCY": "GLEN.L",
    "NSRGY": "NESN.SW",
    "GSK": "GSK.L",
    "AZN": "AZN.L",
    "UL": "ULVR.L",
    "BTI": "BATS.L",
    "DEO": "DGE.L",
    "NVO": "NOVO-B.CO",
    "AIR.DE": "AIR.PA",
    "PETR4.SA": "PBR",
    "SHELL.AS": "SHEL.L",
    "REN.AS": "REL.L",
    "AGL.JO": "AAL.L",
    "HSBC": "HSBA.L"
}

# Nicht gescannt, weil der Fehlergrund unklar ist (Yahoo liefert keine Daten). Bei Bedarf prüfen und zurück in eine Gruppe.
UNVERIFIED_TICKERS = ["ROG.SW", "BITF", "LC", "DIC.DE", "B7A.DE", "C3X.DE", "GNL.DE", "HHE.DE", "HORN.DE", "SAF.DE", "STR.DE", "VIP.DE", "VAR1.DE", "FCMA.MI"]

# Hauptfokus eines Unternehmens (Gold vs. Silber, Uran-Förderer vs. Reaktorbauer, Kupfer, Lithium …)
TICKER_FOCUS = {
    "NEM": "Gold (größter Goldproduzent)",
    "B": "Gold + Kupfer",
    "AEM": "Gold",
    "GFI": "Gold (Südafrika/Australien)",
    "AU": "Gold (AngloGold Ashanti)",
    "KGC": "Gold",
    "AGI": "Gold",
    "HMY": "Gold (Südafrika)",
    "WPM": "Streaming: Silber + Gold",
    "FNV": "Royalty/Streaming: Gold",
    "RGLD": "Royalty: Gold",
    "PAAS": "Silber (+Gold)",
    "HL": "Silber (+Gold)",
    "AG": "Silber",
    "FRES.L": "Silber + Gold (Mexiko)",
    "NST.AX": "Gold (Australien)",
    "CCJ": "Uran-Förderer + Brennstoffkreislauf",
    "UEC": "Uran-Förderer (USA)",
    "UUUU": "Uran + Seltene Erden",
    "DNN": "Uran-Entwickler (Kanada)",
    "NXE": "Uran-Entwickler (Kanada)",
    "LEU": "Urananreicherung",
    "OKLO": "Kleinreaktoren (SMR) Entwickler",
    "SMR": "Kleinreaktoren (SMR) NuScale",
    "NNE": "Mikroreaktoren",
    "BWXT": "Reaktorkomponenten",
    "CEG": "Kernkraft-Betreiber (Versorger)",
    "PDN.AX": "Uran-Förderer (Namibia)",
    "BOE.AX": "Uran (In-situ)",
    "FCX": "Kupfer (+Gold)",
    "SCCO": "Kupfer",
    "TECK": "Kupfer/Zink",
    "ANTO.L": "Kupfer",
    "RIO.L": "Eisenerz/Aluminium/Kupfer",
    "BHP.L": "Eisenerz/Kupfer/Kohle",
    "VALE": "Eisenerz/Nickel",
    "GLEN.L": "Rohstoffhandel + Kupfer/Kohle",
    "AAL.L": "Kupfer/Platin/Diamanten",
    "NUE": "Stahl",
    "STLD": "Stahl",
    "AA": "Aluminium",
    "1088.HK": "Kohle",
    "005490.KS": "Stahl (POSCO)",
    "MT.AS": "Stahl",
    "ALB": "Lithium",
    "SQM": "Lithium",
    "LAC": "Lithium",
    "PLS.AX": "Lithium",
    "MP": "Seltene Erden"
}

# Lesbare Namen der Gruppen (die Gruppen-Schlüssel bleiben technisch unverändert)
GROUP_LABELS = {
    "GOLD_SILVER": "Gold & Silber",
    "URANIUM_NUCLEAR": "Uran & Kernenergie",
    "BATTERY_METALS_RARE_EARTHS": "Lithium & Seltene Erden",
    "METALS_MINING_DIVERSIFIED": "Bergbau & Metalle (Kupfer, Eisenerz, Stahl)",
    "INDUSTRIAL_CYCLICALS": "Industrie-Zykliker",
    "CHEMICALS_FERTILIZER": "Chemie & Dünger",
    "AUTOS": "Autos",
    "GENUSS_TABAK_ALKOHOL_SUESSES": "Genuss: Tabak, Alkohol, Süßes, Getränke",
    "MEDIA_ENTERTAINMENT": "Medien & Unterhaltung",
    "AI_SEMICONDUCTORS": "KI & Halbleiter",
    "SOFTWARE_CLOUD_CYBER": "Software, Cloud, Cybersecurity",
    "TELECOM": "Telekom",
    "TRAVEL_AIRLINES_LOGISTICS": "Reise, Airlines, Logistik",
    "CANADA_TSX": "Kanada (TSX)",
    "AUSTRALIA_ASX": "Australien (ASX)",
    "ISRAEL_TECH_DEFENSE": "Israel Tech & Verteidigung",
    "CHINA_FINANCE_ENERGY_TELECOM": "China Finanzen, Energie, Telekom",
    "CHINA_EV_AUTO": "China E-Autos",
    "CHINA_CONSUMER_EDUCATION": "China Konsum & Bildung",
    "REAL_ESTATE_REITS_EUROPE": "Immobilien Europa",
    "CONSTRUCTION_INFRA_EUROPE": "Bau & Infrastruktur Europa",
    "SPECULATIVE_AI_QUANTUM": "Spekulativ: KI/Quanten",
    "ENERGY_OIL_GAS": "Öl & Gas",
    "DEFENSE_AEROSPACE": "Rüstung & Luftfahrt",
    "EV_CLEAN_ENERGY": "E-Mobilität & Saubere Energie",
    "CRYPTO_FINTECH": "Krypto & Fintech",
    "UTILITIES_INFRASTRUCTURE": "Versorger & Infrastruktur",
    "EMERGING_MARKETS": "Schwellenländer",
    "HEALTHCARE_PHARMA": "Gesundheit & Pharma",
    "FINANCIALS_PAYMENTS": "Finanzen & Zahlungen",
    "CONSUMER_STAPLES": "Basiskonsum",
    "LUXURY_FASHION": "Luxus & Mode",
    "NASDAQ_GROWTH_TECH": "Nasdaq Wachstum/Tech",
    "KOREA_ADR": "Korea (US-ADR)",
    "KOSPI_KOREA": "KOSPI Korea",
    "NIKKEI_SELECTION": "Nikkei Auswahl",
    "CASINOS_GAMING": "Casinos & Glücksspiel",
    "HOTELS_LODGING": "Hotels & Lodging",
    "CRUISE_LINES": "Kreuzfahrtlinien",
    "LEASING_TRANSPORT_EQUIPMENT": "Leasing: Flugzeuge, Züge, Container, Fahrzeuge",
    "SHIPPING_TANKERS_BULK": "Schifffahrt: Tanker & Massengut",
}


def get_focus(ticker: str) -> str:
    """Hauptfokus (z. B. 'Silber (+Gold)') oder ''."""
    return TICKER_FOCUS.get(ticker, "")


def get_group_label(group: str) -> str:
    return GROUP_LABELS.get(group, group.replace("_", " ").title())


def resolve_alias(ticker: str) -> str:
    """ADR-/Zweitlisting -> Hauptticker (sonst unverändert)."""
    return ADR_ALIASES.get(ticker.upper(), ADR_ALIASES.get(ticker, ticker))


# ─────────────────────────────────────────────────────────────────────────────
# REGIONEN (für die zwei Tages-Mails) und Mehrfach-Gruppen
# ─────────────────────────────────────────────────────────────────────────────
# Europa/Asien-Mail: alles, was nach Xetra/Euronext/LSE-Schluss + Asien/Australien-Schluss fertig ist.
# USA-Mail: US-Listings (inkl. ADRs wie PBR), Kanada (.TO) und Amerika (.SA, .MX).
_EU_ASIA_SUFFIXES = (".DE", ".PA", ".MC", ".MI", ".AS", ".VI", ".LS", ".HE", ".L", ".SW", ".OL", ".ST", ".CO",
                     ".BR", ".IS", ".JO", ".SR", ".T", ".HK", ".KS", ".AX")

REGION_LABELS = {"eu_asia": "Europa/Asien", "us": "USA/Amerika", "all": "Alle Märkte"}


def get_region(ticker: str) -> str:
    """'eu_asia' oder 'us' (Ticker ohne Börsen-Suffix = US-Listing)."""
    return "eu_asia" if ticker.upper().endswith(_EU_ASIA_SUFFIXES) else "us"


def get_tickers_by_region(region: str) -> list:
    if region == "all":
        return get_all_tickers()
    return [t for t in get_all_tickers() if get_region(t) == region]


def get_groups_for_ticker(ticker: str) -> list:
    """ALLE Gruppen, in denen ein Ticker geführt wird (Überlappungen sind gewollt: Index + Sektor + Thema)."""
    return [g for g, ts in TICKER_LISTS.items() if ticker in ts]
