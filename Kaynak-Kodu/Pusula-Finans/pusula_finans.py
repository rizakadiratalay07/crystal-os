import os
import sys, warnings, traceback
import numpy as np, pandas as pd, yfinance as yf, mplfinance as mpf, matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from PyQt6 import QtWidgets, QtCore, QtGui
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

matplotlib.use("QtAgg"); warnings.filterwarnings("ignore")
INITIAL_CAPITAL, COOLDOWN_SECONDS = 10_000.0, 20
E = 1e-9
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.764, 1.0]

def atr_sma(df, p):
    pc = df["Close"].shift(1)
    return pd.concat([(df["High"]-df["Low"]), (df["High"]-pc).abs(), (df["Low"]-pc).abs()], axis=1).max(axis=1).rolling(p, min_periods=1).mean()

def rsi(s, p):
    d = s.diff()
    return 100 - 100 / (1 + d.clip(lower=0).ewm(alpha=1/p, adjust=False, min_periods=1).mean() /
                         (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False, min_periods=1).mean().replace(0, np.nan))

def mfi(df, p):
    tp = (df["High"]+df["Low"]+df["Close"]) / 3; mf = tp * df["Volume"]; prev = tp.shift(1)
    pos = mf.where(tp > prev, 0.0).rolling(p, min_periods=1).sum()
    neg = mf.where(tp <= prev, 0.0).rolling(p, min_periods=1).sum()
    return (100*pos/(pos+neg).replace(0, np.nan)).fillna(50)

def barssince(c):
    r, cnt = np.full(len(c), np.nan), np.nan
    for i, v in enumerate(c):
        cnt = 0 if v else (np.nan if np.isnan(cnt) else cnt+1); r[i] = cnt
    return pd.Series(r, index=c.index)

def alternating_signals(buy, sell):
    bc, sc, in_pos = buy.copy(), sell.copy(), False
    for idx in buy.index:
        hb, hs = not pd.isna(buy.loc[idx]), not pd.isna(sell.loc[idx])
        if hb:
            if not in_pos: in_pos = True
            else: bc.loc[idx] = np.nan
        if hs:
            if in_pos: in_pos = False
            else: sc.loc[idx] = np.nan
    return bc, sc

def alpha_trend(df, AP=14, coeff=1.0):
    atr = atr_sma(df, AP)
    cond = (mfi(df, AP) if ("Volume" in df.columns and (df["Volume"]>0).any()) else rsi(df["Close"], AP)) >= 50
    upT, dnT = df["Low"]-atr*coeff, df["High"]+atr*coeff
    at, prev = np.full(len(df), np.nan), np.nan
    for i in range(len(df)):
        c = bool(cond.iloc[i])
        if i == 0: at[i] = upT.iloc[i] if c else dnT.iloc[i]
        elif c:    at[i] = prev if not np.isnan(prev) and upT.iloc[i] < prev else upT.iloc[i]
        else:      at[i] = prev if not np.isnan(prev) and dnT.iloc[i] > prev else dnT.iloc[i]
        prev = at[i]
    df["AlphaTrend"] = at; df["AT2"] = df["AlphaTrend"].shift(2)
    s3 = df["AlphaTrend"].shift(3)
    bx = ((df["AlphaTrend"]>df["AT2"]) & (df["AlphaTrend"].shift(1)<=s3)).fillna(False)
    sx = ((df["AlphaTrend"]<df["AT2"]) & (df["AlphaTrend"].shift(1)>=s3)).fillna(False)
    bs = bx & (barssince(bx.shift(1).fillna(False)) > barssince(sx))
    ss = sx & (barssince(sx.shift(1).fillna(False)) > barssince(bx))
    df["AT_BUY"], df["AT_SELL"] = alternating_signals(
        pd.Series(np.where(bs, df["AT2"]*0.9999, np.nan), index=df.index),
        pd.Series(np.where(ss, df["AT2"]*1.0001, np.nan), index=df.index))
    return df

def macd_rsi(df, fast=12, slow=26, sig=9, rp=14):
    s = df["Close"]; m = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    g = m.ewm(span=sig, adjust=False).mean()
    df["MACD"], df["MACD_SIG"], df["MACD_H"], df["RSI"] = m, g, m-g, rsi(s, rp)
    return df

def _linreg(series, length):
    result, arr = np.full(len(series), np.nan), series.values.astype(float)
    x = np.arange(length, dtype=float); xm = x.mean(); ss = ((x - xm) ** 2).sum()
    for i in range(length - 1, len(arr)):
        y = arr[i - length + 1:i + 1]
        if np.any(np.isnan(y)): continue
        ym = y.mean(); result[i] = ym + ((x - xm) * (y - ym)).sum() / ss * (length - 1 - xm)
    return pd.Series(result, index=series.index)

def squeeze_momentum(df, bb_len=20, bb_m=2.0, kc_len=20, kc_m=1.5):
    cl, hi, lo = df["Close"], df["High"], df["Low"]
    basis = cl.rolling(bb_len, min_periods=1).mean()
    dev = bb_m * cl.rolling(bb_len, min_periods=1).std().fillna(0)
    uBB, lBB = basis + dev, basis - dev
    ma = cl.rolling(kc_len, min_periods=1).mean()
    tr = pd.concat([(hi-lo), (hi-cl.shift(1)).abs(), (lo-cl.shift(1)).abs()], axis=1).max(axis=1)
    rng = tr.rolling(kc_len, min_periods=1).mean()
    uKC, lKC = ma + rng*kc_m, ma - rng*kc_m
    df["SQZ_ON"]  = ((lBB > lKC) & (uBB < uKC)).astype(float)
    df["SQZ_OFF"] = ((lBB < lKC) & (uBB > uKC)).astype(float)
    hh, ll = hi.rolling(kc_len, min_periods=1).max(), lo.rolling(kc_len, min_periods=1).min()
    df["SQZ_VAL"] = _linreg(cl - ((hh + ll) / 2 + ma) / 2, kc_len)
    return df

def fib_bb(df, length=200, mult=3.0):
    src = (df["High"] + df["Low"] + df["Close"]) / 3
    vo = df["Volume"].replace(0, np.nan).fillna(1)
    basis = (src * vo).rolling(length, min_periods=1).sum() / vo.rolling(length, min_periods=1).sum()
    dev = mult * src.rolling(length, min_periods=1).std().fillna(0)
    df["FBB_BASIS"] = basis
    for fl in FIB_LEVELS:
        tag = str(fl).replace(".", "")
        df[f"FBB_U{tag}"] = basis + fl * dev
        df[f"FBB_L{tag}"] = basis - fl * dev
    return df

class MLEngine:
    def __init__(self, horizon=5, threshold=0.005):
        self.horizon, self.threshold, self.model, self.scaler = horizon, threshold, None, StandardScaler()

    def _features(self, df):
        f = pd.DataFrame(index=df.index)
        cl, hi, lo = df["Close"], df["High"], df["Low"]
        vo = df["Volume"] if ("Volume" in df.columns and (df["Volume"]>0).any()) else None
        h, rs = df["MACD_H"], df["RSI"]; ret = cl.pct_change()
        f["rsi"], f["macd"], f["macd_sig"], f["macd_h"] = rs, df["MACD"], df["MACD_SIG"], h
        f["at_dir"]    = (df["AlphaTrend"] > df["AT2"]).astype(float)
        f["at_slope"]  = df["AlphaTrend"].diff()
        f["close_vs_at"] = (cl - df["AlphaTrend"]) / (df["AlphaTrend"].abs() + E)
        f["hl_ratio"]  = (hi - lo) / cl.abs().clip(lower=E)
        f["body_ratio"]= (cl - df["Open"]).abs() / (hi - lo).abs().clip(lower=E)
        for p in [2,3,5,10,20]: f[f"mom_{p}"] = cl.pct_change(p)
        for p in [5,10,20]: f[f"vol_{p}"] = ret.rolling(p).std()
        for p in [10,20,50,100]:
            sma = cl.rolling(p, min_periods=1).mean()
            f[f"above_sma{p}"]    = (cl > sma).astype(float)
            f[f"sma{p}_slope"]    = sma.pct_change(max(p//5,2))
            f[f"close_vs_sma{p}"] = (cl - sma) / (sma.abs() + E)
        for p in [9,21]:
            ema = cl.ewm(span=p, adjust=False).mean()
            f[f"close_vs_ema{p}"] = (cl - ema) / (ema.abs() + E)
        for p in [9,14]:
            lo_p, hi_p = lo.rolling(p, min_periods=1).min(), hi.rolling(p, min_periods=1).max()
            sk = 100 * (cl - lo_p) / (hi_p - lo_p).clip(lower=E)
            f[f"stoch{p}_k"], f[f"stoch{p}_d"] = sk, sk.rolling(3, min_periods=1).mean()
            f[f"stoch{p}_diff"] = sk - f[f"stoch{p}_d"]
        atr = atr_sma(df, 14)
        f["atr_ratio"], f["atr_slope"] = atr / (cl.abs() + E), atr.pct_change(5)
        if vo is not None:
            f["vol_ratio"] = vo / (vo.rolling(20, min_periods=1).mean() + E)
            f["vol_slope"] = vo.pct_change(5)
            f["obv_slope"] = (np.sign(cl.diff()) * vo).rolling(10).sum()
            vwap = (vo*cl).rolling(20).sum() / vo.rolling(20).sum().clip(lower=E)
            f["vwap_ratio"] = (cl / vwap.clip(lower=E)).clip(0.5, 2.0)
        sv = df["SQZ_VAL"]
        f["sqz_val"], f["sqz_val_pos"] = sv, (sv > 0).astype(float)
        f["sqz_val_rising"] = (sv > sv.shift(1)).astype(float)
        f["sqz_val_accel"], f["sqz_val_accel2"] = sv.diff(), sv.diff().diff()
        f["sqz_on"], f["sqz_off"], f["sqz_val_slope"] = df["SQZ_ON"], df["SQZ_OFF"], sv.diff(3)
        fb = df["FBB_BASIS"]; u6 = df["FBB_U10"]; l6 = df["FBB_L10"]
        bw = (u6 - l6).replace(0, np.nan)
        f["fbb_pct"]   = (cl - l6) / bw.fillna(1)
        f["fbb_width"] = bw / (fb.abs() + E)
        f["fbb_basis_dist"] = (cl - fb) / (fb.abs() + E)
        f["fbb_upper_dist"] = (cl - u6) / (cl.abs() + E)
        f["fbb_lower_dist"] = (cl - l6) / (cl.abs() + E)
        f["fbb_squeeze"] = bw / (bw.rolling(50, min_periods=1).mean() + E)
        u4 = df["FBB_U0618"]; l4 = df["FBB_L0618"]
        f["fbb_inner_pct"] = (cl - l4) / ((u4 - l4).replace(0, np.nan).fillna(1))
        f["fbb_above_basis"] = (cl > fb).astype(float)
        f["fbb_zone"] = pd.cut(f["fbb_pct"], bins=[-np.inf, 0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0, np.inf],
                               labels=False).astype(float)
        f["macd_h_pos"]     = (h > 0).astype(float)
        f["macd_h_rising"]  = (h > h.shift(1)).astype(float)
        f["macd_h_accel"]   = h.diff(); f["macd_h_accel2"] = h.diff().diff()
        f["macd_cross_up"]  = ((h > 0) & (h.shift(1) <= 0)).astype(float)
        f["macd_cross_down"]= ((h < 0) & (h.shift(1) >= 0)).astype(float)
        f["macd_h_vs_sma5"] = h - h.rolling(5, min_periods=1).mean()
        hc = (h > 0).astype(int)
        f["macd_h_streak"]  = hc.groupby((hc != hc.shift()).cumsum()).cumcount() * hc
        f["rsi_slope"]  = rs.diff(3)
        f["rsi_ob"]     = (rs > 70).astype(float); f["rsi_os"] = (rs < 30).astype(float)
        f["rsi_vs_sma"] = rs - rs.rolling(14, min_periods=1).mean()
        at_bull = (df["AlphaTrend"] > df["AT2"]).astype(int)
        f["at_bull_streak"] = at_bull.groupby((at_bull != at_bull.shift()).cumsum()).cumcount() * at_bull
        f["rsi_x_macdh"] = (rs/100)*h; f["at_x_macdh"] = f["at_dir"]*f["macd_h_pos"]
        f["rsi_x_at"]    = (rs/100)*f["at_dir"]
        f["sqz_x_at"]    = f["at_dir"]*f["sqz_val_pos"]; f["sqz_x_rsi"] = (rs/100)*sv
        f["fbb_x_at"]    = f["at_dir"] * f["fbb_above_basis"]
        f["fbb_x_rsi"]   = (rs/100) * f["fbb_pct"]
        for col in ["rsi","macd_h","at_dir","mom_5","vol_10","macd_h_pos","macd_h_rising",
                     "close_vs_at","sqz_val","sqz_val_pos","sqz_on","fbb_pct","fbb_above_basis"]:
            if col in f.columns:
                for lag in [1,2,3]: f[f"{col}_lag{lag}"] = f[col].shift(lag)
        return f.replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)

    def _labels(self, df):
        cl, atr, H = df["Close"], atr_sma(df, 14), self.horizon
        buy_lbl, sell_lbl = pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        ca, aa = cl.values, atr.values
        for i in range(len(ca) - H):
            e, w = ca[i], ca[i+1:i+H+1]
            thr = float(np.clip(aa[i] / (e + E), 0.003, 0.025))
            mfe = (np.max(w) - e) / (e + E); mae = (e - np.min(w)) / (e + E)
            if mfe >= thr and mfe >= mae*1.5: buy_lbl.iloc[i] = 1.0
            elif mae >= thr and mae >= mfe*1.5: sell_lbl.iloc[i] = 1.0
        labels = pd.Series(0.0, index=df.index)
        labels[buy_lbl == 1.0] = 1.0; labels[sell_lbl == 1.0] = -1.0
        return labels

    def fit_predict(self, df):
        feat = self._features(df).replace([np.inf,-np.inf], np.nan).clip(-1e9,1e9).dropna()
        y = self._labels(df).loc[feat.index].dropna(); X = feat.loc[y.index]; n = len(X)
        if n < 100: return None, {"error": "Yetersiz veri"}
        CONF = 0.50
        gb = HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.04,
                min_samples_leaf=10, l2_regularization=0.2, class_weight="balanced", random_state=42)
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=8,
                class_weight="balanced", max_features="sqrt", random_state=42, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=300, max_depth=8, min_samples_leaf=8,
                class_weight="balanced", max_features="sqrt", random_state=42, n_jobs=-1)
        X_arr, y_arr = X.values, y.values
        med = np.nanmedian(X_arr, axis=0)
        for j in range(X_arr.shape[1]):
            m = ~np.isfinite(X_arr[:,j])
            if m.any(): X_arr[m,j] = med[j]
        fold_accs, fold_f1s = [], []
        for tr_i, te_i in TimeSeriesSplit(n_splits=5, gap=self.horizon).split(X_arr):
            sc = StandardScaler()
            Xtr_s, Xte_s = sc.fit_transform(X_arr[tr_i]), sc.transform(X_arr[te_i])
            ytr, yte = y_arr[tr_i], y_arr[te_i]
            gb.fit(Xtr_s, ytr); rf.fit(Xtr_s, ytr); et.fit(Xtr_s, ytr)
            avg = (gb.predict_proba(Xte_s) + rf.predict_proba(Xte_s) + et.predict_proba(Xte_s)) / 3
            cls = gb.classes_
            preds = np.array([cls[np.argmax(r)] if np.max(r) >= CONF else 0 for r in avg])
            fold_accs.append(float((preds == yte).mean()*100))
            fold_f1s.append(float(f1_score(yte, preds, average="weighted", zero_division=0)*100))
        acc, f1_avg = float(np.mean(fold_accs)), float(np.mean(fold_f1s))
        self.scaler.fit(X_arr); Xs = self.scaler.transform(X_arr)
        gb.fit(Xs, y_arr); rf.fit(Xs, y_arr); et.fit(Xs, y_arr)
        avg_pa = (gb.predict_proba(Xs) + rf.predict_proba(Xs) + et.predict_proba(Xs)) / 3
        cls = gb.classes_
        raw = np.array([cls[np.argmax(r)] if np.max(r) >= CONF else 0 for r in avg_pa])
        confirmed = np.zeros_like(raw)
        for i in range(1, len(raw)):
            w = raw[i-1:i+1]
            if np.all(w == 1): confirmed[i] = 1
            elif np.all(w == -1): confirmed[i] = -1
        pred_s = pd.Series(confirmed, index=X.index, name="ML_SIGNAL")
        imp = (rf.feature_importances_ + et.feature_importances_) / 2
        self.model = (gb, rf, et)
        return pred_s, {"accuracy": acc, "f1_score": f1_avg,
                        "train_size": int(n*0.8), "test_size": int(n*0.2),
                        "top_features": sorted(zip(X.columns, imp), key=lambda x: -x[1])[:6],
                        "split_date": X.index[int(n*0.8)] if int(n*0.8) < n else None}

class Backtester:
    def __init__(self, initial_capital=INITIAL_CAPITAL, commission=0.0002):
        self.initial_capital, self.commission = initial_capital, commission

    def run(self, df, signals, name="Strateji", min_hold_hours=0, min_profit_pct=0.0):
        signals = signals.reindex(df.index).fillna(0)
        cap, shares, ep, in_pos = float(self.initial_capital), 0.0, 0.0, False
        trades, equity, entry_dt, ec = [], [], None, 0.0
        mh = pd.Timedelta(hours=min_hold_hours)
        for idx, row in df.iterrows():
            px, sig = float(row["Close"]), float(signals.loc[idx])
            if sig == 1 and not in_pos:
                ef = cap * self.commission; shares = (cap-ef)/px
                ec, ep, cap, in_pos, entry_dt = ef, px, 0.0, True, idx
            elif sig == -1 and in_pos and (min_hold_hours == 0 or (idx-entry_dt) >= mh) and (px-ep)/ep >= min_profit_pct:
                pr = shares*px; xf = pr*self.commission; tc = ec+xf
                trades.append({"entry_price": ep, "exit_price": px, "exit_date": idx,
                                "pnl": pr-xf-shares*ep, "return": (px-ep)/ep,
                                "entry_commission": ec, "exit_commission": xf, "commission": tc})
                cap, shares, in_pos, ec = pr-xf, 0.0, False, 0.0
            equity.append(cap + shares*px)
        if in_pos and shares > 0:
            px = float(df["Close"].iloc[-1]); pr = shares*px; xf = pr*self.commission; tc = ec+xf
            trades.append({"entry_price": ep, "exit_price": px, "exit_date": df.index[-1],
                           "pnl": pr-xf-shares*ep, "return": (px-ep)/ep,
                           "entry_commission": ec, "exit_commission": xf, "commission": tc})
            cap = pr-xf
            if equity: equity[-1] = cap
        eq = pd.Series(equity, index=df.index)
        dd = (eq - eq.expanding().max()) / eq.expanding().max() * 100
        dr = eq.pct_change().dropna()
        sharpe = float(dr.mean()/dr.std()*np.sqrt(252)) if dr.std() > 0 else 0.0
        tc = sum(t["commission"] for t in trades)
        if trades:
            td = pd.DataFrame(trades)
            wr, ar = float((td["return"]>0).mean()*100), float(td["return"].mean()*100)
            bt, wt, tp = float(td["return"].max()*100), float(td["return"].min()*100), float(td["pnl"].sum())
        else: wr = ar = bt = wt = tp = 0.0
        return {"name": name, "total_return": (float(eq.iloc[-1])-self.initial_capital)/self.initial_capital*100,
                "bh_return": (float(df["Close"].iloc[-1])-float(df["Close"].iloc[0]))/float(df["Close"].iloc[0])*100,
                "max_drawdown": float(dd.min()), "sharpe": sharpe, "num_trades": len(trades),
                "win_rate": wr, "avg_return": ar, "best_trade": bt, "worst_trade": wt,
                "total_pnl": tp, "total_commission": tc, "equity_curve": eq, "trades": trades}

def _tbl_style(tbl):
    tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
    tbl.verticalHeader().setVisible(False); tbl.setAlternatingRowColors(True)
    tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    tbl.setStyleSheet("QTableWidget{gridline-color:#2a2a4a;}QHeaderView::section{background:#ff9800;padding:6px;}")

class BacktestDialog(QtWidgets.QDialog):
    def __init__(self, at_r, ml_r, ml_info, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Backtest & ML Analiz Sonuçları"); self.resize(1050, 680)
        main = QtWidgets.QVBoxLayout(self); main.setContentsMargins(10,10,10,10)
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet("QTabBar::tab{padding:8px 18px;}QTabBar::tab:selected{color:#ff9800;font-weight:bold;}")
        main.addWidget(tabs)
        for title, fn, args in [("Performans Karşılaştırma", self._metrics, (at_r, ml_r, ml_info)),
                                ("Equity Eğrisi", self._equity, (at_r, ml_r, df)),
                                ("İşlem Geçmişi (ML)", self._trades, (ml_r,))]:
            w = QtWidgets.QWidget(); tabs.addTab(w, title); fn(w, *args)

    def _metrics(self, w, at_r, ml_r, info):
        vl = QtWidgets.QVBoxLayout(w); vl.setSpacing(8)
        if info and "accuracy" in info:
            sd = info.get("split_date"); sd_s = pd.Timestamp(sd).strftime("%d.%m.%Y") if sd else "?"
            lbl = QtWidgets.QLabel(f"Model: GradientBoosting  │  Eğitim: {info.get('train_size',0)} bar  │  "
                                   f"Test: {info.get('test_size',0)} bar  │  Ayrım: {sd_s}")
            lbl.setTextFormat(QtCore.Qt.TextFormat.RichText); vl.addWidget(lbl)
        ml_only = {"f1_score","accuracy"}
        rows = [("Toplam Getiri (%)","total_return",True),("Al & Tut Getiri (%)","bh_return",True),
                ("İşlem Sayısı","num_trades",None),("Kazanma Oranı (%)","win_rate",True),
                ("Maks Drawdown (%)","max_drawdown",False),("Sharpe","sharpe",True),
                ("Ort İşlem Getiri (%)","avg_return",True),("En İyi İşlem (%)","best_trade",True),
                ("En Kötü İşlem (%)","worst_trade",False),("Toplam K/Z","total_pnl",True),
                ("Toplam Komisyon (₺)","total_commission",None),
                ("OOS Doğruluk (%)","accuracy",True),("F1 (%)","f1_score",True)]
        cols = ["Metrik","AlphaTrend"] + (["ML"] if ml_r else [])
        tbl = QtWidgets.QTableWidget(len(rows), len(cols)); tbl.setHorizontalHeaderLabels(cols); _tbl_style(tbl)
        def mi(val, hb):
            it = QtWidgets.QTableWidgetItem(f"{val:.2f}" if isinstance(val, float) else str(val))
            if hb is not None: it.setForeground(QtGui.QColor("#4caf50" if (val > 0 if hb else val > -5) else "#f44336"))
            return it
        for r, (label, key, hb) in enumerate(rows):
            tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(label))
            mo = key in ml_only
            if mo:
                d = QtWidgets.QTableWidgetItem("—"); d.setForeground(QtGui.QColor("#666")); tbl.setItem(r, 1, d)
            else: tbl.setItem(r, 1, mi(at_r.get(key, 0) if at_r else 0, hb))
            if ml_r:
                mv = float(info.get(key, 0)) if mo and info else ml_r.get(key, 0)
                ac = 0 if mo else (at_r.get(key, 0) if at_r else 0)
                it = mi(mv, hb)
                if hb is True and mv > ac: ft = it.font(); ft.setBold(True); it.setFont(ft)
                tbl.setItem(r, 2, it)
        vl.addWidget(tbl)
        if info and info.get("top_features"):
            l2 = QtWidgets.QLabel("En Önemli Özellikler:")
            l2.setStyleSheet("font-weight:bold;margin-top:6px;color:#ff9800;"); vl.addWidget(l2)
            vl.addWidget(QtWidgets.QLabel("  │  ".join(f"{k}: {v:.3f}" for k, v in info["top_features"])))
        if "error" in info: vl.addWidget(QtWidgets.QLabel(info["error"]))

    def _equity(self, w, at_r, ml_r, df):
        vl = QtWidgets.QVBoxLayout(w); vl.setContentsMargins(0,0,0,0)
        fig = Figure(figsize=(10,4.5), dpi=100); fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111); ax.set_facecolor("white")
        bh = df["Close"] / float(df["Close"].iloc[0]) * INITIAL_CAPITAL
        ax.plot(bh.index, bh.values, color="#888", lw=1.2, ls="--", label="Al & Tut", alpha=0.7)
        if at_r: ax.plot(at_r["equity_curve"].index, at_r["equity_curve"].values, color="#2196F3", lw=1.8, label="AlphaTrend")
        if ml_r: ax.plot(ml_r["equity_curve"].index, ml_r["equity_curve"].values, color="#FF9800", lw=1.8, label="ML")
        ax.axhline(INITIAL_CAPITAL, color="#555", lw=0.7, ls=":")
        ax.fill_between(bh.index, INITIAL_CAPITAL, bh.values, where=bh.values < INITIAL_CAPITAL, alpha=0.08, color="red")
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.tick_params(colors="black", labelsize=9)
        for o in (ax.xaxis.label, ax.yaxis.label, ax.title): o.set_color("black")
        ax.set_title(f"Equity Eğrisi (₺{INITIAL_CAPITAL:,.0f})", color="black", fontsize=11)
        ax.legend(facecolor="white", labelcolor="black", edgecolor="#aaa", framealpha=0.9)
        ax.grid(True, alpha=0.25, color="#aaa"); fig.tight_layout(pad=1.5)
        vl.addWidget(FigureCanvas(fig))

    def _trades(self, w, result):
        vl = QtWidgets.QVBoxLayout(w)
        if not result or not result.get("trades"): vl.addWidget(QtWidgets.QLabel("İşlem bulunamadı.")); return
        trades = result["trades"]
        tbl = QtWidgets.QTableWidget(len(trades), 6)
        tbl.setHorizontalHeaderLabels(["#","Giriş","Çıkış","Getiri (%)","K/Z (₺)","Komisyon (₺)"]); _tbl_style(tbl)
        wins, tc = 0, 0.0
        for i, t in enumerate(trades):
            win = t["return"] > 0
            if win: wins += 1
            clr = QtGui.QColor("#4caf50" if win else "#f44336"); cm = t.get("commission", 0.0); tc += cm
            for c, v in enumerate([str(i+1), f"{t['entry_price']:.2f}", f"{t['exit_price']:.2f}"]):
                tbl.setItem(i, c, QtWidgets.QTableWidgetItem(v))
            for c, v in [(3, f"{t['return']*100:.2f}%"), (4, f"₺{t['pnl']:.2f}")]:
                it = QtWidgets.QTableWidgetItem(v); it.setForeground(clr); tbl.setItem(i, c, it)
            ci = QtWidgets.QTableWidgetItem(f"₺{cm:.2f}"); ci.setForeground(QtGui.QColor("#ff9800")); tbl.setItem(i, 5, ci)
        vl.addWidget(tbl)
        s = QtWidgets.QLabel(f"{len(trades)} işlem  │  {wins} kazanan  │  {len(trades)-wins} kaybeden  │  Komisyon: ₺{tc:.2f}")
        s.setStyleSheet("padding:6px;color:#ff9800;font-weight:bold;"); vl.addWidget(s)

class Canvas(FigureCanvas):
    _L, _R, _B, _T, _VH = 0.01, 0.95, 0.06, 0.98, 0.22
    def __init__(self, fig, parent=None):
        super().__init__(fig)
        if parent: self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    def resizeEvent(self, ev):
        w, h = ev.size().width(), ev.size().height(); dpi = float(getattr(self.figure, "dpi", 100.0))
        if w > 0 and h > 0 and dpi > 0:
            try: self.figure.set_size_inches(w/dpi, h/dpi, forward=True)
            except TypeError: self.figure.set_size_inches(w/dpi, h/dpi)
            ax = list(self.figure.axes)
            if ax:
                ax[0].set_position([self._L, self._B+self._VH+0.01, self._R-self._L, max((self._T-self._B)-self._VH-0.01, 0.05)])
                if len(ax) > 1: ax[-1].set_position([self._L, self._B, self._R-self._L, self._VH])
        super().resizeEvent(ev); self.draw_idle()

class Worker(QtCore.QThread):
    ready = QtCore.pyqtSignal(object, float); error = QtCore.pyqtSignal(str)
    def __init__(self, sym, tf, currency="TRY"):
        super().__init__(); self.sym, self.tf, self.currency = sym, tf, currency
    def _fetch_usdtry(self):
        try:
            fx = yf.download("USDTRY=X", period="5d", interval="1d", progress=False, threads=False, auto_adjust=False)
            if fx is not None and not fx.empty:
                if isinstance(fx.columns, pd.MultiIndex): fx.columns = fx.columns.get_level_values(0)
                return float(fx["Close"].dropna().iloc[-1])
        except Exception: pass
        return 1.0
    @staticmethod
    def _fix(df):
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index); return df

    @staticmethod
    def _resample(df, rule):
        return df.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

    def run(self):
        try:
            kw = dict(auto_adjust=False, progress=False, threads=False)
            fx = self._fetch_usdtry() if self.currency == "USD" else 1.0
            raw = yf.download(self.sym, interval="1d", period="5y", **kw)
            if raw is None or raw.empty: self.error.emit("Veri yok: " + self.sym); return
            raw = self._fix(raw).sort_index()
            if self.tf == "2D":
                raw = self._resample(raw, "2D")
            elif self.tf == "1W":
                raw = self._resample(raw, "W")
            self.ready.emit(raw, fx)
        except Exception as e:
            self.error.emit(str(e))

class MLWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(object, object, object, object); error = QtCore.pyqtSignal(str)

    def __init__(self, df, tf="1D"):
        super().__init__(); self._df, self._tf = df.copy(), tf

    def run(self):
        try:
            df = self._df; at_sig = pd.Series(0.0, index=df.index)
            at_sig[df["AT_BUY"].notna()] = 1.0; at_sig[df["AT_SELL"].notna()] = -1.0
            pred, info = MLEngine(horizon=5, threshold=0.005).fit_predict(df)
            bt = Backtester(initial_capital=INITIAL_CAPITAL, commission=0.0002)
            at_r = bt.run(df, at_sig, "AlphaTrend")
            ml_r = bt.run(df, pred.reindex(df.index).fillna(0) if pred is not None else at_sig,
                          "ML", min_hold_hours=4, min_profit_pct=0.02)
            if pred is not None:
                p = pred.reindex(df.index, fill_value=0)
                rb = pd.Series(np.where((p==1)&(p.shift(1)!=1), df["Close"]*0.986, np.nan), index=df.index)
                rs = pd.Series(np.where((p==-1)&(p.shift(1)!=-1), df["Close"]*1.014, np.nan), index=df.index)
                df["ML_BUY"], df["ML_SELL"] = alternating_signals(rb, rs)
            else: df["ML_BUY"] = df["ML_SELL"] = np.nan
            self.done.emit(df, pred, info, (at_r, ml_r))
        except Exception as e:
            print(f"\n[ML HATA] {e}", file=sys.stderr); print(traceback.format_exc(), file=sys.stderr)
            self.error.emit(str(e))

class DisclaimerDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Uyarı")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(
            "Pusula Finans programında yer alan tüm göstergeler, analizler ve tahminler yalnızca eğitim ve bilgilendirme amaçlıdır. "
            "Bunlar hiçbir şekilde yatırım tavsiyesi niteliği taşımaz. Programın kullanımından doğabilecek yatırım kararları ve sonuçları tamamen kullanıcıya aittir."
            "<b>&quot;Kabul&nbsp;Ediyorum&quot;</b> düğmesine basarak bu koşulları okuduğunuzu, anladığınızı ve kabul ettiğinizi beyan etmiş olursunuz. "
        )
        label.setWordWrap(True)
        label.setFixedWidth(360)
        layout.addWidget(label)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_reject = QtWidgets.QPushButton("Kabul Etmiyorum")
        btn_reject.clicked.connect(self.reject)
        btn_row.addWidget(btn_reject)
        btn_accept = QtWidgets.QPushButton("Kabul Ediyorum")
        btn_accept.clicked.connect(self.accept)
        btn_row.addWidget(btn_accept)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        self.adjustSize()
        self.setFixedSize(self.sizeHint())

class App(QtWidgets.QMainWindow):
    _STEP_VALUES = [1, 4, 8, 16]

    def __init__(self):
        super().__init__(); self.setWindowTitle("Pusula Finans V1.3"); self.resize(1000, 680)
        self._at_result = self._ml_result = self._df = self._ml_worker = None
        self._ml_info = {}; self._cooldown_remaining = 0
        self._view_offset = 0
        self._step_index = 0
        self._cooldown_timer = QtCore.QTimer(self); self._cooldown_timer.setInterval(1000)
        self._cooldown_timer.timeout.connect(self._tick_cooldown)

        c = QtWidgets.QWidget(); c.setContentsMargins(0,0,0,0); self.setCentralWidget(c)
        v = QtWidgets.QVBoxLayout(c); v.setContentsMargins(0,0,0,0); v.setSpacing(0)
        h = QtWidgets.QHBoxLayout(); h.setContentsMargins(6,6,6,6); v.addLayout(h)

        def add(w, sp=8): h.addWidget(w); (h.addSpacing(sp) if sp else None)

        add(QtWidgets.QLabel("Hisse:"), 0)
        self.sym = QtWidgets.QLineEdit("XU100"); self.sym.setFixedWidth(160); add(self.sym)

        add(QtWidgets.QLabel("Periyot:"), 0)
        self.tf = QtWidgets.QComboBox()
        for n, d in [("1 Gün", "1D"), ("2 Gün", "2D"), ("1 Hafta", "1W")]:
            self.tf.addItem(n, d)
        add(self.tf)

        add(QtWidgets.QLabel("Grafik Türü:"), 0)
        self.chart_type = QtWidgets.QComboBox()
        self.chart_type.addItem("Klasik Mum", "candle")
        self.chart_type.addItem("Heiken Ashi", "heikinashi")
        self.chart_type.currentIndexChanged.connect(self._on_chart_type_changed)
        add(self.chart_type)

        add(QtWidgets.QLabel("Para Birimi:"), 0)
        self.ccy = QtWidgets.QComboBox(); self.ccy.addItem("TRY","TRY"); self.ccy.addItem("USD","USD"); add(self.ccy, 8)

        self.btn_prev = QtWidgets.QPushButton("<"); self.btn_prev.setFixedWidth(28)
        self.btn_prev.setToolTip("Geri sar"); self.btn_prev.clicked.connect(self._step_left); add(self.btn_prev, 2)

        self.btn_step = QtWidgets.QPushButton(str(self._STEP_VALUES[self._step_index]))
        self.btn_step.setFixedWidth(32)
        self.btn_step.setToolTip("Adım değeri (tıklayarak değiştir: 1 → 4 → 8 → 16)")
        self.btn_step.clicked.connect(self._cycle_step)
        add(self.btn_step, 2)

        self.btn_next = QtWidgets.QPushButton(">"); self.btn_next.setFixedWidth(28)
        self.btn_next.setToolTip("İleri sar"); self.btn_next.clicked.connect(self._step_right); add(self.btn_next)

        self.btn = QtWidgets.QPushButton("Yükle"); self.btn.setFixedWidth(90); self.btn.clicked.connect(self.load); add(self.btn)
        self.bt_btn = QtWidgets.QPushButton("Backtest-ML Raporu"); self.bt_btn.setEnabled(False)
        self.bt_btn.clicked.connect(self.show_backtest); add(self.bt_btn)

        self.cooldown_lbl = QtWidgets.QLabel(""); self.cooldown_lbl.setFixedWidth(150)
        self.cooldown_lbl.setStyleSheet("color:#ff9800;font-weight:bold;padding:0 4px;")
        add(self.cooldown_lbl, 0); h.addStretch()

        self.progress = QtWidgets.QLabel(""); self.progress.setStyleSheet("color:#ff9800;font-weight:bold;padding:0 8px;")
        h.addWidget(self.progress)

        self.pw = QtWidgets.QWidget(); self.pw.setContentsMargins(0,0,0,0)
        self.pl = QtWidgets.QVBoxLayout(self.pw); self.pl.setContentsMargins(0,0,0,0); self.pl.setSpacing(0)
        v.addWidget(self.pw, 1); self.canvas = self.toolbar = self.worker = self.axes = None

    def _cycle_step(self):
        self._step_index = (self._step_index + 1) % len(self._STEP_VALUES)
        val = self._STEP_VALUES[self._step_index]
        self.btn_step.setText(str(val))
        self.btn_prev.setToolTip(f"{val} mum geri")
        self.btn_next.setToolTip(f"{val} mum ileri")

    def _current_step_size(self):
        return self._STEP_VALUES[self._step_index]

    def _current_view_df(self):
        if self._df is None: return None
        n = len(self._df)
        end = max(n - self._view_offset, 1)
        return self._df.iloc[:end]

    def _capture_view_limits(self):
        if not self.axes: return None
        ax_list = self.axes if isinstance(self.axes, (list, tuple)) else [self.axes]
        flat = [a for x in ax_list for a in (x if isinstance(x, (list, tuple)) else [x])]
        try:
            return [(a.get_xlim(), a.get_ylim()) for a in flat]
        except Exception:
            return None

    def _step_left(self):
        if self._df is None: return
        max_offset = len(self._df) - 1
        step = self._current_step_size()
        if self._view_offset >= max_offset: return
        limits = self._capture_view_limits()
        self._view_offset = min(self._view_offset + step, max_offset)
        self._render(self._current_view_df(), view_limits=limits)

    def _step_right(self):
        if self._df is None or self._view_offset <= 0: return
        step = self._current_step_size()
        limits = self._capture_view_limits()
        self._view_offset = max(self._view_offset - step, 0)
        self._render(self._current_view_df(), view_limits=limits)

    @staticmethod
    def _to_heiken_ashi(df):
        ha = df.copy()
        ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
        ha_open_vals = np.empty(len(df), dtype=float)
        ha_open_vals[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
        ha_close_vals = ha_close.values
        for i in range(1, len(df)):
            ha_open_vals[i] = (ha_open_vals[i-1] + ha_close_vals[i-1]) / 2
        ha_open = pd.Series(ha_open_vals, index=df.index)
        ha["Close"] = ha_close
        ha["Open"]  = ha_open
        ha["High"]  = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
        ha["Low"]   = pd.concat([df["Low"],  ha_open, ha_close], axis=1).min(axis=1)
        return ha

    def _on_chart_type_changed(self):
        if self._df is not None:
            self._render(self._current_view_df())

    def _start_cooldown(self):
        self._cooldown_remaining = COOLDOWN_SECONDS; self.btn.setEnabled(False)
        self.cooldown_lbl.setText(f"⏳ Yükle: {self._cooldown_remaining} sn"); self._cooldown_timer.start()

    def _tick_cooldown(self):
        self._cooldown_remaining -= 1
        if self._cooldown_remaining <= 0:
            self._cooldown_timer.stop(); self._cooldown_remaining = 0
            self.btn.setEnabled(True); self.cooldown_lbl.setText("")
        else: self.cooldown_lbl.setText(f"⏳ Yükle: {self._cooldown_remaining} sn")

    def _sym(self):
        s = self.sym.text().strip().upper(); return s if s.endswith(".IS") else s+".IS"

    @staticmethod
    def _update_sqz_colors(df):
        vals = df["SQZ_VAL"].values
        prev_val = np.nan
        bcolors = []
        for i, v in enumerate(vals):
            if np.isnan(v) or np.isinf(v):
                bcolors.append('gray')
            else:
                if v > 0:
                    if not np.isnan(prev_val) and v > prev_val:
                        bcolors.append('#00FF00')
                    else:
                        bcolors.append('#008000')
                elif v < 0:
                    if not np.isnan(prev_val) and v < prev_val:
                        bcolors.append('#FF0000')
                    else:
                        bcolors.append('#800000')
                else:
                    bcolors.append('gray')
            if not np.isnan(v) and not np.isinf(v):
                prev_val = v
        return bcolors

    def load(self):
        s = self._sym()
        if not s: QtWidgets.QMessageBox.warning(self, "Uyarı", "Sembol girin."); return
        self.sym.setText(s); self.btn.setEnabled(False); self.cooldown_lbl.setText("Veri çekiliyor...")
        self.bt_btn.setEnabled(False); self.progress.setText("")
        if self.worker and self.worker.isRunning(): self.worker.terminate()
        self.worker = Worker(s, self.tf.currentData(), self.ccy.currentData())
        self.worker.ready.connect(self.on_data); self.worker.error.connect(self.on_err); self.worker.start()

    def on_err(self, m): print(f"[VERİ HATA] {m}", file=sys.stderr); self._start_cooldown()

    def on_data(self, df, fx_rate):
        if not {"Open","High","Low","Close"}.issubset(df.columns):
            QtWidgets.QMessageBox.critical(self, "Hata", f"Kolonlar eksik: {list(df.columns)}")
            self._start_cooldown(); return

        years = 5
        df = df.loc[df.index >= df.index.max() - pd.DateOffset(years=years)].dropna(
            subset=["Open", "High", "Low", "Close"])

        if self.ccy.currentData() == "USD" and fx_rate > 1.0:
            for col in ["Open","High","Low","Close"]: df[col] = df[col]/fx_rate

        df = fib_bb(squeeze_momentum(macd_rsi(alpha_trend(df)))); df["ML_BUY"] = df["ML_SELL"] = np.nan
        self._df = df; self._view_offset = 0; self._render(df); self._start_cooldown()
        self.progress.setText("ML eğitiliyor...")
        if self._ml_worker and self._ml_worker.isRunning(): self._ml_worker.terminate()
        self._ml_worker = MLWorker(df, tf="1D")
        self._ml_worker.done.connect(self._on_ml_done); self._ml_worker.error.connect(self._on_ml_err)
        self._ml_worker.start()

    def _on_ml_err(self, m): self.progress.setText(f"ML Hata: {m}"); print(f"[ML HATA] {m}", file=sys.stderr)

    def _on_ml_done(self, df, pred, info, results):
        self._at_result, self._ml_result = results; self._ml_info = info; self._df = df
        a, ta, tm = info.get("accuracy",0), (self._at_result or {}).get("total_return",0), (self._ml_result or {}).get("total_return",0)
        self.progress.setText(f"ML Doğruluk: {a:.1f}%  │  AT: {ta:+.1f}%  │  ML: {tm:+.1f}%")
        self.bt_btn.setEnabled(True)
        self._render(self._current_view_df())

    def show_backtest(self):
        if self._df is None: return
        BacktestDialog(self._at_result or {}, self._ml_result, self._ml_info or {}, self._df, parent=self).exec()

    def _clear(self):
        for w in (self.toolbar, self.canvas):
            if w: self.pl.removeWidget(w); w.setParent(None); w.deleteLater()
        self.toolbar = self.canvas = self.axes = None

    def _render(self, df, view_limits=None):
        self._clear()
        chart_t = self.chart_type.currentData()
        plot_df = self._to_heiken_ashi(df) if chart_t == "heikinashi" else df
        sqz_colors = self._update_sqz_colors(plot_df)
        apds = [mpf.make_addplot(plot_df["AlphaTrend"], type="line", width=1.5, panel=0),
                mpf.make_addplot(plot_df["AT2"], type="line", width=1.5, panel=0, linestyle="dashdot"),
                mpf.make_addplot(plot_df["FBB_BASIS"], type="line", panel=0, width=1.8, color="#FF00FF"),
                mpf.make_addplot(plot_df["FBB_U10"],   type="line", panel=0, width=1.5, color="#F23645"),
                mpf.make_addplot(plot_df["FBB_L10"],   type="line", panel=0, width=1.5, color="#089981"),
                mpf.make_addplot(plot_df["FBB_U0764"], type="line", panel=0, width=0.7, color="#787B86", linestyle="solid"),
                mpf.make_addplot(plot_df["FBB_L0764"], type="line", panel=0, width=0.7, color="#787B86", linestyle="solid"),
                mpf.make_addplot(plot_df["FBB_U0618"], type="line", panel=0, width=0.7, color="#787B86", linestyle="solid"),
                mpf.make_addplot(plot_df["FBB_L0618"], type="line", panel=0, width=0.7, color="#787B86", linestyle="solid"),
                mpf.make_addplot(plot_df["FBB_U05"],   type="line", panel=0, width=0.7, color="#787B86", linestyle="solid"),
                mpf.make_addplot(plot_df["FBB_L05"],   type="line", panel=0, width=0.7, color="#787B86", linestyle="solid"),
                mpf.make_addplot(plot_df["FBB_U0382"], type="line", panel=0, width=0.7, color="#787B86", linestyle="solid"),
                mpf.make_addplot(plot_df["FBB_L0382"], type="line", panel=0, width=0.7, color="#787B86", linestyle="solid"),
                mpf.make_addplot(plot_df["FBB_U0236"], type="line", panel=0, width=0.7, color="#787B86", linestyle="solid"),
                mpf.make_addplot(plot_df["FBB_L0236"], type="line", panel=0, width=0.7, color="#787B86", linestyle="solid"),
                mpf.make_addplot(plot_df["SQZ_VAL"], type="bar", panel=1, width=0.7, color=sqz_colors)]
        for col, mk, sz, cl in [("AT_BUY","^",60,"#00e676"),("AT_SELL","v",60,"#ff1744"),
                                  ("ML_BUY","^",90,"#2979ff"),("ML_SELL","v",90,"#ff6d00")]:
            if plot_df[col].notna().any():
                apds.append(mpf.make_addplot(plot_df[col], type="scatter", markersize=sz, marker=mk, panel=0, color=cl))
        st = mpf.make_mpf_style(base_mpf_style="charles",
            marketcolors=mpf.make_marketcolors(up="green", down="red", edge="inherit", wick="inherit", volume="in"))
        fig, axes = mpf.plot(plot_df, type="candle", volume=False, style=st, addplot=apds, returnfig=True, figsize=(12,6))
        flat = [a for x in (axes if isinstance(axes,(list,tuple)) else [axes]) for a in (x if isinstance(x,(list,tuple)) else [x])]
        L, R, B, T, VH = 0.01, 0.95, 0.03, 0.98, 0.22
        if flat:
            flat[0].set_position([L, B+VH+0.01, R-L, max((T-B)-VH-0.01, 0.05)])
            for ax in flat:
                ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
                ax.tick_params(axis="y", which="both", right=True, labelright=True, left=False, labelleft=False)
            if len(flat) > 1:
                flat[-1].set_position([L, B, R-L, VH])
                flat[-1].tick_params(axis="y", which="both", right=False, labelright=False, left=False, labelleft=False)
            flat[0].fill_between(range(len(plot_df)), plot_df["FBB_U10"].values, plot_df["FBB_L10"].values,
                                 alpha=0.03, color="#2196F3", zorder=0)
        self.canvas = Canvas(fig, parent=self.pw); self.pl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self); self.pl.insertWidget(0, self.toolbar)
        self.axes = axes; self.canvas.mpl_connect("scroll_event", self._scroll)
        if view_limits and len(view_limits) == len(flat):
            for ax, (xl, yl) in zip(flat, view_limits):
                try:
                    ax.set_xlim(xl); ax.set_ylim(yl)
                except Exception: pass
        self.canvas.draw()

    def _scroll(self, ev):
        if ev.inaxes is None or ev.xdata is None: return
        try:
            l, r = ev.inaxes.get_xlim(); s = 1/1.2 if ev.button == "up" else 1.2
            nl, nr = ev.xdata-(ev.xdata-l)*s, ev.xdata+(r-ev.xdata)*s
            for a in (self.axes if isinstance(self.axes,(list,tuple)) else [self.axes]):
                try: a.set_xlim(nl, nr)
                except Exception: pass
            if self.canvas: self.canvas.draw_idle()
        except Exception: pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    settings = QtCore.QSettings("PusulaFinans", "PusulaFinansV1")
    if not settings.value("disclaimer_accepted", False, type=bool):
        dlg = DisclaimerDialog()
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            settings.setValue("disclaimer_accepted", True)
        else:
            sys.exit(0)
    win = App(); win.show(); sys.exit(app.exec())
