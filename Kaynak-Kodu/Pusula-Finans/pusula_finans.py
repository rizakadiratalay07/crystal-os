import os,sys,warnings,traceback,numpy as np,pandas as pd,yfinance as yf,mplfinance as mpf,matplotlib,subprocess
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas,NavigationToolbar2QT as NavigationToolbar
from PyQt6 import QtWidgets,QtCore,QtGui
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

# Programcı: Rıza Kadir ATALAY, Ali Emre ERYILMAZ
# Grafiği Paylaşmaya yarayan API uygulamasını yaptığı için Ali Emre ERYILMAZ'a teşekkürler.

matplotlib.use("QtAgg")
matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.weight":"normal","axes.titleweight":"normal","axes.labelweight":"normal","figure.titleweight":"normal","legend.fontsize":9,"path.simplify":True,"path.simplify_threshold":0.1,"agg.path.chunksize":10000,"figure.dpi":100})
CHART_DPI=100
warnings.filterwarnings("ignore")
INITIAL_CAPITAL,COOLDOWN_SECONDS=10_000.0,20
E=1e-9

def alternating_signals(buy,sell):
    bc,sc,in_pos=buy.copy(),sell.copy(),False
    for idx in buy.index:
        hb,hs=not pd.isna(buy.loc[idx]),not pd.isna(sell.loc[idx])
        if hb:
            if not in_pos:in_pos=True
            else:bc.loc[idx]=np.nan
        if hs:
            if in_pos:in_pos=False
            else:sc.loc[idx]=np.nan
    return bc,sc

def _linreg(series,length):
    result=np.full(len(series),np.nan); arr=series.values.astype(float); x=np.arange(length,dtype=float); xm=x.mean(); ss=((x-xm)**2).sum()
    for i in range(length-1,len(arr)):
        y=arr[i-length+1:i+1]
        if np.any(np.isnan(y)):continue
        ym=y.mean(); result[i]=ym+((x-xm)*(y-ym)).sum()/ss*(length-1-xm)
    return pd.Series(result,index=series.index)

def squeeze_momentum(df,bb_len=20,bb_m=2.0,kc_len=20,kc_m=1.5):
    cl,hi,lo=df["Close"],df["High"],df["Low"]
    basis=cl.rolling(bb_len,min_periods=1).mean(); dev=bb_m*cl.rolling(bb_len,min_periods=1).std().fillna(0); uBB,lBB=basis+dev,basis-dev
    ma=cl.rolling(kc_len,min_periods=1).mean(); tr=pd.concat([hi-lo,(hi-cl.shift(1)).abs(),(lo-cl.shift(1)).abs()],axis=1).max(axis=1); rng=tr.rolling(kc_len,min_periods=1).mean(); uKC,lKC=ma+rng*kc_m,ma-rng*kc_m
    df["SQZ_ON"]=((lBB>lKC)&(uBB<uKC)).astype(float); df["SQZ_OFF"]=((lBB<lKC)&(uBB>uKC)).astype(float)
    hh=hi.rolling(kc_len,min_periods=1).max(); ll=lo.rolling(kc_len,min_periods=1).min(); df["SQZ_VAL"]=_linreg(cl-((hh+ll)/2+ma)/2,kc_len)
    return df

def donchian(df,length=20):
    upper=df["High"].rolling(length,min_periods=1).max(); lower=df["Low"].rolling(length,min_periods=1).min(); basis=(upper+lower)/2
    df["DC_UPPER"],df["DC_LOWER"],df["DC_BASIS"]=upper,lower,basis
    return df

class MLEngine:
    def __init__(self,horizon=5,threshold=0.005):self.horizon,self.threshold,self.model,self.scaler=horizon,threshold,None,StandardScaler()
    def _features(self,df):
        f=pd.DataFrame(index=df.index); cl=df["Close"]; sv=df["SQZ_VAL"]
        f["sqz_val"]=sv; f["sqz_val_pos"]=(sv>0).astype(float); f["sqz_val_rising"]=(sv>sv.shift(1)).astype(float); f["sqz_val_accel"]=sv.diff(); f["sqz_val_accel2"]=sv.diff().diff(); f["sqz_on"]=df["SQZ_ON"]; f["sqz_off"]=df["SQZ_OFF"]; f["sqz_val_slope"]=sv.diff(3)
        f["sqz_val_sma5"]=sv.rolling(5,min_periods=1).mean(); f["sqz_val_sma10"]=sv.rolling(10,min_periods=1).mean(); f["sqz_val_std10"]=sv.rolling(10,min_periods=1).std(); f["sqz_val_zscore"]=(sv-f["sqz_val_sma10"])/(f["sqz_val_std10"]+E); f["sqz_val_high5"]=sv.rolling(5,min_periods=1).max(); f["sqz_val_low5"]=sv.rolling(5,min_periods=1).min(); f["sqz_val_rank20"]=sv.rolling(20,min_periods=1).apply(lambda x:(x[-1]>x[:-1]).mean(),raw=True)
        dc_upper,dc_lower,dc_basis=df["DC_UPPER"],df["DC_LOWER"],df["DC_BASIS"]; dc_width=(dc_upper-dc_lower).replace(0,np.nan)
        f["dc_pct"]=(cl-dc_lower)/dc_width.fillna(1); f["dc_width"]=dc_width/(dc_basis.abs()+E); f["dc_basis_dist"]=(cl-dc_basis)/(dc_basis.abs()+E); f["dc_upper_dist"]=(cl-dc_upper)/(cl.abs()+E); f["dc_lower_dist"]=(cl-dc_lower)/(cl.abs()+E); f["dc_squeeze"]=dc_width/(dc_width.rolling(50,min_periods=1).mean()+E); f["dc_above_basis"]=(cl>dc_basis).astype(float); f["dc_zone"]=pd.cut(f["dc_pct"],bins=[-np.inf,0,0.25,0.5,0.75,1.0,np.inf],labels=False).astype(float)
        f["dc_width_change5"]=dc_width.pct_change(5); f["dc_width_change10"]=dc_width.pct_change(10); f["dc_basis_slope5"]=dc_basis.diff(5)/(dc_basis.abs()+E); f["dc_upper_slope5"]=dc_upper.diff(5)/(dc_upper.abs()+E); f["dc_lower_slope5"]=dc_lower.diff(5)/(dc_lower.abs()+E); f["dc_pct_ma5"]=f["dc_pct"].rolling(5,min_periods=1).mean(); f["dc_pct_ma10"]=f["dc_pct"].rolling(10,min_periods=1).mean(); f["dc_breakout_up"]=((cl>dc_upper.shift(1))&(cl.shift(1)<=dc_upper.shift(2))).astype(float); f["dc_breakout_down"]=((cl<dc_lower.shift(1))&(cl.shift(1)>=dc_lower.shift(2))).astype(float); f["dc_width_rank20"]=dc_width.rolling(20,min_periods=1).apply(lambda x:(x[-1]<x[:-1]).mean(),raw=True)
        for col in ["sqz_val","sqz_val_pos","sqz_on","dc_pct","dc_above_basis","sqz_val_zscore"]:
            if col in f.columns:
                for lag in [1,2,3]:f[f"{col}_lag{lag}"]=f[col].shift(lag)
        return f.replace([np.inf,-np.inf],np.nan).clip(-1e6,1e6)
    def _labels(self,df):
        cl=df["Close"]; H=self.horizon; buy_lbl=pd.Series(0.0,index=df.index); sell_lbl=pd.Series(0.0,index=df.index); ca=cl.values
        for i in range(len(ca)-H):
            e=ca[i]; w=ca[i+1:i+H+1]; thr=0.008; mfe=(np.max(w)-e)/(e+E); mae=(e-np.min(w))/(e+E)
            if mfe>=thr and mfe>=mae*1.5:buy_lbl.iloc[i]=1.0
            elif mae>=thr and mae>=mfe*1.5:sell_lbl.iloc[i]=1.0
        labels=pd.Series(0.0,index=df.index); labels[buy_lbl==1.0]=1.0; labels[sell_lbl==1.0]=-1.0
        return labels
    def fit_predict(self,df):
        feat=self._features(df).replace([np.inf,-np.inf],np.nan).clip(-1e9,1e9).dropna(); y=self._labels(df).loc[feat.index].dropna(); X=feat.loc[y.index]; n=len(X)
        if n<100:return None,{"error":"Yetersiz veri"}
        CONF=0.55; unique_labels=np.unique(y.values); label_map={orig:new for new,orig in enumerate(sorted(unique_labels))}; inv_label_map={new:orig for orig,new in label_map.items()}; y_mapped=y.map(label_map).values; num_class=len(unique_labels)
        cuda_available=False
        try:cuda_available=bool(xgb.build_info().get('USE_CUDA',False))
        except:pass
        device='cuda' if cuda_available else 'cpu'; device_info='CUDA' if cuda_available else 'CPU'
        params=dict(n_estimators=500,max_depth=6,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=0.1,min_child_weight=5,tree_method='hist',device=device,n_jobs=-1,random_state=42,objective='multi:softprob',num_class=num_class,verbosity=0)
        xgb1=xgb.XGBClassifier(**params); params.update(n_estimators=400,max_depth=7,learning_rate=0.05,subsample=0.7,colsample_bytree=0.7,reg_alpha=0.05,reg_lambda=0.2,min_child_weight=3); xgb2=xgb.XGBClassifier(**params); params.update(n_estimators=600,max_depth=5,learning_rate=0.02,subsample=0.9,colsample_bytree=0.9,reg_alpha=0.2,reg_lambda=0.05,min_child_weight=1); xgb3=xgb.XGBClassifier(**params)
        models=[xgb1,xgb2,xgb3]; X_arr=X.values; y_arr=y_mapped; med=np.nanmedian(X_arr,axis=0)
        for j in range(X_arr.shape[1]):
            m=~np.isfinite(X_arr[:,j])
            if m.any():X_arr[m,j]=med[j]
        fold_accs=[]; fold_f1s=[]; weights={"xgb1":[],"xgb2":[],"xgb3":[]}
        for tr_i,te_i in TimeSeriesSplit(n_splits=5,gap=self.horizon).split(X_arr):
            sc=StandardScaler(); Xtr_s=sc.fit_transform(X_arr[tr_i]); Xte_s=sc.transform(X_arr[te_i]); ytr=y_arr[tr_i]; yte=y_arr[te_i]
            for model in models:model.fit(Xtr_s,ytr)
            preds_list=[model.predict_proba(Xte_s) for model in models]; accs=[np.mean(np.argmax(pred,axis=1)==yte) for pred in preds_list]; total=sum(accs); w=[1/3]*3 if total==0 else [acc/total for acc in accs]
            for key,val in zip(weights.keys(),w):weights[key].append(val)
            ensemble_proba=np.zeros_like(preds_list[0])
            for proba,wi in zip(preds_list,w):ensemble_proba+=proba*wi
            ensemble_preds=np.argmax(ensemble_proba,axis=1); max_prob=np.max(ensemble_proba,axis=1); final_preds=np.where(max_prob>=CONF,ensemble_preds,label_map[0]); fold_accs.append(float(np.mean(final_preds==yte)*100)); fold_f1s.append(float(f1_score(yte,final_preds,average="weighted",zero_division=0)*100))
        acc=float(np.mean(fold_accs)); f1_avg=float(np.mean(fold_f1s)); w1,w2,w3=[float(np.mean(weights[k])) for k in weights]
        self.scaler.fit(X_arr); Xs=self.scaler.transform(X_arr)
        for model in models:model.fit(Xs,y_arr)
        proba_list=[model.predict_proba(Xs) for model in models]; ensemble_proba=np.zeros_like(proba_list[0])
        for proba,wi in zip(proba_list,[w1,w2,w3]):ensemble_proba+=proba*wi
        raw=np.argmax(ensemble_proba,axis=1); max_prob=np.max(ensemble_proba,axis=1); raw=np.where(max_prob>=CONF,raw,label_map[0]); confirmed=np.zeros_like(raw)
        for i in range(1,len(raw)):
            if raw[i]==raw[i-1]==label_map[1]:confirmed[i]=label_map[1]
            elif raw[i]==raw[i-1]==label_map[-1]:confirmed[i]=label_map[-1]
            else:confirmed[i]=label_map[0]
        confirmed_orig=np.vectorize(inv_label_map.get)(confirmed); pred_s=pd.Series(confirmed_orig,index=X.index,name="ML_SIGNAL")
        imp=(xgb1.feature_importances_+xgb2.feature_importances_+xgb3.feature_importances_)/3; self.model=models
        return pred_s,{"accuracy":acc,"f1_score":f1_avg,"train_size":int(n*0.8),"test_size":int(n*0.2),"top_features":sorted(zip(X.columns,imp),key=lambda x:-x[1])[:6],"split_date":(X.index[int(n*0.8)] if int(n*0.8)<n else None),"device":device_info}

class Backtester:
    def __init__(self,initial_capital=INITIAL_CAPITAL,commission=0.0002):self.initial_capital,self.commission=initial_capital,commission
    def run(self,df,signals,name="Strateji",min_hold_hours=0,min_profit_pct=0.0):
        signals=signals.reindex(df.index).fillna(0); cap=float(self.initial_capital); shares=0.0; ep=0.0; in_pos=False; trades=[]; equity=[]; entry_dt=None; ec=0.0; mh=pd.Timedelta(hours=min_hold_hours)
        for idx,row in df.iterrows():
            px=float(row["Close"]); sig=float(signals.loc[idx])
            if sig==1 and not in_pos:
                ef=cap*self.commission; shares=(cap-ef)/px; ec=ef; ep=px; cap=0.0; in_pos=True; entry_dt=idx
            elif sig==-1 and in_pos and (min_hold_hours==0 or (idx-entry_dt)>=mh) and ((px-ep)/ep>=min_profit_pct):
                pr=shares*px; xf=pr*self.commission; tc=ec+xf; trades.append({"entry_price":ep,"exit_price":px,"exit_date":idx,"pnl":pr-xf-shares*ep,"return":(px-ep)/ep,"entry_commission":ec,"exit_commission":xf,"commission":tc}); cap=pr-xf; shares=0.0; in_pos=False; ec=0.0
            equity.append(cap+shares*px)
        if in_pos and shares>0:
            px=float(df["Close"].iloc[-1]); pr=shares*px; xf=pr*self.commission; tc=ec+xf; trades.append({"entry_price":ep,"exit_price":px,"exit_date":df.index[-1],"pnl":pr-xf-shares*ep,"return":(px-ep)/ep,"entry_commission":ec,"exit_commission":xf,"commission":tc}); cap=pr-xf
            if equity:equity[-1]=cap
        eq=pd.Series(equity,index=df.index); dd=(eq-eq.expanding().max())/eq.expanding().max()*100; dr=eq.pct_change().dropna(); sharpe=float(dr.mean()/dr.std()*np.sqrt(252)) if dr.std()>0 else 0.0; tc=sum(t["commission"] for t in trades)
        if trades:
            td=pd.DataFrame(trades); wr=float((td["return"]>0).mean()*100); ar=float(td["return"].mean()*100); bt=float(td["return"].max()*100); wt=float(td["return"].min()*100); tp=float(td["pnl"].sum())
        else:wr=ar=bt=wt=tp=0.0
        return {"name":name,"total_return":(float(eq.iloc[-1])-self.initial_capital)/self.initial_capital*100,"bh_return":(float(df["Close"].iloc[-1])-float(df["Close"].iloc[0]))/float(df["Close"].iloc[0])*100,"max_drawdown":float(dd.min()),"sharpe":sharpe,"num_trades":len(trades),"win_rate":wr,"avg_return":ar,"best_trade":bt,"worst_trade":wt,"total_pnl":tp,"total_commission":tc,"equity_curve":eq,"trades":trades}

def _tbl_style(tbl):
    tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch); tbl.verticalHeader().setVisible(False); tbl.setAlternatingRowColors(True); tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers); tbl.setStyleSheet("QTableWidget{gridline-color:#2a2a4a;}QHeaderView::section{background:#ff9800;padding:6px;}")

class BacktestDialog(QtWidgets.QDialog):
    def __init__(self,ml_r,ml_info,df,parent=None):
        super().__init__(parent); self.setWindowTitle("Backtest & ML Analiz Sonuçları"); self.resize(1050,680); main=QtWidgets.QVBoxLayout(self); main.setContentsMargins(10,10,10,10); tabs=QtWidgets.QTabWidget(); tabs.setStyleSheet("QTabBar::tab{padding:8px 18px;}QTabBar::tab:selected{color:#ff9800;font-weight:bold;}"); main.addWidget(tabs)
        for title,fn,args in [("Performans",self._metrics,(ml_r,ml_info)),("Equity Eğrisi",self._equity,(ml_r,df)),("İşlem Geçmişi (ML)",self._trades,(ml_r,)),("Ek Bilgi",self._ek_bilgi,(ml_info,))]:
            w=QtWidgets.QWidget(); tabs.addTab(w,title); fn(w,*args)
    def _copy_table_to_clipboard(self, tbl):
        if tbl is None: return
        rows = tbl.rowCount()
        cols = tbl.columnCount()
        if rows == 0 or cols == 0: return
        lines = []
        header = []
        for c in range(cols):
            item = tbl.horizontalHeaderItem(c)
            header.append(item.text() if item else "")
        lines.append("\t".join(header))
        for r in range(rows):
            row_data = []
            for c in range(cols):
                item = tbl.item(r, c)
                row_data.append(item.text() if item else "")
            lines.append("\t".join(row_data))
        text = "\n".join(lines)
        QtWidgets.QApplication.clipboard().setText(text)
    def _metrics(self,w,ml_r,info):
        vl=QtWidgets.QVBoxLayout(w); vl.setSpacing(8)
        if info and "accuracy" in info:
            sd=info.get("split_date"); sd_s=pd.Timestamp(sd).strftime("%d.%m.%Y") if sd else "?"; device=info.get("device","Bilinmiyor"); lbl=QtWidgets.QLabel(f"Model: XGBoost Ensemble  │  Eğitim: {info.get('train_size',0)} bar  │  Test: {info.get('test_size',0)} bar  │  Ayrım: {sd_s}  │  Cihaz: {device}"); lbl.setTextFormat(QtCore.Qt.TextFormat.RichText); vl.addWidget(lbl)
        ml_only={"f1_score","accuracy"}; rows=[("Toplam Getiri (%)","total_return",True),("Al & Tut Getiri (%)","bh_return",True),("İşlem Sayısı","num_trades",None),("Kazanma Oranı (%)","win_rate",True),("Maks Drawdown (%)","max_drawdown",False),("Sharpe","sharpe",True),("Ort İşlem Getiri (%)","avg_return",True),("En İyi İşlem (%)","best_trade",True),("En Kötü İşlem (%)","worst_trade",False),("Toplam K/Z","total_pnl",True),("Toplam Komisyon (₺)","total_commission",None),("OOS Doğruluk (%)","accuracy",True),("F1 (%)","f1_score",True)]; cols=["Metrik","ML"]; tbl=QtWidgets.QTableWidget(len(rows),len(cols)); tbl.setHorizontalHeaderLabels(cols); _tbl_style(tbl)
        def mi(val,hb):
            it=QtWidgets.QTableWidgetItem(f"{val:.2f}" if isinstance(val,float) else str(val))
            if hb is not None:it.setForeground(QtGui.QColor("#4caf50" if (val>0 if hb else val>-5) else "#f44336"))
            return it
        for r,(label,key,hb) in enumerate(rows):
            tbl.setItem(r,0,QtWidgets.QTableWidgetItem(label)); mo=key in ml_only; mv=float(info.get(key,0)) if (mo and info) else (ml_r.get(key,0) if ml_r else 0); tbl.setItem(r,1,mi(mv,hb))
        vl.addWidget(tbl)
        if "error" in info:vl.addWidget(QtWidgets.QLabel(info["error"]))
        copy_btn = QtWidgets.QPushButton("CTRL+C (Kopyala)")
        copy_btn.clicked.connect(lambda: self._copy_table_to_clipboard(tbl))
        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(copy_btn)
        vl.addLayout(hbox)
    def _ek_bilgi(self,w,info):
        vl=QtWidgets.QVBoxLayout(w)
        if not info or "top_features" not in info:
            vl.addWidget(QtWidgets.QLabel("Ek bilgi bulunamadı."))
            return
        top_features=info["top_features"]; tbl=QtWidgets.QTableWidget(len(top_features),2); tbl.setHorizontalHeaderLabels(["Özellik","Önem"]); _tbl_style(tbl)
        for i,(feat,imp) in enumerate(top_features):
            tbl.setItem(i,0,QtWidgets.QTableWidgetItem(feat)); tbl.setItem(i,1,QtWidgets.QTableWidgetItem(f"{imp:.6f}"))
        vl.addWidget(tbl)
        copy_btn = QtWidgets.QPushButton("CTRL+C (Kopyala)")
        copy_btn.clicked.connect(lambda: self._copy_table_to_clipboard(tbl))
        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(copy_btn)
        vl.addLayout(hbox)
    def _equity(self,w,ml_r,df):
        vl=QtWidgets.QVBoxLayout(w); vl.setContentsMargins(0,0,0,0); fig=Figure(figsize=(10,4.5),dpi=100); fig.patch.set_facecolor("white"); ax=fig.add_subplot(111); ax.set_facecolor("white"); bh=df["Close"]/float(df["Close"].iloc[0])*INITIAL_CAPITAL; ax.plot(bh.index,bh.values,color="#888",lw=1.2,ls="--",label="Al & Tut",alpha=0.7)
        if ml_r:ax.plot(ml_r["equity_curve"].index,ml_r["equity_curve"].values,color="#FF9800",lw=1.8,label="ML")
        ax.axhline(INITIAL_CAPITAL,color="#555",lw=0.7,ls=":"); ax.fill_between(bh.index,INITIAL_CAPITAL,bh.values,where=bh.values<INITIAL_CAPITAL,alpha=0.08,color="red")
        for sp in ax.spines.values():sp.set_color("#333")
        ax.tick_params(colors="black",labelsize=9)
        for o in (ax.xaxis.label,ax.yaxis.label,ax.title):o.set_color("black")
        ax.set_title(f"Equity Eğrisi (₺{INITIAL_CAPITAL:,.0f})",color="black",fontsize=11); ax.legend(facecolor="white",labelcolor="black",edgecolor="#aaa",framealpha=0.9); ax.grid(True,alpha=0.25,color="#aaa"); fig.tight_layout(pad=1.5); vl.addWidget(FigureCanvas(fig))
    def _trades(self,w,result):
        vl=QtWidgets.QVBoxLayout(w)
        if not result or not result.get("trades"):
            vl.addWidget(QtWidgets.QLabel("İşlem bulunamadı."))
            return
        trades=result["trades"]; tbl=QtWidgets.QTableWidget(len(trades),6); tbl.setHorizontalHeaderLabels(["#","Giriş","Çıkış","Getiri (%)","K/Z (₺)","Komisyon (₺)"]); _tbl_style(tbl); wins=0; tc=0.0
        for i,t in enumerate(trades):
            win=t["return"]>0
            if win:wins+=1
            clr=QtGui.QColor("#4caf50" if win else "#f44336"); cm=t.get("commission",0.0); tc+=cm
            for c,v in enumerate([str(i+1),f"{t['entry_price']:.2f}",f"{t['exit_price']:.2f}"]):tbl.setItem(i,c,QtWidgets.QTableWidgetItem(v))
            for c,v in [(3,f"{t['return']*100:.2f}%"),(4,f"₺{t['pnl']:.2f}")]:
                it=QtWidgets.QTableWidgetItem(v); it.setForeground(clr); tbl.setItem(i,c,it)
            ci=QtWidgets.QTableWidgetItem(f"₺{cm:.2f}"); ci.setForeground(QtGui.QColor("#ff9800")); tbl.setItem(i,5,ci)
        vl.addWidget(tbl)
        s=QtWidgets.QLabel(f"{len(trades)} işlem  │  {wins} kazanan  │  {len(trades)-wins} kaybeden  │  Komisyon: ₺{tc:.2f}")
        s.setStyleSheet("padding:6px;color:#ff9800;font-weight:bold;")
        copy_btn = QtWidgets.QPushButton("CTRL+C (Kopyala)")
        copy_btn.clicked.connect(lambda: self._copy_table_to_clipboard(tbl))
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(s)
        hbox.addStretch()
        hbox.addWidget(copy_btn)
        vl.addLayout(hbox)

class Canvas(FigureCanvas):
    _L,_R,_B,_T,_VH=0.04,0.95,0.06,0.98,0.22
    def __init__(self,fig,parent=None):
        super().__init__(fig)
        if parent:self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(1,1)
        self.updateGeometry()
    def resizeEvent(self,ev):
        w,h=ev.size().width(),ev.size().height()
        if w>0 and h>0 and self.figure is not None:
            dpi=self.figure.get_dpi() or CHART_DPI
            self.figure.set_size_inches(w/dpi,h/dpi,forward=False)
        super().resizeEvent(ev)
        self.draw_idle()

class Worker(QtCore.QThread):
    ready=QtCore.pyqtSignal(object,float); error=QtCore.pyqtSignal(str)
    def __init__(self,sym,tf,currency="TRY"):super().__init__(); self.sym,self.tf,self.currency=sym,tf,currency
    def _fetch_usdtry(self):
        try:
            fx=yf.download("USDTRY=X",period="5d",interval="1d",progress=False,threads=False,auto_adjust=False)
            if fx is not None and not fx.empty:
                if isinstance(fx.columns,pd.MultiIndex):fx.columns=fx.columns.get_level_values(0)
                return float(fx["Close"].dropna().iloc[-1])
        except:pass
        return 1.0
    @staticmethod
    def _fix(df):
        if isinstance(df.columns,pd.MultiIndex):df.columns=df.columns.get_level_values(0)
        df.index=pd.to_datetime(df.index); return df
    @staticmethod
    def _resample(df,rule):return df.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    def run(self):
        try:
            kw={"auto_adjust":False,"progress":False,"threads":False}; fx=self._fetch_usdtry() if self.currency=="USD" else 1.0; raw=yf.download(self.sym,interval="1d",period="5y",**kw)
            if raw is None or raw.empty:self.error.emit("Veri yok: "+self.sym); return
            raw=self._fix(raw).sort_index()
            if self.tf=="2D":raw=self._resample(raw,"2D")
            elif self.tf=="1W":raw=self._resample(raw,"W")
            self.ready.emit(raw,fx)
        except Exception as e:self.error.emit(str(e))

class MLWorker(QtCore.QThread):
    done=QtCore.pyqtSignal(object,object,object,object,str); error=QtCore.pyqtSignal(str)
    def __init__(self,df,tf="1D",chart_type="candle"):super().__init__(); self._df,self._tf,self._chart_type=df.copy(),tf,chart_type
    def run(self):
        try:
            df=self._df; pred,info=MLEngine(horizon=5,threshold=0.005).fit_predict(df); bt=Backtester(initial_capital=INITIAL_CAPITAL,commission=0.0002); ml_r=bt.run(df,pred.reindex(df.index).fillna(0) if pred is not None else pd.Series(0,index=df.index),"ML",min_hold_hours=4,min_profit_pct=0.02)
            if pred is not None:
                p=pred.reindex(df.index,fill_value=0); rb=pd.Series(np.where((p==1)&(p.shift(1)!=1),df["Close"]*0.986,np.nan),index=df.index); rs=pd.Series(np.where((p==-1)&(p.shift(1)!=-1),df["Close"]*1.014,np.nan),index=df.index); df["ML_BUY"],df["ML_SELL"]=alternating_signals(rb,rs)
            else:df["ML_BUY"],df["ML_SELL"]=np.nan,np.nan
            self.done.emit(df,pred,info,ml_r,self._chart_type)
        except Exception as e:
            print(f"\n[ML HATA] {e}",file=sys.stderr); print(traceback.format_exc(),file=sys.stderr); self.error.emit(str(e))

class DisclaimerDialog(QtWidgets.QDialog):
    def __init__(self,parent=None):
        super().__init__(parent); self.setWindowTitle("Uyarı"); self.setModal(True); layout=QtWidgets.QVBoxLayout(self); label=QtWidgets.QLabel("Pusula Finans programında yer alan tüm göstergeler, analizler ve tahminler yalnızca eğitim ve bilgilendirme amaçlıdır. Bunlar hiçbir şekilde yatırım tavsiyesi niteliği taşımaz. Programın kullanımından doğabilecek yatırım kararları ve sonuçları tamamen kullanıcıya aittir.<b>&quot;Kabul&nbsp;Ediyorum&quot;</b> düğmesine basarak bu koşulları okuduğunuzu, anladığınızı ve kabul ettiğinizi beyan etmiş olursunuz."); label.setWordWrap(True); label.setFixedWidth(360); layout.addWidget(label); btn_row=QtWidgets.QHBoxLayout(); btn_row.addStretch(); btn_reject=QtWidgets.QPushButton("Kabul Etmiyorum"); btn_reject.clicked.connect(self.reject); btn_row.addWidget(btn_reject); btn_accept=QtWidgets.QPushButton("Kabul Ediyorum"); btn_accept.clicked.connect(self.accept); btn_row.addWidget(btn_accept); btn_row.addStretch(); layout.addLayout(btn_row); self.adjustSize(); self.setFixedSize(self.sizeHint())

class App(QtWidgets.QMainWindow):
    _STEP_VALUES=[1,4,8,16]; MAX_BARS=350
    def __init__(self):
        super().__init__(); self.setWindowTitle("Pusula Finans V1.3"); self.resize(1000,680)
        self._ml_result_candle=None; self._ml_info_candle={}; self._ml_result_ha=None; self._ml_info_ha={}; self._df_candle=None; self._df_ha=None; self._ml_worker_candle=None; self._ml_worker_ha=None
        self._cooldown_remaining=0; self._view_offset=0; self._step_index=0; self._cooldown_timer=QtCore.QTimer(self); self._cooldown_timer.setInterval(1000); self._cooldown_timer.timeout.connect(self._tick_cooldown)
        c=QtWidgets.QWidget(); c.setContentsMargins(0,0,0,0); self.setCentralWidget(c); v=QtWidgets.QVBoxLayout(c); v.setContentsMargins(0,0,0,0); v.setSpacing(0); h=QtWidgets.QHBoxLayout(); h.setContentsMargins(6,6,6,6); v.addLayout(h)
        def add(w,sp=8):h.addWidget(w); h.addSpacing(sp) if sp else None
        add(QtWidgets.QLabel("Hisse:"),0); self.sym=QtWidgets.QLineEdit("XU100"); self.sym.setFixedWidth(160); add(self.sym); add(QtWidgets.QLabel("Periyot:"),0); self.tf=QtWidgets.QComboBox()
        for n,d in [("1 Gün","1D"),("2 Gün","2D"),("1 Hafta","1W")]:self.tf.addItem(n,d)
        add(self.tf); add(QtWidgets.QLabel("Grafik Türü:"),0); self.chart_type=QtWidgets.QComboBox(); self.chart_type.addItem("Klasik Mum","candle"); self.chart_type.addItem("Heiken Ashi","heikinashi"); self.chart_type.currentIndexChanged.connect(self._on_chart_type_changed); add(self.chart_type); add(QtWidgets.QLabel("Para Birimi:"),0); self.ccy=QtWidgets.QComboBox(); self.ccy.addItem("TRY","TRY"); self.ccy.addItem("USD","USD"); add(self.ccy,8)
        self.btn_prev=QtWidgets.QPushButton("<"); self.btn_prev.setFixedWidth(28); self.btn_prev.setToolTip("Geri sar"); self.btn_prev.clicked.connect(self._step_left); add(self.btn_prev,2); self.btn_step=QtWidgets.QPushButton(str(self._STEP_VALUES[self._step_index])); self.btn_step.setFixedWidth(32); self.btn_step.setToolTip("Adım değeri (tıklayarak değiştir: 1 → 4 → 8 → 16)"); self.btn_step.clicked.connect(self._cycle_step); add(self.btn_step,2); self.btn_next=QtWidgets.QPushButton(">"); self.btn_next.setFixedWidth(28); self.btn_next.clicked.connect(self._step_right); add(self.btn_next)
        self.btn=QtWidgets.QPushButton("Yükle"); self.btn.setFixedWidth(45); self.btn.clicked.connect(self.load); add(self.btn); self.bt_btn=QtWidgets.QPushButton("Backtest-ML Raporu"); self.bt_btn.setEnabled(False); self.bt_btn.clicked.connect(self.show_backtest); add(self.bt_btn); self.share_btn=QtWidgets.QPushButton("Grafiği Paylaş"); self.share_btn.setFixedWidth(90); self.share_btn.clicked.connect(self.share_graph); add(self.share_btn); h.addStretch()
        self.cooldown_lbl=QtWidgets.QLabel(""); self.cooldown_lbl.setFixedWidth(150); self.cooldown_lbl.setStyleSheet("color:#ff9800;font-weight:bold;padding:0 4px;"); self.cooldown_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignVCenter); add(self.cooldown_lbl,0); self.progress=QtWidgets.QLabel(""); self.progress.setStyleSheet("color:#ff9800;font-weight:bold;padding:0 8px;"); self.progress.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight|QtCore.Qt.AlignmentFlag.AlignVCenter); h.addWidget(self.progress)
        self.pw=QtWidgets.QWidget(); self.pw.setContentsMargins(0,0,0,0); self.pl=QtWidgets.QVBoxLayout(self.pw); self.pl.setContentsMargins(0,0,0,0); self.pl.setSpacing(0); v.addWidget(self.pw,1); self.canvas=None; self.toolbar=None; self.worker=None; self.axes=None
    def share_graph(self):
        try:
            script_dir=os.path.dirname(os.path.abspath(__file__)); grafik_paylas_path=os.path.join(script_dir,"grafiği_paylaş.py"); subprocess.Popen([sys.executable,grafik_paylas_path])
        except Exception as e:QtWidgets.QMessageBox.warning(self,"Hata",f"grafiği_paylaş.py çalıştırılamadı: {e}")
    def _cycle_step(self):
        self._step_index=(self._step_index+1)%len(self._STEP_VALUES); val=self._STEP_VALUES[self._step_index]; self.btn_step.setText(str(val)); self.btn_prev.setToolTip(f"{val} mum geri"); self.btn_next.setToolTip(f"{val} mum ileri")
    def _current_step_size(self):return self._STEP_VALUES[self._step_index]
    def _get_current_df(self):return self._df_ha if self.chart_type.currentData()=="heikinashi" else self._df_candle
    def _current_view_df(self):
        df=self._get_current_df()
        if df is None:return None
        n=len(df); end=n-self._view_offset; start=max(0,end-self.MAX_BARS); return df.iloc[start:end]
    def _capture_view_limits(self):
        if not self.axes:return None
        ax_list=self.axes if isinstance(self.axes,(list,tuple)) else [self.axes]
        try:return [(a.get_xlim(),a.get_ylim()) for a in ax_list]
        except:return None
    def _step_left(self):
        if self._get_current_df() is None:return
        max_offset=max(0,len(self._get_current_df())-self.MAX_BARS); step=self._current_step_size()
        if self._view_offset>=max_offset:return
        limits=self._capture_view_limits(); self._view_offset=min(self._view_offset+step,max_offset); self._render(self._current_view_df(),view_limits=limits)
    def _step_right(self):
        if self._get_current_df() is None or self._view_offset<=0:return
        step=self._current_step_size(); limits=self._capture_view_limits(); self._view_offset=max(self._view_offset-step,0); self._render(self._current_view_df(),view_limits=limits)
    @staticmethod
    def _to_heiken_ashi(df):
        ha=df.copy(); ha_close=(df["Open"]+df["High"]+df["Low"]+df["Close"])/4; ha_open_vals=np.empty(len(df),dtype=float); ha_open_vals[0]=(df["Open"].iloc[0]+df["Close"].iloc[0])/2; ha_close_vals=ha_close.values
        for i in range(1,len(df)):ha_open_vals[i]=(ha_open_vals[i-1]+ha_close_vals[i-1])/2
        ha_open=pd.Series(ha_open_vals,index=df.index); ha["Close"]=ha_close; ha["Open"]=ha_open; ha["High"]=pd.concat([df["High"],ha_open,ha_close],axis=1).max(axis=1); ha["Low"]=pd.concat([df["Low"],ha_open,ha_close],axis=1).min(axis=1); return ha
    def _on_chart_type_changed(self):
        if self._get_current_df() is not None:self._view_offset=0; self._render(self._current_view_df()); self._update_progress_text()
    def _start_cooldown(self):
        self._cooldown_remaining=COOLDOWN_SECONDS; self.btn.setEnabled(False); self.cooldown_lbl.setText(f"⏳ Yükle: {self._cooldown_remaining} sn"); self._cooldown_timer.start()
    def _tick_cooldown(self):
        self._cooldown_remaining-=1
        if self._cooldown_remaining<=0:self._cooldown_timer.stop(); self._cooldown_remaining=0; self.btn.setEnabled(True); self.cooldown_lbl.setText("")
        else:self.cooldown_lbl.setText(f"⏳ Yükle: {self._cooldown_remaining} sn")
    def _sym(self):s=self.sym.text().strip().upper(); return s if s.endswith(".IS") else s+".IS"
    @staticmethod
    def _update_sqz_colors(df):
        vals=df["SQZ_VAL"].values; prev_val=np.nan; bcolors=[]
        for i,v in enumerate(vals):
            if np.isnan(v) or np.isinf(v):bcolors.append("gray")
            else:
                if v>0:
                    if not np.isnan(prev_val) and v>prev_val:bcolors.append("#00FF00")
                    else:bcolors.append("#008000")
                elif v<0:
                    if not np.isnan(prev_val) and v<prev_val:bcolors.append("#FF0000")
                    else:bcolors.append("#800000")
                else:bcolors.append("gray")
            if not np.isnan(v) and not np.isinf(v):prev_val=v
        return bcolors
    def load(self):
        s=self._sym()
        if not s:QtWidgets.QMessageBox.warning(self,"Uyarı","Sembol girin."); return
        self.sym.setText(s); self.btn.setEnabled(False); self.cooldown_lbl.setText("Veri çekiliyor..."); self.bt_btn.setEnabled(False); self.progress.setText("")
        if self.worker and self.worker.isRunning():self.worker.terminate()
        self.worker=Worker(s,self.tf.currentData(),self.ccy.currentData()); self.worker.ready.connect(self.on_data); self.worker.error.connect(self.on_err); self.worker.start()
    def on_err(self,m):print(f"[VERİ HATA] {m}",file=sys.stderr); self._start_cooldown()
    def on_data(self,df,fx_rate):
        if not {"Open","High","Low","Close"}.issubset(df.columns):QtWidgets.QMessageBox.critical(self,"Hata",f"Kolonlar eksik: {list(df.columns)}"); self._start_cooldown(); return
        years=5; df=df.loc[df.index>=df.index.max()-pd.DateOffset(years=years)].dropna(subset=["Open","High","Low","Close"])
        if self.ccy.currentData()=="USD" and fx_rate>1.0:
            for col in ["Open","High","Low","Close"]:df[col]=df[col]/fx_rate
        df_candle=donchian(squeeze_momentum(df.copy())); df_candle["ML_BUY"]=np.nan; df_candle["ML_SELL"]=np.nan; df_ha=donchian(squeeze_momentum(self._to_heiken_ashi(df.copy()))); df_ha["ML_BUY"]=np.nan; df_ha["ML_SELL"]=np.nan; self._df_candle,self._df_ha=df_candle,df_ha; self._view_offset=0; self._render(self._current_view_df()); self._start_cooldown(); self.progress.setText("ML eğitiliyor...")
        if self._ml_worker_candle and self._ml_worker_candle.isRunning():self._ml_worker_candle.terminate()
        if self._ml_worker_ha and self._ml_worker_ha.isRunning():self._ml_worker_ha.terminate()
        self._ml_worker_candle=MLWorker(df_candle,tf="1D",chart_type="candle"); self._ml_worker_candle.done.connect(self._on_ml_done); self._ml_worker_candle.error.connect(self._on_ml_err); self._ml_worker_ha=MLWorker(df_ha,tf="1D",chart_type="heikinashi"); self._ml_worker_ha.done.connect(self._on_ml_done); self._ml_worker_ha.error.connect(self._on_ml_err); self._ml_worker_candle.start(); self._ml_worker_ha.start()
    def _on_ml_err(self,m):self.progress.setText(f"ML Hata: {m}"); print(f"[ML HATA] {m}",file=sys.stderr)
    def _on_ml_done(self,df,pred,info,ml_r,chart_type):
        if chart_type=="candle":self._ml_result_candle,self._ml_info_candle,self._df_candle=ml_r,info,df
        else:self._ml_result_ha,self._ml_info_ha,self._df_ha=ml_r,info,df
        if self._ml_result_candle is not None and self._ml_result_ha is not None:self.bt_btn.setEnabled(True); self._update_progress_text()
        self._render(self._current_view_df())
    def _update_progress_text(self):
        if self._ml_result_candle is not None and self._ml_result_ha is not None:
            a_c=self._ml_info_candle.get("accuracy",0); t_c=self._ml_result_candle.get("total_return",0); a_h=self._ml_info_ha.get("accuracy",0); t_h=self._ml_result_ha.get("total_return",0); self.progress.setText(f"ML C: %{a_c:.1f} / %{t_c:+.1f}  │  HA: %{a_h:.1f} / %{t_h:+.1f}")
        elif self._ml_result_candle is not None:
            a=self._ml_info_candle.get("accuracy",0); t=self._ml_result_candle.get("total_return",0); self.progress.setText(f"ML Candle: %{a:.1f} / %{t:+.1f}  │  HA eğitiliyor...")
        elif self._ml_result_ha is not None:
            a=self._ml_info_ha.get("accuracy",0); t=self._ml_result_ha.get("total_return",0); self.progress.setText(f"Candle eğitiliyor...  │  ML HA: %{a:.1f} / %{t:+.1f}")
    def show_backtest(self):
        current_df=self._get_current_df()
        if current_df is None:return
        if self.chart_type.currentData()=="heikinashi":ml_r,ml_info=self._ml_result_ha,self._ml_info_ha
        else:ml_r,ml_info=self._ml_result_candle,self._ml_info_candle
        BacktestDialog(ml_r,ml_info,current_df,parent=self).exec()
    def _clear(self):
        for w in (self.toolbar,self.canvas):
            if w:self.pl.removeWidget(w); w.setParent(None); w.deleteLater()
        self.toolbar,self.canvas,self.axes=None,None,None
    def _render(self,df,view_limits=None):
        self._clear()
        if df is None:return
        plot_df=df; sqz_colors=self._update_sqz_colors(plot_df); apds=[mpf.make_addplot(plot_df["DC_BASIS"],type="line",panel=0,width=0.8,color="#FF6D00",secondary_y=False),mpf.make_addplot(plot_df["DC_UPPER"],type="line",panel=0,width=0.8,color="#2962FF",secondary_y=False),mpf.make_addplot(plot_df["DC_LOWER"],type="line",panel=0,width=0.8,color="#2962FF",secondary_y=False),mpf.make_addplot(plot_df["SQZ_VAL"],type="bar",panel=1,width=0.7,color=sqz_colors,secondary_y=False)]
        for col,mk,sz,cl in [("ML_BUY","^",90,"#2979ff"),("ML_SELL","v",90,"#ff6d00")]:
            if plot_df[col].notna().any():apds.append(mpf.make_addplot(plot_df[col],type="scatter",markersize=sz,marker=mk,panel=0,color=cl,secondary_y=False))
        marketcolors=mpf.make_marketcolors(up="green",down="red",edge="inherit",wick="inherit",volume="in"); st=mpf.make_mpf_style(base_mpf_style="classic",marketcolors=marketcolors,rc={"font.family":"DejaVu Sans","font.weight":"normal","axes.titleweight":"normal","axes.labelweight":"normal","figure.titleweight":"normal"})
        pw_size=self.pw.size(); w_px=max(pw_size.width(),400); h_px=max(pw_size.height(),300); fig_w,fig_h=w_px/CHART_DPI,h_px/CHART_DPI
        fig,axes=mpf.plot(plot_df,type="candle",volume=False,style=st,addplot=apds,returnfig=True,figsize=(fig_w,fig_h),xrotation=0,tight_layout=False,scale_padding=0.0)
        fig.set_dpi(CHART_DPI); fig.patch.set_facecolor("white")
        if isinstance(axes,(list,tuple)):ax_list=list(axes)
        else:ax_list=[axes]
        main_ax=ax_list[0]; panel_ax=None
        if len(ax_list)>=4:panel_ax=ax_list[2]; ax_list[1].set_visible(False); ax_list[3].set_visible(False)
        elif len(ax_list)>=2:panel_ax=ax_list[-1]; [extra_ax.set_visible(False) for extra_ax in ax_list[1:-1]]
        L,R,B,T,VH=0.01,0.95,0.04,0.97,0.22; main_height=max((T-B-VH-0.015),0.05); main_ax.set_position([L,B+VH+0.015,R-L,main_height]); main_ax.yaxis.tick_right(); main_ax.yaxis.set_label_position("right"); main_ax.tick_params(axis="y",which="both",right=True,labelright=True,left=False,labelleft=False,pad=3); main_ax.tick_params(axis="x",which="both",bottom=False,labelbottom=False); main_ax.set_ylabel("")
        if panel_ax is not None:
            panel_ax.set_position([L,B,R-L,VH]); panel_ax.yaxis.tick_right(); panel_ax.tick_params(axis="y",which="both",left=False,labelleft=False,right=True,labelright=True,pad=3); panel_ax.tick_params(axis="x",which="both",bottom=True,labelbottom=True); panel_ax.set_ylabel("")
        for ax in [main_ax,panel_ax]:
            if ax is not None:
                for spine in ax.spines.values():spine.set_linewidth(0.5)
                ax.tick_params(labelsize=10)
        dc_lo=pd.to_numeric(plot_df["DC_LOWER"],errors="coerce").to_numpy(); dc_hi=pd.to_numeric(plot_df["DC_UPPER"],errors="coerce").to_numpy(); valid=np.isfinite(dc_lo)&np.isfinite(dc_hi)
        if valid.any():main_ax.fill_between(np.arange(len(plot_df)),dc_lo,dc_hi,where=valid,alpha=0.05,color="#2962FF",zorder=0)
        n_bars=len(plot_df); tight_xlim=(-0.5,n_bars-0.5)
        if view_limits:
            if len(view_limits)>0:
                try:main_ax.set_xlim(view_limits[0][0]); main_ax.set_ylim(view_limits[0][1])
                except:main_ax.set_xlim(tight_xlim)
            else:main_ax.set_xlim(tight_xlim)
            if panel_ax is not None:
                if len(view_limits)>1:
                    try:panel_ax.set_xlim(view_limits[-1][0]); panel_ax.set_ylim(view_limits[-1][1])
                    except:panel_ax.set_xlim(tight_xlim)
                else:panel_ax.set_xlim(tight_xlim)
        else:
            main_ax.set_xlim(tight_xlim)
            if panel_ax is not None:panel_ax.set_xlim(tight_xlim)
        main_ax.margins(x=0)
        if panel_ax is not None:panel_ax.margins(x=0)
        self.canvas=Canvas(fig,parent=self.pw); self.pl.addWidget(self.canvas); self.toolbar=NavigationToolbar(self.canvas,self); self.pl.insertWidget(0,self.toolbar); self.axes=[main_ax]+([panel_ax] if panel_ax is not None else []); self.canvas.mpl_connect("scroll_event",self._scroll); self.canvas.draw_idle()
    def _scroll(self,ev):
        if ev.inaxes is None or ev.xdata is None:return
        try:
            l,r=ev.inaxes.get_xlim(); s=1/1.2 if ev.button=="up" else 1.2; nl=ev.xdata-(ev.xdata-l)*s; nr=ev.xdata+(r-ev.xdata)*s
            for a in (self.axes if isinstance(self.axes,(list,tuple)) else [self.axes]):
                try:a.set_xlim(nl,nr)
                except:pass
            if self.canvas:self.canvas.draw_idle()
        except:pass

if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv); app.setFont(QtGui.QFont("DejaVu Sans",9)); settings=QtCore.QSettings("PusulaFinans","PusulaFinansV1")
    if not settings.value("disclaimer_accepted",False,type=bool):
        dlg=DisclaimerDialog()
        if dlg.exec()==QtWidgets.QDialog.DialogCode.Accepted:settings.setValue("disclaimer_accepted",True)
        else:sys.exit(0)
    win=App(); win.show(); sys.exit(app.exec())
