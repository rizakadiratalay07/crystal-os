from llama_cpp import Llama
import glob, yfinance as yf, pandas as pd, numpy as np, re, sys, traceback
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection, PolyCollection
from PyQt6 import QtWidgets, QtCore, QtGui
from huggingface_hub import snapshot_download

class DownloadThread(QtCore.QThread):
    finished = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)
    def run(self):
        try:
            snapshot_download(
                repo_id="ogulcanaydogan/Turkish-LLM-32B-Instruct-GGUF",
                local_dir="./turkish-llm-32b-gguf",
                local_dir_use_symlinks=False,
                allow_patterns=["*Q4_K_M*.gguf"],
            )
            self.finished.emit()
        except Exception as e:
            self.failed.emit(str(e))

model = None

INVALID_STOCK_WORDS = {
    "HOCAM", "HOCA", "SELAM", "MERHABA", "NASIL", "NEDEN", "BENCE", "SENCE",
    "BURDA", "ORADA", "ALLAH", "DOLAR", "EURO", "BIST", "HISSE", "SENET",
    "ANALIZ", "YORUM", "TEKNIK", "TEMEL", "GRAFIK", "MUM", "BORSA", "ALTIN"
}

def extract_symbol_from_text(text):
    tokens = text.split()
    candidates = []
    for token in tokens:
        token_clean = token.strip('.,!?;:()[]{}"').upper()
        if token_clean.endswith('.IS'):
            token_clean = token_clean[:-3]
        if re.fullmatch(r'[A-Z0-9]{4,5}', token_clean):
            if token_clean not in INVALID_STOCK_WORDS:
                candidates.append(token_clean)
    if not candidates:
        return None
    if len(tokens) == 1:
        return candidates[0] + '.IS' if len(candidates) == 1 else None
    for candidate in candidates:
        if candidate not in INVALID_STOCK_WORDS:
            return candidate + '.IS'
    return None

def is_market_query(text):
    keywords = ["piyasa", "borsa", "endeks", "xu100", "bist", "bist100", "bugün piyasalar", "piyasalar nasıl", "borsa nasıl", "endeks nasıl", "dolar", "euro"]
    return any(kw in text.lower() for kw in keywords)

def heiken_ashi(hist):
    ha_close = (hist['Open'] + hist['High'] + hist['Low'] + hist['Close']) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (hist['Open'].iloc[0] + hist['Close'].iloc[0]) / 2
    for i in range(1, len(hist)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_high = pd.concat([hist['High'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([hist['Low'], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({'Open': ha_open, 'High': ha_high, 'Low': ha_low, 'Close': ha_close, 'Volume': hist.get('Volume', 0)}, index=hist.index)

def sqzmom(hist):
    length, multKC, lengthKC = 20, 1.5, 20
    basis = hist['Close'].rolling(length).mean()
    dev = multKC * hist['Close'].rolling(length).std()
    upperBB, lowerBB = basis + dev, basis - dev
    ma_kc = hist['Close'].rolling(lengthKC).mean()
    tr = pd.concat([hist['High'] - hist['Low'], (hist['High'] - hist['Close'].shift(1)).abs(), (hist['Low'] - hist['Close'].shift(1)).abs()], axis=1).max(axis=1)
    rangema = tr.rolling(lengthKC).mean()
    upperKC, lowerKC = ma_kc + rangema * multKC, ma_kc - rangema * multKC
    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
    noSqz = ~sqzOn & ~sqzOff
    hh = hist['High'].rolling(lengthKC).max()
    ll = hist['Low'].rolling(lengthKC).min()
    avg_h_l = (hh + ll) / 2
    avg_close = hist['Close'].rolling(lengthKC).mean()
    source_adj = hist['Close'] - ((avg_h_l + avg_close) / 2)
    def linreg(series, window):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            y = series.iloc[i-window+1:i+1].values
            if not np.isnan(y).any():
                slope, intercept = np.polyfit(np.arange(window), y, 1)
                result.iloc[i] = intercept + slope * (window-1)
        return result
    val = linreg(source_adj, lengthKC)
    latest, prev = val.iloc[-1] if len(val) > 0 else 0.0, val.iloc[-2] if len(val) > 1 else 0.0
    color = "lime" if latest > 0 and latest > prev else "green" if latest > 0 else "red" if latest < 0 and latest < prev else "maroon"
    squeeze = "no" if noSqz.iloc[-1] else "on" if sqzOn.iloc[-1] else "off"
    return {'value': latest, 'color': color, 'squeeze': squeeze, 'series': val}

def indicators(hist, symbol_name):
    hist = hist.copy()
    for col in ['Open','High','Low','Close','Volume']:
        if col not in hist.columns:
            if col == 'Volume': hist[col] = 0
            else: return None
    hist['DC_upper'] = hist['High'].rolling(20).max()
    hist['DC_lower'] = hist['Low'].rolling(20).min()
    hist['DC_basis'] = (hist['DC_upper'] + hist['DC_lower']) / 2
    hist['BB_middle'] = hist['Close'].rolling(20).mean()
    bb_std = hist['Close'].rolling(20).std()
    hist['BB_upper'] = hist['BB_middle'] + 2 * bb_std
    hist['BB_lower'] = hist['BB_middle'] - 2 * bb_std
    latest = hist.iloc[-1]
    vol_avg = hist['Volume'].rolling(20).mean().iloc[-1]
    sq = sqzmom(hist)
    change = ((latest['Close'] - hist.iloc[-5]['Close']) / hist.iloc[-5]['Close'] * 100) if len(hist) >= 5 else 0.0
    return {'symbol': symbol_name, 'price': float(latest['Close']), 'volume': float(latest['Volume']),
            'dc_upper': float(latest['DC_upper']), 'dc_lower': float(latest['DC_lower']), 'dc_basis': float(latest['DC_basis']),
            'bb_upper': float(latest['BB_upper']), 'bb_lower': float(latest['BB_lower']),
            'bb_middle': float(latest['BB_middle']), 'price_change': change, 'volume_avg': float(vol_avg),
            'sqzmom_value': sq['value'], 'sqzmom_color': sq['color'], 'sqzmom_squeeze': sq['squeeze'],
            'sqzmom_series': sq['series']}

class InteractiveChartCanvas(FigureCanvas):
    def __init__(self, figure, owner):
        super().__init__(figure)
        self.owner = owner
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
        self._last_draw_ms = 0
        self._draw_interval_ms = 24

    def _axis_geometry(self):
        axes = getattr(self.owner, "_axes", [])
        if not axes: return None
        ax = axes[0]
        pos = ax.get_position()
        width = max(1.0, float(self.width()))
        height = max(1.0, float(self.height()))
        return (float(pos.x0) * width, float(pos.x1) * width, float(pos.y0) * height, float(pos.y1) * height, width, height)

    def _throttled_draw(self, force=False):
        now = QtCore.QTime.currentTime().msecsSinceStartOfDay()
        if not force and self._last_draw_ms:
            delta = now - self._last_draw_ms
            if delta < 0: delta += 24 * 60 * 60 * 1000
            if delta < self._draw_interval_ms: return
        self._last_draw_ms = now
        self.draw_idle()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            axes = getattr(self.owner, "_axes", [])
            chart_df = getattr(self.owner, "_chart_df", None)
            if not axes or chart_df is None or chart_df.empty:
                event.ignore(); return
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            self.owner._dragging = True
            self.owner._drag_start_x = float(event.position().x())
            self.owner._drag_start_y = float(event.position().y())
            self.owner._drag_start_xlim = tuple(axes[0].get_xlim())
            self.owner._drag_start_ylim = tuple(axes[0].get_ylim())
            try: self.grabMouse()
            except Exception: pass
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ClosedHandCursor))
            event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.owner._dragging:
            super().mouseMoveEvent(event); return
        if not (event.buttons() & QtCore.Qt.MouseButton.LeftButton):
            self._finish_drag(); event.accept(); return
        axes = getattr(self.owner, "_axes", [])
        chart_df = getattr(self.owner, "_chart_df", None)
        if not axes or chart_df is None or chart_df.empty: return
        geom = self._axis_geometry()
        if geom is None: return
        axis_left, axis_right, axis_bottom, axis_top, _, _ = geom
        axis_width = max(1.0, axis_right - axis_left)
        axis_height = max(1.0, axis_top - axis_bottom)
        current_x = float(event.position().x())
        current_y = float(event.position().y())
        dx = current_x - float(self.owner._drag_start_x)
        dy = current_y - float(self.owner._drag_start_y)
        left, right = self.owner._drag_start_xlim
        bottom, top = self.owner._drag_start_ylim
        xspan = right - left
        yspan = top - bottom
        x_delta = -dx * xspan / axis_width
        new_left = left + x_delta
        new_right = right + x_delta
        y_delta = dy * yspan / axis_height
        new_bottom = bottom + y_delta
        new_top = top + y_delta
        n = max(1, len(chart_df))
        x_margin = max(500.0, n * 2.0)
        x_floor = -x_margin
        x_ceiling = float(n - 1) + x_margin
        if xspan < x_ceiling - x_floor:
            if new_left < x_floor:
                shift = x_floor - new_left
                new_left += shift; new_right += shift
            if new_right > x_ceiling:
                shift = new_right - x_ceiling
                new_left -= shift; new_right -= shift
        y_margin = max(abs(yspan) * 20.0, 1.0)
        center = (bottom + top) / 2.0
        if new_bottom < center - y_margin:
            delta = (center - y_margin) - new_bottom
            new_bottom += delta; new_top += delta
        if new_top > center + y_margin:
            delta = new_top - (center + y_margin)
            new_bottom -= delta; new_top -= delta
        for ax in axes: ax.set_xlim(new_left, new_right)
        axes[0].set_ylim(new_bottom, new_top)
        self._throttled_draw(force=False)
        event.accept()

    def _finish_drag(self):
        self.owner._dragging = False
        self.owner._drag_start_x = None
        self.owner._drag_start_y = None
        self.owner._drag_start_xlim = None
        self.owner._drag_start_ylim = None
        try: self.releaseMouse()
        except Exception: pass
        self.owner._update_visible_xticks()
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
        self.draw_idle()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._finish_drag(); event.accept(); return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        axes = getattr(self.owner, "_axes", [])
        chart_df = getattr(self.owner, "_chart_df", None)
        if not axes or chart_df is None or chart_df.empty:
            super().wheelEvent(event); return
        geom = self._axis_geometry()
        if geom is None: return
        axis_left, axis_right, axis_bottom, axis_top, _, _ = geom
        axis_width = max(1.0, axis_right - axis_left)
        axis_height = max(1.0, axis_top - axis_bottom)
        left, right = axes[0].get_xlim()
        bottom, top = axes[0].get_ylim()
        xspan = right - left
        yspan = top - bottom
        x = float(event.position().x())
        y = float(event.position().y())
        xr = min(1.0, max(0.0, (x - axis_left) / axis_width))
        yr = min(1.0, max(0.0, (axis_top - y) / axis_height))
        cursor_x = left + xr * xspan
        cursor_y = bottom + yr * yspan
        angle = event.angleDelta().y()
        if angle > 0: scale = 0.80
        elif angle < 0: scale = 1.25
        else: return
        new_left = cursor_x - (cursor_x - left) * scale
        new_right = cursor_x + (right - cursor_x) * scale
        new_bottom = cursor_y - (cursor_y - bottom) * scale
        new_top = cursor_y + (top - cursor_y) * scale
        min_xspan = 15.0
        min_yspan = max(yspan * 0.03, 1e-6)
        if new_right - new_left < min_xspan:
            half = min_xspan / 2.0
            new_left = cursor_x - half
            new_right = cursor_x + half
        if new_top - new_bottom < min_yspan:
            half = min_yspan / 2.0
            new_bottom = cursor_y - half
            new_top = cursor_y + half
        for ax in axes: ax.set_xlim(new_left, new_right)
        axes[0].set_ylim(new_bottom, new_top)
        self.owner._update_visible_xticks()
        self.draw_idle()
        event.accept()

class MarketChartWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._dragging = False
        self._drag_start_x = None
        self._drag_start_y = None
        self._drag_start_xlim = None
        self._drag_start_ylim = None
        self._chart_df = pd.DataFrame()
        self._axes = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        top_bar = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel("Mum Grafiği")
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-size:15px;font-weight:bold;padding:4px;color:#000;")
        top_bar.addSpacing(68)
        top_bar.addStretch()
        top_bar.addWidget(self.title)
        top_bar.addStretch()
        self.zoom_in_btn = QtWidgets.QPushButton("+")
        self.zoom_in_btn.setFixedSize(32, 32)
        self.zoom_in_btn.setStyleSheet("QPushButton{font-size:18px;font-weight:bold;background:#ffffff;border:1px solid #cfd4dc;border-radius:4px;} QPushButton:hover{background:#f0f0f0;}")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        top_bar.addWidget(self.zoom_in_btn)
        self.zoom_out_btn = QtWidgets.QPushButton("-")
        self.zoom_out_btn.setFixedSize(32, 32)
        self.zoom_out_btn.setStyleSheet("QPushButton{font-size:18px;font-weight:bold;background:#ffffff;border:1px solid #cfd4dc;border-radius:4px;} QPushButton:hover{background:#f0f0f0;}")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        top_bar.addWidget(self.zoom_out_btn)
        layout.addLayout(top_bar)
        self.figure = Figure(figsize=(7, 6), dpi=100)
        self.figure.patch.set_facecolor("#ffffff")
        self.canvas = InteractiveChartCanvas(self.figure, self)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self.canvas, 1)
        self.show_placeholder()

    def show_placeholder(self):
        self.figure.clear()
        self._axes = []
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#ffffff")
        ax.text(0.5, 0.5, "Endeks veya hisse analizi\nbaşlatıldığında grafik burada görünecek.", ha="center", va="center", color="#6b7280", fontsize=12, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        self.figure.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.02)
        self.canvas.draw_idle()

    def _update_visible_xticks(self):
        if self._chart_df.empty or len(self._axes) < 2: return
        dates = pd.to_datetime(self._chart_df.index)
        ax = self._axes[-1]
        left, right = self._axes[0].get_xlim()
        first, last = max(0, int(np.floor(left))), min(len(dates) - 1, int(np.ceil(right)))
        if last < first: ax.set_xticks([]); return
        visible = last - first + 1
        slots = max(5, min(8, int(self.canvas.width() / 95)))
        step = max(1, int(np.ceil(visible / slots)))
        ticks = list(range(first, last + 1, step))
        if not ticks: ticks = [first]
        if last - ticks[-1] >= max(2, step // 2): ticks.append(last)
        ax.set_xticks(ticks)
        ax.set_xticklabels([dates[i].strftime("%d/%m") for i in ticks], rotation=45, ha="right")

    def _build_candles(self, ax, opens, highs, lows, closes):
        wick_up_segments = []
        wick_down_segments = []
        body_polys_up = []
        body_polys_down = []
        width = 0.62
        min_body = 1e-12
        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            if not all(np.isfinite(v) for v in (o, h, l, c)): continue
            if c >= o:
                wick_up_segments.append([(i, l), (i, h)])
            else:
                wick_down_segments.append([(i, l), (i, h)])
            lo = min(o, c); hi = max(o, c)
            if hi - lo < min_body: hi = lo + min_body
            poly = [(i - width / 2.0, lo), (i - width / 2.0, hi), (i + width / 2.0, hi), (i + width / 2.0, lo)]
            if c >= o:
                body_polys_up.append(poly)
            else:
                body_polys_down.append(poly)
        if wick_up_segments:
            ax.add_collection(LineCollection(wick_up_segments, linewidths=0.8, colors="#26a69a", zorder=2, capstyle="butt"))
        if wick_down_segments:
            ax.add_collection(LineCollection(wick_down_segments, linewidths=0.8, colors="#ef5350", zorder=2, capstyle="butt"))
        if body_polys_up:
            ax.add_collection(PolyCollection(body_polys_up, facecolors="#26a69a", edgecolors="#26a69a", linewidths=0.5, zorder=3))
        if body_polys_down:
            ax.add_collection(PolyCollection(body_polys_down, facecolors="#ef5350", edgecolors="#ef5350", linewidths=0.5, zorder=3))

    def _build_sqz_bars(self, ax, values):
        polys = []
        facecolors = []
        width = 0.72
        prev = np.nan
        for i, value in enumerate(values):
            if not np.isfinite(value):
                prev = value; continue
            color = "#808080"
            if value > 0:
                color = "#00FF00" if np.isfinite(prev) and value > prev else "#008000"
            elif value < 0:
                color = "#FF0000" if np.isfinite(prev) and value < prev else "#800000"
            y0, y1 = (0.0, value) if value >= 0 else (value, 0.0)
            polys.append([(i - width / 2.0, y0), (i - width / 2.0, y1), (i + width / 2.0, y1), (i + width / 2.0, y0)])
            facecolors.append(color)
            prev = value
        if polys:
            ax.add_collection(PolyCollection(polys, facecolors=facecolors, edgecolors=facecolors, linewidths=0.0, antialiaseds=False, zorder=2))

    def plot(self, hist, symbol):
        if hist is None or hist.empty:
            self.show_placeholder(); return
        self._chart_df = hist.copy()
        df = self._chart_df
        opens = pd.to_numeric(df["Open"], errors="coerce").to_numpy()
        highs = pd.to_numeric(df["High"], errors="coerce").to_numpy()
        lows = pd.to_numeric(df["Low"], errors="coerce").to_numpy()
        closes = pd.to_numeric(df["Close"], errors="coerce").to_numpy()
        sq = sqzmom(df)
        sq_series = pd.to_numeric(sq["series"].reindex(df.index), errors="coerce")
        self.figure.clear()
        self.figure.patch.set_facecolor("#ffffff")
        gs = self.figure.add_gridspec(2, 1, height_ratios=[3.55, 0.72], hspace=0.07)
        main_ax = self.figure.add_subplot(gs[0, 0])
        sqz_ax = self.figure.add_subplot(gs[1, 0], sharex=main_ax)
        self._axes = [main_ax, sqz_ax]
        for axis in self._axes:
            axis.set_facecolor("#ffffff")
            axis.grid(True, alpha=0.16, linewidth=0.7, color="#9ca3af")
            axis.tick_params(axis="both", colors="#4b5563", labelsize=8)
            for spine in axis.spines.values(): spine.set_color("#d1d5db")
        dc_upper = pd.to_numeric(df["High"], errors="coerce").rolling(20).max()
        dc_lower = pd.to_numeric(df["Low"], errors="coerce").rolling(20).min()
        dc_basis = (dc_upper + dc_lower) / 2
        x = np.arange(len(df))
        main_ax.fill_between(x, dc_lower, dc_upper, color="#2196F3", alpha=0.05, zorder=1)
        main_ax.plot(x, dc_upper.to_numpy(), linewidth=0.5, alpha=0.9, color="#2962FF", label="DC Upper", zorder=4)
        main_ax.plot(x, dc_lower.to_numpy(), linewidth=0.5, alpha=0.9, color="#2962FF", label="DC Lower", zorder=4)
        main_ax.plot(x, dc_basis.to_numpy(), linewidth=0.5, alpha=0.9, color="#FF6D00", label="DC Basis", zorder=4)
        self._build_candles(main_ax, opens, highs, lows, closes)
        main_ax.set_title(f"{symbol} • Heiken-Ashi", color="#20242b", fontsize=11, pad=4)
        main_ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#4b5563")
        main_ax.tick_params(axis="x", labelbottom=False)
        values = sq_series.to_numpy(dtype=float)
        self._build_sqz_bars(sqz_ax, values)
        sqz_ax.axhline(0, color="#6b7280", linewidth=0.8, alpha=0.85)
        sqz_ax.set_title("SQZMOM", loc="left", color="#20242b", fontsize=9, pad=1)
        n = len(df)
        visible = min(180, max(20, n))
        start = max(-0.7, n - visible - 0.7)
        end = float(max(n - 1, start + visible - 1)) + 0.7
        main_ax.set_xlim(start, end)
        sqz_ax.set_xlim(start, end)

        dc_upper_values = pd.to_numeric(dc_upper, errors="coerce").to_numpy(dtype=float)
        dc_lower_values = pd.to_numeric(dc_lower, errors="coerce").to_numpy(dtype=float)
        finite_prices = np.concatenate([
            highs[np.isfinite(highs)],
            lows[np.isfinite(lows)],
            dc_upper_values[np.isfinite(dc_upper_values)],
            dc_lower_values[np.isfinite(dc_lower_values)]
        ])
        if finite_prices.size:
            p_low, p_high = float(np.min(finite_prices)), float(np.max(finite_prices))
            p_pad = max((p_high - p_low) * 0.04, 0.01)
            main_ax.set_ylim(p_low - p_pad, p_high + p_pad)

        finite_sqz = values[np.isfinite(values)]
        if finite_sqz.size:
            s_low, s_high = float(np.min(finite_sqz)), float(np.max(finite_sqz))
            s_low, s_high = min(s_low, 0.0), max(s_high, 0.0)
            s_pad = max((s_high - s_low) * 0.12, 0.15)
            sqz_ax.set_ylim(s_low - s_pad, s_high + s_pad)
        else:
            sqz_ax.set_ylim(-1.0, 1.0)

        self.figure.subplots_adjust(left=0.055, right=0.995, top=0.968, bottom=0.075, hspace=0.045)
        self._update_visible_xticks()
        self.canvas.draw_idle()

    def zoom_in(self):
        if not self._axes: return
        self._zoom(0.8)

    def zoom_out(self):
        if not self._axes: return
        self._zoom(1.25)

    def _zoom(self, factor):
        main_ax = self._axes[0]
        left, right = main_ax.get_xlim()
        bottom, top = main_ax.get_ylim()
        x_center = (left + right) / 2
        y_center = (bottom + top) / 2
        new_xspan = (right - left) * factor
        new_yspan = (top - bottom) * factor
        new_left = x_center - new_xspan / 2
        new_right = x_center + new_xspan / 2
        new_bottom = y_center - new_yspan / 2
        new_top = y_center + new_yspan / 2
        for ax in self._axes:
            ax.set_xlim(new_left, new_right)
        main_ax.set_ylim(new_bottom, new_top)
        self._update_visible_xticks()
        self.canvas.draw_idle()

def _fix_yahoo_data(data):
    if data is None or data.empty: return None
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index)
    if 'Volume' not in data.columns: data['Volume'] = 0
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required[:4]): return None
    return data[required].sort_index()

def get_data_yahoo(symbol, period='5y'):
    try:
        kw = {'interval': '1d', 'period': period, 'auto_adjust': False, 'progress': False, 'threads': False}
        data = yf.download(symbol, **kw)
        if data is None or data.empty:
            print(f"⚠️ Yahoo veri yok: {symbol}"); return None
        data = _fix_yahoo_data(data)
        if data is None or data.empty:
            print(f"⚠️ Yahoo kolonları geçersiz: {symbol}"); return None
        return data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    except Exception as e:
        print(f"❌ Yahoo veri hatası ({symbol}): {e}")
        return None

def get_technical(symbol, period='5y'):
    symbol = symbol.upper().strip()
    if not symbol.endswith('.IS'):
        symbol += '.IS'
    base = symbol[:-3]
    index_candidates = {
        'XU100': ['XU100.IS', '^XU100', 'XU100', 'BIST100.IS', '^BIST100'],
        'BIST100': ['XU100.IS', '^XU100', 'BIST100.IS', '^BIST100', 'XU100'],
        'XU030': ['XU030.IS', '^XU030', 'XU030'],
        'XU050': ['XU050.IS', '^XU050', 'XU050'],
        'XUTUM': ['XUTUM.IS', '^XUTUM', 'XUTUM']
    }
    if base in index_candidates:
        candidates = index_candidates[base]
    else:
        candidates = [symbol]
    seen = set()
    for sym in candidates:
        if sym in seen: continue
        seen.add(sym)
        print(f"🔎 Yahoo deneniyor: {sym}")
        data = get_data_yahoo(sym, period)
        if data is None or data.empty: continue
        try:
            ha = heiken_ashi(data)
            result = indicators(ha, sym)
            if result is not None:
                result["_chart_df"] = ha
            return result
        except Exception as e:
            print(f"❌ Teknik analiz hatası ({sym}): {e}")
            traceback.print_exc()
    print(f"❌ Yahoo'dan veri alınamadı: {symbol}")
    return None

def format_technical_data_for_prompt(data):
    destek = max(data["dc_lower"], data["bb_lower"])
    direnc = min(data["dc_upper"], data["bb_upper"])
    return f"""
Sembol: {data['symbol']}
Güncel Fiyat: {data['price']:.2f}
5 Günlük Değişim: %{data['price_change']:.2f}
Donchian Üst: {data['dc_upper']:.2f} | Alt: {data['dc_lower']:.2f} | Basis: {data['dc_basis']:.2f}
Bollinger Üst: {data['bb_upper']:.2f} | Orta: {data['bb_middle']:.2f} | Alt: {data['bb_lower']:.2f}
Yakın Destek: {destek:.2f} | Yakın Direnç: {direnc:.2f}
Hacim: {data['volume']:.0f} | Ortalama Hacim: {data['volume_avg']:.0f}
SQZMOM: {data['sqzmom_value']:.3f} | Renk: {data['sqzmom_color']} | Squeeze: {data['sqzmom_squeeze']}
"""

def ask_llm_analysis(data):
    prompt = f"""
Sen profesyonel bir teknik analiz asistanısın. Verilerden hareketle Türkçe, objektif ve detaylı bir hisse değerlendirmesi yap. Yatırım tavsiyesi verme; "olası/izlenebilecek" seviyeler olarak konuş. 10-12 cümle kullan.
Şunları mutlaka belirt:
1) Ana trend ve momentum.
2) Donchian, Bollinger, hacim ve SQZMOM yorumu.
3) Yakın destek ve direnç seviyeleri.
4) Olası alım bölgesi: hangi destek çevresi ve hangi koşulda alım teyidi aranır?
5) Olası satış/kar alma bölgesi: hangi direnç çevresi ve hangi koşulda çıkış düşünülür?
6) Destek kırılımında izlenecek risk/stop seviyesi.
7) Kısa vadeli ve orta vadeli senaryoyu ayrı ayrı özetle.
Seviyeleri gereksiz yere yuvarlama ve verilen fiyatlardan sapma.

{format_technical_data_for_prompt(data)}
"""
    try:
        r = model.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.4,
            top_p=0.85,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Analiz üretilemedi: {e}"

def analyze_stock(symbol):
    symbol = symbol.upper().strip()
    data = get_technical(symbol)
    if not data: return None
    report = ask_llm_analysis(data)
    return {"report": report, "data": data, "symbol": data["symbol"], "chart": data.get("_chart_df")}

def analyze_market():
    data = get_technical("XU100")
    if not data: return None
    report = ask_llm_analysis(data)
    return {"report": report, "data": data, "symbol": data["symbol"], "chart": data.get("_chart_df")}

system = "Sen profesyonel bir borsa analistisin. Yanıtların Türkçe, objektif, net ve kısa olmalı. Yatırım tavsiyesi verme."
messages = [{"role": "system", "content": system}]

class FinanceWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    def __init__(self, kind, value):
        super().__init__()
        self.kind = kind
        self.value = value
    def run(self):
        try:
            result = analyze_market() if self.kind == "market" else analyze_stock(self.value)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))

class ChatWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(str)
    def __init__(self, user_text, history):
        super().__init__()
        self.user_text = user_text
        self.history = [dict(m) for m in history]
    def run(self):
        try:
            history = self.history + [{"role": "user", "content": self.user_text}]
            r = model.create_chat_completion(
                messages=history,
                max_tokens=500,
                temperature=0.5,
                top_p=0.9,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            self.finished.emit(r["choices"][0]["message"]["content"].strip())
        except Exception as e:
            self.finished.emit(f"Model hatası: {e}")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pusula AI")
        self.resize(1280, 760)
        self.setMinimumSize(900, 600)
        self.finance_worker = None
        self.chat_worker = None
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        header = QtWidgets.QFrame()
        header.setFixedHeight(52)
        header.setStyleSheet("QFrame{background:#f3f4f6;border-bottom:1px solid #d7dbe2;}")
        hv = QtWidgets.QHBoxLayout(header)
        hv.setContentsMargins(14, 6, 14, 6)
        title = QtWidgets.QLabel("Pusula AI • Atalay Teknoloji ve Yatırım Girişimi")
        title.setStyleSheet("color:#ff9800;font-size:16px;font-weight:bold;")
        hv.addWidget(title)
        hv.addStretch()
        self.status = QtWidgets.QLabel("Hazır")
        self.status.setStyleSheet("color:#6b7280;font-size:11px;")
        hv.addWidget(self.status)
        root.addWidget(header)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(4)
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter, 1)
        self.chat_panel = QtWidgets.QFrame()
        self.chat_panel.setStyleSheet("QFrame{background:#ffffff;}")
        chat_layout = QtWidgets.QVBoxLayout(self.chat_panel)
        chat_layout.setContentsMargins(10, 10, 10, 10)
        chat_layout.setSpacing(8)
        self.chat_title = QtWidgets.QLabel("Sohbet Alanı")
        self.chat_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.chat_title.setStyleSheet("font-size:14px;font-weight:bold;color:#20242b;padding:4px;")
        chat_layout.addWidget(self.chat_title)
        self.chat = QtWidgets.QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setStyleSheet("QTextEdit{background:#ffffff;color:#20242b;border:1px solid #d7dbe2;border-radius:6px;padding:8px;font-size:12px;}")
        chat_layout.addWidget(self.chat, 1)
        row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Mesaj yazın...")
        self.input.setStyleSheet("QLineEdit{background:#ffffff;color:#20242b;border:1px solid #cfd4dc;border-radius:6px;padding:9px;}")
        self.input.returnPressed.connect(self.send_message)
        row.addWidget(self.input, 1)
        self.send_btn = QtWidgets.QPushButton("Gönder")
        self.send_btn.setFixedWidth(80)
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setStyleSheet("QPushButton{background:#ff9800;color:#111;border:0;border-radius:6px;padding:9px;font-weight:bold;}QPushButton:hover{background:#ffad33;}")
        row.addWidget(self.send_btn)
        chat_layout.addLayout(row)
        self.chart_panel = MarketChartWidget()
        self.chart_panel.setMinimumWidth(400)
        self.splitter.addWidget(self.chat_panel)
        self.splitter.addWidget(self.chart_panel)
        self.chart_panel.hide()

    def append_system(self, text):
        safe = text.replace("\n", "<br>")
        self.chat.append(f'<div style="color:#ff9800;"><b>PUSULA AI</b></div><div style="color:#aaaaaa;">{safe}</div><br>')

    def append_user(self, text):
        safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        self.chat.append(f'<div style="color:#1976d2;"><b>SİZ</b></div><div style="color:#20242b;">{safe}</div><br>')

    def append_ai(self, text):
        safe = (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>"))
        self.chat.append(f'<div style="color:#2e7d32;"><b>PUSULA AI</b></div><div style="color:#20242b;">{safe}</div><br>')

    def set_chat_mode(self):
        self.chart_panel.hide()
        self.splitter.setSizes([self.width(), 0])
        self.chat_title.setText("Sohbet Alanı")
        self.status.setText("Sohbet modu")

    def set_finance_mode(self, chart_df, symbol):
        self.chart_panel.show()
        self.splitter.setSizes([self.width() // 2, self.width() // 2])
        self.chat_title.setText("Sohbet Alanı")
        self.status.setText(f"Finans analizi • {symbol}")
        if chart_df is not None: self.chart_panel.plot(chart_df, symbol)

    def send_message(self):
        global messages
        text = self.input.text().strip()
        if not text: return
        self.append_user(text)
        self.input.clear()
        self.send_btn.setEnabled(False)
        self.input.setEnabled(False)
        if is_market_query(text):
            self.set_finance_mode(None, "XU100")
            self.chat.append('<div style="color:#ff9800;"><b>🔎 XU100 analizi yapılıyor...</b></div><br>')
            self.finance_worker = FinanceWorker("market", "XU100")
            self.finance_worker.finished.connect(self.on_finance_finished)
            self.finance_worker.failed.connect(self.on_worker_error)
            self.finance_worker.start()
            return
        sym = extract_symbol_from_text(text)
        if sym:
            self.set_finance_mode(None, sym)
            self.chat.append(f'<div style="color:#ff9800;"><b>🔎 {sym} analizi yapılıyor...</b></div><br>')
            self.finance_worker = FinanceWorker("stock", sym)
            self.finance_worker.finished.connect(self.on_finance_finished)
            self.finance_worker.failed.connect(self.on_worker_error)
            self.finance_worker.start()
            return
        self.set_chat_mode()
        messages.append({"role": "user", "content": text})
        self.chat_worker = ChatWorker(text, messages[:-1])
        self.chat_worker.finished.connect(self.on_chat_finished)
        self.chat_worker.start()

    def on_finance_finished(self, result):
        global messages
        self.send_btn.setEnabled(True)
        self.input.setEnabled(True)
        if not result:
            self.set_chat_mode()
            self.append_ai("Veri alınamadı veya analiz yapılamadı.")
            return
        report = result["report"]
        symbol = result["symbol"]
        chart_df = result["chart"]
        self.set_finance_mode(chart_df, symbol)
        self.append_ai(report + "\n\n⚠️ Bu analiz yatırım tavsiyesi değildir.")
        messages.append({"role": "assistant", "content": report})

    def on_chat_finished(self, reply):
        global messages
        self.send_btn.setEnabled(True)
        self.input.setEnabled(True)
        self.append_ai(reply)
        messages.append({"role": "assistant", "content": reply})

    def on_worker_error(self, error):
        self.send_btn.setEnabled(True)
        self.input.setEnabled(True)
        self.append_ai(f"Hata: {error}")
        self.status.setText("Hata")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#f5f6f8"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#f0f2f5"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#20242b"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#20242b"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#20242b"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#ff9800"))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#111111"))
    app.setPalette(palette)

    files = glob.glob("./turkish-llm-32b-gguf/*.gguf")
    if not files:
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Pusula AI")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel("Pusula AI modeli bulunamadı! Model indiriliyor...")
        progress = QtWidgets.QProgressBar()
        progress.setRange(0, 0)
        layout.addWidget(label)
        layout.addWidget(progress)
        dialog.show()
        thread = DownloadThread()
        thread.finished.connect(dialog.accept)
        thread.failed.connect(lambda err: (QtWidgets.QMessageBox.critical(dialog, "Hata", f"Model indirilemedi: {err}"), dialog.reject()))
        thread.start()
        dialog.exec()
        files = glob.glob("./turkish-llm-32b-gguf/*.gguf")
        if not files:
            QtWidgets.QMessageBox.critical(None, "Hata", "Model indirildi ancak dosya bulunamadı.")
            sys.exit(1)

    try:
        model = Llama(model_path=files[0], n_ctx=8192, n_threads=8, n_gpu_layers=-1, verbose=False)
        print("✅ Model tamamen GPU'da yüklendi.")
    except Exception as e:
        print(f"⚠️ Tam GPU yükleme hatası: {e}")
        try:
            model = Llama(model_path=files[0], n_ctx=8192, n_threads=8, n_gpu_layers=28, verbose=False)
            print("✅ Model kısmi GPU (28 katman) ile yüklendi.")
        except Exception as e2:
            print(f"⚠️ Kısmi GPU yükleme hatası: {e2}")
            model = Llama(model_path=files[0], n_ctx=8192, n_threads=8, n_gpu_layers=0, verbose=False)
            print("✅ Model CPU'da yüklendi.")

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
