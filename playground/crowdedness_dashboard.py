"""
交易拥挤度 (Trading Crowdedness) Dashboard
============================================
抓取 NVDA / META / GOOGL / MSFT / TSLA / AAPL / AMZN 过去一年量价数据，
计算交易拥挤度指标并用 Plotly + Streamlit 展示。

指标设计
--------
交易拥挤度由 5 个子指标等权合成，每个子指标先做滚动百分位归一化 (0-100):

1. 成交量激增比  Volume / MA(Volume, short_window)
2. 波动率扩张比  RealizedVol(5d) / RealizedVol(long_window)
3. 量价相关性    RollingCorr(|Return|, Volume, short_window)
4. 收益自相关    RollingAutoCorr(Return, short_window)
5. 换手集中度    RollingStd(Volume / MA(Volume, long_window), short_window)

运行: streamlit run crowdedness_dashboard.py
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = ["NVDA", "META", "GOOGL", "MSFT", "TSLA", "AAPL", "AMZN"]

TICKER_NAMES = {
    "NVDA": "NVIDIA",
    "META": "Meta",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "AMZN": "Amazon",
}

DEFAULT_COLORS = {
    "NVDA": "#76B900",
    "META": "#0081FB",
    "GOOGL": "#EA4335",
    "MSFT": "#00A4EF",
    "TSLA": "#CC0000",
    "AAPL": "#A2AAAD",
    "AMZN": "#FF9900",
}

# Plotly 默认调色板，用于为用户自定义的股票分配颜色
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def get_color(ticker: str, idx: int) -> str:
    """返回 ticker 对应的颜色，优先使用预设颜色，否则从调色板分配。"""
    return DEFAULT_COLORS.get(ticker, _PALETTE[idx % len(_PALETTE)])


# ---------------------------------------------------------------------------
# 数据获取
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="正在从 Yahoo Finance 拉取数据 ...")
def fetch_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """拉取指定股票在 [start, end] 区间的日线 OHLCV 数据，返回 {ticker: DataFrame}。"""
    if len(tickers) == 1:
        raw = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        df = raw.dropna()
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
        return {tickers[0]: df}
    raw = yf.download(tickers, start=start, end=end, interval="1d", group_by="ticker", auto_adjust=True, progress=False)
    result = {}
    for tk in tickers:
        try:
            df = raw[tk].dropna()
            df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
            result[tk] = df
        except KeyError:
            pass
    return result


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """滚动百分位排名，将值映射到 [0, 100]。"""
    def _pct(arr):
        if np.isnan(arr[-1]):
            return np.nan
        return (arr[:-1] <= arr[-1]).sum() / (len(arr) - 1) * 100
    return series.rolling(window + 1, min_periods=window + 1).apply(_pct, raw=True)


def compute_crowdedness(df: pd.DataFrame, short_win: int = 20, long_win: int = 60) -> pd.DataFrame:
    """
    计算单只股票的交易拥挤度。

    Parameters
    ----------
    df : OHLCV DataFrame
    short_win : 短期窗口 (默认 20 天)
    long_win  : 长期窗口 (默认 60 天)

    Returns
    -------
    DataFrame 包含 5 个子指标 + 合成拥挤度
    """
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze().astype(float)
    ret = close.pct_change()

    # 1. 成交量激增比
    vol_surge = volume / volume.rolling(short_win).mean()

    # 2. 波动率扩张比
    vol_short = ret.rolling(5).std() * np.sqrt(252)
    vol_long = ret.rolling(long_win).std() * np.sqrt(252)
    vol_expansion = vol_short / vol_long

    # 3. 量价相关性
    abs_ret = ret.abs()
    pv_corr = abs_ret.rolling(short_win).corr(volume)

    # 4. 收益自相关
    autocorr_win = max(short_win * 2, 40)
    ret_autocorr = ret.rolling(autocorr_win).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 2 else np.nan,
        raw=False,
    )

    # 5. 换手集中度
    turnover_ratio = volume / volume.rolling(long_win).mean()
    turnover_conc = turnover_ratio.rolling(short_win).std()

    # 百分位归一化 (使用 long_win 做历史参考窗口)
    rank_win = min(252, max(long_win, len(df) - long_win))
    sub = pd.DataFrame({
        "vol_surge": rolling_percentile(vol_surge, rank_win),
        "vol_expansion": rolling_percentile(vol_expansion, rank_win),
        "pv_corr": rolling_percentile(pv_corr, rank_win),
        "ret_autocorr": rolling_percentile(ret_autocorr, rank_win),
        "turnover_conc": rolling_percentile(turnover_conc, rank_win),
    }, index=df.index)

    sub["crowdedness"] = sub.mean(axis=1)
    sub["close"] = close.values
    sub["volume"] = volume.values
    sub["ret"] = ret.values
    return sub


def detect_sell_signals(
    sub: pd.DataFrame,
    crowd_threshold: int = 75,
    rsi_period: int = 14,
    lookback: int = 20,
) -> pd.Series:
    """
    识别"坏的"拥挤度卖点信号。

    卖点 = 高拥挤度 + 价格近期高位 + 至少一个衰竭确认:
      1. 成交量 climax (Z-score > 2)
      2. RSI 顶背离 (价格创新高但 RSI 未创新高)
      3. 量价背离 (价格上涨但成交量趋势下降)

    Returns
    -------
    bool Series, True 表示该日出现卖出信号
    """
    close = sub["close"]
    volume = sub["volume"]
    ret = sub["ret"]

    # ---- RSI ----
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    # ---- 条件 1: 高拥挤度 ----
    high_crowd = sub["crowdedness"] >= crowd_threshold

    # ---- 条件 2: 价格接近近期高点 (在 lookback 日最高价的 95% 以上) ----
    rolling_max = close.rolling(lookback, min_periods=1).max()
    near_high = close >= rolling_max * 0.95

    # ---- 衰竭确认 (至少满足 1 个) ----

    # 2a. 成交量 climax: volume Z-score > 2
    vol_ma = volume.rolling(lookback).mean()
    vol_std = volume.rolling(lookback).std()
    vol_zscore = (volume - vol_ma) / vol_std.replace(0, np.nan)
    volume_climax = vol_zscore > 2

    # 2b. RSI 顶背离: 价格创 lookback 日新高，但 RSI 较前期明显衰减
    price_at_high = close >= close.rolling(lookback, min_periods=1).max()
    rsi_momentum_decay = rsi < rsi.shift(lookback // 2).rolling(5).mean() - 5
    rsi_divergence = price_at_high & rsi_momentum_decay & (rsi > 60)

    # 2c. 量价背离: 近 lookback/2 日收益为正，但均量趋势下降
    half = max(lookback // 2, 5)
    price_up = close.pct_change(half) > 0.02  # 价格上涨 >2%
    vol_recent_avg = volume.rolling(half).mean()
    vol_prior_avg = vol_recent_avg.shift(half)
    vol_declining = (vol_recent_avg / vol_prior_avg - 1) < -0.1  # 均量下降 >10%
    pv_divergence = price_up & vol_declining

    exhaustion = volume_climax | rsi_divergence | pv_divergence

    sell_signal = high_crowd & near_high & exhaustion
    return sell_signal.fillna(False)


# ---------------------------------------------------------------------------
# Plotly 绘图
# ---------------------------------------------------------------------------
def plot_all_crowdedness(crowd_dict: dict[str, pd.DataFrame], sell_dict: dict[str, pd.Series]) -> go.Figure:
    """所有股票拥挤度时序对比折线图，红点标注卖出信号。"""
    fig = go.Figure()
    first_sell = True  # 只在图例中显示一次"卖出信号"
    for i, tk in enumerate(crowd_dict):
        s = crowd_dict[tk]["crowdedness"].dropna()
        label = f"{tk} ({TICKER_NAMES[tk]})" if tk in TICKER_NAMES else tk
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines", name=label,
            line=dict(color=get_color(tk, i), width=2),
            hovertemplate=f"{tk}<br>日期: %{{x|%Y-%m-%d}}<br>拥挤度: %{{y:.1f}}<extra></extra>",
        ))
        # 卖出信号红点
        sell_mask = sell_dict[tk].reindex(s.index).fillna(False)
        sell_pts = s[sell_mask]
        if len(sell_pts) > 0:
            fig.add_trace(go.Scatter(
                x=sell_pts.index, y=sell_pts.values,
                mode="markers",
                name="卖出信号" if first_sell else None,
                showlegend=first_sell,
                marker=dict(color="red", size=9, symbol="circle",
                            line=dict(color="darkred", width=1)),
                hovertemplate=f"{tk} 卖出信号<br>日期: %{{x|%Y-%m-%d}}<br>拥挤度: %{{y:.1f}}<extra></extra>",
            ))
            first_sell = False
    fig.update_layout(
        title="交易拥挤度时序对比",
        xaxis_title="日期",
        yaxis_title="拥挤度 (0-100)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=520,
        template="plotly_white",
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="高拥挤", annotation_position="bottom right")
    fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5,
                  annotation_text="低拥挤", annotation_position="top right")
    return fig


def plot_latest_ranking(crowd_dict: dict[str, pd.DataFrame]) -> go.Figure:
    """最新拥挤度排名柱状图。"""
    latest = {}
    for tk in crowd_dict:
        s = crowd_dict[tk]["crowdedness"].dropna()
        if len(s) > 0:
            latest[tk] = s.iloc[-1]
    sorted_items = sorted(latest.items(), key=lambda x: x[1], reverse=True)
    tickers = [t for t, _ in sorted_items]
    values = [v for _, v in sorted_items]
    all_tks = list(crowd_dict.keys())
    colors = [get_color(t, all_tks.index(t)) for t in tickers]
    labels = [f"{t} ({TICKER_NAMES[t]})" if t in TICKER_NAMES else t for t in tickers]

    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.1f}" for v in values], textposition="outside",
        hovertemplate="%{x}<br>拥挤度: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title="最新交易拥挤度排名",
        yaxis_title="拥挤度 (0-100)",
        yaxis_range=[0, 105],
        height=450,
        template="plotly_white",
    )
    return fig


def plot_single_detail(tk: str, sub: pd.DataFrame, sell_signal: pd.Series, color_idx: int = 0) -> go.Figure:
    """单只股票详情: 拥挤度 + 收盘价 + 成交量 三子图联动，红点标注卖出信号。"""
    display_name = f"{tk} ({TICKER_NAMES[tk]})" if tk in TICKER_NAMES else tk
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=(
            f"{display_name} 拥挤度",
            "收盘价",
            "成交量",
        ),
        row_heights=[0.4, 0.35, 0.25],
    )
    data = sub.dropna(subset=["crowdedness"])
    sell_mask = sell_signal.reindex(data.index).fillna(False)

    # 拥挤度
    tk_color = get_color(tk, color_idx)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["crowdedness"],
        mode="lines", name="拥挤度",
        line=dict(color=tk_color, width=2),
        hovertemplate="拥挤度: %{y:.1f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.4, row=1, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.4, row=1, col=1)

    # 拥挤度面板上的卖出信号红点
    sell_crowd = data.loc[sell_mask, "crowdedness"]
    if len(sell_crowd) > 0:
        fig.add_trace(go.Scatter(
            x=sell_crowd.index, y=sell_crowd.values,
            mode="markers", name="卖出信号",
            marker=dict(color="red", size=10, symbol="circle",
                        line=dict(color="darkred", width=1.5)),
            hovertemplate="卖出信号<br>拥挤度: %{y:.1f}<extra></extra>",
        ), row=1, col=1)

    # 子指标区域 (半透明)
    for col_name, label in [
        ("vol_surge", "成交量激增"),
        ("vol_expansion", "波动率扩张"),
        ("pv_corr", "量价相关"),
        ("ret_autocorr", "收益自相关"),
        ("turnover_conc", "换手集中"),
    ]:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[col_name],
            mode="lines", name=label,
            line=dict(width=1, dash="dot"), opacity=0.5,
            hovertemplate=f"{label}: %{{y:.1f}}<extra></extra>",
        ), row=1, col=1)

    # 收盘价
    fig.add_trace(go.Scatter(
        x=data.index, y=data["close"],
        mode="lines", name="收盘价",
        line=dict(color="#333", width=1.5),
        hovertemplate="价格: $%{y:.2f}<extra></extra>",
    ), row=2, col=1)

    # 收盘价面板上的卖出信号红点
    sell_price = data.loc[sell_mask, "close"]
    if len(sell_price) > 0:
        fig.add_trace(go.Scatter(
            x=sell_price.index, y=sell_price.values,
            mode="markers", name="卖出信号 (价格)",
            marker=dict(color="red", size=10, symbol="triangle-down",
                        line=dict(color="darkred", width=1.5)),
            hovertemplate="卖出信号<br>价格: $%{y:.2f}<extra></extra>",
            showlegend=False,
        ), row=2, col=1)

    # 成交量
    fig.add_trace(go.Bar(
        x=data.index, y=data["volume"],
        name="成交量", marker_color=tk_color, opacity=0.6,
        hovertemplate="成交量: %{y:,.0f}<extra></extra>",
    ), row=3, col=1)

    fig.update_layout(
        height=750, template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="拥挤度", row=1, col=1)
    fig.update_yaxes(title_text="USD", row=2, col=1)
    fig.update_yaxes(title_text="成交量", row=3, col=1)
    return fig


# ---------------------------------------------------------------------------
# Streamlit 主界面
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="交易拥挤度 Dashboard", layout="wide")
    st.title("交易拥挤度 Dashboard")
    st.caption("数据来源: Yahoo Finance | 指标: 5 维度合成拥挤度")

    # ---- 侧边栏 ----
    st.sidebar.header("股票选择")
    ticker_input = st.sidebar.text_input(
        "股票代码 (逗号分隔)",
        value=", ".join(DEFAULT_TICKERS),
        help="输入 Yahoo Finance 股票代码，用逗号分隔，例如: NVDA, AAPL, 005930.KS",
    )
    # 解析用户输入的 ticker
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if not tickers:
        st.error("请输入至少一个股票代码")
        return

    st.sidebar.header("日期范围")
    default_end = date.today()
    default_start = default_end - timedelta(days=365)
    col_start, col_end = st.sidebar.columns(2)
    start_date = col_start.date_input("开始日期", value=default_start)
    end_date = col_end.date_input("结束日期", value=default_end)
    if start_date >= end_date:
        st.error("开始日期必须早于结束日期")
        return

    st.sidebar.header("参数设置")
    short_win = st.sidebar.slider("短期窗口 (天)", 10, 40, 20)
    long_win = st.sidebar.slider("长期窗口 (天)", 40, 120, 60)

    st.sidebar.header("卖出信号")
    crowd_threshold = st.sidebar.slider("拥挤度阈值", 50, 95, 75,
                                        help="拥挤度超过此值才可能触发卖出信号")

    # 拉取数据
    data_dict = fetch_data(tickers, str(start_date), str(end_date))
    if not data_dict:
        st.error("未能获取任何股票数据，请检查代码是否正确")
        return

    # 过滤掉拉取失败的 ticker
    valid_tickers = list(data_dict.keys())

    # 单只详情选择器 (放在数据加载后，只显示有效 ticker)
    st.sidebar.header("详情查看")
    selected = st.sidebar.selectbox(
        "单只股票详情",
        valid_tickers,
        format_func=lambda t: f"{t} ({TICKER_NAMES[t]})" if t in TICKER_NAMES else t,
    )

    # 计算拥挤度 + 卖出信号
    crowd_dict: dict[str, pd.DataFrame] = {}
    sell_dict: dict[str, pd.Series] = {}
    for tk in valid_tickers:
        crowd_dict[tk] = compute_crowdedness(data_dict[tk], short_win, long_win)
        sell_dict[tk] = detect_sell_signals(crowd_dict[tk], crowd_threshold, lookback=short_win)

    # 全局对比
    st.plotly_chart(plot_all_crowdedness(crowd_dict, sell_dict), use_container_width=True)

    # 排名
    st.plotly_chart(plot_latest_ranking(crowd_dict), use_container_width=True)

    # 单只详情
    display_name = f"{selected} ({TICKER_NAMES[selected]})" if selected in TICKER_NAMES else selected
    st.subheader(f"{display_name} 详细分析")
    color_idx = valid_tickers.index(selected)
    st.plotly_chart(plot_single_detail(selected, crowd_dict[selected], sell_dict[selected], color_idx), use_container_width=True)

    # 指标说明
    with st.expander("指标说明"):
        st.markdown("""
**交易拥挤度** 由以下 5 个子指标等权合成，每个子指标先做滚动百分位归一化 (0-100):

| 子指标 | 公式 | 含义 |
|--------|------|------|
| 成交量激增比 | Volume / MA(Volume, 短期窗口) | 短期放量程度 |
| 波动率扩张比 | RealizedVol(5d) / RealizedVol(长期窗口) | 短期波动相对长期的偏离 |
| 量价相关性 | RollingCorr(\\|Return\\|, Volume, 短期窗口) | 价格与成交量的同步性 |
| 收益自相关 | AutoCorr(Return, lag=1, 短期窗口) | 动量追逐 / 趋势拥挤 |
| 换手集中度 | RollingStd(Volume / MA(Volume, 长期窗口), 短期窗口) | 成交量分布的集中程度 |

- **拥挤度 > 80**: 交易过度集中，注意风险
- **拥挤度 < 20**: 交易清淡，可能存在机会

---

**卖出信号 (红点)** 区分"好的拥挤"与"坏的拥挤"，仅在同时满足以下条件时触发:

1. **高拥挤度**: 合成拥挤度 >= 阈值 (侧边栏可调)
2. **价格近期高位**: 收盘价位于滚动最高价的 95% 以上
3. **至少一个衰竭确认**:
   - 成交量 Climax: 成交量 Z-score > 2 (相对滚动均值的极端放量)
   - RSI 顶背离: 价格创新高但 RSI 未同步创新高 (动量衰减)
   - 量价背离: 价格上涨但成交量趋势下降 (买盘枯竭)

> "好的拥挤"(健康突破、持续放量) 不会触发卖出信号，只有出现衰竭特征的"坏的拥挤"才会标记。
""")


if __name__ == "__main__":
    main()
