"""
交易拥挤度 (Trading Crowdedness) Dashboard
============================================
抓取 NVDA / META / GOOGL / MSFT / TSLA / AAPL / AMZN 过去一年量价数据，
计算交易拥挤度指标并用 Plotly + Streamlit 展示。

指标设计
--------
交易拥挤度由 5 个子指标等权合成，每个子指标先做滚动百分位归一化 (0-100):

1. 成交量激增比  Volume / MA(Volume, short_window)
2. 波动率压缩比  RealizedVol(5d) / RealizedVol(long_window)
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
        raw = yf.download(tickers, start=start, end=end, interval="1d", progress=False)
        df = raw.dropna()
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
        return {tickers[0]: df}
    raw = yf.download(tickers, start=start, end=end, interval="1d", group_by="ticker", progress=False)
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

    # 2. 波动率压缩比
    vol_short = ret.rolling(5).std() * np.sqrt(252)
    vol_long = ret.rolling(long_win).std() * np.sqrt(252)
    vol_ratio = vol_short / vol_long

    # 3. 量价相关性
    abs_ret = ret.abs()
    pv_corr = abs_ret.rolling(short_win).corr(volume)

    # 4. 收益自相关
    ret_autocorr = ret.rolling(short_win).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 2 else np.nan,
        raw=False,
    )

    # 5. 换手集中度
    turnover_ratio = volume / volume.rolling(long_win).mean()
    turnover_conc = turnover_ratio.rolling(short_win).std()

    # 百分位归一化 (使用 long_win 做历史参考窗口)
    rank_win = long_win
    sub = pd.DataFrame({
        "vol_surge": rolling_percentile(vol_surge, rank_win),
        "vol_ratio": rolling_percentile(vol_ratio, rank_win),
        "pv_corr": rolling_percentile(pv_corr, rank_win),
        "ret_autocorr": rolling_percentile(ret_autocorr, rank_win),
        "turnover_conc": rolling_percentile(turnover_conc, rank_win),
    }, index=df.index)

    sub["crowdedness"] = sub.mean(axis=1)
    sub["close"] = close.values
    sub["volume"] = volume.values
    return sub


# ---------------------------------------------------------------------------
# Plotly 绘图
# ---------------------------------------------------------------------------
def plot_all_crowdedness(crowd_dict: dict[str, pd.DataFrame]) -> go.Figure:
    """所有股票拥挤度时序对比折线图。"""
    fig = go.Figure()
    for i, tk in enumerate(crowd_dict):
        s = crowd_dict[tk]["crowdedness"].dropna()
        label = f"{tk} ({TICKER_NAMES[tk]})" if tk in TICKER_NAMES else tk
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines", name=label,
            line=dict(color=get_color(tk, i), width=2),
            hovertemplate=f"{tk}<br>日期: %{{x|%Y-%m-%d}}<br>拥挤度: %{{y:.1f}}<extra></extra>",
        ))
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


def plot_single_detail(tk: str, sub: pd.DataFrame, color_idx: int = 0) -> go.Figure:
    """单只股票详情: 拥挤度 + 收盘价 + 成交量 三子图联动。"""
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

    # 子指标区域 (半透明)
    for col_name, label in [
        ("vol_surge", "成交量激增"),
        ("vol_ratio", "波动率压缩"),
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

    # 计算拥挤度
    crowd_dict: dict[str, pd.DataFrame] = {}
    for tk in valid_tickers:
        crowd_dict[tk] = compute_crowdedness(data_dict[tk], short_win, long_win)

    # 全局对比
    st.plotly_chart(plot_all_crowdedness(crowd_dict), use_container_width=True)

    # 排名
    st.plotly_chart(plot_latest_ranking(crowd_dict), use_container_width=True)

    # 单只详情
    display_name = f"{selected} ({TICKER_NAMES[selected]})" if selected in TICKER_NAMES else selected
    st.subheader(f"{display_name} 详细分析")
    color_idx = valid_tickers.index(selected)
    st.plotly_chart(plot_single_detail(selected, crowd_dict[selected], color_idx), use_container_width=True)

    # 指标说明
    with st.expander("指标说明"):
        st.markdown("""
**交易拥挤度** 由以下 5 个子指标等权合成，每个子指标先做滚动百分位归一化 (0-100):

| 子指标 | 公式 | 含义 |
|--------|------|------|
| 成交量激增比 | Volume / MA(Volume, 短期窗口) | 短期放量程度 |
| 波动率压缩比 | RealizedVol(5d) / RealizedVol(长期窗口) | 短期波动相对长期的偏离 |
| 量价相关性 | RollingCorr(\\|Return\\|, Volume, 短期窗口) | 价格与成交量的同步性 |
| 收益自相关 | AutoCorr(Return, lag=1, 短期窗口) | 动量追逐 / 趋势拥挤 |
| 换手集中度 | RollingStd(Volume / MA(Volume, 长期窗口), 短期窗口) | 成交量分布的集中程度 |

- **拥挤度 > 80**: 交易过度集中，注意风险
- **拥挤度 < 20**: 交易清淡，可能存在机会
""")


if __name__ == "__main__":
    main()
