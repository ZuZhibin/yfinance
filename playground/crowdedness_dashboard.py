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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
TICKERS = ["NVDA", "META", "GOOGL", "MSFT", "TSLA", "AAPL", "AMZN"]

TICKER_NAMES = {
    "NVDA": "NVIDIA",
    "META": "Meta",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "AMZN": "Amazon",
}

COLORS = {
    "NVDA": "#76B900",
    "META": "#0081FB",
    "GOOGL": "#EA4335",
    "MSFT": "#00A4EF",
    "TSLA": "#CC0000",
    "AAPL": "#A2AAAD",
    "AMZN": "#FF9900",
}


# ---------------------------------------------------------------------------
# 数据获取
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="正在从 Yahoo Finance 拉取数据 ...")
def fetch_data() -> dict[str, pd.DataFrame]:
    """为每只股票拉取过去 1 年日线 OHLCV 数据，返回 {ticker: DataFrame}。"""
    raw = yf.download(TICKERS, period="1y", interval="1d", group_by="ticker", progress=False)
    result = {}
    for tk in TICKERS:
        df = raw[tk].dropna()
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
        result[tk] = df
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
    """7 只股票拥挤度时序对比折线图。"""
    fig = go.Figure()
    for tk in TICKERS:
        s = crowd_dict[tk]["crowdedness"].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines", name=f"{tk} ({TICKER_NAMES[tk]})",
            line=dict(color=COLORS[tk], width=2),
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
    for tk in TICKERS:
        s = crowd_dict[tk]["crowdedness"].dropna()
        if len(s) > 0:
            latest[tk] = s.iloc[-1]
    sorted_items = sorted(latest.items(), key=lambda x: x[1], reverse=True)
    tickers = [t for t, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [COLORS[t] for t in tickers]
    labels = [f"{t} ({TICKER_NAMES[t]})" for t in tickers]

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


def plot_single_detail(tk: str, sub: pd.DataFrame) -> go.Figure:
    """单只股票详情: 拥挤度 + 收盘价 + 成交量 三子图联动。"""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=(
            f"{tk} ({TICKER_NAMES[tk]}) 拥挤度",
            "收盘价",
            "成交量",
        ),
        row_heights=[0.4, 0.35, 0.25],
    )
    data = sub.dropna(subset=["crowdedness"])

    # 拥挤度
    fig.add_trace(go.Scatter(
        x=data.index, y=data["crowdedness"],
        mode="lines", name="拥挤度",
        line=dict(color=COLORS[tk], width=2),
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
        name="成交量", marker_color=COLORS[tk], opacity=0.6,
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

    # 侧边栏参数
    st.sidebar.header("参数设置")
    short_win = st.sidebar.slider("短期窗口 (天)", 10, 40, 20)
    long_win = st.sidebar.slider("长期窗口 (天)", 40, 120, 60)
    selected = st.sidebar.selectbox(
        "单只股票详情",
        TICKERS,
        format_func=lambda t: f"{t} ({TICKER_NAMES[t]})",
    )

    # 拉取数据
    data_dict = fetch_data()

    # 计算拥挤度
    crowd_dict: dict[str, pd.DataFrame] = {}
    for tk in TICKERS:
        crowd_dict[tk] = compute_crowdedness(data_dict[tk], short_win, long_win)

    # 全局对比
    st.plotly_chart(plot_all_crowdedness(crowd_dict), use_container_width=True)

    # 排名
    st.plotly_chart(plot_latest_ranking(crowd_dict), use_container_width=True)

    # 单只详情
    st.subheader(f"{selected} ({TICKER_NAMES[selected]}) 详细分析")
    st.plotly_chart(plot_single_detail(selected, crowd_dict[selected]), use_container_width=True)

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
