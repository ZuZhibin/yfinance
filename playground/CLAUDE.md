# Playground - 量化分析工具集

## 运行环境

- Python 3.12+
- 依赖: `pip install -r requirements.txt` (yfinance, streamlit, plotly, pandas, numpy)
- 主入口: `streamlit run crowdedness_dashboard.py`
- Docker: `docker build -t crowdedness . && docker run -p 8501:8501 crowdedness`

## 项目结构

```
playground/
├── crowdedness_dashboard.py   # 交易拥挤度 Dashboard (Streamlit)
├── capex_to_revenue.py        # CapEx/FCF 比率分析脚本 (matplotlib)
├── requirements.txt
├── Dockerfile
└── CLAUDE.md
```

## 核心逻辑: crowdedness_dashboard.py

### 数据流

```
用户输入 (侧边栏: 股票代码, 日期, 参数)
  → fetch_data()          通过 yfinance 拉取 OHLCV 日线数据 (缓存 1h)
  → compute_crowdedness() 计算 5 维合成拥挤度
  → detect_sell_signals() 识别衰竭型卖出信号
  → plot_*()              Plotly 可视化
```

### 拥挤度指标 (5 维等权合成, 滚动百分位归一化 0-100)

1. **成交量激增比** - Volume / MA(Volume, short_win)
2. **波动率扩张比** - RealizedVol(5d) / RealizedVol(long_win)
3. **量价相关性** - RollingCorr(|Return|, Volume, short_win)
4. **收益自相关** - AutoCorr(Return, lag=1, short_win)
5. **换手集中度** - RollingStd(Volume/MA(Volume, long_win), short_win)

### 卖出信号算法 (detect_sell_signals)

区分"好的拥挤"(健康放量突破) 和"坏的拥挤"(衰竭见顶), 三重条件 AND:

1. **高拥挤度** - crowdedness >= 阈值 (默认 75, 侧边栏可调)
2. **价格近期高位** - close >= rolling_max(lookback) * 0.95
3. **衰竭确认** (至少 1 个):
   - Volume Climax: volume Z-score > 2
   - RSI 顶背离: 价格创新高但 RSI 较前期衰减且 RSI > 60
   - 量价背离: 价格涨 >2% 但均量降 >10%

### 关键函数

| 函数 | 作用 |
|------|------|
| `fetch_data(tickers, start, end)` | yfinance 拉取数据, 支持单/多股票, st.cache_data 缓存 |
| `rolling_percentile(series, window)` | 滚动百分位归一化 |
| `compute_crowdedness(df, short_win, long_win)` | 返回含 5 个子指标 + 合成拥挤度 + close/volume/ret 的 DataFrame |
| `detect_sell_signals(sub, crowd_threshold, rsi_period, lookback)` | 返回 bool Series, True = 卖出信号 |
| `plot_all_crowdedness(crowd_dict, sell_dict)` | 所有股票拥挤度对比 + 红点卖出信号 |
| `plot_latest_ranking(crowd_dict)` | 最新拥挤度排名柱状图 |
| `plot_single_detail(tk, sub, sell_signal, color_idx)` | 单股三面板详情 (拥挤度/价格/成交量) + 红点/倒三角卖出信号 |

### 开发注意事项

- 颜色系统: 7 只预设股票有固定颜色 (`DEFAULT_COLORS`), 自定义股票从 `_PALETTE` 轮转分配
- yfinance 单股票 vs 多股票 download 返回结构不同, `fetch_data` 已处理
- `compute_crowdedness` 返回的 DataFrame index 是 DatetimeIndex, 与 sell_signal 的 index 对齐
- 侧边栏参数变化会自动触发 Streamlit 重新计算 (响应式)
