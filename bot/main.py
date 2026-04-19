"""
BTC 自動売買ダッシュボード — bot/main.py
==========================================
GitHub Actions で毎日自動実行 → GitHub Pages に反映

機能:
  ・5戦略バックテスト結果を同時表示
      1) RSI逆張り（日足1年）
      2) ボリンジャーバンド（日足1年）
      3) MAクロス MA25/75（日足1年）
      4) RSI高頻度（1時間足60日）  ← 現在最強
      5) アダプティブ（相場環境自動判定で1〜4を切替）
  ・現在の相場環境バッジ（トレンド／レンジ）を自動判定
  ・現在価格・RSI・MA などのリアルタイム指標
  ・ペーパートレード（模擬売買）ログ

使い方:
  python bot/main.py
  生成物: btc_dashboard.html  →  docs/index.html にコピーして GitHub Pages に公開
"""

import os, json, base64, io, hashlib, hmac, time, requests
from datetime import datetime, timezone, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# 定数
# ============================================================
INITIAL_CAPITAL = 1_000_000
RISK_PER_TRADE  = 0.40
FEE_RATE        = 0.001

PAPER_MODE = os.environ.get("PAPER_MODE", "true").lower() != "false"

API_KEY    = os.environ.get("BITFLYER_API_KEY", "")
API_SECRET = os.environ.get("BITFLYER_API_SECRET", "")

BG   = '#0d1117'
CARD = '#161b22'
LINE = '#30363d'

# ============================================================
# テクニカル指標
# ============================================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al))

def calc_bb(series, period=20, sigma=2.0):
    ma  = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma, ma + sigma * std, ma - sigma * std

# ============================================================
# バックテスト エンジン
# ============================================================
def backtest(df, signal_fn, label=""):
    """
    signal_fn(df, i, row) → "buy" | "sell" | None
    """
    cap, pos, entry = INITIAL_CAPITAL, 0.0, 0.0
    trades, eq_vals, eq_dates = [], [], []
    buys, sells = [], []

    rows = list(df.iterrows())
    for i, (idx, row) in enumerate(rows):
        price  = float(row['Close'])
        signal = signal_fn(df, i, row)

        if signal == "buy" and pos == 0:
            invest = cap * RISK_PER_TRADE
            pos    = invest / price * (1 - FEE_RATE)
            entry  = price
            cap   -= invest
            buys.append((idx, price))

        elif signal == "sell" and pos > 0:
            proc = pos * price * (1 - FEE_RATE)
            trades.append({
                "entry":   entry,
                "exit":    price,
                "pnl":     proc - pos * entry,
                "pnl_pct": (price / entry - 1) * 100,
                "date":    str(idx.date()) if hasattr(idx, 'date') else str(idx)
            })
            cap += proc
            pos  = 0.0
            sells.append((idx, price))

        eq_vals.append(cap + pos * price)
        eq_dates.append(idx)

    # 未決済を最終日で強制決済
    if pos > 0:
        lp = float(df['Close'].iloc[-1])
        proc = pos * lp * (1 - FEE_RATE)
        trades.append({
            "entry":   entry,
            "exit":    lp,
            "pnl":     proc - pos * entry,
            "pnl_pct": (lp / entry - 1) * 100,
            "date":    "(未決済・強制)"
        })
        if eq_vals:
            eq_vals[-1] = cap + proc

    n   = len(trades)
    wr  = sum(1 for t in trades if t['pnl'] > 0) / n * 100 if n else 0
    ap  = np.mean([t['pnl_pct'] for t in trades]) if n else 0
    final = eq_vals[-1] if eq_vals else INITIAL_CAPITAL
    ret = (final / INITIAL_CAPITAL - 1) * 100

    eq_s = pd.Series(eq_vals, index=eq_dates)
    md   = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min() if len(eq_s) > 1 else 0

    return {
        "label":  label,
        "stats":  {"total_return": ret, "n_trades": n, "win_rate": wr,
                   "avg_pnl_pct": ap, "max_dd": md, "final_equity": final},
        "eq":     eq_s,
        "trades": trades,
        "buys":   buys,
        "sells":  sells,
        "df":     df,
    }


# ── シグナル関数 ──────────────────────────────────────────────
def sig_rsi_daily(df, i, row):
    rsi = float(row.get('RSI', 50))
    if rsi < 35: return "buy"
    if rsi > 65: return "sell"
    return None

def sig_bb_daily(df, i, row):
    price = float(row['Close'])
    lb = float(row.get('LB', 0))
    ub = float(row.get('UB', 1e12))
    if price < lb: return "buy"
    if price > ub: return "sell"
    return None

def sig_ma_cross(df, i, row):
    if i == 0: return None
    prev = df.iloc[i - 1]
    mas_now  = float(row.get('MA_S', 0))
    mal_now  = float(row.get('MA_L', 0))
    mas_prev = float(prev.get('MA_S', 0))
    mal_prev = float(prev.get('MA_L', 0))
    if mas_prev <= mal_prev and mas_now > mal_now: return "buy"
    if mas_prev >= mal_prev and mas_now < mal_now: return "sell"
    return None

def sig_rsi_hourly(df, i, row):
    rsi = float(row.get('RSI', 50))
    if rsi < 40: return "buy"
    if rsi > 60: return "sell"
    return None


# ============================================================
# 相場環境判定（アダプティブ戦略の核心）
# ============================================================
def detect_regime(df_daily):
    """
    日足MA75の傾きで相場環境を判定
    Returns: 'trend_up' | 'trend_down' | 'range'
    """
    if len(df_daily) < 30:
        return 'range'
    ma75 = df_daily['Close'].rolling(75).mean()
    ma75 = ma75.dropna()
    if len(ma75) < 22:
        return 'range'
    slope = (ma75.iloc[-1] - ma75.iloc[-22]) / ma75.iloc[-22] * 100
    if slope > 2.0:
        return 'trend_up'
    elif slope < -2.0:
        return 'trend_down'
    else:
        return 'range'


def run_adaptive(df_daily, df_hourly, regime):
    """
    相場環境に応じて戦略を切り替える
      trend_up   → MAクロス（日足）
      range      → RSI高頻度（1時間足）  ← 現在最強
      trend_down → ボリンジャーバンド（日足）
    """
    if regime == 'trend_up':
        sub_label = "MAクロス（トレンド相場）"
        df = df_daily.copy()
        df['MA_S'] = df['Close'].rolling(25).mean()
        df['MA_L'] = df['Close'].rolling(75).mean()
        df.dropna(inplace=True)
        return backtest(df, sig_ma_cross, sub_label), sub_label

    elif regime == 'trend_down':
        sub_label = "ボリンジャーバンド（下降トレンド）"
        df = df_daily.copy()
        df['MA'], df['UB'], df['LB'] = calc_bb(df['Close'])
        df.dropna(inplace=True)
        return backtest(df, sig_bb_daily, sub_label), sub_label

    else:  # range
        sub_label = "RSI高頻度（レンジ相場） ★現在採用"
        df = df_hourly.copy()
        df['RSI'] = calc_rsi(df['Close'])
        df.dropna(inplace=True)
        return backtest(df, sig_rsi_hourly, sub_label), sub_label


# ============================================================
# bitFlyer API（現在価格取得）
# ============================================================
def get_ticker():
    try:
        r = requests.get(
            "https://api.bitflyer.com/v1/ticker?product_code=BTC_JPY",
            timeout=10
        )
        d = r.json()
        return {
            "price":     d.get("ltp", 0),
            "best_bid":  d.get("best_bid", 0),
            "best_ask":  d.get("best_ask", 0),
            "volume_24": d.get("volume_by_product", 0),
        }
    except Exception:
        return {"price": 0, "best_bid": 0, "best_ask": 0, "volume_24": 0}


# ============================================================
# チャート生成
# ============================================================
CHART_STYLE = {
    'bg': '#1a1f2e', 'card': '#232938', 'grid': '#2d3347',
    'buy': '#39d353', 'sell': '#f85149', 'price': '#58a6ff',
    'eq_up': '#39d353', 'eq_dn': '#f85149',
}
C = CHART_STYLE

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=C['bg'])
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _style_ax(ax):
    ax.set_facecolor(C['card'])
    ax.tick_params(colors='#8b9dc3', labelsize=7)
    ax.grid(color=C['grid'], lw=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(C['grid'])

def make_price_eq_chart(result, extra_lines=None, title_suffix=""):
    df      = result['df']
    buys    = result['buys']
    sells   = result['sells']
    eq      = result['eq']
    label   = result['label']

    rows = 3 if 'RSI' in df.columns else 2
    ratios = [3, 1, 1.2] if rows == 3 else [2, 1.2]
    fig, axes = plt.subplots(rows, 1, figsize=(10, 6 if rows == 3 else 5),
                             gridspec_kw={'height_ratios': ratios},
                             facecolor=C['bg'])
    if rows == 2:
        ax1, ax3 = axes
        ax2 = None
    else:
        ax1, ax2, ax3 = axes

    fig.suptitle(f"{label}{title_suffix}", color='#e6edf3',
                 fontsize=11, fontweight='bold', y=0.99)

    # 価格
    _style_ax(ax1)
    ax1.plot(df.index, df['Close'], color=C['price'], lw=1.1, label='BTC/JPY')
    if extra_lines:
        for (col, color, ls, lbl) in extra_lines:
            if col in df.columns:
                ax1.plot(df.index, df[col], color=color, lw=0.8,
                         ls=ls, label=lbl, alpha=0.8)
    for d, p in buys:
        ax1.scatter(d, p, color=C['buy'], marker='^', zorder=5, s=60)
    for d, p in sells:
        ax1.scatter(d, p, color=C['sell'], marker='v', zorder=5, s=60)
    ax1.set_ylabel('価格（円）', color='#8b9dc3', fontsize=8)
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    if extra_lines:
        ax1.legend(fontsize=7, facecolor=C['card'], labelcolor='#8b9dc3',
                   loc='upper left')

    # RSI（あれば）
    if ax2 is not None and 'RSI' in df.columns:
        _style_ax(ax2)
        ax2.plot(df.index, df['RSI'], color='#e3b341', lw=0.9)
        ax2.axhline(40, color=C['buy'],  ls='--', lw=0.7, alpha=0.6)
        ax2.axhline(60, color=C['sell'], ls='--', lw=0.7, alpha=0.6)
        ax2.fill_between(df.index, df['RSI'], 40,
                         where=df['RSI'] < 40, alpha=0.25, color=C['buy'])
        ax2.fill_between(df.index, df['RSI'], 60,
                         where=df['RSI'] > 60, alpha=0.25, color=C['sell'])
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('RSI', color='#8b9dc3', fontsize=8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))

    # 資産推移
    _style_ax(ax3)
    color_eq = C['eq_up'] if eq.iloc[-1] >= INITIAL_CAPITAL else C['eq_dn']
    ax3.plot(eq.index, eq.values, color=color_eq, lw=1.1)
    ax3.axhline(INITIAL_CAPITAL, color='#484f58', ls=':', lw=0.8)
    ax3.fill_between(eq.index, eq.values, INITIAL_CAPITAL,
                     where=eq.values >= INITIAL_CAPITAL,
                     alpha=0.18, color=C['eq_up'])
    ax3.fill_between(eq.index, eq.values, INITIAL_CAPITAL,
                     where=eq.values < INITIAL_CAPITAL,
                     alpha=0.18, color=C['eq_dn'])
    ax3.set_ylabel('資産（円）', color='#8b9dc3', fontsize=8)
    ax3.yaxis.get_major_formatter().set_scientific(False)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# ============================================================
# HTML ダッシュボード
# ============================================================
def color_for(v, pos_good=True):
    if pos_good:
        return '#39d353' if v >= 0 else '#f85149'
    return '#f85149' if v <= 0 else '#39d353'

REGIME_BADGE = {
    'trend_up':   ('<span class="regime-badge trend-up">📈 トレンド相場（上昇）</span>', '上昇トレンドを検出。MAクロス戦略を採用中。'),
    'trend_down': ('<span class="regime-badge trend-dn">📉 トレンド相場（下降）</span>', '下降トレンドを検出。ボリンジャーバンド戦略を採用中。'),
    'range':      ('<span class="regime-badge range">↔️ レンジ相場</span>', 'レンジ相場を検出。RSI高頻度（1時間足）戦略を採用中。'),
}

STRATEGY_META = [
    {"icon": "📉", "desc": "RSI<35で買い、RSI>65で売り。売られすぎ・買われすぎを狙う逆張り基本戦略（日足1年）"},
    {"icon": "📊", "desc": "-2σ（下限）割れで買い、+2σ（上限）越えで売り。バンド幅を活かす逆張り（日足1年）"},
    {"icon": "✂️", "desc": "MA25がMA75を上抜け（GC）で買い、下抜け（DC）で売り。トレンドフォロー型（日足1年）"},
    {"icon": "⚡", "desc": "1時間足でRSI<40買い・>60売り。取引回数が多い積極型。現環境（レンジ相場）で最強"},
    {"icon": "🤖", "desc": "相場環境を毎日自動判定し、最適な戦略に切り替えるアダプティブ戦略"},
]

def build_card(result, meta, rank=None):
    s = result['stats']
    ret_c = color_for(s['total_return'])
    dd_c  = color_for(s['max_dd'], pos_good=False)
    wr_c  = '#39d353' if s['win_rate'] >= 50 else '#f85149'
    border = 'border:2px solid #58a6ff' if rank == 1 else ''
    crown  = '<span class="crown">👑 現在最強</span>' if rank == 1 else ''

    # トレード履歴（最新5件）
    trade_rows = ""
    for t in result['trades'][-5:]:
        pnl_c = '#39d353' if t['pnl'] > 0 else '#f85149'
        trade_rows += f"""
        <tr>
          <td>{t.get('date','')}</td>
          <td>{t['entry']:,.0f}</td>
          <td>{t['exit']:,.0f}</td>
          <td style="color:{pnl_c}">{t['pnl_pct']:+.1f}%</td>
          <td style="color:{pnl_c}">{t['pnl']:+,.0f}円</td>
        </tr>"""

    trade_table = f"""
    <div class="trade-history">
      <div class="th-title">直近取引履歴（最新5件）</div>
      <table>
        <thead><tr><th>日付</th><th>買値</th><th>売値</th><th>損益率</th><th>損益額</th></tr></thead>
        <tbody>{trade_rows if trade_rows else '<tr><td colspan="5" style="color:#8b949e;text-align:center">取引なし</td></tr>'}</tbody>
      </table>
    </div>""" if result['trades'] else ""

    return f"""
    <div class="card" style="{border}">
      <div class="card-header">
        <div>
          <h2>{meta['icon']} {result['label']} {crown}</h2>
          <span class="desc">{meta['desc']}</span>
        </div>
      </div>
      <div class="metrics">
        <div class="metric">
          <span class="ml">トータルリターン</span>
          <span class="mv" style="color:{ret_c}">{s['total_return']:+.1f}%</span>
        </div>
        <div class="metric">
          <span class="ml">取引回数</span>
          <span class="mv">{s['n_trades']} 回</span>
        </div>
        <div class="metric">
          <span class="ml">勝率</span>
          <span class="mv" style="color:{wr_c}">{s['win_rate']:.0f}%</span>
        </div>
        <div class="metric">
          <span class="ml">平均損益/取引</span>
          <span class="mv" style="color:{color_for(s['avg_pnl_pct'])}">{s['avg_pnl_pct']:+.1f}%</span>
        </div>
        <div class="metric">
          <span class="ml">最大ドローダウン</span>
          <span class="mv" style="color:{dd_c}">{s['max_dd']:.1f}%</span>
        </div>
        <div class="metric">
          <span class="ml">最終資産（模擬）</span>
          <span class="mv">{s['final_equity']:,.0f}円</span>
        </div>
      </div>
      <img src="data:image/png;base64,{result['img']}" style="width:100%;border-radius:8px;margin-top:12px;">
      {trade_table}
    </div>"""


def build_html(results_list, ticker, regime, adaptive_label, current_signal):
    now = datetime.now(timezone(timedelta(hours=9))).strftime("%Y年%m月%d日 %H:%M JST")

    regime_badge, regime_desc = REGIME_BADGE.get(regime, REGIME_BADGE['range'])

    # 価格フォーマット
    price_str = f"¥{ticker['price']:,.0f}" if ticker['price'] else "取得中..."

    # 全カード生成（戦略4を1位扱いで強調）
    cards_html = ""
    best_idx = 3  # 戦略4（RSI高頻度）
    for i, (res, meta) in enumerate(zip(results_list, STRATEGY_META)):
        rank = 1 if i == best_idx else None
        cards_html += build_card(res, meta, rank)

    # 現在のシグナル
    sig_color = '#39d353' if current_signal == 'BUY' else '#f85149' if current_signal == 'SELL' else '#e3b341'
    sig_emoji = '🟢' if current_signal == 'BUY' else '🔴' if current_signal == 'SELL' else '🟡'

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>BTC 自動売買ダッシュボード</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:{BG};color:#e6edf3;font-family:'Segoe UI',sans-serif;padding:20px 16px}}

  /* ヘッダー */
  .header{{text-align:center;margin-bottom:24px}}
  .header h1{{font-size:1.6rem;color:#58a6ff;margin-bottom:6px}}
  .header .sub{{color:#8b949e;font-size:.82rem}}
  .badge{{background:#21262d;color:#8b949e;font-size:.7rem;padding:3px 10px;
           border-radius:10px;display:inline-block;margin:2px}}

  /* ライブパネル */
  .live-panel{{background:#161b22;border:1px solid #30363d;border-radius:12px;
               padding:18px 22px;margin-bottom:22px;
               display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px}}
  .live-item .ll{{font-size:.7rem;color:#8b949e;margin-bottom:4px}}
  .live-item .lv{{font-size:1.3rem;font-weight:700}}

  /* 相場環境バナー */
  .regime-banner{{background:#161b22;border:1px solid #30363d;border-radius:10px;
                  padding:14px 18px;margin-bottom:22px;display:flex;
                  align-items:center;gap:14px;flex-wrap:wrap}}
  .regime-badge{{font-size:.85rem;font-weight:600;padding:5px 14px;border-radius:20px}}
  .trend-up{{background:rgba(57,211,83,.15);color:#39d353;border:1px solid #39d35340}}
  .trend-dn{{background:rgba(248,81,73,.15);color:#f85149;border:1px solid #f8514940}}
  .range{{background:rgba(88,166,255,.15);color:#58a6ff;border:1px solid #58a6ff40}}
  .regime-desc{{font-size:.8rem;color:#8b9dc3}}
  .adaptive-label{{font-size:.78rem;color:#e3b341;margin-top:3px}}

  /* グリッド */
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}}

  /* カード */
  .card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px}}
  .card-header{{margin-bottom:14px;border-bottom:1px solid #30363d;padding-bottom:10px}}
  .card-header h2{{font-size:1rem;color:#f0f6fc;display:flex;align-items:center;gap:8px}}
  .crown{{font-size:.72rem;background:rgba(88,166,255,.2);color:#58a6ff;
           padding:2px 8px;border-radius:8px}}
  .desc{{font-size:.72rem;color:#8b949e;margin-top:4px;display:block;line-height:1.5}}

  /* メトリクス */
  .metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
  .metric{{background:{BG};border-radius:8px;padding:10px 8px;text-align:center}}
  .ml{{display:block;font-size:.65rem;color:#8b949e;margin-bottom:4px}}
  .mv{{display:block;font-size:1.05rem;font-weight:700}}

  /* トレード履歴 */
  .trade-history{{margin-top:12px;background:{BG};border-radius:8px;padding:10px}}
  .th-title{{font-size:.72rem;color:#8b949e;margin-bottom:6px}}
  table{{width:100%;border-collapse:collapse;font-size:.72rem}}
  th{{color:#8b949e;padding:4px 6px;border-bottom:1px solid #30363d;text-align:left}}
  td{{color:#c9d1d9;padding:4px 6px;border-bottom:1px solid #21262d}}

  .footer{{text-align:center;color:#484f58;font-size:.7rem;margin-top:24px;line-height:1.8}}

  @media(max-width:600px){{
    .grid{{grid-template-columns:1fr}}
    .metrics{{grid-template-columns:repeat(2,1fr)}}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>📊 BTC/JPY 自動売買ダッシュボード</h1>
  <p class="sub">
    <span class="badge">ペーパートレード（模擬）モード</span>
    <span class="badge">初期資金 1,000,000円</span>
    <span class="badge">手数料 0.1%/片道</span>
    <span class="badge">最終更新: {now}</span>
  </p>
</div>

<!-- ライブパネル -->
<div class="live-panel">
  <div class="live-item">
    <div class="ll">現在価格（BTC/JPY）</div>
    <div class="lv" style="color:#58a6ff">{price_str}</div>
  </div>
  <div class="live-item">
    <div class="ll">現在のシグナル</div>
    <div class="lv" style="color:{sig_color}">{sig_emoji} {current_signal}</div>
  </div>
  <div class="live-item">
    <div class="ll">採用中の戦略</div>
    <div class="lv" style="font-size:.85rem;color:#e3b341">{adaptive_label}</div>
  </div>
  <div class="live-item">
    <div class="ll">24h出来高</div>
    <div class="lv" style="font-size:1rem">{ticker.get('volume_24', 0):.2f} BTC</div>
  </div>
</div>

<!-- 相場環境バナー -->
<div class="regime-banner">
  <div>
    {regime_badge}
    <div class="adaptive-label">🤖 アダプティブ戦略が自動選択中: {adaptive_label}</div>
  </div>
  <div class="regime-desc">{regime_desc}</div>
</div>

<!-- 戦略カード群 -->
<div class="grid">
{cards_html}
</div>

<div class="footer">
  bitFlyer BTC/JPY | データ: Yahoo Finance (yfinance) |
  戦略: RSI逆張り・ボリンジャーバンド・MAクロス・RSI高頻度・アダプティブ<br>
  ※このダッシュボードはペーパートレード（模擬取引）です。実際の取引は行っていません。
</div>

</body>
</html>"""


# ============================================================
# メイン処理
# ============================================================
def main():
    print("=" * 55)
    print("  BTC 自動売買ダッシュボード 生成開始")
    print("=" * 55)

    # ── データ取得 ──────────────────────────────────────────
    print("\n[1] データ取得中...")
    df_daily = yf.download("BTC-JPY", period="1y",  interval="1d", progress=False)
    df_1h    = yf.download("BTC-JPY", period="60d", interval="1h", progress=False)
    for df in [df_daily, df_1h]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    print(f"  日足: {len(df_daily)}件  1時間足: {len(df_1h)}件")

    # ── 現在価格・ティッカー ──────────────────────────────
    print("\n[2] 現在価格取得中...")
    ticker = get_ticker()
    print(f"  現在価格: ¥{ticker['price']:,.0f}")

    # ── 相場環境判定 ────────────────────────────────────────
    print("\n[3] 相場環境判定...")
    regime = detect_regime(df_daily)
    regime_names = {'trend_up': '上昇トレンド', 'trend_down': '下降トレンド', 'range': 'レンジ相場'}
    print(f"  判定結果: {regime_names[regime]}")

    # ── 各戦略バックテスト ──────────────────────────────────
    print("\n[4] 各戦略バックテスト実行中...")
    results = []

    # 戦略1: RSI逆張り（日足）
    d1 = df_daily.copy()
    d1['RSI'] = calc_rsi(d1['Close'])
    d1.dropna(inplace=True)
    r1 = backtest(d1, sig_rsi_daily, "RSI逆張り（日足1年）")
    r1['img'] = make_price_eq_chart(r1)
    print(f"  戦略1 {r1['stats']['total_return']:+.1f}%  {r1['stats']['n_trades']}回")
    results.append(r1)

    # 戦略2: ボリンジャーバンド（日足）
    d2 = df_daily.copy()
    d2['MA'], d2['UB'], d2['LB'] = calc_bb(d2['Close'])
    d2.dropna(inplace=True)
    r2 = backtest(d2, sig_bb_daily, "ボリンジャーバンド（日足1年）")
    r2['img'] = make_price_eq_chart(
        r2,
        extra_lines=[('UB','#f85149','--','+2σ'),('LB','#39d353','--','-2σ'),('MA','#e3b341','--','MA20')]
    )
    print(f"  戦略2 {r2['stats']['total_return']:+.1f}%  {r2['stats']['n_trades']}回")
    results.append(r2)

    # 戦略3: MAクロス（日足）
    d3 = df_daily.copy()
    d3['MA_S'] = d3['Close'].rolling(25).mean()
    d3['MA_L'] = d3['Close'].rolling(75).mean()
    d3.dropna(inplace=True)
    r3 = backtest(d3, sig_ma_cross, "MAクロス MA25/75（日足1年）")
    r3['img'] = make_price_eq_chart(
        r3,
        extra_lines=[('MA_S','#e3b341','--','MA25'),('MA_L','#ff7b72','--','MA75')]
    )
    print(f"  戦略3 {r3['stats']['total_return']:+.1f}%  {r3['stats']['n_trades']}回")
    results.append(r3)

    # 戦略4: RSI高頻度（1時間足）
    d4 = df_1h.copy()
    d4['RSI'] = calc_rsi(d4['Close'])
    d4.dropna(inplace=True)
    r4 = backtest(d4, sig_rsi_hourly, "RSI高頻度（1時間足60日）")
    r4['img'] = make_price_eq_chart(r4)
    print(f"  戦略4 {r4['stats']['total_return']:+.1f}%  {r4['stats']['n_trades']}回")
    results.append(r4)

    # 戦略5: アダプティブ（自動切替）
    print(f"\n[5] アダプティブ戦略（相場環境: {regime_names[regime]}）...")
    r5, adaptive_label = run_adaptive(df_daily, d4, regime)
    r5['label'] = f"アダプティブ戦略（自動切替）"
    r5['img']   = make_price_eq_chart(r5, title_suffix=f"\n現在: {adaptive_label}")
    print(f"  戦略5 {r5['stats']['total_return']:+.1f}%  {r5['stats']['n_trades']}回")
    results.append(r5)

    # ── 現在シグナル計算（戦略4ベース）──────────────────────
    print("\n[6] 現在のシグナル計算中...")
    current_signal = "HOLD"
    if len(d4) >= 2:
        latest_rsi = float(d4['RSI'].iloc[-1])
        if   latest_rsi < 40: current_signal = "BUY"
        elif latest_rsi > 60: current_signal = "SELL"
        print(f"  最新RSI(1h): {latest_rsi:.1f}  → シグナル: {current_signal}")

    # ── HTML生成 ─────────────────────────────────────────────
    print("\n[7] ダッシュボードHTML生成中...")
    html = build_html(results, ticker, regime, adaptive_label, current_signal)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "btc_dashboard.html")
    out = os.path.normpath(out)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  保存完了: {out}")

    # ── サマリー表示 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'戦略':<28} {'リターン':>8} {'取引':>5} {'勝率':>6} {'最大DD':>8}")
    print("=" * 60)
    for r in results:
        s = r['stats']
        print(f"  {r['label']:<28} {s['total_return']:>+7.1f}% {s['n_trades']:>4}回 "
              f"{s['win_rate']:>5.0f}% {s['max_dd']:>7.1f}%")
    print("=" * 60)
    print(f"\n✅ 完了！ btc_dashboard.html を確認してください。")
    print(f"   GitHub Actionsが docs/index.html にコピーしてPagesに公開します。")


if __name__ == "__main__":
    main()
