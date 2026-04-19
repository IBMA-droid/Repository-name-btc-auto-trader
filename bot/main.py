# ========================================
# BTC自動売買ボット + ダッシュボード生成
# GitHub Actions で毎日自動実行される
# ========================================

import os, json, ftplib, hashlib, hmac, time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # サーバー環境ではGUI不要
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
from datetime import datetime
from io import BytesIO

plt.rcParams["font.family"] = "DejaVu Sans"  # サーバー用フォント

# ── 環境変数からキーを取得（GitHub Secrets）──────
API_KEY    = os.environ.get("BITFLYER_API_KEY", "")
API_SECRET = os.environ.get("BITFLYER_API_SECRET", "")
FTP_SERVER = os.environ.get("FTP_SERVER", "")
FTP_USER   = os.environ.get("FTP_USERNAME", "")
FTP_PASS   = os.environ.get("FTP_PASSWORD", "")
BASE_URL   = "https://api.bitflyer.com"

# ── 戦略パラメータ ────────────────────────────────
P = {"rsi_buy": 40, "rsi_sell": 70, "ma_period": 75,
     "stop_pct": -0.07, "tp_pct": 0.08, "risk": 0.4}

# ── RSI計算 ───────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / loss))

# ── bitFlyer API ──────────────────────────────────
def private_get(endpoint, params=None):
    q   = ("?" + "&".join(f"{k}={v}" for k,v in params.items())) if params else ""
    ts  = str(time.time())
    sig = hmac.new(API_SECRET.encode(),
                   (ts+"GET"+endpoint+q).encode(),
                   hashlib.sha256).hexdigest()
    h   = {"ACCESS-KEY":API_KEY,"ACCESS-TIMESTAMP":ts,"ACCESS-SIGN":sig}
    try:
        return requests.get(BASE_URL+endpoint+q, headers=h, timeout=10).json()
    except:
        return {}

# ── データ取得・分析 ──────────────────────────────
print("📥 データ取得中...")
df = yf.download("BTC-JPY", period="120d", interval="1d",
                 auto_adjust=True, progress=False)
df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
close   = df["Close"]
rsi     = calc_rsi(close)
ma      = close.rolling(P["ma_period"]).mean()

cur_price  = float(close.iloc[-1])
cur_rsi    = float(rsi.iloc[-1])
prev_rsi   = float(rsi.iloc[-2])
cur_ma     = float(ma.iloc[-1])
above_ma   = cur_price > cur_ma
buy_signal = (prev_rsi < P["rsi_buy"] <= cur_rsi) and above_ma
sell_signal= (prev_rsi < P["rsi_sell"] <= cur_rsi)

if buy_signal:
    signal_text  = "🟢 買いシグナル発生！"
    signal_color = "#22c55e"
elif sell_signal:
    signal_text  = "🔴 売りシグナル発生！"
    signal_color = "#ef4444"
else:
    signal_text  = "⏸️ 待機中（シグナルなし）"
    signal_color = "#6b7280"

# ── 残高取得 ──────────────────────────────────────
bals = private_get("/v1/me/getbalance")
jpy  = next((b["available"] for b in bals if b.get("currency_code")=="JPY"), 0) if isinstance(bals,list) else 0
btc  = next((b["available"] for b in bals if b.get("currency_code")=="BTC"), 0) if isinstance(bals,list) else 0
total= float(jpy) + float(btc) * cur_price

print(f"✅ 現在価格: {cur_price:,.0f}円 / RSI: {cur_rsi:.1f} / {signal_text}")

# ── グラフ生成（Base64埋め込み用）────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                 sharex=True,
                                 gridspec_kw={"height_ratios":[3,1]})
fig.patch.set_facecolor("#0f172a")
for ax in [ax1, ax2]:
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#334155")

# 価格チャート
ax1.plot(close.index, close.values, color="#f97316", linewidth=1.2)
ax1.plot(ma.index,    ma.values,    color="#60a5fa", linewidth=1.5,
         linestyle="--", label=f"MA{P['ma_period']}")
ax1.set_ylabel("Price (JPY)", color="white", fontsize=10)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M"))
ax1.legend(facecolor="#334155", labelcolor="white", fontsize=9)
ax1.grid(True, alpha=0.2, color="#475569")

# RSIチャート
ax2.plot(rsi.index, rsi.values, color="#a78bfa", linewidth=1.2)
ax2.axhline(P["rsi_buy"],  color="#22c55e", linestyle="--",
            linewidth=1, alpha=0.8)
ax2.axhline(P["rsi_sell"], color="#ef4444", linestyle="--",
            linewidth=1, alpha=0.8)
ax2.fill_between(rsi.index, rsi, P["rsi_buy"],
                 where=rsi<P["rsi_buy"], alpha=0.3, color="#22c55e")
ax2.fill_between(rsi.index, rsi, P["rsi_sell"],
                 where=rsi>P["rsi_sell"], alpha=0.3, color="#ef4444")
ax2.set_ylabel("RSI", color="white", fontsize=10)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.2, color="#475569")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
fig.autofmt_xdate()
plt.tight_layout()

buf = BytesIO()
plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
            facecolor="#0f172a")
buf.seek(0)
chart_b64 = base64.b64encode(buf.read()).decode()
plt.close()
print("✅ グラフ生成完了")

# ── HTMLダッシュボード生成 ────────────────────────
now_str  = datetime.now().strftime("%Y年%m月%d日 %H:%M")
rsi_bar  = max(0, min(100, cur_rsi))
rsi_color= "#22c55e" if cur_rsi < 40 else "#ef4444" if cur_rsi > 70 else "#f97316"

html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="3600">
<title>BTC自動売買ダッシュボード</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI',sans-serif;padding:16px}}
  .card{{background:#1e293b;border-radius:12px;padding:20px;margin-bottom:16px;
         border:1px solid #334155}}
  .title{{font-size:1.3em;font-weight:bold;color:#f97316;margin-bottom:4px}}
  .update{{color:#64748b;font-size:0.85em;margin-bottom:16px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
  .stat{{background:#0f172a;border-radius:8px;padding:14px;text-align:center}}
  .stat-label{{font-size:0.75em;color:#94a3b8;margin-bottom:4px}}
  .stat-value{{font-size:1.4em;font-weight:bold}}
  .signal-box{{border-radius:8px;padding:16px;text-align:center;font-size:1.1em;
               font-weight:bold;margin:12px 0;background:{signal_color}22;
               border:2px solid {signal_color};color:{signal_color}}}
  .rsi-bar-bg{{background:#334155;border-radius:99px;height:8px;margin-top:8px}}
  .rsi-bar{{height:8px;border-radius:99px;background:{rsi_color};
             width:{rsi_bar}%}}
  .chart{{width:100%;border-radius:8px;margin-top:8px}}
  .footer{{text-align:center;color:#475569;font-size:0.75em;margin-top:12px}}
</style>
</head>
<body>
<div class="card">
  <div class="title">🟠 BTC/JPY 自動売買ダッシュボード</div>
  <div class="update">最終更新：{now_str} JST（毎朝9時自動更新）</div>

  <div class="signal-box">{signal_text}</div>

  <div class="grid">
    <div class="stat">
      <div class="stat-label">現在価格</div>
      <div class="stat-value" style="color:#f97316">{cur_price:,.0f}<span style="font-size:0.5em">円</span></div>
    </div>
    <div class="stat">
      <div class="stat-label">RSI (14)</div>
      <div class="stat-value" style="color:{rsi_color}">{cur_rsi:.1f}</div>
      <div class="rsi-bar-bg"><div class="rsi-bar"></div></div>
    </div>
    <div class="stat">
      <div class="stat-label">MA{P['ma_period']}</div>
      <div class="stat-value" style="color:#60a5fa">{cur_ma:,.0f}<span style="font-size:0.5em">円</span></div>
    </div>
    <div class="stat">
      <div class="stat-label">トレンド</div>
      <div class="stat-value">{'📈' if above_ma else '📉'}</div>
    </div>
    <div class="stat">
      <div class="stat-label">BTC残高</div>
      <div class="stat-value" style="color:#a78bfa">{float(btc):.4f}<span style="font-size:0.5em">BTC</span></div>
    </div>
    <div class="stat">
      <div class="stat-label">総資産（概算）</div>
      <div class="stat-value" style="color:#34d399">{total:,.0f}<span style="font-size:0.5em">円</span></div>
    </div>
  </div>
</div>

<div class="card">
  <div style="font-weight:bold;margin-bottom:8px">📈 価格チャート（直近120日）</div>
  <img class="chart" src="data:image/png;base64,{chart_b64}" alt="BTC Chart">
</div>

<div class="card">
  <div style="font-weight:bold;margin-bottom:8px">⚙️ 戦略パラメータ</div>
  <div class="grid">
    <div class="stat"><div class="stat-label">買い閾値</div>
      <div class="stat-value">RSI &lt; {P['rsi_buy']}</div></div>
    <div class="stat"><div class="stat-label">売り閾値</div>
      <div class="stat-value">RSI &gt; {P['rsi_sell']}</div></div>
    <div class="stat"><div class="stat-label">損切り</div>
      <div class="stat-value" style="color:#ef4444">{P['stop_pct']*100:.0f}%</div></div>
    <div class="stat"><div class="stat-label">利確</div>
      <div class="stat-value" style="color:#22c55e">+{P['tp_pct']*100:.0f}%</div></div>
  </div>
</div>

<div class="footer">
  Powered by Python / bitFlyer API / GitHub Actions<br>
  ※ 投資は自己責任でお願いします
</div>
</body>
</html>"""

with open("btc_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)
print("✅ HTMLダッシュボード生成完了")

# ── FTPアップロード（改良版）────────────────────────
print("📤 さくらサーバーへアップロード中...")

def try_ftp_upload():
    import ftplib, socket

    # 試みるパスのリスト（さくらのよくある構成）
    paths_to_try = [
        "/home/kenchik/www/",
        "www/",
        "/www/",
        "/home/kenchik/",
        "",
    ]

    # ① まずFTPS（暗号化）で試みる
    for path in paths_to_try:
        try:
            print(f"  FTPS試行中... パス: '{path}'")
            ftp = ftplib.FTP_TLS(timeout=30)
            ftp.connect(FTP_SERVER, 21)
            ftp.auth()
            ftp.login(FTP_USER, FTP_PASS)
            ftp.prot_p()
            ftp.set_pasv(True)

            # 現在のフォルダとファイル一覧を表示（デバッグ用）
            print(f"  現在地: {ftp.pwd()}")
            print(f"  フォルダ内容: {ftp.nlst()}")

            if path:
                ftp.cwd(path)
                print(f"  移動後: {ftp.pwd()}")

            with open("btc_dashboard.html", "rb") as f:
                ftp.storbinary("STOR btc_dashboard.html", f)
            ftp.quit()
            print(f"✅ FTPSアップロード成功！パス: '{path}'")
            return True

        except Exception as e:
            print(f"  FTPSエラー（パス:{path}）: {e}")
            try:
                ftp.quit()
            except:
                pass

    # ② FTPSが全滅したら通常FTPで試みる
    for path in paths_to_try:
        try:
            print(f"  FTP試行中... パス: '{path}'")
            with ftplib.FTP(timeout=30) as ftp:
                ftp.connect(FTP_SERVER, 21)
                ftp.login(FTP_USER, FTP_PASS)
                ftp.set_pasv(True)

                print(f"  現在地: {ftp.pwd()}")
                print(f"  フォルダ内容: {ftp.nlst()}")

                if path:
                    ftp.cwd(path)

                with open("btc_dashboard.html", "rb") as f:
                    ftp.storbinary("STOR btc_dashboard.html", f)
            print(f"✅ FTPアップロード成功！パス: '{path}'")
            return True

        except Exception as e:
            print(f"  FTPエラー（パス:{path}）: {e}")

    print("❌ 全FTP試行が失敗しました")
    return False

try_ftp_upload()
