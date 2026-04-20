"""
BTC 自動売買ボット — trader.py
================================
戦略: RSI高頻度（1時間足）
  ・RSI < 40 → 買い
  ・RSI > 60 → 売り

GitHub Actions から毎時間呼ばれます。
実行: python bot/trader.py

環境変数（GitHub Secrets に登録）:
  BITFLYER_API_KEY    bitFlyer の API キー
  BITFLYER_API_SECRET bitFlyer の API シークレット
  PAPER_MODE          "true" で模擬取引（デフォルト true）
  TRADE_BUDGET_JPY    1回の取引上限（円）デフォルト 15000
"""

import os, json, time, hmac, hashlib, math, traceback
from datetime import datetime, timezone, timedelta
import requests
import yfinance as yf
import pandas as pd

# ============================================================
# 設定
# ============================================================
API_KEY    = os.environ.get("BITFLYER_API_KEY", "")
API_SECRET = os.environ.get("BITFLYER_API_SECRET", "")
PAPER_MODE = os.environ.get("PAPER_MODE", "true").lower() != "false"

# 1回の取引予算（円）。この金額でBTC数量を自動計算
TRADE_BUDGET_JPY = float(os.environ.get("TRADE_BUDGET_JPY", "15000"))

RSI_BUY   = 40   # これ以下で買い
RSI_SELL  = 60   # これ以上で売り
RSI_PERIOD = 14

PRODUCT_CODE = "BTC_JPY"

# ファイルパス（bot/ フォルダ基準）
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
POSITION_FILE = os.path.join(BASE_DIR, "position.json")
LOG_FILE      = os.path.join(BASE_DIR, "trade_log.csv")

JST = timezone(timedelta(hours=9))

# ============================================================
# ログ
# ============================================================
def log(msg):
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")
    line = f"[{now}] {msg}"
    print(line)

# ============================================================
# bitFlyer API
# ============================================================
BASE_URL = "https://api.bitflyer.com"

def _auth_headers(method, path, body=""):
    ts     = str(time.time())
    text   = ts + method + path + body
    sign   = hmac.new(
        API_SECRET.encode(),
        text.encode(),
        hashlib.sha256
    ).hexdigest()
    return {
        "ACCESS-KEY":       API_KEY,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-SIGN":      sign,
        "Content-Type":     "application/json",
    }

def get_ticker():
    """現在価格を取得"""
    r = requests.get(f"{BASE_URL}/v1/ticker?product_code={PRODUCT_CODE}", timeout=10)
    r.raise_for_status()
    return r.json()

def get_balance():
    """残高を取得 → JPYとBTCを返す"""
    path = "/v1/me/getbalance"
    headers = _auth_headers("GET", path)
    r = requests.get(BASE_URL + path, headers=headers, timeout=10)
    r.raise_for_status()
    balances = r.json()
    jpy = next((b["available"] for b in balances if b["currency_code"] == "JPY"), 0)
    btc = next((b["available"] for b in balances if b["currency_code"] == "BTC"), 0)
    return jpy, btc

def place_order(side, size):
    """
    成行注文を発注
    side: "BUY" or "SELL"
    size: BTC数量（0.001単位）
    """
    path = "/v1/me/sendchildorder"
    body_dict = {
        "product_code":     PRODUCT_CODE,
        "child_order_type": "MARKET",   # 成行注文
        "side":             side,
        "size":             size,
        "minute_to_expire": 10000,
        "time_in_force":    "GTC",
    }
    body = json.dumps(body_dict)
    headers = _auth_headers("POST", path, body)
    r = requests.post(BASE_URL + path, headers=headers, data=body, timeout=10)
    r.raise_for_status()
    return r.json()

# ============================================================
# ポジション管理
# ============================================================
def load_position():
    """保有ポジションを読み込む"""
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, "r") as f:
            return json.load(f)
    return {"holding": False, "size": 0.0, "entry_price": 0.0, "entry_time": ""}

def save_position(pos):
    with open(POSITION_FILE, "w") as f:
        json.dump(pos, f, ensure_ascii=False, indent=2)

# ============================================================
# 取引ログ
# ============================================================
def append_log(action, price, size, rsi, pnl=None, note=""):
    """取引ログ（BUY/SELL のみ）"""
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    pnl_str = f"{pnl:+.0f}" if pnl is not None else ""
    line = f"{now},{action},{price:.0f},{size:.4f},{rsi:.1f},{pnl_str},{note}\n"
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if write_header:
            f.write("datetime,action,price,size,rsi,pnl_jpy,note\n")
        f.write(line)
    log(f"📝 ログ記録: {line.strip()}")

# ============================================================
# テクニカル指標
# ============================================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = ag / al
    return 100 - (100 / (1 + rs))

def get_rsi_signal():
    """
    yfinance から1時間足を取得し、最新の RSI を計算
    Returns: (rsi_value, current_price)
    """
    df = yf.download("BTC-JPY", period="10d", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < RSI_PERIOD + 2:
        raise ValueError("データ取得失敗またはデータ不足")
    df['RSI'] = calc_rsi(df['Close'], RSI_PERIOD)
    df.dropna(inplace=True)

    latest_rsi   = float(df['RSI'].iloc[-1])
    latest_price = float(df['Close'].iloc[-1])
    log(f"📊 最新RSI(1h): {latest_rsi:.1f}  価格: ¥{latest_price:,.0f}")
    return latest_rsi, latest_price

def calc_btc_size(budget_jpy, price):
    """
    予算（円）と現在価格からBTC数量を計算
    bitFlyer の最小単位 0.001 BTC に切り捨て
    """
    raw  = budget_jpy / price
    size = math.floor(raw * 1000) / 1000   # 0.001単位に切り捨て
    return max(size, 0.001)               # 最低 0.001 BTC

# ============================================================
# メイン処理
# ============================================================
def main():
    log("=" * 50)
    log(f"  BTC 自動売買ボット 起動")
    log(f"  モード: {'⚠️ ペーパートレード（模擬）' if PAPER_MODE else '🔴 本番取引'}")
    log(f"  予算上限: ¥{TRADE_BUDGET_JPY:,.0f} / 取引")
    log("=" * 50)

    # ── RSI 計算 ──────────────────────────────────────────
    try:
        rsi, price = get_rsi_signal()
    except Exception as e:
        log(f"❌ RSI取得エラー: {e}")
        return

    # ── ポジション読み込み ──────────────────────────────
    pos = load_position()
    log(f"💼 現在のポジション: {'保有中 ({:.4f} BTC, 取得単価 ¥{:,.0f})'.format(pos['size'], pos['entry_price']) if pos['holding'] else 'なし'}")

    # ── シグナル判定 ──────────────────────────────────────
    signal = None
    if rsi < RSI_BUY and not pos["holding"]:
        signal = "BUY"
    elif rsi > RSI_SELL and pos["holding"]:
        signal = "SELL"

    if signal is None:
        log(f"⏸️ シグナルなし（RSI: {rsi:.1f}）。待機します。")
        return

    log(f"🚨 シグナル発生: {signal}  RSI={rsi:.1f}")

    # ── 取引実行 ──────────────────────────────────────────
    if signal == "BUY":
        size = calc_btc_size(TRADE_BUDGET_JPY, price)
        cost = size * price
        log(f"🟢 買い注文: {size} BTC  ≈ ¥{cost:,.0f}")

        if PAPER_MODE:
            log("   [ペーパートレード] 実際の注文は送信しません。")
        else:
            # 実際の残高確認
            jpy_balance, _ = get_balance()
            if jpy_balance < cost:
                log(f"⚠️ 残高不足: ¥{jpy_balance:,.0f} < 必要額 ¥{cost:,.0f}")
                return
            result = place_order("BUY", size)
            log(f"✅ 注文送信完了: {result}")

        # ポジション更新
        pos = {
            "holding":     True,
            "size":        size,
            "entry_price": price,
            "entry_time":  datetime.now(JST).isoformat(),
        }
        save_position(pos)
        append_log("BUY", price, size, rsi, note="ペーパー" if PAPER_MODE else "本番")

    elif signal == "SELL":
        size       = pos["size"]
        entry_price= pos["entry_price"]
        pnl        = (price - entry_price) * size
        pct        = (price / entry_price - 1) * 100
        log(f"🔴 売り注文: {size} BTC  損益: ¥{pnl:+,.0f}（{pct:+.2f}%）")

        if PAPER_MODE:
            log("   [ペーパートレード] 実際の注文は送信しません。")
        else:
            _, btc_balance = get_balance()
            if btc_balance < size:
                log(f"⚠️ BTC残高不足: {btc_balance} < {size}")
                return
            result = place_order("SELL", size)
            log(f"✅ 注文送信完了: {result}")

        # ポジションリセット
        save_position({"holding": False, "size": 0.0,
                        "entry_price": 0.0, "entry_time": ""})
        append_log("SELL", price, size, rsi, pnl=pnl,
                   note=f"ペーパー {pct:+.2f}%" if PAPER_MODE else f"本番 {pct:+.2f}%")

    log("✅ 処理完了。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"❌ 予期しないエラー: {e}")
        traceback.print_exc()
