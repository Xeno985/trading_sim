import streamlit as st
from streamlit_autorefresh import st_autorefresh
import asyncio
import websockets
import json
import numpy as np
import threading
import time
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
from datetime import datetime
import matplotlib.pyplot as plt

# --- Spot pairs fetch ---
@st.cache_data(show_spinner=False)
def fetch_okx_spot_pairs():
    url = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        pairs = [x['instId'] for x in data['data']]
        return sorted(pairs)
    except Exception as e:
        st.warning(f"Could not fetch spot pairs: {e}")
        return ['BTC-USDT']

# --- Shared Data Singleton ---
class SharedData:
    def __init__(self):
        self.lock = threading.Lock()
        self.orderbook = {'bids': [], 'asks': []}
        self.latency = 0.0
        self.latency_series = []
        self.mid_prices = []
        self.snapshots = []
        self.volatility = 0.0
        self.error = None

def websocket_thread(shared, ws_url):
    async def ws_loop():
        try:
            async with websockets.connect(ws_url, ping_interval=None) as ws:
                while True:
                    start = time.time()
                    msg = await ws.recv()
                    data = json.loads(msg)
                    with shared.lock:
                        shared.orderbook = {
                            'bids': data.get('bids', []),
                            'asks': data.get('asks', [])
                        }
                        # Latency
                        end = time.time()
                        shared.latency = (end - start) * 1000
                        shared.latency_series.append(shared.latency)
                        if len(shared.latency_series) > 200:
                            shared.latency_series.pop(0)
                        # Mid price and volatility
                        if shared.orderbook['bids'] and shared.orderbook['asks']:
                            best_bid = float(shared.orderbook['bids'][0][0])
                            best_ask = float(shared.orderbook['asks'][0][0])
                            mid = (best_bid + best_ask) / 2
                            shared.mid_prices.append(mid)
                            if len(shared.mid_prices) > 100:
                                shared.mid_prices.pop(0)
                            shared.volatility = float(np.std(shared.mid_prices[-50:])) if len(shared.mid_prices) > 2 else 0.0
                        # Snapshots for TWAP/VWAP
                        shared.snapshots.append(shared.orderbook.copy())
                        if len(shared.snapshots) > 100:
                            shared.snapshots.pop(0)
        except Exception as e:
            with shared.lock:
                shared.error = str(e)
    asyncio.run(ws_loop())

@st.cache_resource
def get_shared(ws_url):
    shared = SharedData()
    t = threading.Thread(target=websocket_thread, args=(shared, ws_url), daemon=True)
    t.start()
    return shared

# --- Trade Models ---
class TradeModels:
    def __init__(self, slippage_mode='Linear'):
        self.slippage_mode = slippage_mode
        # Dummy data for quantile regression
        X = np.random.rand(100, 2)
        y = 0.0001 * X[:,0] + 0.05 * X[:,1] + 0.001 + np.random.normal(0, 0.0001, 100)
        self.linreg = LinearRegression().fit(X, y)
        self.qreg = QuantileRegressor(quantile=0.5, solver='highs').fit(X, y)
    def estimate_slippage(self, trade_size, volatility):
        X = np.array([[trade_size, volatility]])
        if self.slippage_mode == 'Quantile':
            return float(self.qreg.predict(X)[0])
        else:
            return float(self.linreg.predict(X)[0])
    def calculate_fees(self, trade_value, fee_tier, fee_type='taker'):
        fee_rates = {
            'taker': {'Tier 1': 0.001, 'Tier 2': 0.0008, 'Tier 3': 0.0005},
            'maker': {'Tier 1': 0.0008, 'Tier 2': 0.0006, 'Tier 3': 0.0003}
        }
        fee_rate = fee_rates[fee_type].get(fee_tier, 0.001)
        return float(trade_value * fee_rate)
    def estimate_market_impact(self, trade_size, volatility, liquidity):
        eta = 0.01
        gamma = 0.02
        impact_cost = eta * trade_size + 0.5 * gamma * (trade_size ** 2) / max(liquidity, 1)
        return float(impact_cost)
    def predict_maker_taker(self, features):
        return 0.5

# --- TWAP/VWAP Simulation ---
def simulate_twap_vwap(snapshots, total_usd, n_slices):
    slice_usd = total_usd / n_slices
    twap_prices = []
    vwap_prices = []
    for ob in snapshots[-n_slices:]:
        if ob['bids'] and ob['asks']:
            best_bid = float(ob['bids'][0][0])
            best_ask = float(ob['asks'][0][0])
            mid = (best_bid + best_ask) / 2
            twap_prices.append(mid)
            vwap_prices.append(best_ask)
    twap_avg = np.mean(twap_prices) if twap_prices else 0
    vwap_avg = np.mean(vwap_prices) if vwap_prices else 0
    return twap_avg, vwap_avg

# --- PnL Simulator State ---
def init_pnl_state():
    if 'pnl_trades' not in st.session_state:
        st.session_state['pnl_trades'] = []
    if 'position' not in st.session_state:
        st.session_state['position'] = 0.0
    if 'cash' not in st.session_state:
        st.session_state['cash'] = 10000.0
    if 'avg_entry' not in st.session_state:
        st.session_state['avg_entry'] = 0.0
    if 'pnl_history' not in st.session_state:
        st.session_state['pnl_history'] = []

# --- Main App ---
def main():
    st.set_page_config(page_title="Trade Simulator", layout="wide")
    st.title("High-Performance Crypto Trade Simulator")

    # --- Auto-refresh every 2s ---
    st_autorefresh(interval=1000, key="datarefresh")

    spot_pairs = fetch_okx_spot_pairs()
    col1, col2 = st.columns([1,2])

    # --- Inputs ---
    with col1:
        st.header("Input Parameters")
        exchange = st.text_input("Exchange", value="OKX", disabled=True)
        spot_asset = st.selectbox("Spot Asset", spot_pairs, index=spot_pairs.index('BTC-USDT') if 'BTC-USDT' in spot_pairs else 0)
        order_type = st.text_input("Order Type", value="market", disabled=True)
        quantity = st.number_input("Quantity (USD)", min_value=1.0, value=100.0, step=1.0)
        fee_tier = st.selectbox("Fee Tier", ['Tier 1', 'Tier 2', 'Tier 3'])
        slippage_mode = st.selectbox("Slippage Model", ['Linear', 'Quantile'])
        n_slices = st.number_input("TWAP/VWAP Slices", min_value=2, max_value=20, value=5, step=1)

    # --- Shared data singleton per asset ---
    ws_url = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{spot_asset}"
    shared = get_shared(ws_url)

    # --- Read latest data ---
    with shared.lock:
        orderbook = shared.orderbook.copy()
        latency = shared.latency
        latency_series = shared.latency_series.copy()
        volatility = shared.volatility
        snapshots = shared.snapshots.copy()
        error = shared.error

    with col1:
        st.number_input("Volatility", value=volatility, disabled=True, format="%.6f")

    # --- Model Calculations ---
    models = TradeModels(slippage_mode=slippage_mode)
    if orderbook['bids'] and orderbook['asks']:
        best_bid = float(orderbook['bids'][0][0])
        best_ask = float(orderbook['asks'][0][0])
        mid_price = (best_bid + best_ask) / 2
        trade_size = quantity / mid_price if mid_price > 0 else 0.0
        slippage = models.estimate_slippage(trade_size, volatility)
        fees_taker = models.calculate_fees(trade_size * mid_price, fee_tier, 'taker')
        fees_maker = models.calculate_fees(trade_size * mid_price, fee_tier, 'maker')
        liquidity = sum(float(x[1]) for x in orderbook['asks'][:10])
        market_impact = models.estimate_market_impact(trade_size, volatility, liquidity)
        net_cost = slippage + fees_taker + market_impact
        maker_taker = models.predict_maker_taker([best_bid, best_ask])
        sum_bids = sum(float(x[1]) for x in orderbook['bids'][:10])
        sum_asks = sum(float(x[1]) for x in orderbook['asks'][:10])
        imbalance = (sum_bids - sum_asks) / (sum_bids + sum_asks) if (sum_bids + sum_asks) > 0 else 0
    else:
        best_bid = best_ask = mid_price = trade_size = slippage = fees_taker = fees_maker = market_impact = net_cost = maker_taker = imbalance = 0.0

    # --- TWAP/VWAP Simulation ---
    if len(snapshots) >= n_slices:
        twap_avg, vwap_avg = simulate_twap_vwap(snapshots, quantity, int(n_slices))
    else:
        twap_avg = vwap_avg = 0.0

    # --- Output Table ---
    with col2:
        st.header("Simulation Output")
        if error:
            st.error(f"WebSocket error: {error}")
        output_data = {
            "Expected Slippage": [slippage],
            "Taker Fees": [fees_taker],
            "Maker Fees": [fees_maker],
            "Expected Market Impact": [market_impact],
            "Net Cost": [net_cost],
            "Maker/Taker Proportion": [maker_taker],
            "Orderbook Imbalance": [imbalance],
            "TWAP Simulated Avg Price": [twap_avg],
            "VWAP Simulated Avg Price": [vwap_avg],
            "Internal Latency (ms)": [latency]
        }
        df = pd.DataFrame(output_data)
        st.dataframe(df.T, use_container_width=True)
        # --- Export to CSV ---
        csv = df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='trade_simulation_results.csv',
            mime='text/csv',
        )
        # --- Orderbook Visualization ---
        st.subheader("Orderbook Depth (Top 20)")
        try:
            bids = np.array([[float(x[0]), float(x[1])] for x in orderbook['bids'][:20]])
            asks = np.array([[float(x[0]), float(x[1])] for x in orderbook['asks'][:20]])
            if bids.size > 0 and asks.size > 0:
                bids = bids[np.argsort(-bids[:,0])]
                asks = asks[np.argsort(asks[:,0])]
                bid_prices = bids[:,0]
                bid_sizes = np.cumsum(bids[:,1])
                ask_prices = asks[:,0]
                ask_sizes = np.cumsum(asks[:,1])
                fig, ax = plt.subplots()
                ax.plot(bid_prices, bid_sizes, color='green', label='Bids')
                ax.plot(ask_prices, ask_sizes, color='red', label='Asks')
                ax.set_title("Orderbook Depth")
                ax.set_xlabel("Price")
                ax.set_ylabel("Cumulative Size")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Waiting for orderbook data...")
        except Exception as e:
            st.warning(f"Orderbook plot error: {e}")
        # --- Latency Time-Series Plot ---
        st.subheader("Internal Latency Over Time")
        if latency_series:
            fig2, ax2 = plt.subplots()
            ax2.plot(latency_series, color='blue')
            ax2.set_title("Internal Latency (ms)")
            ax2.set_xlabel("Tick")
            ax2.set_ylabel("Latency (ms)")
            st.pyplot(fig2)
        else:
            st.info("Waiting for latency data...")

    # --- PnL Simulator ---
    with col2:
        st.header("PnL Simulator")
        init_pnl_state()
        buy_qty = st.number_input("Buy Quantity (asset)", min_value=0.0, value=0.0, step=0.0001, format="%.4f", key="buy_qty")
        sell_qty = st.number_input("Sell Quantity (asset)", min_value=0.0, value=0.0, step=0.0001, format="%.4f", key="sell_qty")
        col_buy, col_sell = st.columns(2)
        with col_buy:
            if st.button("Buy at Ask"):
                if buy_qty > 0 and best_ask > 0:
                    cost = buy_qty * best_ask
                    prev_pos = st.session_state['position']
                    st.session_state['cash'] -= cost
                    st.session_state['position'] += buy_qty
                    if prev_pos == 0:
                        st.session_state['avg_entry'] = best_ask
                    else:
                        st.session_state['avg_entry'] = (
                            st.session_state['avg_entry'] * prev_pos + best_ask * buy_qty
                        ) / (prev_pos + buy_qty)
                    st.session_state['pnl_trades'].append({
                        'time': datetime.now(),
                        'side': 'Buy',
                        'qty': buy_qty,
                        'price': best_ask,
                        'cash': st.session_state['cash'],
                        'position': st.session_state['position']
                    })
        with col_sell:
            if st.button("Sell at Bid"):
                if sell_qty > 0 and best_bid > 0 and st.session_state['position'] >= sell_qty:
                    proceeds = sell_qty * best_bid
                    st.session_state['cash'] += proceeds
                    st.session_state['position'] -= sell_qty
                    realized = (best_bid - st.session_state['avg_entry']) * sell_qty
                    if st.session_state['position'] == 0:
                        st.session_state['avg_entry'] = 0.0
                    st.session_state['pnl_trades'].append({
                        'time': datetime.now(),
                        'side': 'Sell',
                        'qty': sell_qty,
                        'price': best_bid,
                        'cash': st.session_state['cash'],
                        'position': st.session_state['position'],
                        'realized_pnl': realized
                    })
        # Calculate PnL
        unrealized_pnl = (mid_price - st.session_state['avg_entry']) * st.session_state['position'] if st.session_state['position'] > 0 else 0.0
        realized_pnl = sum(trade.get('realized_pnl', 0.0) for trade in st.session_state['pnl_trades'])
        total_pnl = realized_pnl + unrealized_pnl
        st.session_state['pnl_history'].append({
            'time': datetime.now(),
            'realized': realized_pnl,
            'unrealized': unrealized_pnl,
            'total': total_pnl
        })
        st.write(f"**Cash:** ${st.session_state['cash']:.2f}")
        st.write(f"**Position:** {st.session_state['position']:.6f} units")
        st.write(f"**Avg Entry Price:** ${st.session_state['avg_entry']:.2f}")
        st.write(f"**Unrealized PnL:** ${unrealized_pnl:.2f}")
        st.write(f"**Realized PnL:** ${realized_pnl:.2f}")
        st.write(f"**Total PnL:** ${total_pnl:.2f}")
        # Plot PnL over time
        df_pnl = pd.DataFrame(st.session_state['pnl_history'])
        if not df_pnl.empty:
            st.line_chart(df_pnl.set_index('time')[['realized', 'unrealized', 'total']])
        st.subheader("Trade Log")
        df_trades = pd.DataFrame(st.session_state['pnl_trades'])
        if not df_trades.empty:
            st.dataframe(df_trades)
        if st.button("Reset PnL Simulator"):
            st.session_state['pnl_trades'] = []
            st.session_state['position'] = 0.0
            st.session_state['cash'] = 10000.0
            st.session_state['avg_entry'] = 0.0
            st.session_state['pnl_history'] = []

if __name__ == '__main__':
    main()
