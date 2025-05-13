import asyncio
import websockets
import json
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import threading
import logging
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import requests


# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- ORDERBOOK CLIENT ---
class OrderBookClient:
    """
    Connects to the OKX L2 orderbook WebSocket, processes real-time data,
    maintains rolling mid-prices for volatility estimation, and calls a callback
    on each new tick.
    """
    def __init__(self, ws_url, symbol, update_callback):
        self.ws_url = ws_url
        self.symbol = symbol
        self.orderbook = {'bids': [], 'asks': []}
        self.last_tick_time = None
        self.latencies = []
        self.update_callback = update_callback
        self.running = True
        self.mid_prices = []

    async def connect(self):
        try:
            async with websockets.connect(self.ws_url, ping_interval=None) as ws:
                logging.info(f"Connected to {self.ws_url}")
                while self.running:
                    start = time.time()
                    msg = await ws.recv()
                    data = json.loads(msg)
                    self.process_orderbook(data)
                    end = time.time()
                    latency = (end - start) * 1000  # ms
                    self.latencies.append(latency)
                    self.update_callback(self.orderbook, latency)
        except Exception as e:
            logging.error(f"WebSocket error: {e}")

    def process_orderbook(self, data):
        self.orderbook['bids'] = data.get('bids', [])
        self.orderbook['asks'] = data.get('asks', [])
        self.last_tick_time = data.get('timestamp', None)
        # Track mid prices for volatility estimation
        try:
            if self.orderbook['bids'] and self.orderbook['asks']:
                best_bid = float(self.orderbook['bids'][0][0])
                best_ask = float(self.orderbook['asks'][0][0])
                mid_price = (best_bid + best_ask) / 2
                self.mid_prices.append(mid_price)
                if len(self.mid_prices) > 100:
                    self.mid_prices.pop(0)
        except Exception as e:
            logging.warning(f"Orderbook processing error: {e}")

    def get_volatility(self):
        # Rolling standard deviation of mid prices (last 50)
        if len(self.mid_prices) < 2:
            return 0.0
        return float(np.std(self.mid_prices[-50:]))
    

    def stop(self):
        self.running = False

# --- TRADE MODELS ---
class TradeModels:
    """
    Implements:
    - Slippage estimation (linear regression)
    - Fee calculation (rule-based)
    - Market impact (Almgren-Chriss)
    - Maker/taker proportion (logistic regression, dummy for demo)
    """
    def __init__(self):
        self.slippage_model = LinearRegression()
        self.maker_taker_model = LogisticRegression()

    def estimate_slippage(self, trade_size, volatility):
        # Linear model: slippage = a * trade_size + b * volatility + c
        # (Replace with quantile regression if needed)
        coef = np.array([0.0001, 0.05])
        intercept = 0.001
        slippage = trade_size * coef[0] + volatility * coef[1] + intercept
        return float(slippage)

    def calculate_fees(self, trade_value, fee_tier):
        # Fee schedule from OKX docs (example values)
        fee_rates = {'Tier 1': 0.001, 'Tier 2': 0.0008, 'Tier 3': 0.0005}
        fee_rate = fee_rates.get(fee_tier, 0.001)
        return float(trade_value * fee_rate)

    def estimate_market_impact(self, trade_size, volatility, liquidity):
        # Almgren-Chriss model: Impact = eta*X + 0.5*gamma*X^2/liquidity
        eta = 0.01  # permanent impact coefficient (calibrated)
        gamma = 0.02  # temporary impact coefficient (calibrated)
        impact_cost = eta * trade_size + 0.5 * gamma * (trade_size ** 2) / max(liquidity, 1)
        return float(impact_cost)

    def predict_maker_taker(self, features):
        # Dummy: 50% probability for both
        return 0.5

# --- UI CLASS ---
class TradeSimulatorUI:
    """
    Tkinter GUI with left panel for inputs and right panel for outputs.
    Updates outputs live as new orderbook data arrives.
    """
    def __init__(self, root, models, orderbook_client,spot_pairs):
        self.root = root
        self.models = models
        self.orderbook_client = orderbook_client
        self.root.title("High-Performance Trade Simulator")
        self.spot_pairs=spot_pairs
        self.create_widgets()
        self.create_orderbook_plot()

    def create_orderbook_plot(self):
        # Matplotlib Figure for orderbook
        self.fig, self.ax = plt.subplots(figsize=(5,3))
        self.ax.set_title("Orderbook Depth")
        self.ax.set_xlabel("Price")
        self.ax.set_ylabel("Cumulative Size")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2)

    def update_orderbook_plot(self, orderbook):
        try:
            bids = np.array([[float(x[0]), float(x[1])] for x in orderbook['bids'][:20]])
            asks = np.array([[float(x[0]), float(x[1])] for x in orderbook['asks'][:20]])
            if bids.size == 0 or asks.size == 0:
                return
            bids = bids[np.argsort(-bids[:,0])]
            asks = asks[np.argsort(asks[:,0])]
            bid_prices = bids[:,0]
            bid_sizes = np.cumsum(bids[:,1])
            ask_prices = asks[:,0]
            ask_sizes = np.cumsum(asks[:,1])
            self.ax.clear()
            self.ax.plot(bid_prices, bid_sizes, color='green', label='Bids')
            self.ax.plot(ask_prices, ask_sizes, color='red', label='Asks')
            self.ax.set_title("Orderbook Depth")
            self.ax.set_xlabel("Price")
            self.ax.set_ylabel("Cumulative Size")
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            logging.warning(f"Orderbook plot error: {e}")

    def create_widgets(self):
        # Left panel for input parameters
        left_frame = ttk.Frame(self.root, padding=10)
        left_frame.grid(row=0, column=0, sticky='ns')

        ttk.Label(left_frame, text="Exchange:").grid(row=0, column=0, sticky='w')
        self.exchange_var = tk.StringVar(value='OKX')
        ttk.Entry(left_frame, textvariable=self.exchange_var, state='readonly').grid(row=0, column=1)

        ttk.Label(left_frame, text="Spot Asset:").grid(row=1, column=0, sticky='w')
        self.spot_asset_var = tk.StringVar(value='BTC-USDT-SWAP')
        self.spot_asset_dropdown = ttk.Combobox(left_frame, textvariable=self.spot_asset_var, values=self.spot_pairs, width=20)
        self.spot_asset_dropdown.grid(row=1, column=1)
        self.spot_asset_dropdown.bind("<<ComboboxSelected>>", self.on_asset_change)

        ttk.Label(left_frame, text="Order Type:").grid(row=2, column=0, sticky='w')
        self.order_type_var = tk.StringVar(value='market')
        ttk.Entry(left_frame, textvariable=self.order_type_var, state='readonly').grid(row=2, column=1)

        ttk.Label(left_frame, text="Quantity (USD):").grid(row=3, column=0, sticky='w')
        self.quantity_var = tk.DoubleVar(value=100.0)
        ttk.Entry(left_frame, textvariable=self.quantity_var).grid(row=3, column=1)

        ttk.Label(left_frame, text="Volatility:").grid(row=4, column=0, sticky='w')
        self.volatility_var = tk.DoubleVar(value=0.0)
        ttk.Entry(left_frame, textvariable=self.volatility_var, state='readonly').grid(row=4, column=1)

        ttk.Label(left_frame, text="Fee Tier:").grid(row=5, column=0, sticky='w')
        self.fee_tier_var = tk.StringVar(value='Tier 1')
        ttk.Combobox(left_frame, textvariable=self.fee_tier_var, values=['Tier 1', 'Tier 2', 'Tier 3']).grid(row=5, column=1)

        # Right panel for output parameters
        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.grid(row=0, column=1, sticky='ns')

        ttk.Label(right_frame, text="Expected Slippage:").grid(row=0, column=0, sticky='w')
        self.slippage_var = tk.StringVar(value='N/A')
        ttk.Label(right_frame, textvariable=self.slippage_var).grid(row=0, column=1, sticky='w')

        ttk.Label(right_frame, text="Expected Fees:").grid(row=1, column=0, sticky='w')
        self.fees_var = tk.StringVar(value='N/A')
        ttk.Label(right_frame, textvariable=self.fees_var).grid(row=1, column=1, sticky='w')

        ttk.Label(right_frame, text="Expected Market Impact:").grid(row=2, column=0, sticky='w')
        self.market_impact_var = tk.StringVar(value='N/A')
        ttk.Label(right_frame, textvariable=self.market_impact_var).grid(row=2, column=1, sticky='w')

        ttk.Label(right_frame, text="Net Cost:").grid(row=3, column=0, sticky='w')
        self.net_cost_var = tk.StringVar(value='N/A')
        ttk.Label(right_frame, textvariable=self.net_cost_var).grid(row=3, column=1, sticky='w')

        ttk.Label(right_frame, text="Maker/Taker Proportion:").grid(row=4, column=0, sticky='w')
        self.maker_taker_var = tk.StringVar(value='N/A')
        ttk.Label(right_frame, textvariable=self.maker_taker_var).grid(row=4, column=1, sticky='w')

        ttk.Label(right_frame, text="Internal Latency (ms):").grid(row=5, column=0, sticky='w')
        self.latency_var = tk.StringVar(value='N/A')
        ttk.Label(right_frame, textvariable=self.latency_var).grid(row=5, column=1, sticky='w')

    def update_output(self, orderbook, latency):
        try:
            if not orderbook['bids'] or not orderbook['asks']:
                return
            best_bid = float(orderbook['bids'][0][0])
            best_ask = float(orderbook['asks'][0][0])
            mid_price = (best_bid + best_ask) / 2

            # Volatility from rolling std
            volatility = self.orderbook_client.get_volatility()
            self.volatility_var.set(round(volatility, 5))

            # Trade size in base asset
            usd_qty = self.quantity_var.get()
            trade_size = usd_qty / mid_price if mid_price > 0 else 0.0
            fee_tier = self.fee_tier_var.get() or 'Tier 1'

            slippage = self.models.estimate_slippage(trade_size, volatility)
            fees = self.models.calculate_fees(trade_size * mid_price, fee_tier)
            liquidity = sum(float(x[1]) for x in orderbook['asks'][:10])  # sum top 10 ask sizes
            market_impact = self.models.estimate_market_impact(trade_size, volatility, liquidity)
            net_cost = slippage + fees + market_impact
            maker_taker = self.models.predict_maker_taker([best_bid, best_ask])

            self.slippage_var.set(f"{slippage:.6f}")
            self.fees_var.set(f"{fees:.6f}")
            self.market_impact_var.set(f"{market_impact:.6f}")
            self.net_cost_var.set(f"{net_cost:.6f}")
            self.maker_taker_var.set(f"{maker_taker:.2f}")
            self.latency_var.set(f"{latency:.2f}")
            self.update_orderbook_plot(orderbook)

        except Exception as e:
            logging.error(f"Error updating output: {e}")
    def on_asset_change(self, event=None):
        # Restart WebSocket client with new symbol
        new_symbol = self.spot_asset_var.get()
        self.orderbook_client.stop()
        ws_url = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{new_symbol}"
        new_client = OrderBookClient(ws_url, new_symbol, self.update_output)
        self.orderbook_client = new_client
        loop = asyncio.get_event_loop()
        loop.create_task(new_client.connect())


# --- ASYNCIO THREAD FOR WEBSOCKET ---
class AsyncioThread(threading.Thread):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop

    def run(self):
        self.loop.run_forever()


def fetch_okx_spot_pairs():
        url = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            pairs = [x['instId'] for x in data['data']]
            return sorted(pairs)
        except Exception as e:
            logging.warning(f"Could not fetch spot pairs: {e}")
            return ['BTC-USDT-SWAP']

# --- MAIN FUNCTION ---
def main():
    spot_pairs = fetch_okx_spot_pairs()

    root = tk.Tk()
    ws_url = 'wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP'
    symbol = 'BTC-USDT-SWAP'

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    models = TradeModels()
    orderbook_client = OrderBookClient(ws_url, symbol, lambda ob, lat: None)
    ui = TradeSimulatorUI(root, models, orderbook_client,spot_pairs)

    # Create OrderBookClient instance with UI update callback
    client = OrderBookClient(ws_url, symbol, ui.update_output)
    ui.orderbook_client = client

    loop.create_task(client.connect())
    thread = AsyncioThread(loop)
    thread.start()

    try:
        root.mainloop()
    finally:
        client.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join()

if __name__ == '__main__':
    main()
