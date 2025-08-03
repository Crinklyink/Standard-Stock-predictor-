import json
import hashlib
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import r2_score, mean_absolute_error

# Simple CLI chatbot loop
def chatbot():
    print("Welcome to StockBot! Ask me to predict a stock price. (e.g., 'AAPL 7' for 7 days ahead)")
    while True:
        user_input = input('You: ')
        if user_input.lower() in ['exit', 'quit']:
            print('Goodbye!')
            break
        response = handle_query(user_input)
        print('Bot:', response)

def handle_query(query, model_name='Linear Regression'):
    # Parse input like 'AAPL 7' or 'TSLA 1'
    import re
    query = query.strip()
    # Accept both 'AAPL 7', 'AAPL 12h', 'AAPL 1d', etc.
    match = re.match(r'^(\w+)\s+(\d+)([dh]?)$', query, re.IGNORECASE)
    if not match:
        return ("Sorry, I didn't understand. Try: 'AAPL 7' for 7 days or 'AAPL 12h' for 12 hours ahead.", None, "")
    symbol = match.group(1).upper()
    value = int(match.group(2))
    unit = match.group(3).lower()
    # --- Ensemble and Model Comparison ---
    model_names = ["Linear Regression", "Random Forest", "SVR"]
    if model_name == "Ensemble":
        preds = []
        accs = []
        figs = []
        for m in model_names:
            res = handle_query(query, m)
            # Always expect a tuple of 3, but handle if not
            if not isinstance(res, tuple) or len(res) != 3:
                res = ("Error", None, "")
            r, fig, acc = res
            if fig is not None:
                try:
                    preds.append(float(r.split('$')[-1].split()[0]))
                except Exception:
                    preds.append(np.nan)
                accs.append(acc)
                figs.append((m, fig, r, acc))
        if preds:
            avg_pred = np.nanmean(preds)
            html = "<div style='margin-bottom:1em;'><b>Ensemble (average of all models):</b> <span style='color:#00c6fb;'>${:.2f}</span></div>".format(avg_pred)
            html += "<table style='width:100%;background:#10131a;border-radius:12px;border:1.5px solid #005bea55;color:#eaf6ff;font-size:1.08em;'><tr><th style='padding:0.5em;'>Model</th><th style='padding:0.5em;'>Prediction</th><th style='padding:0.5em;'>Accuracy</th></tr>"
            for (m, _, r, acc) in figs:
                html += f"<tr><td style='padding:0.5em;'>{m}</td><td style='padding:0.5em;'>{r.split('using')[0].split(':')[-1]}</td><td style='padding:0.5em;'>{acc}</td></tr>"
            html += "</table>"
            return (html, figs[0][1], "Ensemble: ${:.2f}".format(avg_pred))
        else:
            return ("Error: Could not compute ensemble.", None, "")
    # --- Single model prediction logic ---
    try:
        # Parse time unit
        days_ahead = 0
        hours_ahead = 0
        if unit == 'h':
            hours_ahead = value
        else:
            days_ahead = value

        # Predict and get data
        pred, data, future_dt = predict_stock(symbol, days_ahead, hours_ahead, model_name, return_data=True)

        # Compute accuracy (R2, MAE) on last 10 points
        X = data[['DateHour']]
        y = data['Close']
        if model_name == 'Random Forest':
            model = RandomForestRegressor().fit(X, y)
        elif model_name == 'SVR':
            model = SVR().fit(X, y)
        else:
            model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        acc_str = f"R²: {r2:.3f}, MAE: {mae:.2f}"

        # Log accuracy
        try:
            with open("accuracy_log.txt", "a") as f:
                horizon = f"{days_ahead}d" if days_ahead else f"{hours_ahead}h"
                f.write(f"{datetime.now().isoformat()} | {symbol} | {model_name} | {horizon} | {pred:.2f} | {r2:.4f} | {mae:.4f}\n")
        except Exception:
            pass

        # Build result string
        result = f"Predicted price for <b>{symbol}</b> {days_ahead}d {hours_ahead}h ahead: <span style='color:#00c6fb;'>${pred:.2f}</span> (using <b>{model_name}</b>)"

        # Build chart (ensure x-axis is datetime and present)
        fig = go.Figure()
        x_vals = pd.to_datetime(data['Datetime']) if 'Datetime' in data.columns else data.index
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=data['Close'],
            mode='lines',
            name='History',
            line=dict(color='#00c6fb', width=3),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        # Add prediction point
        fig.add_trace(go.Scatter(
            x=[future_dt],
            y=[pred],
            mode='markers',
            marker=dict(color='red', size=16, line=dict(width=2, color='black')),
            name='Prediction',
            hovertemplate='Prediction<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        fig.update_layout(
            title={
                'text': f"{symbol} Price Prediction ({model_name})",
                'y':0.95,
                'xanchor': 'center',
                'font': dict(size=22, color='#fff')
            },
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(30,30,30,0.7)', bordercolor='gray', borderwidth=1, font=dict(size=14, color='#fff')),
            plot_bgcolor='#181a20',
            paper_bgcolor='#181a20',
            xaxis=dict(showgrid=True, gridcolor='#333', tickangle=45, tickfont=dict(size=12, color='#fff'), linecolor='#fff', zerolinecolor='#333', title=dict(font=dict(color='#fff'))),
            yaxis=dict(showgrid=True, gridcolor='#333', tickfont=dict(size=12, color='#fff'), linecolor='#fff', zerolinecolor='#333', title=dict(font=dict(color='#fff'))),
            font=dict(color='#fff'),
            margin=dict(l=40, r=20, t=60, b=80)
        )

        return (result, fig, acc_str)
    except Exception as e:
        return (f"Error: {e}", None, "")

def predict_stock(symbol, days_ahead=0, hours_ahead=0, model_name='Linear Regression', return_data=False):
    data = yf.download(symbol, period='60d', interval='1h')
    if data.empty:
        raise ValueError('No data found for symbol')
    data = data.reset_index()
    # Ensure datetime column is present and named 'Datetime'
    if 'Datetime' not in data.columns:
        # Try to find the datetime column (could be 'index' or first column)
        possible_dt = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
        if possible_dt:
            data = data.rename(columns={possible_dt[0]: 'Datetime'})
        else:
            data = data.rename(columns={data.columns[0]: 'Datetime'})
    # Remove rows with missing closing prices to avoid training errors
    data = data.dropna(subset=['Close'])
    if data.empty:
        raise ValueError('No valid data after dropping missing Close values')
    # Use date+hour/24 as a single float feature for more granularity
    data['DateHour'] = pd.to_datetime(data['Datetime']).map(lambda d: d.toordinal() + d.hour/24.0)
    X = data[['DateHour']]
    y = data['Close']
    if model_name == 'Random Forest':
        model = RandomForestRegressor().fit(X, y)
    elif model_name == 'SVR':
        model = SVR().fit(X, y)
    else:
        model = LinearRegression().fit(X, y)
    future_dt = datetime.now() + timedelta(days=days_ahead, hours=hours_ahead)
    future_datehour = future_dt.toordinal() + future_dt.hour/24.0
    pred = model.predict([[future_datehour]])
    if return_data:
        return float(pred.flatten()[0]), data, future_dt
    return float(pred.flatten()[0])

def get_realtime_chart(symbol):
    try:
        data = yf.download(symbol, period='1d', interval='1m')
        if data.empty or 'Close' not in data.columns or data['Close'].dropna().empty:
            return go.Figure(), f"No data for {symbol}"
        data = data.reset_index()
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        # Remove rows with missing Close values
        data = data.dropna(subset=['Close'])
        if data.empty:
            return go.Figure(), f"No data for {symbol}"
        # Format x-axis as string for Plotly/Gradio compatibility
        x_hist_str = data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        y_hist = data['Close'].astype(float).tolist()
        last_idx = data['Close'].last_valid_index()
        last_price = data.loc[last_idx, 'Close']
        last_time = data.loc[last_idx, 'Datetime']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_hist_str,
            y=y_hist,
            mode='lines',
            name='Real-Time',
            line=dict(color='royalblue', width=3),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        # Add a marker for the latest price
        fig.add_trace(go.Scatter(
            x=[last_time.strftime('%Y-%m-%d %H:%M:%S')],
            y=[last_price],
            mode='markers',
            marker=dict(color='red', size=18, line=dict(width=2, color='black')),
            name='Latest',
            hovertemplate='Latest<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        fig.update_layout(
            title={
                'text': f"{symbol} Real-Time Price",
                'x':0.5,
                'xanchor': 'center',
                'font': dict(size=22, color='#fff')
            },
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(30,30,30,0.7)', bordercolor='gray', borderwidth=1, font=dict(size=14, color='#fff')),
            plot_bgcolor='#181a20',
            paper_bgcolor='#181a20',
            xaxis=dict(showgrid=True, gridcolor='#333', tickangle=45, tickfont=dict(size=12, color='#fff'), linecolor='#fff', zerolinecolor='#333', title=dict(font=dict(color='#fff'))),
            yaxis=dict(showgrid=True, gridcolor='#333', tickfont=dict(size=12, color='#fff'), linecolor='#fff', zerolinecolor='#333', title=dict(font=dict(color='#fff'))),
            font=dict(color='#fff'),
            margin=dict(l=40, r=20, t=60, b=80)
        )
        return fig, f"Last: ${last_price:.2f} at {last_time.strftime('%Y-%m-%d %H:%M')}"
    except Exception as e:
        return go.Figure(), f"Error: {e}"

# --- Move these helper functions to top-level scope so Gradio can access them ---
def realtime_tab(symbol):
    fig, status = get_realtime_chart(symbol)
    return fig, status

def watchlist_tab(watchlist):
    results = []
    for symbol in watchlist:
        try:
            # Try 1d/1m, fallback to 5d/5m if no data (e.g. ETF, after hours, or weekends)
            data = yf.download(symbol, period='1d', interval='1m')
            if data.empty or 'Close' not in data.columns or data['Close'].dropna().empty:
                data = yf.download(symbol, period='5d', interval='5m')
            if data.empty or 'Close' not in data.columns or data['Close'].dropna().empty:
                results.append(f"{symbol}: No data")
                continue
            data = data.dropna(subset=['Close'])
            last = data['Close'].iloc[-1]
            prev = data['Close'].iloc[-2] if len(data['Close']) > 1 else last
            change = last - prev
            trend = '↑' if change > 0 else ('↓' if change < 0 else '→')
            results.append(f"{symbol}: ${last:.2f} {trend} ({change:+.2f})")
        except Exception as e:
            results.append(f"{symbol}: Error")
    return results

def get_watchlist(username=None):
    return []

def set_watchlist(username, watchlist):
    return True

import gradio as gr
import time
import plotly.graph_objs as go

import os
import pandas as pd

def gradio_predict(user_input, model_name):
    result, fig, acc_str = handle_query(user_input, model_name)
    # Format the result as a styled HTML card
    if result and not result.startswith("Error"):
        html_result = f'''
        <div style="background:linear-gradient(90deg,#10131a,#181c24);border-radius:16px;padding:1.3em 1.5em 1.1em 1.5em;margin:0.5em 0 0.7em 0;box-shadow:0 2px 16px #005bea33,0 1.5px 0 #00c6fb22;border:1.5px solid #005bea55;">
            <span style="font-size:1.25em;font-weight:700;color:#00c6fb;">Prediction Result</span><br>
            <span style="font-size:1.13em;color:#eaf6ff;">{result}</span>
        </div>
        '''
    else:
        html_result = f'<div style="background:#2a1a1a;border-radius:12px;padding:1em 1.2em;color:#ffb3b3;font-weight:600;">{result}</div>'
    return html_result, fig, acc_str

def show_accuracy_log():
    if not os.path.exists("accuracy_log.txt"):
        return pd.DataFrame(columns=["Timestamp", "Symbol", "Model", "Horizon", "Price", "R2", "MAE"])
    try:
        df = pd.read_csv(
            "accuracy_log.txt",
            sep=r"\|",
            header=None,
            names=["Timestamp", "Symbol", "Model", "Horizon", "Price", "R2", "MAE"],
            engine="python"
        )
        # Clean up whitespace
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        return df.tail(20)  # Show last 20 entries
    except Exception as e:
        return pd.DataFrame({"Error":[str(e)]})

custom_css = """
body, .gradio-container {
    background: #0a0c10 !important;
    color: #eaf6ff !important;
}
.main-card {
    background: #10131a;
    border-radius: 22px;
    box-shadow: 0 6px 32px #005bea55, 0 1.5px 0 #00c6fb33;
    padding: 2.7rem 2.2rem 2.2rem 2.2rem;
    margin-top: 2.2rem;
    border: 1.5px solid #005bea55;
}
.header-title {
    color: #eaf6ff;
    font-size: 2.5rem;
    font-weight: 900;
    margin-bottom: 0.7rem;
    letter-spacing: 1.5px;
    text-shadow: 0 2px 12px #005bea44;
}
.header-desc {
    color: #b0cfff;
    font-size: 1.18rem;
    margin-bottom: 1.7rem;
}
.footer {
    margin-top: 2.7rem;
    color: #b0cfff;
    font-size: 1.05rem;
    text-align: center;
}
.gr-button {
    background: linear-gradient(90deg,#00c6fb,#005bea) !important;
    color: #fff !important;
    border-radius: 12px !important;
    font-weight: 800;
    font-size: 1.13rem;
    box-shadow: 0 2px 12px #005bea44, 0 1.5px 0 #00c6fb33;
    border: none !important;
    letter-spacing: 0.5px;
    transition: background 0.2s;
}
.gr-button:hover {
    background: linear-gradient(90deg,#005bea,#00c6fb) !important;
    color: #fff !important;
    box-shadow: 0 4px 18px #00c6fb55;
}
/* Styled box for all input/output components (matches Prediction Result) */
.gr-input, .gr-textbox, .gr-dropdown, .gr-plot, .gr-box, .gr-markdown, .gr-list, .gr-dataframe {
    background: linear-gradient(90deg,#10131a,#181c24) !important;
    color: #eaf6ff !important;
    border-radius: 16px !important;
    border: 1.5px solid #005bea55 !important;
    box-shadow: 0 2px 16px #005bea33, 0 1.5px 0 #00c6fb22 !important;
    padding: 1.3em 1.5em 1.1em 1.5em !important;
    margin: 0.5em 0 0.7em 0 !important;
    font-size: 1.13em !important;
    transition: box-shadow 0.2s, border 0.2s;
}
.gr-input:focus-within, .gr-textbox:focus-within, .gr-dropdown:focus-within, .gr-plot:focus-within, .gr-box:focus-within, .gr-markdown:focus-within, .gr-list:focus-within, .gr-dataframe:focus-within {
    box-shadow: 0 0 0 3px #00c6fbcc, 0 2px 16px #005bea33, 0 1.5px 0 #00c6fb22 !important;
    border: 1.5px solid #00c6fb !important;
}
.gr-input input, .gr-textbox textarea, .gr-dropdown select {
    background: transparent !important;
    color: #eaf6ff !important;
}
.gr-html {
    padding: 0 !important;
    background: none !important;
    border: none !important;
    box-shadow: none !important;
}
.gr-label, label {
    color: #b0cfff !important;
    font-weight: 700;
    letter-spacing: 0.2px;
}
.gr-plot {
    box-shadow: 0 2px 16px #005bea33;
    border: 1.5px solid #005bea55 !important;
    background: #10131a !important;
}
/* Custom tab styling */
.gr-tabs {
    background: none !important;
    border-bottom: none !important;
    margin-bottom: 0.5rem;
    display: flex !important;
    gap: 0.7em;
    justify-content: flex-start;
    padding-left: 0.5em;
}
.gr-tabitem {
    background: linear-gradient(90deg,#10131a,#181c24) !important;
    color: #eaf6ff !important;
    border-radius: 14px !important;
    border: 2.5px solid #005bea55 !important;
    margin-right: 0.2em;
    font-weight: 800;
    font-size: 1.13rem;
    letter-spacing: 0.5px;
    transition: background 0.2s, color 0.2s, border 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 12px #005bea22;
    padding: 0.7em 1.7em !important;
    min-width: 120px;
    text-align: center;
    outline: none !important;
    border-bottom: 3.5px solid transparent !important;
}
.gr-tabitem.selected {
    background: linear-gradient(90deg,#005bea,#00c6fb) !important;
    color: #fff !important;
    border: 2.5px solid #00c6fb !important;
    border-bottom: 3.5px solid #00c6fb !important;
    box-shadow: 0 4px 18px #00c6fb55;
    z-index: 2;
}
.gr-list {
    background: #181c24 !important;
    color: #eaf6ff !important;
    border-radius: 10px !important;
    border: 1.5px solid #005bea55 !important;
}
.gr-markdown {
    background: transparent !important;
    color: #eaf6ff !important;
}
::selection {
    background: #005bea99;
    color: #fff;
}
"""

def watchlist_tab(watchlist):
    if not watchlist:
        return "<div class='empty-watchlist'>Your watchlist is empty. Add stocks to track them here.</div>"
    html = "<div class='watchlist-container'>"
    for ticker in watchlist:
        html += f"<div class='watchlist-item'>{ticker} <span class='remove-btn' data-ticker='{ticker}'>×</span></div>"
    html += "</div>"
    return html

def add_to_watchlist(ticker, user, state):
    if not ticker:
        return state, watchlist_tab(state), "Please enter a ticker symbol"
    ticker = ticker.upper().strip()
    if ticker in state:
        return state, watchlist_tab(state), f"{ticker} is already in your watchlist"
    state.append(ticker)
    return state, watchlist_tab(state), gr.update(visible=False)

def remove_from_watchlist(ticker, user, state):
    if ticker in state:
        state.remove(ticker)
        return state, watchlist_tab(state), gr.update(visible=False)
    return state, watchlist_tab(state), gr.update(visible=False)

def update_watchlist_on_login(user):
    return [], watchlist_tab([]), gr.update(visible=False)

if __name__ == "__main__":
    print("Launching StockBot web interface...")
    with gr.Blocks(css=custom_css, title="StockBot - AI Stock Predictions") as demo:
        # Global state
        user_state = gr.State("guest")
        watchlist_state = gr.State([])
        
        with gr.Row():
            # Sidebar
            with gr.Column(scale=1, min_width=250, elem_classes="sidebar"):
                gr.Markdown('''
                <div style='padding:1.5em; background:linear-gradient(120deg,#10131a 80%,#005bea22 100%); border-radius:18px; box-shadow:0 2px 16px #005bea22;'>
                    <div style='display:flex;align-items:center;gap:0.7em;margin-bottom:1em;'>
                        <span style='font-size:2em;font-weight:900;color:#00c6fb;'>StockBot</span>
                    </div>
                    <div style='color:#b0cfff;font-size:1.1em;margin-bottom:1.5em;'>
                        AI Stock Market Predictions<br>
                        <span style='color:#00c6fb;font-size:0.9em;'>Powered by yfinance and scikit-learn</span>
                    </div>
                    <div style='color:#b0cfff;font-size:0.95em;'>
                        <b>Quick Start</b><br>
                        <ul style='margin:0.5em 0 0 1.2em;padding:0;'>
                            <li>📈 <b>Predict</b>: Get stock predictions</li>
                            <li>⏱️ <b>Real-Time</b>: Live market data</li>
                            <li>⭐ <b>Watchlist</b>: Track your stocks</li>
                            <li>📊 <b>Accuracy Log</b>: Model performance</li>
                        </ul>
                    </div>
                    <div style='margin-top:1.5em;font-size:0.9em;color:#b0cfff;'>
                        <i>This is a demo and not financial advice.</i>
                    </div>
                </div>
                ''')
            
            # Main content area with tabs
            with gr.Column(scale=3):
                with gr.Tabs() as tabs:
                    with gr.Tab("📈 Predict", id="predict_tab"):
                        with gr.Column(elem_classes="main-card"):
                            gr.Markdown(
                                '''<div class="header-title">📈 Stock Price Predictor</div>
                                <div class="header-desc">
                                Get AI-powered stock price predictions. Enter a stock symbol and time horizon below.
                                </div>'''
                            )
                            # Define the input box first
                            chatbot_box = gr.Textbox(
                                label="Stock & Timeframe",
                                placeholder="e.g., AAPL 7 (for 7 days) or TSLA 12h (for 12 hours)",
                                scale=3,
                            )
                            
                            with gr.Row():
                                with gr.Column(scale=3):
                                    gr.Markdown("### How to Use")
                                    gr.Markdown("""
                                    1. Enter a stock symbol (e.g., `AAPL`, `TSLA`)
                                    2. Add time horizon (e.g., `7` for 7 days, `12h` for 12 hours)
                                    3. Select a prediction model
                                    4. Click "Predict & Show Chart"
                                    
                                    **Examples:**
                                    - `AAPL 7` - Predicts Apple's price 7 days from now
                                    - `TSLA 12h` - Predicts Tesla's price 12 hours from now
                                    - `SPY 3d` - Predicts S&P 500 ETF price in 3 days
                                    """)
                                with gr.Column(scale=2):
                                    gr.Markdown("### Popular Tickers")
                                    popular_tickers = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "NVDA", "AMD", "SPY", "QQQ"]
                                    with gr.Row():
                                        for ticker in popular_tickers[:3]:
                                            gr.Button(ticker, size="sm").click(
                                                lambda t=ticker: t + " 7",  # Default to 7 days
                                                inputs=None,
                                                outputs=chatbot_box
                                            )
                                    with gr.Row():
                                        for ticker in popular_tickers[3:6]:
                                            gr.Button(ticker, size="sm").click(
                                                lambda t=ticker: t + " 7",
                                                inputs=None,
                                                outputs=chatbot_box
                                            )
                                    with gr.Row():
                                        for ticker in popular_tickers[6:]:
                                            gr.Button(ticker, size="sm").click(
                                                lambda t=ticker: t + " 7",
                                                inputs=None,
                                                outputs=chatbot_box
                                            )
                            with gr.Row():
                                # chatbot_box is now defined above
                                model_dropdown = gr.Dropdown(
                                    choices=["Linear Regression", "Random Forest", "SVR", "Ensemble"],
                                    value="Linear Regression",
                                    label="Prediction Model",
                                    info="Choose the prediction algorithm",
                                    scale=1,
                                    container=False
                                )
                            
                            submit_btn = gr.Button("Predict & Show Chart", variant="primary", size="lg")
                            
                            with gr.Row():
                                with gr.Column(scale=2):
                                    output_box = gr.HTML(label="Prediction Result")
                                    acc_box = gr.Textbox(label="Model Accuracy", interactive=False, visible=True)
                                with gr.Column(scale=3):
                                    plot_box = gr.Plot(label="Price Chart")
                            
                            submit_btn.click(
                                fn=gradio_predict,
                                inputs=[chatbot_box, model_dropdown],
                                outputs=[output_box, plot_box, acc_box]
                            )
                    with gr.Tab("⏱️ Real-Time", id="realtime_tab"):
                        with gr.Column(elem_classes="main-card"):
                            gr.Markdown(
                                '''<div class="header-title">⏱️ Real-Time Market Data</div>
                                <div class="header-desc">View live price charts and market data for any stock.</div>'''
                            )
                            with gr.Row():
                                realtime_ticker = gr.Textbox(
                                    label="Stock Symbol",
                                    value="AAPL",
                                    placeholder="Enter a stock symbol (e.g., AAPL, TSLA, SPY)",
                                    scale=3
                                )
                                realtime_btn = gr.Button("Refresh Data", variant="secondary", scale=1)
                            
                            # Hidden components for period/interval control
                            realtime_period = gr.Textbox("1d", visible=False)
                            realtime_interval = gr.Textbox("1m", visible=False)
                            
                            with gr.Row():
                                with gr.Column(scale=3):
                                    realtime_plot = gr.Plot(label="Price Chart")
                                with gr.Column(scale=1):
                                    realtime_status = gr.Textbox(
                                        label="Latest Price",
                                        interactive=False,
                                        elem_classes=["price-display"]
                                    )
                                    gr.Markdown("### Quick Actions")
                                    with gr.Row():
                                        for period in ["1d", "5d", "1mo", "3mo"]:
                                            gr.Button(period, size="sm").click(
                                                lambda p=period: (p, "1m" if p in ["1d", "5d"] else "1d"),
                                                inputs=None,
                                                outputs=[realtime_period, realtime_interval]
                                            )
                            def update_realtime(symbol):
                                return realtime_tab(symbol)
                            realtime_btn.click(fn=update_realtime, inputs=[realtime_ticker], outputs=[realtime_plot, realtime_status])
                    with gr.Tab("⭐ Watchlist", id="watchlist_tab"):
                        with gr.Column(elem_classes="main-card"):
                            gr.Markdown('''
                                <div class="header-title">⭐ My Watchlist</div>
                                <div class="header-desc">Track your favorite stocks and view their performance at a glance.</div>
                            ''')
                            
                            # Watchlist management
                            with gr.Row():
                                with gr.Column(scale=4):
                                    watchlist_input = gr.Textbox(
                                        placeholder="Enter a stock symbol (e.g., AAPL, TSLA)",
                                        show_label=False
                                    )
                                with gr.Column(scale=1):
                                    with gr.Row():
                                        add_btn = gr.Button("Add", variant="primary")
                                        remove_btn = gr.Button("Clear All", variant="secondary")
                            
                            # Watchlist display
                            watchlist_display = gr.HTML(
                                "<div class='empty-watchlist'>Your watchlist is empty. Add stocks to track them here.</div>"
                            )
                            
                            # Watchlist summary cards
                            watchlist_summary = gr.HTML()
                            
                            # Hidden state
                            wl_msg = gr.Markdown(visible=False)
                            remove_input = gr.Textbox(visible=False)
                            
                            # Initialize watchlist list
                            watchlist_list = watchlist_display
                            
                            # Set up button callbacks
                            add_btn.click(
                                fn=add_to_watchlist, 
                                inputs=[watchlist_input, user_state, watchlist_state], 
                                outputs=[watchlist_state, watchlist_display, wl_msg]
                            )
                            remove_btn.click(
                                fn=remove_from_watchlist, 
                                inputs=[remove_input, user_state, watchlist_state], 
                                outputs=[watchlist_state, watchlist_display, wl_msg]
                            )
                            user_state.change(
                                fn=update_watchlist_on_login, 
                                inputs=[user_state], 
                                outputs=[watchlist_state, watchlist_display, wl_msg]
                            )
                    with gr.Tab("📊 Analytics", id="analytics_tab"):
                        with gr.Column(elem_classes="main-card"):
                            gr.Markdown('''
                                <div class="header-title">📊 Performance Analytics</div>
                                <div class="header-desc">Track prediction accuracy and model performance metrics.</div>
                            ''')
                            with gr.Tabs():
                                with gr.Tab("Accuracy Log"):
                                    accuracy_table = gr.Dataframe(
                                        value=show_accuracy_log(),
                                        headers=["Timestamp", "Symbol", "Model", "Horizon", "Price", "R2", "MAE"],
                                        label="Recent Predictions",
                                        interactive=False,
                                        wrap=True
                                    )
                                
                                with gr.Tab("Model Comparison"):
                                    model_comparison_plot = gr.Plot()
                                    gr.Markdown("### Model Performance Over Time")
                                    # Placeholder for future model comparison visualization
                            
                            with gr.Row():
                                refresh_btn = gr.Button("🔄 Refresh Data", variant="secondary")
                                gr.Markdown("*Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "*")
                            
                            refresh_btn.click(
                                fn=show_accuracy_log,
                                inputs=[],
                                outputs=[accuracy_table]
                            )
    demo.launch(server_name="localhost", server_port=7860, share=True)
