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

if __name__ == '__main__':

    USERS_FILE = "users.json"

    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    def load_users():
        if not os.path.exists(USERS_FILE):
            return {}
        with open(USERS_FILE, "r") as f:
            return json.load(f)

    def save_users(users):
        with open(USERS_FILE, "w") as f:
            json.dump(users, f)

    def signup(username, password):
        users = load_users()
        if username in users:
            return False, "Username already exists."
        users[username] = {"password": hash_password(password), "watchlist": []}
        save_users(users)
        return True, "Signup successful! Please log in."

    def login(username, password):
        users = load_users()
        if username not in users:
            return False, "User not found."
        if users[username]["password"] != hash_password(password):
            return False, "Incorrect password."
        return True, "Login successful!"

    def get_watchlist(username):
        users = load_users()
        return users.get(username, {}).get("watchlist", [])

    def set_watchlist(username, watchlist):
        users = load_users()
        if username in users:
            users[username]["watchlist"] = watchlist
            save_users(users)

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

    # Remove duplicate/old UI definition block
    # --- UI is now only defined in the main block below ---
    # (UI definition block removed; only the main block below is used)


if __name__ == "__main__":
    print("Launching StockBot web interface...")
    with gr.Blocks(css=custom_css) as demo:
        # Use gradio's localStorage to persist username (simulate cookies)
        user_state = gr.State("")
        watchlist_state = gr.State([])
        # Add a hidden HTML component to sync username to/from localStorage
        # Only set the script once, never update dynamically
        user_cookie = gr.HTML(
            """
            <script>
            window.addEventListener('DOMContentLoaded', function() {
                let u = localStorage.getItem('stockbot_user');
                if (u && window.gradioApp) {
                    let el = document.querySelector('[data-testid=\"state\"] input');
                    if (el) {
                        el.value = u;
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            });
            </script>
            """,
            visible=False
        )
        # --- Modernized Layout: Add a sidebar, logo, and better tab structure ---
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("""
                <div style='padding:2.2em 1.2em 1.2em 1.2em; background:linear-gradient(120deg,#10131a 80%,#005bea22 100%); border-radius:18px; box-shadow:0 2px 16px #005bea22;'>
                    <div style='display:flex;align-items:center;gap:0.7em;'>
                        <img src='https://img.icons8.com/fluency/48/000000/line-chart.png' style='width:44px;height:44px;border-radius:10px;background:#0a0c10;padding:4px;'>
                        <span style='font-size:2.1em;font-weight:900;color:#00c6fb;letter-spacing:1.5px;'>StockBot</span>
                    </div>
                    <div style='margin-top:1.2em;color:#b0cfff;font-size:1.13em;'>
                        <b>AI Stock Market Chatbot</b><br>
                        Predict, track, and analyze stocks with a modern, interactive dashboard.<br>
                        <span style='color:#00c6fb;'>Powered by yfinance, scikit-learn, and Plotly.</span>
                    </div>
                    <hr style='border:0;border-top:1.5px solid #005bea55;margin:1.5em 0;'>
                    <div style='color:#b0cfff;font-size:1.05em;'>
                        <b>Quick Links</b><br>
                        <ul style='margin:0 0 0 1.2em;padding:0;'>
                            <li>🔐 <b>Sign Up / Log In</b>: Manage your account & watchlist</li>
                            <li>📈 <b>Predict</b>: Chatbot stock prediction</li>
                            <li>⏱️ <b>Real-Time Data</b>: Live charts & prices</li>
                            <li>⭐ <b>Watchlist</b>: Track your favorite tickers</li>
                            <li>📊 <b>Accuracy Log</b>: Model performance</li>
                        </ul>
                    </div>
                    <div style='margin-top:2.2em;font-size:0.98em;color:#b0cfff;'>
                        <i>This is a demo and not financial advice.</i>
                    </div>
                </div>
                """, elem_id="sidebar-card")
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Sign Up / Log In"):
                        with gr.Column(elem_classes="main-card"):
                            gr.Markdown("""<div class='header-title'>🔐 Sign Up / Log In</div><div class='header-desc'>Create an account or log in to save your watchlist.</div>""")
                            auth_mode = gr.Radio(["Sign Up", "Log In"], value="Log In", label="Mode")
                            username_box = gr.Textbox(label="Username")
                            password_box = gr.Textbox(label="Password", type="password")
                            auth_btn = gr.Button("Submit")
                            auth_msg = gr.Markdown(visible=False)
                            def handle_auth(mode, username, password):
                                if not username or not password:
                                    return gr.update(visible=True, value="Please enter both username and password."), ""
                                if mode == "Sign Up":
                                    ok, msg = signup(username, password)
                                    return gr.update(visible=True, value=msg), ""
                                else:
                                    ok, msg = login(username, password)
                                    if ok:
                                        wl = get_watchlist(username)
                                        js = f"""
                                        <script>
                                        localStorage.setItem('stockbot_user', '{username}');
                                        </script>
                                        """
                                        return gr.update(visible=True, value=msg), username + "|COOKIESET|" + js
                                    else:
                                        return gr.update(visible=True, value=msg), ""
                            def sync_cookie_to_state(cookie_val):
                                if cookie_val:
                                    username = cookie_val
                                    wl = get_watchlist(username)
                                    return username, wl, gr.update(visible=False)
                                else:
                                    return "", [], gr.update(visible=True, value="Please log in to use your watchlist.")
                            auth_btn.click(handle_auth, [auth_mode, username_box, password_box], [auth_msg, user_state])
                            user_cookie.change(sync_cookie_to_state, [user_cookie], [user_state, watchlist_state, gr.Markdown(visible=False)])
                    with gr.Tab("Predict"):
                        with gr.Column(elem_classes="main-card"):
                            gr.Markdown(
                                '''<div class="header-title">📈 StockBot: Stock Price Predictor</div>
                                <div class="header-desc">
                                <span style="color:#00c6fb;">AI-powered, modern, and functional stock price prediction.</span><br>
                                Enter a stock symbol and time ahead (e.g., <b>AAPL 7</b>, <b>TSLA 1</b>, <b>AMZN 5</b>, <b>QFIN 3</b>, <b>NVDA 2</b>, <b>AMD 4</b>, <b>SPY 3</b>, <b>AAPL 12h</b>, <b>SPY 36h</b>).<br>
                                You can use days (e.g., <b>AAPL 7</b> or <b>AAPL 7d</b>) or hours (e.g., <b>AAPL 12h</b>).<br>
                                Select a prediction model and click <b>Predict & Show Chart</b>.<br>
                                </div>
                                <details>
                                <summary><b>Popular tickers</b></summary>
                                <ul>
                                <li><b>AAPL</b> (Apple)</li>
                                <li><b>TSLA</b> (Tesla)</li>
                                <li><b>AMZN</b> (Amazon)</li>
                                <li><b>QFIN</b> (360 DigiTech)</li>
                                <li><b>NVDA</b> (NVIDIA)</li>
                                <li><b>AMD</b> (Advanced Micro Devices)</li>
                                <li><b>SPY</b> (S&P 500 ETF)</li>
                                </ul>
                                </details>
                                <div style="margin-bottom:1.2rem; color:#b0b3c6; font-size:1rem;">
                                <b>Examples:</b> <code>SPY 3</code> predicts the S&P 500 ETF price 3 days from now.<br>
                                <code>AAPL 12h</code> predicts Apple's price 12 hours from now.
                                </div>''', elem_id="header"
                            )
                            with gr.Row():
                                chatbot_box = gr.Textbox(label="Your question", placeholder="e.g. NVDA 5 or AAPL 12h", scale=2)
                                model_dropdown = gr.Dropdown(
                                    choices=["Linear Regression", "Random Forest", "SVR", "Ensemble"],
                                    value="Linear Regression",
                                    label="Model",
                                    info="Choose the prediction model or 'Ensemble' for average & comparison",
                                    scale=1
                                )
                            with gr.Row():
                                output_box = gr.HTML(label="Prediction Result")
                            with gr.Row():
                                plot_box = gr.Plot(label="Price Chart", scale=2)
                            with gr.Row():
                                acc_box = gr.Textbox(label="Model Accuracy", scale=1, interactive=False)
                            submit_btn = gr.Button("Predict & Show Chart", elem_id="predict-btn")
                            submit_btn.click(fn=gradio_predict, inputs=[chatbot_box, model_dropdown], outputs=[output_box, plot_box, acc_box])
                        gr.Markdown(
                            '''<div class="footer">
                            <b>StockBot</b> &copy; 2025 &mdash; Powered by yfinance, scikit-learn, and Plotly.<br>
                            <span style="font-size:0.95em;">This is a demo and not financial advice.</span>
                            </div>'''
                        )
                    with gr.Tab("Real-Time Data"):
                        with gr.Column(elem_classes="main-card"):
                            gr.Markdown(
                                '''<div class="header-title">⏱️ Real-Time Stock Data</div>
                                <div class="header-desc">Enter a ticker to see a live-updating chart and latest price.</div>'''
                            )
                            realtime_ticker = gr.Textbox(label="Ticker", value="AAPL", scale=2)
                            realtime_plot = gr.Plot(label="Real-Time Chart")
                            realtime_status = gr.Textbox(label="Latest Price", interactive=False)
                            realtime_btn = gr.Button("Refresh Data")
                            def update_realtime(symbol):
                                return realtime_tab(symbol)
                            realtime_btn.click(fn=update_realtime, inputs=[realtime_ticker], outputs=[realtime_plot, realtime_status])
                    with gr.Tab("Watchlist"):
                        with gr.Column(elem_classes="main-card"):
                            gr.Markdown('''<div class="header-title">⭐ Watchlist</div><div class="header-desc">Sign up and log in to save your watchlist to your account.</div>''')
                            wl_msg = gr.Markdown(visible=False)
                            watchlist_input = gr.Textbox(label="Add Ticker", placeholder="e.g. MSFT")
                            add_btn = gr.Button("Add to Watchlist")
                            remove_input = gr.Textbox(label="Remove Ticker", placeholder="e.g. TSLA")
                            remove_btn = gr.Button("Remove from Watchlist")
                            watchlist_list = gr.List(label="Watchlist", interactive=False)
                            def add_to_watchlist(ticker, user, state):
                                if not user:
                                    return state, watchlist_tab(state), gr.update(visible=True, value="Please log in to use your watchlist.")
                                ticker = ticker.strip().upper()
                                if ticker and ticker not in state:
                                    state.append(ticker)
                                    set_watchlist(user, state)
                                return state, watchlist_tab(state), gr.update(visible=False)
                            def remove_from_watchlist(ticker, user, state):
                                if not user:
                                    return state, watchlist_tab(state), gr.update(visible=True, value="Please log in to use your watchlist.")
                                ticker = ticker.strip().upper()
                                if ticker in state:
                                    state.remove(ticker)
                                    set_watchlist(user, state)
                                return state, watchlist_tab(state), gr.update(visible=False)
                            def update_watchlist_on_login(user):
                                if user:
                                    wl = get_watchlist(user)
                                    return wl, watchlist_tab(wl), gr.update(visible=False)
                                else:
                                    return [], [], gr.update(visible=True, value="Please log in to use your watchlist.")
                            add_btn.click(add_to_watchlist, [watchlist_input, user_state, watchlist_state], [watchlist_state, watchlist_list, wl_msg])
                            remove_btn.click(remove_from_watchlist, [remove_input, user_state, watchlist_state], [watchlist_state, watchlist_list, wl_msg])
                            user_state.change(update_watchlist_on_login, [user_state], [watchlist_state, watchlist_list, wl_msg])
                            watchlist_list.value = []
                    with gr.Tab("Accuracy Log"):
                        with gr.Column(elem_classes="main-card"):
                            gr.Markdown(
                                '''<div class="header-title">📊 Model Accuracy Log</div>
                                <div class="header-desc">Recent prediction accuracy for all models and tickers.</div>'''
                            )
                            accuracy_table = gr.Dataframe(value=show_accuracy_log, headers=None, label="Recent Accuracy (last 20)", interactive=False, wrap=True)
                            accuracy_table = gr.Dataframe(value=show_accuracy_log, headers=["Timestamp", "Symbol", "Model", "Horizon", "Price", "R2", "MAE"], label="Recent Accuracy (last 20)", interactive=False, wrap=True)
                            refresh_btn.click(fn=show_accuracy_log, inputs=[], outputs=[accuracy_table])
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
