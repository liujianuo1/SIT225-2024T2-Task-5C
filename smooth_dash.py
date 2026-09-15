"""
SIT225 - Data Capture Technologies
Credit Task (5C): Live Smooth Plotly Dash Update for Smartphone Accelerometer Data

Student: JIANUO LIU  |  ID: 225160181

Smooth update mechanism:
  - Data is received at 10 Hz into a sliding deque (200 samples)
  - dcc.Interval fires every 100 ms
  - Each interval tick emits exactly 1 new point via extendData (NOT a full redraw)
  - extendData appends to existing Plotly traces in-place → zero flicker / smooth scroll
"""

import time
import threading
from collections import deque
from datetime import datetime

import numpy as np
import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────
BUFFER_SIZE     = 200    # visible rolling window (points)
UPDATE_MS       = 100    # interval between Dash callbacks (ms)  →  ~10 fps
POINTS_PER_TICK = 1      # emit exactly 1 point per tick for smooth scroll

# ── Shared buffers (written by background thread, read by Dash callback) ──────
buf_time = deque(maxlen=BUFFER_SIZE)
buf_x    = deque(maxlen=BUFFER_SIZE)
buf_y    = deque(maxlen=BUFFER_SIZE)
buf_z    = deque(maxlen=BUFFER_SIZE)

# Cumulative counter — len(deque) caps at BUFFER_SIZE, so we need this separately
_total   = {"n": 0}
# How many points the Dash frontend has already received
_sent    = {"n": 0}


# ── Data source: Arduino IoT Cloud (with simulation fallback) ─────────────────

def _start_cloud():
    try:
        from arduino.iot.cloud import ArduinoCloudClient
        # Replace with your actual credentials
        client = ArduinoCloudClient(device_id="YOUR_DEVICE_ID", secret_key="YOUR_SECRET_KEY")
        def _cb_x(c, v): buf_x.append(v)
        def _cb_y(c, v): buf_y.append(v)
        def _cb_z(c, v):
            buf_z.append(v)
            buf_time.append(datetime.now().strftime("%H:%M:%S.%f")[:-3])
            _total["n"] += 1
        client.register("accelerometer_x", value=None, on_write=_cb_x)
        client.register("accelerometer_y", value=None, on_write=_cb_y)
        client.register("accelerometer_z", value=None, on_write=_cb_z)
        client.start()
    except Exception as e:
        print(f"[Cloud] {e} → using simulated data")
        _simulate()


def _simulate():
    """10 Hz sine-wave simulation — produces 1 sample every 100 ms."""
    t = 0.0
    while True:
        t += 0.1
        buf_x.append(round(np.sin(t)       + np.random.normal(0, 0.03), 4))
        buf_y.append(round(np.cos(t)       + np.random.normal(0, 0.03), 4))
        buf_z.append(round(np.sin(t * 0.5) + np.random.normal(0, 0.03), 4))
        buf_time.append(datetime.now().strftime("%H:%M:%S.%f")[:-3])
        _total["n"] += 1
        time.sleep(0.1)


# ── Wrapper function (Q2) ─────────────────────────────────────────────────────

def smooth_live_dash_update(time_buffer, y_buffers, sent_ref, total_ref,
                            points_per_tick=POINTS_PER_TICK, max_points=BUFFER_SIZE):
    """
    Wrapper for smooth, flicker-free real-time Plotly Dash graph updates.

    Instead of rebuilding the Figure each interval (which causes a full SVG
    redraw and visible flicker), this function returns a Plotly ``extendData``
    3-tuple that **appends** only the newest data points to the existing traces.
    Sending exactly ``points_per_tick`` samples per call (default: 1) at 100 ms
    produces ~10 fps smooth scrolling — analogous to a movie frame stream.

    Parameters
    ----------
    time_buffer : deque   Shared sliding deque of timestamp strings.
    y_buffers   : dict    {trace_label: deque}  e.g. {"X": buf_x, ...}
    sent_ref    : dict    {"n": int}  mutable counter of points already sent.
    total_ref   : dict    {"n": int}  cumulative sample counter (never capped).
    points_per_tick : int  Max new points to send per callback (default 1).
    max_points  : int      Rolling window size shown on graph (default 200).

    Returns
    -------
    dash.no_update   — when no new data has arrived (skips the Dash update).
    tuple            — (data_dict, trace_indices, max_points) ready for
                       the extendData Output property.

    Usage
    -----
    @app.callback(Output("graph", "extendData"), Input("interval", "n_intervals"))
    def update(_):
        return smooth_live_dash_update(buf_time, {"X": buf_x, "Y": buf_y},
                                       _sent, _total)
    """
    n_new = total_ref["n"] - sent_ref["n"]
    if n_new <= 0:
        return dash.no_update          # ← correct: single no_update value

    n_emit     = min(n_new, points_per_tick)
    times_snap = list(time_buffer)
    new_times  = times_snap[-n_emit:]

    xs, ys = [], []
    for buf in y_buffers.values():
        snap = list(buf)
        xs.append(new_times)
        ys.append(snap[-n_emit:])

    sent_ref["n"] += n_emit
    trace_indices  = list(range(len(y_buffers)))

    # extendData 3-tuple: (data_dict, trace_indices, max_points_to_keep)
    return ({"x": xs, "y": ys}, trace_indices, max_points)


# ── Dash layout ───────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="SIT225 – Live Accelerometer Dashboard")

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif",
           "backgroundColor": "#1a1a2e", "minHeight": "100vh", "padding": "20px"},
    children=[
        html.H2("Live Smartphone Accelerometer",
                style={"color": "#e0e0e0", "textAlign": "center"}),
        html.P("Real-time X / Y / Z acceleration streamed from Arduino IoT Cloud",
               style={"color": "#a0a0b0", "textAlign": "center"}),

        dcc.Graph(
            id="live-graph",
            figure={
                "data": [
                    go.Scatter(name="X-axis", x=[], y=[], mode="lines",
                               line=dict(color="#ff6b6b", width=1.5)),
                    go.Scatter(name="Y-axis", x=[], y=[], mode="lines",
                               line=dict(color="#4ecdc4", width=1.5)),
                    go.Scatter(name="Z-axis", x=[], y=[], mode="lines",
                               line=dict(color="#ffe66d", width=1.5)),
                ],
                "layout": go.Layout(
                    paper_bgcolor="#16213e", plot_bgcolor="#0f3460",
                    font=dict(color="#e0e0e0"),
                    xaxis=dict(title="Time", showgrid=True, gridcolor="#2a3f6f",
                               tickangle=-45, nticks=15),
                    yaxis=dict(title="Acceleration (g)", showgrid=True,
                               gridcolor="#2a3f6f", range=[-1.5, 1.5]),
                    margin=dict(l=70, r=20, t=30, b=80),
                    legend=dict(bgcolor="#16213e"),
                    # Disable Plotly's own layout animation to avoid
                    # competing with our smooth extendData scroll
                    transition={"duration": 0},
                    uirevision="constant",   # prevents axis reset on update
                ),
            },
            style={"height": "480px"},
        ),

        dcc.Interval(id="interval", interval=UPDATE_MS, n_intervals=0),

        html.Div(id="status",
                 style={"color": "#a0a0b0", "textAlign": "center", "marginTop": "8px",
                        "fontSize": "13px"}),
    ],
)


# ── Callback ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("live-graph", "extendData"),
    Output("status", "children"),
    Input("interval", "n_intervals"),
)
def update_graph(_n):
    result = smooth_live_dash_update(
        time_buffer=buf_time,
        y_buffers={"X-axis": buf_x, "Y-axis": buf_y, "Z-axis": buf_z},
        sent_ref=_sent,
        total_ref=_total,
    )
    total   = _total["n"]
    visible = min(total, BUFFER_SIZE)
    status  = (f"⏱ Polling every {UPDATE_MS} ms  |  "
               f"Buffer: {visible}/{BUFFER_SIZE} samples  |  "
               f"Total received: {total}  |  "
               f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    return result, status


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=_start_cloud, daemon=True).start()
    time.sleep(1)   # let simulation pre-fill a few points
    print("=" * 60)
    print("  SIT225 5C – Smooth Live Dashboard")
    print("  Open: http://127.0.0.1:8050")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=8050)
