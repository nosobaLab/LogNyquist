# LogNyquist Plot trial 03
# under construction

import sys
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math

MAX_FILES = 3


def read_bode_file(path):
    freqs = []
    logrs = []
    phases = []
    mags_db = []
    phases_deg = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Freq."):
                continue

            # Split into frequency and value parts (whitespace-separated)
            parts = line.split()
            freq = float(parts[0])

            # Example: (4.0037e+01dB,-8.2480e+01ｰ)
            raw = parts[1]

            # Strip the leading "(" and trailing ")"
            raw = raw.strip("()")

            mag_str, phase_str = raw.split(",")

            # Remove "dB" and "ｰ"
            mag_db = float(mag_str.replace("dB", ""))
            phase_deg = float(phase_str.replace("ｰ", ""))

            # dB -> log value
            logr = math.log10(10**(mag_db / 20.0)+1)
            # degree -> radian
            phase_rad = np.deg2rad(phase_deg)

            freqs.append(freq)
            logrs.append(logr)
            phases.append(phase_rad)
            mags_db.append(mag_db)
            phases_deg.append(phase_deg)

    return (
        np.array(freqs),
        np.array(logrs),
        np.array(phases),
        np.array(mags_db),
        np.array(phases_deg),
    )


def interpolate_curve(x, y, freqs, mags_db, phases_deg, n_sub=50):
    """
    Interpolate between adjacent data points by splitting each interval into n_sub steps.
    x, y are linearly interpolated within each interval (to match the plotted polyline).
    freq is interpolated as a geometric sequence (since LTspice AC analysis sweeps
    frequency logarithmically, linear on a log axis means a geometric sequence).
    mag(dB) and phase(deg) are linearly interpolated.
    """
    x_fine = []
    y_fine = []
    freq_fine = []
    mag_db_fine = []
    phase_deg_fine = []

    t = np.linspace(0.0, 1.0, n_sub, endpoint=False)

    for i in range(len(x) - 1):
        x_fine.append(x[i] + (x[i + 1] - x[i]) * t)
        y_fine.append(y[i] + (y[i + 1] - y[i]) * t)
        freq_fine.append(freqs[i] * (freqs[i + 1] / freqs[i]) ** t)
        mag_db_fine.append(mags_db[i] + (mags_db[i + 1] - mags_db[i]) * t)
        phase_deg_fine.append(phases_deg[i] + (phases_deg[i + 1] - phases_deg[i]) * t)

    # Append the last point
    x_fine.append([x[-1]])
    y_fine.append([y[-1]])
    freq_fine.append([freqs[-1]])
    mag_db_fine.append([mags_db[-1]])
    phase_deg_fine.append([phases_deg[-1]])

    return (
        np.concatenate(x_fine),
        np.concatenate(y_fine),
        np.concatenate(freq_fine),
        np.concatenate(mag_db_fine),
        np.concatenate(phase_deg_fine),
    )


def get_input_paths():
    """Use file names given as command-line arguments if present (up to MAX_FILES),
    otherwise prompt via standard input (1 to MAX_FILES, blank Enter to stop)."""
    if len(sys.argv) > 1:
        return sys.argv[1:MAX_FILES + 1]

    paths = []
    for i in range(MAX_FILES):
        prompt = (
            f"Enter the Nyquist data file name ({i + 1}/{MAX_FILES}"
            + (", blank Enter to finish" if i > 0 else "")
            + "): "
        )
        path = input(prompt).strip()
        if not path:
            break
        paths.append(path)

    if not paths:
        print("No file specified. Exiting.")
        sys.exit(1)

    return paths


# -----------------------------
# Data loading: results of an LTspice stability AC analysis with a source at the input
# -----------------------------
input_paths = get_input_paths()

COLORS = ['tab:blue', 'tab:orange', 'tab:green']

curves = []
for i, path in enumerate(input_paths):
    freqs, logrs, phases, mags_db, phases_deg = read_bode_file(path)

    x = logrs * np.cos(phases)
    y = logrs * np.sin(phases)

    # Finely interpolate the curve for mouse-proximity detection and value display
    # (at low frequencies the data spacing is wide, so without this the distance
    #  to the nearest data point can exceed the threshold even when the mouse is
    #  close to the curve, causing the display to disappear)
    x_fine, y_fine, freq_fine, mag_db_fine, phase_deg_fine = interpolate_curve(
        x, y, freqs, mags_db, phases_deg
    )
    abs_fine = 10 ** (mag_db_fine / 20.0)

    curves.append({
        "path": path,
        "label": os.path.basename(path),
        "color": COLORS[i % len(COLORS)],
        "x": x, "y": y,
        "x_fine": x_fine, "y_fine": y_fine,
        "freq_fine": freq_fine,
        "mag_db_fine": mag_db_fine,
        "phase_deg_fine": phase_deg_fine,
        "abs_fine": abs_fine,
    })

# -----------------------------
# Value display (limit digit count based on significant figures)
# -----------------------------
def format_sig(value, sig=3, max_decimals=3):
    """
    Round value to sig significant figures and format it as a string.
    Never exceed max_decimals decimal places. Strip trailing zeros.
    """
    if value == 0:
        return "0"

    order = math.floor(math.log10(abs(value)))
    decimals = sig - order - 1
    decimals = max(0, min(max_decimals, decimals))

    s = f"{value:.{decimals}f}"
    if decimals > 0:
        s = s.rstrip('0').rstrip('.')
    return s


def human_readable_value(n, unit=""):
    """Format a value as a string with k/M/G suffixes, honoring significant figures."""
    if n >= 1_000_000_000:
        return format_sig(n / 1_000_000_000) + "G" + unit
    elif n >= 1_000_000:
        return format_sig(n / 1_000_000) + "M" + unit
    elif n >= 1_000:
        return format_sig(n / 1_000) + "k" + unit
    else:
        return format_sig(n) + unit


def human_readable_freq(n):
    return human_readable_value(n, "Hz")


# -----------------------------
# Plot setup
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal', adjustable='box')

# Hide axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Real and imaginary axes
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# Draw the grid circles
theta = np.linspace(0, 2*np.pi, 400)
grid_powers = []  # exponents p of the drawn grid circles (magnitude 10^p)


def draw_grid_circle(power):
    A = 10 ** power
    radius = math.log10(A + 1)
    ax.plot(radius*np.cos(theta), radius*np.sin(theta), color='gray', linestyle='--')
    # clip_on=True: automatically hide labels that fall outside the view
    ax.text(radius, 0, human_readable_value(A), fontsize=10, va='bottom', ha='left', clip_on=True)
    grid_powers.append(power)


def extend_grid_circles(view_extent):
    """When the view range expands, draw additional grid circles up to the needed decade."""
    while True:
        next_power = max(grid_powers) + 1
        radius = math.log10(10 ** next_power + 1)
        if radius > view_extent:
            break
        draw_grid_circle(next_power)


for p in range(6):  # 1, 10, 100, 1k, 10k, 100k
    draw_grid_circle(p)

# Draw the curves (color-coded per file, with file names shown in the legend)
NORMAL_LINEWIDTH = 1.5
SELECTED_LINEWIDTH = 3.0

for c in curves:
    (line,) = ax.plot(
        c["x"], c["y"],
        color=c["color"],
        linewidth=NORMAL_LINEWIDTH,
        label=c["label"],
        zorder=2,
    )
    c["line"] = line

ax.legend(loc='upper left', fontsize=9)

# Marker showing the point closest to the mouse (same color as the selected curve)
(marker,) = ax.plot(
    [], [], 'o',
    color=curves[0]["color"],
    markersize=8,
    markeredgecolor='none',
    zorder=5,
)

# -----------------------------
# Fixed-position text (top right)
# -----------------------------
info_text = ax.text(
    0.95, 0.95, "", transform=ax.transAxes,
    ha='right', va='top', fontsize=12,
    bbox=dict(facecolor='white', alpha=0.7)
)

# Currently highlighted curve (starts at None to distinguish from the unselected state)
selected_curve = None


def set_selected_curve(curve):
    """Make the selected curve bold and bring it to the front; reset the others to normal."""
    global selected_curve
    if curve is selected_curve:
        return
    for c in curves:
        is_selected = (c is curve)
        c["line"].set_linewidth(SELECTED_LINEWIDTH if is_selected else NORMAL_LINEWIDTH)
        c["line"].set_zorder(4 if is_selected else 2)
    selected_curve = curve


# -----------------------------
# Mouse event handling
# -----------------------------
def on_move(event):
    if not event.inaxes:
        info_text.set_text("")
        marker.set_data([], [])
        set_selected_curve(None)
        fig.canvas.draw_idle()
        return

    # Mouse position
    mx, my = event.xdata, event.ydata

    # Compute the distance to each curve's interpolated data and pick the closest curve/point
    best_curve = None
    best_idx = None
    best_dist = None
    for c in curves:
        dist = np.hypot(c["x_fine"] - mx, c["y_fine"] - my)
        idx = np.argmin(dist)
        if best_dist is None or dist[idx] < best_dist:
            best_dist = dist[idx]
            best_idx = idx
            best_curve = c

    # Only show when close enough (the threshold is adjustable)
    if best_dist is not None and best_dist < 0.2:
        freq_str = human_readable_freq(best_curve["freq_fine"][best_idx])
        abs_str = human_readable_value(best_curve["abs_fine"][best_idx])
        mag_db_str = format_sig(best_curve["mag_db_fine"][best_idx])
        phase_str = format_sig(best_curve["phase_deg_fine"][best_idx])
        info_text.set_text(
            f"{best_curve['label']}\n"
            f"{freq_str}\n{abs_str} ({mag_db_str} dB)\n{phase_str}°"
        )
        marker.set_data([best_curve["x_fine"][best_idx]], [best_curve["y_fine"][best_idx]])
        marker.set_color(best_curve["color"])
        set_selected_curve(best_curve)
    else:
        info_text.set_text("")
        marker.set_data([], [])
        set_selected_curve(None)

    fig.canvas.draw_idle()


# -----------------------------
# Zoom in/out with the mouse wheel
# -----------------------------
def on_scroll(event):
    if not event.inaxes:
        return

    base_scale = 1.2
    scale_factor = 1 / base_scale if event.button == 'up' else base_scale

    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    # Always zoom centered on the origin, regardless of mouse position
    ax.set_xlim(-new_width / 2, new_width / 2)
    ax.set_ylim(-new_height / 2, new_height / 2)

    # If the view range expanded, add grid circles up to the needed decade
    view_extent = math.hypot(new_width / 2, new_height / 2)
    extend_grid_circles(view_extent)

    fig.canvas.draw_idle()


# -----------------------------
# PNG export via the P key
# -----------------------------
def make_png_filename():
    names = [os.path.splitext(c["label"])[0] for c in curves]
    base = "_".join(names)
    if len(base) > 80:
        base = base[:80]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}.png"


def on_key(event):
    if event.key in ('p', 'P'):
        filename = make_png_filename()
        fig.savefig(filename, dpi=150)
        print(f"Plot saved as image: {filename}")


# Register event handlers
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.title("LogNyquist Plot")
plt.show()
