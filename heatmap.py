"""
heatmap.py
==========
Render heatmaps for the RELEASE sample set (on/off pairs from the release
generator), for BOTH appliances (EV and pool pump). Reuses the original
small-font styling -- good for individual on-screen viewing, NOT the enlarged
paper-figure fonts.

Expected layout (place this script INSIDE the release_samples folder, or set
ROOT below to point at it):

  release_samples/
    diffusion/
      ev/
        appliance_off/   00.npy ... 19.npy
        appliance_on/    00.npy ... 19.npy
        temperature/     00.npy ... 19.npy
      pp/
        appliance_off/   00.npy ... 19.npy
        appliance_on/    00.npy ... 19.npy
        temperature/     00.npy ... 19.npy
    vae/   (same structure)
    gan/   (same structure)
    ev_subattribute_conditions.csv     (optional; used for EV titles if present)
    heatmap.py                          <-- this file

Run:
    python heatmap.py

Output (created next to the model folders):
  vis_heatmaps/
    diffusion/
      ev/  00.png ... 19.png            (3-panel: OFF | ON | Temperature)
      pp/  00.png ... 19.png
    vae/  (same)
    gan/  (same)

NOTE ON UNITS: the release .npy files are already DENORMALIZED (real units),
so this script plots them as-is and applies no normalization stats.
"""

import os, csv, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ─────────────────────────── CONFIG ──────────────────────────────────────────
ROOT      = '.'                       # folder containing diffusion/ vae/ gan/
OUT_DIR   = 'vis_heatmaps'
MODELS    = ['diffusion', 'vae', 'gan']
APPS      = ['ev', 'pp']              # appliance subfolders within each model
CSV_NAME  = 'ev_subattribute_conditions.csv'

# Panel titles per appliance
ON_TITLE  = {'ev': 'EV = ON',  'pp': 'Pool pump = ON'}
OFF_TITLE = {'ev': 'EV = OFF', 'pp': 'Pool pump = OFF'}

# Fixed temperature display range (matches plot_experiments.py)
TEMP_VMIN = 20.0
TEMP_VMAX = 120.0

CMAP_ENERGY = 'jet'        # energy panels (OFF / ON)
CMAP_TEMP   = 'coolwarm'   # temperature panel

DPI = 300
# ─────────────────────────────────────────────────────────────────────────────


# ── helpers (original small-font styling) ─────────────────────────────────────

def squeeze(arr):
    """(1,H,W)->(H,W); leave (H,W) untouched."""
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def colorbar(ax, im, fontsize=8):
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='4%', pad=0.07)
    cb  = ax.get_figure().colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=fontsize - 1)
    return cb


def imshow_energy(ax, data, title, vmin=None, vmax=None, first_col=True):
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(data, 99.5))
    im = ax.imshow(data, aspect='auto', cmap=CMAP_ENERGY,
                   vmin=vmin, vmax=vmax, origin='upper', interpolation='nearest')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
    ax.set_xlabel('Time-of-day (0–95)', fontsize=8)
    if first_col:
        ax.set_ylabel('Day (0–359)', fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def imshow_temp(ax, data, title, first_col=False):
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect='auto', cmap=CMAP_TEMP,
                   vmin=TEMP_VMIN, vmax=TEMP_VMAX,
                   origin='upper', interpolation='nearest')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
    ax.set_xlabel('Time-of-day (0–95)', fontsize=8)
    if first_col:
        ax.set_ylabel('Day (0–359)', fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def load_subattr_labels(model):
    """Return {sample_id: 'amp=high, count=low, ...'} for EV on-panel titles.
    Reads ev_subattribute_conditions.csv if present; otherwise returns {}."""
    path = os.path.join(ROOT, CSV_NAME)
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('model', model) != model:
                continue
            sid = row.get('sample_id', '')
            parts = []
            for k in ['amp', 'count', 'avgdur', 'ctype', 'tou']:
                v = row.get(k)
                if v and v != 'NULL':
                    parts.append(f'{k}={v}')
            out[sid] = ', '.join(parts) if parts else 'NULL sub-attrs'
    return out


# ── main render ───────────────────────────────────────────────────────────────

def render_app(model, app):
    """Render all samples for one model / one appliance (ev or pp)."""
    adir = os.path.join(ROOT, model, app)
    off_dir  = os.path.join(adir, 'appliance_off')
    on_dir   = os.path.join(adir, 'appliance_on')
    temp_dir = os.path.join(adir, 'temperature')
    if not (os.path.isdir(off_dir) and os.path.isdir(on_dir) and os.path.isdir(temp_dir)):
        print(f'  [skip] {model}/{app}: off/on/temperature subfolders not found')
        return 0

    dst = os.path.join(ROOT, OUT_DIR, model, app)
    os.makedirs(dst, exist_ok=True)

    # sub-attribute titles only apply to EV
    labels = load_subattr_labels(model) if app == 'ev' else {}

    ids = sorted(os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(on_dir, '*.npy')))
    if not ids:
        print(f'  [skip] {model}/{app}: no .npy files in appliance_on/')
        return 0
    print(f'  [{model}/{app}] {len(ids)} samples')

    for sid in ids:
        try:
            off  = squeeze(np.load(os.path.join(off_dir,  f'{sid}.npy')))
            on   = squeeze(np.load(os.path.join(on_dir,   f'{sid}.npy')))
            temp = squeeze(np.load(os.path.join(temp_dir, f'{sid}.npy')))
        except FileNotFoundError:
            print(f'    [warn] {model}/{app}/{sid}: missing a panel, skipping')
            continue

        # shared colour scale across OFF and ON so the toggle is comparable
        vmax = float(np.nanpercentile(np.concatenate([off.ravel(), on.ravel()]), 99.5))
        vmax = max(vmax, 1e-6)

        sub = labels.get(sid, '')
        on_title = ON_TITLE[app] + (f'\n({sub})' if sub else '')

        fig, axes = plt.subplots(1, 3, figsize=(12, 5),
                                 gridspec_kw={'wspace': 0.28})
        im0 = imshow_energy(axes[0], off, OFF_TITLE[app], vmin=0.0, vmax=vmax, first_col=True)
        colorbar(axes[0], im0)
        im1 = imshow_energy(axes[1], on, on_title, vmin=0.0, vmax=vmax, first_col=False)
        colorbar(axes[1], im1)
        im2 = imshow_temp(axes[2], temp, 'Temperature', first_col=False)
        colorbar(axes[2], im2)

        fig.suptitle(f'{model}  —  {app.upper()}  —  sample {sid}',
                     fontsize=11, fontweight='bold')
        fig.savefig(os.path.join(dst, f'{sid}.png'),
                    dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    print(f'    -> wrote {len(ids)} figures to {dst}')
    return len(ids)


def main():
    total = 0
    for model in MODELS:
        if not os.path.isdir(os.path.join(ROOT, model)):
            continue
        print(f'\n[{model}]')
        for app in APPS:
            total += render_app(model, app)
    if total == 0:
        print(f'\nNo samples found. Expected '
              f'{os.path.abspath(ROOT)}/<model>/<ev|pp>/appliance_(on|off)/*.npy\n'
              f'If your generator only produced EV so far, the pp/ folders will be empty.')
    else:
        print(f'\nDone. {total} heatmaps under {os.path.join(ROOT, OUT_DIR)}/')


if __name__ == '__main__':
    main()