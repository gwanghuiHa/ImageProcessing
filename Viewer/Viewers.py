"""
Image or ICT viewers

@author: Gwanghui

1. image_viewer_simple(images,x_axis=None,y_axis=None,mode="single",cmap="viridis",title=None)
    images: numpy array with the dimension of (N-shots, X, Y)
    x(y)_axis: x- and y-axis array. If None, just use pixel numbers
    mode: single/all/overlap/average
2. ict_viewer(ict_wfm, ns_per_div=200, mode="single", title=None)
    ict_wfm: list including ict waveforms for each shot
    ns_per_div: scope scale
    mode: single/all/overlap/average
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def image_viewer_simple(
    images,
    x_axis=None,
    y_axis=None,
    mode="single",      # 'single', 'all', 'overlap', 'average'
    cmap="viridis",
    title=None,
    max_per_fig=24,     # only used in 'all'
):
    """
    Matplotlib beam image viewer (no colorbar, fixed clean layout).

    Parameters
    ----------
    images : array
        (X, Y) or (Nshot, X, Y)
    x_axis, y_axis : 1D arrays or None
        Optional physical axes (used as extent); if None, pixels.
    mode : {'single','all','overlap','average'}
    cmap : str
    title : str or None
    max_per_fig : int
        For 'all' mode, max number of shots per figure.

    Returns
    -------
    mode in {'single','average','overlap'} : (fig, ax)
    mode == 'all' : list of (fig, axes_array)
    """

    # --------------------------------------------------------
    # normalize input
    # --------------------------------------------------------
    arr = np.asarray(images)
    if arr.ndim == 2:
        arr = arr[None, ...]   # -> (1, X, Y)
    elif arr.ndim != 3:
        raise ValueError("images must be (X,Y) or (Nshot,X,Y)")

    N, H, W = arr.shape
    mode = mode.lower()
    if mode not in ("single", "all", "overlap", "average"):
        raise ValueError("mode must be one of 'single','all','overlap','average'")

    # extent if axes provided
    use_extent = False
    extent = None
    if x_axis is not None and y_axis is not None:
        x = np.asarray(x_axis)
        y = np.asarray(y_axis)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x_axis and y_axis must be 1D arrays")
        if len(x) != W or len(y) != H:
            raise ValueError("x_axis length must match width, y_axis length must match height")
        extent = [x[0], x[-1], y[-1], y[0]]  # origin='upper'
        use_extent = True

    # global vmin/vmax
    data_min = float(np.nanmin(arr))
    data_max = float(np.nanmax(arr))
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        data_min, data_max = 0.0, 1.0

    def _sanitize(vmin, vmax):
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        return vmin, vmax

    # ------------------------------------------------------------------
    # SINGLE / AVERAGE / OVERLAP  (one main axes + 3 slider rows)
    # ------------------------------------------------------------------
    if mode in ("single", "average", "overlap"):
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(10, 7), dpi=100)
        # 4 rows: [image] [shot slider] [vmin] [vmax]
        gs = GridSpec(
            nrows=5,
            ncols=1,
            height_ratios=[10, 3, 0.5, 0.5, 0.5],
            hspace=0.1,
        )

        ax_img = fig.add_subplot(gs[0, 0])
        ax_idx = fig.add_subplot(gs[2, 0])
        ax_vmin = fig.add_subplot(gs[3, 0])
        ax_vmax = fig.add_subplot(gs[4, 0])

        # clean slider axes (no ticks)
        for ax_s in (ax_idx, ax_vmin, ax_vmax):
            ax_s.tick_params(left=False, labelleft=False,
                             bottom=False, labelbottom=False)

        if title is None:
            fig.suptitle(f"Beam image viewer ({mode})")
        else:
            fig.suptitle(title)

        # base image
        if mode == "average":
            base_img = np.nanmean(arr.astype(np.float32), axis=0)
        elif mode == "overlap":
            base_img = np.nansum(arr.astype(np.float32), axis=0)
        else:
            base_img = arr[0].astype(np.float32)

        # global vmin/vmax was computed from arr above
        # but for overlap we should use the summed image range
        if mode == "overlap":
            data_min = float(np.nanmin(base_img))
            data_max = float(np.nanmax(base_img))
            if not np.isfinite(data_min) or not np.isfinite(data_max):
                data_min, data_max = 0.0, 1.0

        im_kwargs = dict(
            cmap=cmap,
            origin="upper",
            vmin=data_min,
            vmax=data_max,
        )
        if use_extent:
            im_kwargs["extent"] = extent

        im = ax_img.imshow(base_img, **im_kwargs)
        main_im_list = [im]

        ax_img.set_xlabel("x (pixel)" if x_axis is None else "x")
        ax_img.set_ylabel("y (pixel)" if y_axis is None else "y")

        # shot slider (only for single & N>1)
        use_idx_slider = (mode == "single" and N > 1)
        if use_idx_slider:
            s_idx = Slider(
                ax=ax_idx,
                label="shot index",
                valmin=0,
                valmax=N - 1,
                valinit=0,
                valstep=1,
            )

            def _update_shot(_val):
                idx = int(s_idx.val)
                img = arr[idx]
                main_im_list[0].set_data(img)
                ax_img.set_title(f"Shot {idx}")
                fig.canvas.draw_idle()

            s_idx.on_changed(_update_shot)
            ax_img.set_title("Shot 0")
        else:
            ax_idx.set_visible(False)
            if mode == "average":
                ax_img.set_title("Average over all shots")
            elif mode == "overlap":
                ax_img.set_title(f"Overlap of {N} shots")

        # vmin / vmax sliders
        s_vmin = Slider(ax=ax_vmin, label="vmin",
                        valmin=data_min, valmax=data_max, valinit=data_min)
        s_vmax = Slider(ax=ax_vmax, label="vmax",
                        valmin=data_min, valmax=data_max, valinit=data_max)

        def _update_clim(_):
            vmin = s_vmin.val
            vmax = s_vmax.val
            vmin, vmax = _sanitize(vmin, vmax)
            for im_obj in main_im_list:
                im_obj.set_clim(vmin=vmin, vmax=vmax)
            fig.canvas.draw_idle()

        s_vmin.on_changed(_update_clim)
        s_vmax.on_changed(_update_clim)

        plt.show()
        return fig, ax_img

    # ------------------------------------------------------------------
    # ALL  (HD-like 16×9, tight grid; sliders in dedicated bottom band)
    # ------------------------------------------------------------------
    if mode == "all":
        from matplotlib.gridspec import GridSpec

        figs = []
        max_per_fig = int(max_per_fig) if max_per_fig > 0 else N
        n_chunks = int(np.ceil(N / max_per_fig))

        for chunk_idx in range(n_chunks):
            start = chunk_idx * max_per_fig
            stop = min((chunk_idx + 1) * max_per_fig, N)
            arr_chunk = arr[start:stop]
            n_local = arr_chunk.shape[0]

            # decide grid (up to 6 columns; rest goes to rows)
            ncols = min(6, n_local)
            nrows = int(np.ceil(n_local / ncols))

            fig = plt.figure(figsize=(16, 9), dpi=100)

            # outer gridspec: images area + slider band
            outer = GridSpec(
                nrows=2,
                ncols=1,
                height_ratios=[10.0, 1.6],
                hspace=0.3,
            )
            gs_imgs = outer[0].subgridspec(nrows, ncols, wspace=0.05, hspace=0.08)
            gs_sliders = outer[1].subgridspec(2, 1, hspace=0.4)

            if title is None:
                fig.suptitle(f"Beam images (shots {start}–{stop-1})")
            else:
                fig.suptitle(f"{title} (shots {start}–{stop-1})")

            ims = []
            axes = np.empty((nrows, ncols), dtype=object)

            im_kwargs = dict(
                cmap=cmap,
                origin="upper",
                vmin=data_min,
                vmax=data_max,
            )
            if use_extent:
                im_kwargs["extent"] = extent

            for k in range(nrows * ncols):
                ax = fig.add_subplot(gs_imgs[k // ncols, k % ncols])
                axes[k // ncols, k % ncols] = ax
                if k < n_local:
                    img = arr_chunk[k]
                    im = ax.imshow(img, **im_kwargs)
                    ims.append(im)
                    ax.set_title(str(start + k), fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis("off")

            # slider axes (no ticks)
            ax_vmin = fig.add_subplot(gs_sliders[0, 0])
            ax_vmax = fig.add_subplot(gs_sliders[1, 0])
            for ax_s in (ax_vmin, ax_vmax):
                ax_s.tick_params(left=False, labelleft=False,
                                 bottom=False, labelbottom=False)

            s_vmin = Slider(ax=ax_vmin, label="vmin",
                            valmin=data_min, valmax=data_max, valinit=data_min)
            s_vmax = Slider(ax=ax_vmax, label="vmax",
                            valmin=data_min, valmax=data_max, valinit=data_max)

            def _update_all(_,
                            ims_local=ims,
                            fig_local=fig,
                            s_vmin_local=s_vmin,
                            s_vmax_local=s_vmax):
                vmin = s_vmin_local.val
                vmax = s_vmax_local.val
                vmin, vmax = _sanitize(vmin, vmax)
                for im_obj in ims_local:
                    im_obj.set_clim(vmin=vmin, vmax=vmax)
                fig_local.canvas.draw_idle()

            s_vmin.on_changed(_update_all)
            s_vmax.on_changed(_update_all)

            plt.show()
            figs.append((fig, axes))

        return figs

from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def image_viewer_profiles_fft(
    images,
    x_axis=None,
    y_axis=None,
    mode="single",      # 'single', 'average', 'overlap'
    cmap="viridis",
    title=None,
):
    """
    Beam image viewer with:
      - threshold (lower-clip: values < thr -> 0)
      - x/y projected profiles
      - 2D FFT of the thresholded image (always visible, auto-updating)

    Parameters
    ----------
    images : array
        (X, Y) or (Nshot, X, Y)
    x_axis, y_axis : 1D arrays or None
        Optional physical axes. If None, pixel indices are used.
    mode : {'single','average','overlap'}
        'single'  : one shot with shot slider if N>1
        'average' : average over all shots
        'overlap' : sum over all shots
    cmap : str
    title : str or None

    Returns
    -------
    fig, ax_img, ax_fft
        Figure, main image axes, FFT axes.
    """
    # --------------- normalize input ---------------
    arr = np.asarray(images)
    if arr.ndim == 2:
        arr = arr[None, ...]  # -> (1, X, Y)
    elif arr.ndim != 3:
        raise ValueError("images must be (X,Y) or (Nshot,X,Y)")

    N, H, W = arr.shape
    mode = mode.lower()
    if mode not in ("single", "average", "overlap"):
        raise ValueError("mode must be one of 'single','average','overlap'")

    # extent if axes provided
    use_extent = False
    extent = None
    if x_axis is not None and y_axis is not None:
        x = np.asarray(x_axis)
        y = np.asarray(y_axis)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x_axis and y_axis must be 1D arrays")
        if len(x) != W or len(y) != H:
            raise ValueError("x_axis length must match width, y_axis length must match height")
        extent = [x[0], x[-1], y[-1], y[0]]  # origin='upper'
        use_extent = True

    # global vmin/vmax from raw array
    data_min = float(np.nanmin(arr))
    data_max = float(np.nanmax(arr))
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min == data_max:
        data_min, data_max = 0.0, 1.0

    def _sanitize(vmin, vmax):
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        return vmin, vmax

    # --------------- figure + layout ---------------
    fig = plt.figure(figsize=(10, 8), dpi=100)
    # rows: [image+profiles] [FFT] [threshold] [shot slider] [vmin] [vmax]
    gs = GridSpec(
        nrows=6,
        ncols=1,
        height_ratios=[10, 5, 0.7, 0.7, 0.5, 0.5],
        hspace=0.15,
    )

    ax_img  = fig.add_subplot(gs[0, 0])
    ax_fft  = fig.add_subplot(gs[1, 0])
    ax_thr  = fig.add_subplot(gs[2, 0])
    ax_idx  = fig.add_subplot(gs[3, 0])
    ax_vmin = fig.add_subplot(gs[4, 0])
    ax_vmax = fig.add_subplot(gs[5, 0])

    # clean slider axes (no ticks)
    for ax_s in (ax_thr, ax_idx, ax_vmin, ax_vmax):
        ax_s.tick_params(left=False, labelleft=False,
                         bottom=False, labelbottom=False)

    if title is None:
        fig.suptitle(f"Beam image viewer (profiles+FFT, {mode})")
    else:
        fig.suptitle(title)

    # base image depending on mode
    arr_f32 = arr.astype(np.float32)
    if mode == "average":
        base_img = np.nanmean(arr_f32, axis=0)
    elif mode == "overlap":
        base_img = np.nansum(arr_f32, axis=0)
    else:
        base_img = arr_f32[0]

    # local vmin/vmax (overlap might have different range)
    if mode == "overlap":
        data_min_local = float(np.nanmin(base_img))
        data_max_local = float(np.nanmax(base_img))
        if not np.isfinite(data_min_local) or not np.isfinite(data_max_local) or data_min_local == data_max_local:
            data_min_local, data_max_local = data_min, data_max
    else:
        data_min_local, data_max_local = data_min, data_max

    # -------- threshold slider (lower-clip) --------
    s_thr = Slider(
        ax=ax_thr,
        label="threshold (lower clip)",
        valmin=data_min_local,
        valmax=data_max_local,
        valinit=data_min_local,   # start with no clipping
    )

    def _apply_threshold(raw, thr=None):
        """
        Lower-clip threshold:
          values < thr -> 0
          values >= thr -> unchanged
        """
        if thr is None:
            thr = s_thr.val
        disp = raw.astype(float).copy()
        # treat NaNs as 0 so they don't mess up FFT
        disp[~np.isfinite(disp)] = 0.0
        mask = disp < thr
        disp[mask] = 0.0
        return disp

    # -------- profiles axes on same canvas --------
    divider = make_axes_locatable(ax_img)
    ax_xprof = divider.append_axes("top", 1.0, pad=0.1, sharex=ax_img)
    ax_yprof = divider.append_axes("right", 1.0, pad=0.1, sharey=ax_img)

    plt.setp(ax_xprof.get_xticklabels(), visible=False)
    plt.setp(ax_yprof.get_yticklabels(), visible=False)

    ax_xprof.set_ylabel("Proj Y")
    ax_yprof.set_xlabel("Proj X")

    # extent for imshow
    im_kwargs = dict(
        cmap=cmap,
        origin="upper",
        vmin=data_min_local,
        vmax=data_max_local,
    )
    if use_extent:
        im_kwargs["extent"] = extent

    # shared state: current raw image (before threshold)
    current_raw = {"img": base_img}

    # initial display with threshold applied (no clipping at start)
    disp0 = _apply_threshold(current_raw["img"], thr=data_min_local)
    im = ax_img.imshow(disp0, **im_kwargs)

    # x, y axes for profiles
    if x_axis is None:
        x_vals = np.arange(W)
    else:
        x_vals = np.asarray(x_axis)

    if y_axis is None:
        y_vals = np.arange(H)
    else:
        y_vals = np.asarray(y_axis)

    # initial profiles
    px0 = np.mean(disp0, axis=0)
    py0 = np.mean(disp0, axis=1)

    line_x, = ax_xprof.plot(x_vals, px0, lw=1.0)
    line_y, = ax_yprof.plot(py0, y_vals, lw=1.0)

    def _update_profiles(disp):
        px = np.mean(disp, axis=0)
        py = np.mean(disp, axis=1)
        line_x.set_ydata(px)
        line_y.set_xdata(py)
        ax_xprof.relim()
        ax_xprof.autoscale_view()
        ax_yprof.relim()
        ax_yprof.autoscale_view()

    ax_img.set_xlabel("x (pixel)" if x_axis is None else "x")
    ax_img.set_ylabel("y (pixel)" if y_axis is None else "y")

    # -------- 2D FFT panel (always visible) --------
    def _update_fft(disp):
        # FFT of thresholded image
        fft = np.fft.fftshift(np.fft.fft2(disp))
        mag = np.log10(np.abs(fft) + 1e-6)
        ax_fft.clear()
        ax_fft.imshow(mag, origin="lower", cmap="viridis")
        ax_fft.set_title("2D FFT log10|F(kx, ky)|")
        ax_fft.set_xlabel("kx")
        ax_fft.set_ylabel("ky")

    _update_fft(disp0)

    # -------- shot slider (single mode) --------
    use_idx_slider = (mode == "single" and N > 1)
    if use_idx_slider:
        s_idx = Slider(
            ax=ax_idx,
            label="shot index",
            valmin=0,
            valmax=N - 1,
            valinit=0,
            valstep=1,
        )

        def _update_shot(_val):
            idx = int(s_idx.val)
            img = arr_f32[idx]
            current_raw["img"] = img
            disp = _apply_threshold(img)
            im.set_data(disp)
            _update_profiles(disp)
            _update_fft(disp)
            ax_img.set_title(f"Shot {idx}")
            fig.canvas.draw_idle()

        s_idx.on_changed(_update_shot)
        ax_img.set_title("Shot 0")
    else:
        ax_idx.set_visible(False)
        if mode == "average":
            ax_img.set_title("Average over all shots")
        elif mode == "overlap":
            ax_img.set_title(f"Overlap of {N} shots")

    # -------- vmin / vmax sliders --------
    s_vmin = Slider(ax=ax_vmin, label="vmin",
                    valmin=data_min_local, valmax=data_max_local, valinit=data_min_local)
    s_vmax = Slider(ax=ax_vmax, label="vmax",
                    valmin=data_min_local, valmax=data_max_local, valinit=data_max_local)

    def _update_clim(_):
        vmin = s_vmin.val
        vmax = s_vmax.val
        vmin, vmax = _sanitize(vmin, vmax)
        im.set_clim(vmin=vmin, vmax=vmax)
        fig.canvas.draw_idle()

    s_vmin.on_changed(_update_clim)
    s_vmax.on_changed(_update_clim)

    # -------- threshold callback (updates everything) --------
    def _update_threshold(_):
        raw = current_raw["img"]
        disp = _apply_threshold(raw)
        im.set_data(disp)
        _update_profiles(disp)
        _update_fft(disp)
        fig.canvas.draw_idle()

    s_thr.on_changed(_update_threshold)

    plt.show()
    return fig, ax_img, ax_fft



#%%
def _make_time_base(n_samples, ns_per_div=None, sec_per_div=None):
    """
    Build time base for ICT waveforms.

    If ns_per_div or sec_per_div is given, assume 10 divisions and
    center at t = 0, i.e., span = 10 * sec_per_div, from -5*span/div to +5*span/div.
    Otherwise, just return index.
    """
    if sec_per_div is None and ns_per_div is not None:
        sec_per_div = ns_per_div * 1e-9

    if sec_per_div is None:
        t = np.arange(n_samples)
        label = "sample index"
        return t, label

    total_span = 10.0 * sec_per_div        # 10 divisions across screen
    t = np.linspace(-0.5 * total_span, 0.5 * total_span, n_samples)
    # use ns on axis
    return t * 1e-9, "time (ns)"

def ict_viewer(
    traces,
    ns_per_div=200,
    sec_per_div=None,
    mode="single",      # 'single', 'all', 'overlap', 'average'
    title=None,
):
    """
    ICT waveform viewer using Matplotlib.

    Parameters
    ----------
    traces : np.ndarray
        Shape (Nshot, Nsample) or (Nsample,).
    ns_per_div, sec_per_div : float or None
        Time scale of the scope. If provided, a 10-division window
        is assumed and time base is centered at t=0.
        If both are None, x-axis is just sample index.
    mode : {'single','all','overlap','average'}
    title : str or None

    Returns
    -------
    For mode in {'single','overlap','average'}:
        fig, ax
    For mode == 'all':
        list of (fig, axes_array)
    """

    # ------------- normalize traces -------------
    arr = np.asarray(traces)
    if arr.ndim == 1:
        arr = arr[None, :]    # (1, Nsample)
    elif arr.ndim != 2:
        raise ValueError("traces must be (Nsample,) or (Nshot, Nsample)")

    Nshot, Nsample = arr.shape
    mode = mode.lower()
    if mode not in ("single", "all", "overlap", "average"):
        raise ValueError("mode must be one of 'single','all','overlap','average'")

    # ------------- time base -------------
    t, xlabel = _make_time_base(Nsample, ns_per_div=ns_per_div, sec_per_div=sec_per_div)

    # ------------- global y-range -------------
    y_min = float(np.nanmin(arr))
    y_max = float(np.nanmax(arr))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = -1.0, 1.0

    # =================================================================
    # SINGLE
    # =================================================================
    if mode == "single":
        # 5-row gridspec: [title (implicit)] [plot] [xlabel spacer] [slider row]
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(10, 6), dpi=100)
        gs = GridSpec(
            nrows=4,
            ncols=1,
            height_ratios=[8.0, 0.8, 0.2, 0.7],  # plot, spacer, (for xlabel space), slider
            hspace=0.15,
        )

        ax = fig.add_subplot(gs[0, 0])
        ax_spacer = fig.add_subplot(gs[1, 0])
        ax_slider = fig.add_subplot(gs[3, 0])

        ax_spacer.axis("off")  # dummy row purely to keep xlabel away from slider
        ax_slider.tick_params(left=False, labelleft=False,
                              bottom=False, labelbottom=False)

        if title is None:
            fig.suptitle("ICT viewer (single)")
        else:
            fig.suptitle(title)

        # initial shot
        line, = ax.plot(t, arr[0], lw=1.0)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("signal (a.u.)")
        ax.set_xlabel(xlabel)
        ax.set_title("Shot 0")

        # shot index slider (if multiple shots)
        if Nshot > 1:
            s_idx = Slider(
                ax=ax_slider,
                label="shot index",
                valmin=0,
                valmax=Nshot - 1,
                valinit=0,
                valstep=1,
            )

            def _update_shot(_):
                idx = int(s_idx.val)
                line.set_ydata(arr[idx])
                ax.set_title(f"Shot {idx}")
                fig.canvas.draw_idle()

            s_idx.on_changed(_update_shot)
        else:
            # no slider needed
            ax_slider.set_visible(False)

        plt.show()
        return fig, ax

    # =================================================================
    # OVERLAP
    # =================================================================
    if mode == "overlap":
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        if title is None:
            fig.suptitle("ICT viewer (sum of shots)")
        else:
            fig.suptitle(title)

        summed = np.nansum(arr, axis=0)       # <-- point-wise sum
        ax.plot(t, summed, lw=1.2)

        y_min_s = float(np.nanmin(summed))
        y_max_s = float(np.nanmax(summed))
        if not np.isfinite(y_min_s) or not np.isfinite(y_max_s):
            y_min_s, y_max_s = -1.0, 1.0

        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(y_min_s, y_max_s)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("sum signal (a.u.)")
        ax.set_title(f"Sum over {Nshot} shots")

        plt.show()
        return fig, ax

    # =================================================================
    # AVERAGE
    # =================================================================
    if mode == "average":
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        if title is None:
            fig.suptitle("ICT viewer (average)")
        else:
            fig.suptitle(title)

        avg = np.nanmean(arr, axis=0)
        ax.plot(t, avg, lw=1.2)

        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("signal (a.u.)")
        ax.set_title("Average over all shots")

        plt.show()
        return fig, ax

    # =================================================================
    # ALL : grid of subplots (no slider)
    # =================================================================
    if mode == "all":
        figs = []

        max_per_fig = 24  # or expose as argument if you like
        max_per_fig = int(max_per_fig) if max_per_fig > 0 else Nshot
        n_chunks = int(np.ceil(Nshot / max_per_fig))

        from matplotlib.gridspec import GridSpec

        for chunk_idx in range(n_chunks):
            start = chunk_idx * max_per_fig
            stop = min((chunk_idx + 1) * max_per_fig, Nshot)
            arr_chunk = arr[start:stop]
            n_local = arr_chunk.shape[0]

            # up to 6 columns, rest into rows
            ncols = min(6, n_local)
            nrows = int(np.ceil(n_local / ncols))

            fig = plt.figure(figsize=(16, 9), dpi=100)
            outer = GridSpec(
                nrows=2,
                ncols=1,
                height_ratios=[9.0, 1.0],   # plots + bottom band (for xlabel space)
                hspace=0.25,
            )
            gs_plots = outer[0].subgridspec(nrows, ncols, wspace=0.15, hspace=0.25)
            ax_bottom = fig.add_subplot(outer[1, 0])  # dummy for shared xlabel

            ax_bottom.axis("off")

            if title is None:
                fig.suptitle(f"ICT viewer (shots {start}–{stop-1})")
            else:
                fig.suptitle(f"{title} (shots {start}–{stop-1})")

            axes = np.empty((nrows, ncols), dtype=object)

            for k in range(nrows * ncols):
                ax = fig.add_subplot(gs_plots[k // ncols, k % ncols])
                axes[k // ncols, k % ncols] = ax

                if k < n_local:
                    idx = start + k
                    ax.plot(t, arr_chunk[k], lw=0.8)
                    ax.set_xlim(t[0], t[-1])
                    ax.set_ylim(y_min, y_max)
                    ax.set_title(str(idx), fontsize=8)

                    # remove axis labels, keep only light ticks if you want
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    # if you also want NO ticks at all, uncomment:
                    # ax.set_xticks([])
                    # ax.set_yticks([])

                else:
                    ax.axis("off")

            plt.show()
            figs.append((fig, axes))

        return figs
