"""
Collection of script that GH may use for processing

@author: Gwanghui

rotate_one_image(img, angle_deg)
rotate_all_image(img, angle_deg)
    img: original image array
    angle_deg: rotation angle in deg


"""
#%% rotations
from scipy.ndimage import rotate
import numpy as np

def rotate_one_image(img, angle_deg):
    """
    Rotate an image without cropping (expands output canvas to fit entire result).
    Positive angle = counter-clockwise.
    """
    rotated = rotate(img, angle_deg, reshape=True, mode='constant', cval=0)
    return rotated
def rotate_all_image(imgs, angle_deg):
    """
    Rotate an image without cropping (expands output canvas to fit entire result).
    Positive angle = counter-clockwise.
    imgs needs to be (Nshot, X, Y)
    """
    N_img = imgs.shape[0]
    temp = rotate_one_image(imgs[0],angle_deg)
    out_shape = (N_img,) + temp.shape
    rotated = np.empty(out_shape,dtype=imgs.dtype)
    
    rotated[0] = temp
    for i in range(1,N_img):
        rotated[i] = rotate_one_image(imgs[i],angle_deg)
    
    return rotated
#%% Calibration
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector
from matplotlib.patches import Ellipse
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from skimage import filters, exposure, feature, transform, measure, morphology, util

def get_ellipse_from_image(
    image: np.ndarray,
    line_color: str = "orange",
    line_style: str = "--",   # ":" dotted, "--" dashed, "-" solid
    line_width: float = 1.5,
    cmap: str = "gray",
    pad_frac: float = 0.12,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Draw/adjust an ellipse. Press ENTER to capture and RETURN its parameters.
    The figure stays open; you can keep working or close it later.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D image array.
    vmin, vmax : float, optional
        Intensity limits for display (passed to imshow).

    Returns
    -------
    dict
        {"center_x","center_y","width","height","angle_deg"} or None if nothing drawn.
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (H,W) or 3D (H,W,C)")

    # Make sure toolbar isn't in pan/zoom mode
    try:
        tb = plt.get_current_fig_manager().toolbar
        if tb is not None:
            tb.mode = ""
    except Exception:
        pass

    H, W = image.shape[:2]
    pad = float(pad_frac) * max(H, W)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    try:
        fig.canvas.manager.set_window_title("Interactive Ellipse")
    except Exception:
        pass

    # -------- Display image with vmin/vmax --------
    ax.imshow(
        image,
        cmap=(cmap if image.ndim == 2 else None),
        origin="upper",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_aspect("equal")
    ax.set_title("Drag to draw/adjust ellipse. Press ENTER to capture (window stays open).")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_xlim(-pad, W + pad)
    ax.set_ylim(H + pad, -pad)

    selector = EllipseSelector(
        ax,
        onselect=lambda *a, **k: None,
        interactive=True,
        useblit=False,
        button=[1],
        grab_range=10,
        props=dict(edgecolor=line_color, facecolor="none",
                   linestyle=line_style, linewidth=line_width, alpha=1.0),
        handle_props=dict(markeredgecolor=line_color, markerfacecolor="none"),
    )
    selector.set_active(True)

    def _read_current_ellipse():
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception:
            pass

        for attr in ("_selection_artist", "_shape_artist", "to_draw"):
            artist = getattr(selector, attr, None)
            if isinstance(artist, Ellipse):
                cx, cy = artist.center
                w = float(getattr(artist, "width", 0.0))
                h = float(getattr(artist, "height", 0.0))
                ang_deg = float(getattr(artist, "angle", 0.0))
                if w > 1e-6 and h > 1e-6:
                    return float(cx), float(cy), w, h, ang_deg

        try:
            x1, x2, y1, y2 = selector.extents
            w = float(abs(x2 - x1))
            h = float(abs(y2 - y1))
            if w > 1e-6 and h > 1e-6:
                cx = float(0.5 * (x1 + x2))
                cy = float(0.5 * (y1 + y2))
                ang = getattr(selector, "angle", 0.0)
                ang_deg = float(np.degrees(ang))
                return cx, cy, w, h, ang_deg
        except Exception:
            pass
        return None

    result_container = {"value": None}

    def _on_key(event):
        if event.key in ("enter", "return"):
            got = _read_current_ellipse()
            if got is None:
                result_container["value"] = None
            else:
                cx, cy, w, h, ang_deg = got
                result_container["value"] = {
                    "center_x": cx,
                    "center_y": cy,
                    "width": w,
                    "height": h,
                    "angle_deg": ang_deg,
                }
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass

    cid = fig.canvas.mpl_connect("key_press_event", _on_key)

    plt.show(block=False)
    try:
        fig.canvas.start_event_loop(timeout=-1)
    except Exception:
        plt.waitforbuttonpress(timeout=-1)

    try:
        fig.canvas.mpl_disconnect(cid)
    except Exception:
        pass

    return result_container["value"]