"""
Collection of script that GH may use for processing

@author: Gwanghui

"""
from . import session_state
#%% rotations
from scipy.ndimage import rotate
import numpy as np

def rotate_one_image(img, angle_deg=None):
    """
    Rotate an image without cropping (expands output canvas to fit entire result).
    Positive angle = counter-clockwise.
    """
    if angle_deg is None or angle_deg == []:
        angle = session_state.processing_info.get("rotation_angle", None)
        if angle is None:
            raise ValueError("No rotation_angle stored in processing_info and no angle_deg provided.")
    else:
        angle = float(angle_deg)
            
    rotated = rotate(img, angle, reshape=True, mode='constant', cval=0)
    return rotated
def rotate_all_image(imgs, angle_deg=None, overwrite=True):
    """
    Rotate an image without cropping (expands output canvas to fit entire result).
    Positive angle = counter-clockwise.
    imgs needs to be (Nshot, X, Y)
    """
    if angle_deg is None or angle_deg == []:
        angle = session_state.processing_info.get("rotation_angle", None)
        if angle is None:
            raise ValueError("No rotation_angle stored in processing_info and no angle_deg provided.")
    else:
        angle = float(angle_deg)
        if overwrite or session_state.processing_info.get("rotation_angle") is None:
            session_state.processing_info["rotation_angle"] = angle
            
    N_img = imgs.shape[0]
    temp = rotate_one_image(imgs[0],angle)
    out_shape = (N_img,) + temp.shape
    rotated = np.empty(out_shape,dtype=imgs.dtype)
    
    rotated[0] = temp
    for i in range(1,N_img):
        rotated[i] = rotate_one_image(imgs[i],angle)
    
    return rotated
#%% Drawing
from matplotlib import pyplot as plt
from matplotlib.widgets import EllipseSelector
from matplotlib.patches import Ellipse

def ellipse_manual(
    ax,
    line_color="orange",
    line_style="--",
    line_width=1.5,
):
    """
    Interactive ellipse selector using ONE global ellipse entry.

    Behavior:
      - Loads previous ellipse from session_state.processing_info["ellipse_info"]
      - After ENTER: overwrites session_state.processing_info["ellipse_info"]
      - Returns the new ellipse dict (or None if nothing valid)
    """
    if ax is None:
        print("[WARNING] No axes provided.")
        return None

    fig = ax.figure

    # avoid tight_layout warning
    try:
        fig.set_tight_layout(False)
    except Exception:
        pass

    # previous ellipse (if any)
    prev = session_state.processing_info.get("ellipse_info", None)

    # make sure toolbar is not in pan/zoom mode
    try:
        tb = fig.canvas.manager.toolbar
        if tb is not None:
            tb.mode = ""
    except Exception:
        pass

    ax.set_title("Draw/adjust ellipse, then press ENTER to capture.")

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

    # --- preload previous ellipse, if it exists ---
    if prev is not None:
        try:
            cx = float(prev["center_x"])
            cy = float(prev["center_y"])
            w  = float(prev["width"])
            h  = float(prev["height"])
            if w > 1e-6 and h > 1e-6:
                x1 = cx - 0.5 * w
                x2 = cx + 0.5 * w
                y1 = cy - 0.5 * h
                y2 = cy + 0.5 * h
                selector.extents = (x1, x2, y1, y2)
                fig.canvas.draw_idle()
        except Exception as e:
            print(f"[ellipse_manual] Could not apply previous ellipse: {e}")

    # --- helper to read the current ellipse from the selector ---
    def _read_current():
        fig.canvas.draw()
        fig.canvas.flush_events()
        for attr in ("_selection_artist", "_shape_artist", "to_draw"):
            obj = getattr(selector, attr, None)
            if isinstance(obj, Ellipse):
                cx, cy = obj.center
                w = obj.width
                h = obj.height
                if w > 1e-6 and h > 1e-6:
                    return {
                        "center_x": float(cx),
                        "center_y": float(cy),
                        "width": float(w),
                        "height": float(h),
                    }
        return None

    result = {"value": None}

    def _on_key(event):
        if event.key in ("enter", "return"):
            result["value"] = _read_current()
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass

    cid = fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.start_event_loop(timeout=-1)
    fig.canvas.mpl_disconnect(cid)

    info = result["value"]

    # ⬅️ save to your global processing_info
    session_state.processing_info["ellipse_info"] = info

    return info

from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle

def rectangle_manual(
    ax,
    line_color="orange",
    line_style="--",
    line_width=1.5,
):
    """
    Interactive rectangle selector using ONE global rectangle entry.

    Behavior:
      - Loads previous rectangle from session_state.processing_info["rectangle_info"]
      - After ENTER: overwrites session_state.processing_info["rectangle_info"]
      - Returns the new rectangle dict (or None if nothing valid)

    Returned dict (pixels):
        {
          "center_x", "center_y",
          "width", "height",
          "x1", "y1", "x2", "y2"
        }
    """
    if ax is None:
        print("[WARNING] No axes provided.")
        return None

    fig = ax.figure

    # avoid tight_layout warning
    try:
        fig.set_tight_layout(False)
    except Exception:
        pass

    # previous rectangle (if any)
    prev = session_state.processing_info.get("rect_info", None)

    # make sure toolbar is not in pan/zoom mode
    try:
        tb = fig.canvas.manager.toolbar
        if tb is not None:
            tb.mode = ""
    except Exception:
        pass

    ax.set_title("Drag/adjust rectangle, then press ENTER to capture.")

    selector = RectangleSelector(
        ax,
        onselect=lambda *a, **k: None,
        interactive=True,
        useblit=False,
        button=[1],
        minspanx=1,
        minspany=1,
        spancoords="pixels",
        grab_range=10,
        props=dict(
            edgecolor=line_color,
            facecolor="none",
            linestyle=line_style,
            linewidth=line_width,
            alpha=1.0,
        ),
        handle_props=dict(markeredgecolor=line_color, markerfacecolor="none"),
    )
    selector.set_active(True)

    # --- preload previous rectangle, if it exists ---
    if prev is not None:
        try:
            cx = float(prev["center_x"])
            cy = float(prev["center_y"])
            w  = float(prev["width"])
            h  = float(prev["height"])
            if w > 1e-6 and h > 1e-6:
                x1 = cx - 0.5 * w
                x2 = cx + 0.5 * w
                y1 = cy - 0.5 * h
                y2 = cy + 0.5 * h
                selector.extents = (x1, x2, y1, y2)
                fig.canvas.draw_idle()
        except Exception as e:
            print(f"[rectangle_manual] Could not apply previous rectangle: {e}")

    # --- helper to read the current rectangle from the selector ---
    def _read_current():
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception:
            pass

        # Try to read from the artist if available
        for attr in ("_selection_artist", "_shape_artist", "to_draw"):
            obj = getattr(selector, attr, None)
            if isinstance(obj, Rectangle):
                x, y = obj.get_xy()
                w = float(obj.get_width())
                h = float(obj.get_height())
                if w > 1e-6 and h > 1e-6:
                    x1, x2 = sorted([x, x + w])
                    y1, y2 = sorted([y, y + h])
                    cx = x1 + 0.5 * (x2 - x1)
                    cy = y1 + 0.5 * (y2 - y1)
                    return {
                        "center_x": cx,
                        "center_y": cy,
                        "width":   x2 - x1,
                        "height":  y2 - y1,
                    }

        # Fallback: use extents
        try:
            x1, x2, y1, y2 = selector.extents
            xlo, xhi = sorted([x1, x2])
            ylo, yhi = sorted([y1, y2])
            if (xhi - xlo) > 1e-6 and (yhi - ylo) > 1e-6:
                cx = xlo + 0.5 * (xhi - xlo)
                cy = ylo + 0.5 * (yhi - ylo)
                return {
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "width":   float(xhi - xlo),
                    "height":  float(yhi - ylo),
                }
        except Exception:
            pass

        return None

    result = {"value": None}

    def _on_key(event):
        if event.key in ("enter", "return"):
            result["value"] = _read_current()
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass

    cid = fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.start_event_loop(timeout=-1)
    fig.canvas.mpl_disconnect(cid)

    info = result["value"]

    # save to your global processing_info
    session_state.processing_info["rect_info"] = info

    return info


from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon


def lasso_manual(
    ax,
    line_color="orange",
    line_style="--",
    line_width=1.5,
):
    """
    Lasso / polygon selector on an existing axes, with previous-lasso display.

    - If session_state.processing_info["lasso_info"] exists, draw it (cyan, dotted)
      as a reference.
    - User clicks to define vertices (PolygonSelector).
    - Double-click closes polygon.
    - Press ENTER to accept polygon.
    - Saves to session_state.processing_info["lasso_info"] as:
          {"vertices": [(x0,y0), (x1,y1), ...]}
    - Returns that dict (or None).
    """
    if ax is None:
        print("[WARNING] No axes provided.")
        return None

    fig = ax.figure

    # avoid tight_layout warning
    try:
        fig.set_tight_layout(False)
    except Exception:
        pass

    # make sure toolbar is not in pan/zoom mode
    try:
        tb = fig.canvas.manager.toolbar
        if tb is not None:
            tb.mode = ""
    except Exception:
        pass

    # ---------- draw previous lasso, if any ----------
    prev = session_state.processing_info.get("lasso_info", None)
    if prev is not None:
        try:
            verts_prev = prev.get("vertices", None)
            if verts_prev is not None and len(verts_prev) >= 3:
                poly_ref = Polygon(
                    verts_prev,
                    closed=True,
                    fill=False,
                    linestyle=":",
                    linewidth=line_width,
                    edgecolor="cyan",
                    alpha=0.0,
                )
                ax.add_patch(poly_ref)
                fig.canvas.draw_idle()
        except Exception as e:
            print(f"[lasso_manual] Could not draw previous lasso: {e}")

    ax.set_title("Lasso: click vertices, double-click to close, ENTER to accept")

    # where PolygonSelector will store the latest polygon
    verts_holder = {"verts": None}

    def _onselect(verts):
        # verts is list of (x, y)
        verts_holder["verts"] = [(float(x), float(y)) for x, y in verts]

    # ---------- create PolygonSelector with minimal args ----------
    selector = PolygonSelector(
        ax,
        _onselect,
        useblit=False,
        props = dict(color=line_color,linestyle=line_style,linewidth=line_width)
    )
    selector.set_active(True)

    # try to style the line AFTER creation (best-effort only)
    try:
        # different versions use different attribute names; try common ones
        line = getattr(selector, "line", None) or getattr(selector, "_line", None)
        if line is not None:
            line.set_color(line_color)
            line.set_linestyle(line_style)
            line.set_linewidth(line_width)
    except Exception:
        # styling is purely cosmetic; ignore failures
        pass

    result = {"value": None}

    def _on_key(event):
        if event.key in ("enter", "return"):
            verts = verts_holder["verts"]
            if verts is None or len(verts) < 3:
                result["value"] = None
            else:
                result["value"] = {"vertices": verts}
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass

    cid = fig.canvas.mpl_connect("key_press_event", _on_key)

    # block until ENTER
    fig.canvas.start_event_loop(timeout=-1)
    fig.canvas.mpl_disconnect(cid)

    info = result["value"]

    # save into global processing_info (overwrites previous)
    session_state.processing_info["lasso_info"] = info

    return info



#%% Calibration
def conversion_yag(yag=50e-3,ellipse_info=None,overwrite=True):
    """
    From ellipse information, obtain pixel-to-m conversion factor.
    """
    if overwrite == False:
        cal = session_state.processing_info.get("calibration", None)
        fiducial = session_state.processing_info.get("fiducial", None)
        if (cal is None) or (fiducial is None):
            raise ValueError("If you don't want to overwrite anything, calibration and fiducial information must exist, but it doesn't.")
    else:
        if ellipse_info is None:
            ellipse_info = session_state.processing_info.get("ellipse_info", None)
            if ellipse_info is None:
                raise ValueError("No ellipse information is provided.")
        cal = {"cal_x": yag/ellipse_info['width'], "cal_y": yag/ellipse_info['height'] }
        fiducial = {"center_x": ellipse_info['center_x'],"center_y": ellipse_info['center_y']}
        session_state.processing_info["calibration"] = cal
        session_state.processing_info["fiducial"] = fiducial
    return cal, fiducial
#%% Background substraction
def bg_substraction(img_main, img_bg):
    """
    average background image and substract it from main images
    images must be in the form of (Nshot, X, Y)
    """
    if img_bg.ndim == 3:
        bg = np.mean(img_bg,axis=0)
    else:
        bg = img_bg
    out = img_main-bg
    out[out<0] = 0
    return out

def build_bg_template(bg_stack, method="median"):
    """
    bg_stack : (N_bg, H, W)
    method   : 'median' or 'mean'
    """
    bg_stack = bg_stack.astype(np.float32)

    if method == "median":
        bg_template = np.median(bg_stack, axis=0)
    elif method == "mean":
        bg_template = np.mean(bg_stack, axis=0)
    else:
        raise ValueError("method must be 'median' or 'mean'")
    return bg_template

def subtract_bg_scaled(main_img, bg_template, roi_mask=None):
    """
    main_img : (H,W) or (N,H,W)
    """
    main = main_img.astype(np.float32)
    bg   = bg_template.astype(np.float32)

    # ensure 3D: (N,H,W)
    if main.ndim == 2:
        main = main[None, ...]   # (1,H,W)

    N, H, W = main.shape

    if roi_mask is None:
        roi_mask = np.ones((H, W), dtype=bool)

    m = roi_mask.ravel()
    x = bg.ravel()[m]   # template in ROI

    resid = np.empty_like(main, dtype=np.float32)
    ab_list = []

    for i in range(N):
        y = main[i].ravel()[m]   # this shot's data in ROI

        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]

        bg_est = a * bg + b
        r = main[i] - bg_est
        r[r < 0] = 0

        resid[i] = r
        ab_list.append((a, b))

    # if input was 2D, return 2D
    if main_img.ndim == 2:
        return resid[0], ab_list[0]
    else:
        return resid, ab_list


#%% ROI
from matplotlib.path import Path

def apply_roi_mask(
    images,
    roi_info=None,
    roi_type="ellipse",   # "ellipse", "rect", or "lasso"
    outside_value=0,
):
    """
    Zero all pixels outside of ROI.

    Parameters
    ----------
    images : array
        2D (H, W) or 3D (N, H, W) image(s).
    roi_info : dict or [] or None
        If dict:
          ellipse/rect: {'center_x','center_y','width','height'} in pixels
          lasso:        {'vertices': [(x0,y0), (x1,y1), ...]}
        If [] or None, ROI will be loaded from
          roi_type='ellipse' -> session_state.processing_info["ellipse_info"]
          roi_type='rect'    -> session_state.processing_info["rect_info"]
          roi_type='lasso'   -> session_state.processing_info["lasso_info"]
    roi_type : {'ellipse','rect','lasso'}
        Shape of the ROI (axis-aligned for ellipse/rect).
    outside_value : scalar
        Value to assign outside the ROI (default 0).

    Returns
    -------
    masked : array
        Same shape and dtype as input, with outside-ROI pixels set to outside_value.
    """
    # ----- normalize input images -----
    arr = np.asarray(images)
    if arr.ndim == 2:
        arr = arr[None, ...]   # -> (N, H, W)
        squeeze_back = True
    elif arr.ndim == 3:
        squeeze_back = False
    else:
        raise ValueError("images must be 2D (H,W) or 3D (N,H,W)")

    N, H, W = arr.shape

    roi_type_low = roi_type.lower()

    # ----- fetch ROI info if not provided -----
    if roi_info is None or roi_info == []:
        if roi_type_low == "ellipse":
            roi_info = session_state.processing_info.get("ROI_ellipse", None)
        elif roi_type_low in ("rect", "rectangle"):
            roi_info = session_state.processing_info.get("ROI_rect", None)
        elif roi_type_low == "lasso":
            roi_info = session_state.processing_info.get("ROI_lasso", None)
        else:
            raise ValueError("roi_type must be 'ellipse', 'rect', or 'lasso'")

        if roi_info is None:
            raise ValueError(
                f"No ROI info provided and "
                f"session_state.processing_info does not contain "
                f"a key for roi_type='{roi_type_low}'."
            )

    # pixel grids: y = rows, x = columns
    yy, xx = np.ogrid[0:H, 0:W]

    # ----- build 2D mask -----
    if roi_type_low == "ellipse":
        cx = float(roi_info["center_x"])
        cy = float(roi_info["center_y"])
        w  = float(roi_info["width"])
        h  = float(roi_info["height"])

        rx = w / 2.0
        ry = h / 2.0
        if rx <= 0 or ry <= 0:
            raise ValueError("ROI width/height must be positive for ellipse.")

        normx = (xx - cx) / rx
        normy = (yy - cy) / ry
        mask2d = (normx**2 + normy**2) <= 1.0

    elif roi_type_low in ("rect", "rectangle"):
        cx = float(roi_info["center_x"])
        cy = float(roi_info["center_y"])
        w  = float(roi_info["width"])
        h  = float(roi_info["height"])

        half_w = w / 2.0
        half_h = h / 2.0
        if half_w <= 0 or half_h <= 0:
            raise ValueError("ROI width/height must be positive for rectangle.")

        mask2d = (
            (xx >= cx - half_w) &
            (xx <= cx + half_w) &
            (yy >= cy - half_h) &
            (yy <= cy + half_h)
        )

    elif roi_type_low == "lasso":
        verts = np.asarray(roi_info["vertices"], dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
            raise ValueError("lasso roi_info['vertices'] must be list of (x,y) with len>=3")

        # Path expects (x,y). Build mask by testing all pixel centers.
        path = Path(verts)
        # construct all (x,y) positions
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        points = np.column_stack([xs.ravel(), ys.ravel()])
        mask_flat = path.contains_points(points)
        mask2d = mask_flat.reshape(H, W)

    else:
        raise ValueError("roi_type must be 'ellipse', 'rect', or 'lasso'")

    # ----- apply mask -----
    mask3d = np.broadcast_to(mask2d, (N, H, W))
    out = arr.copy()
    out[~mask3d] = outside_value

    if squeeze_back:
        return out[0]
    return out

#%%
def apply_roi_threshold(
    images,
    roi_info=None,
    roi_type="ellipse",          # "ellipse" or "rectangle"
    scaling=1.0,
    save_scaling=True,
):
    """
    ROI-based background thresholding.

    For each shot:
      1) compute mean intensity inside ROI,
      2) threshold = scaling * mean,
      3) new_image = image - threshold, with negatives set to 0.

    Parameters
    ----------
    images : array
        2D (H, W) or 3D (N, H, W).
    roi_info : dict or [] or None
        If dict, must be:
            {'center_x', 'center_y', 'width', 'height'} in pixels.
        If [] or None:
            - roi_type == "ellipse"   -> use session_state.processing_info["ellipse_info"]
            - roi_type == "rect" -> use session_state.processing_info["rect_info"]
    roi_type : {'ellipse','rect'}
        Shape of ROI (axis-aligned).
    scaling : float
        Factor to multiply the ROI mean before subtracting.
    save_scaling : bool
        If True, save scaling into session_state.processing_info[threshold_scaling].

    Returns
    -------
    out : array
        Thresholded image(s), same shape and dtype as input.
    """

    # ----- normalize images -----
    arr = np.asarray(images)
    orig_dtype = arr.dtype

    if arr.ndim == 2:
        arr = arr[None, ...]   # -> (N, H, W)
        squeeze_back = True
    elif arr.ndim == 3:
        squeeze_back = False
    else:
        raise ValueError("images must be 2D (H,W) or 3D (N,H,W)")

    N, H, W = arr.shape

    # ----- get ROI info -----
    if roi_info is None or roi_info == []:
        if roi_type.lower() == "ellipse":
            roi_info = session_state.processing_info.get("ellipse_info", None)
        elif roi_type.lower() == "rect":
            roi_info = session_state.processing_info.get("rect_info", None)
        else:
            raise ValueError("roi_type must be 'ellipse' or 'rect'")

        if roi_info is None:
            raise ValueError(
                f"No ROI info provided and no stored ROI for roi_type='{roi_type}'."
            )

    cx = float(roi_info["center_x"])
    cy = float(roi_info["center_y"])
    w  = float(roi_info["width"])
    h  = float(roi_info["height"])

    # ----- build 2D ROI mask -----
    yy, xx = np.ogrid[0:H, 0:W]
    roi_type_low = roi_type.lower()

    if roi_type_low == "ellipse":
        rx = w / 2.0
        ry = h / 2.0
        if rx <= 0 or ry <= 0:
            raise ValueError("ROI width/height must be positive for ellipse.")
        normx = (xx - cx) / rx
        normy = (yy - cy) / ry
        mask2d = (normx**2 + normy**2) <= 1.0

    elif roi_type_low == "rect":
        half_w = w / 2.0
        half_h = h / 2.0
        if half_w <= 0 or half_h <= 0:
            raise ValueError("ROI width/height must be positive for rect.")
        mask2d = (
            (xx >= cx - half_w) &
            (xx <= cx + half_w) &
            (yy >= cy - half_h) &
            (yy <= cy + half_h)
        )
    else:
        raise ValueError("roi_type must be 'ellipse' or 'rect'")

    # ----- apply threshold per shot -----
    # work in float, then cast back to original dtype
    arr_f = arr.astype(float, copy=False)
    out_f = np.empty_like(arr_f)

    roi_pixels = mask2d

    for i in range(N):
        shot = arr_f[i]
        vals = shot[roi_pixels]
        if vals.size == 0:
            mean_roi = 0.0
        else:
            mean_roi = float(vals.mean())

        thr = scaling * mean_roi

        temp = shot - thr
        temp[temp < 0.0] = 0.0
        out_f[i] = temp

    # ----- save scaling factor -----
    if save_scaling:
        session_state.processing_info['threshold_scaling'] = float(scaling)

    # ----- cast back to original dtype -----
    if np.issubdtype(orig_dtype, np.integer):
        # clip to valid range for integer type
        info = np.iinfo(orig_dtype)
        out_f = np.clip(out_f, info.min, info.max)
        out = out_f.astype(orig_dtype)
    else:
        out = out_f.astype(orig_dtype, copy=False)

    if squeeze_back:
        return out[0]
    return out
#%% median filter
from scipy.ndimage import median_filter
def apply_median_filter(
    images,
    window_size=3,         # must be odd number: 3,5,7,...
    save=True,
):
    """
    Apply median filter to 2D or 3D images.
    Saves the window size to session_state.processing_info[save_key].

    Parameters
    ----------
    images : ndarray
        (H,W) or (N,H,W)
    window_size : int
        Median filter window size (must be odd).
    save : bool
        If True, save window_size into session_state.processing_info.

    Returns
    -------
    filtered : ndarray
        Same shape as input, with median filtering applied.
    """

    # ---------- sanity check ----------
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    if window_size % 2 == 0:
        raise ValueError("window_size must be an *odd* integer: 3,5,7,...")

    arr = np.asarray(images)

    # ---------- apply filter ----------
    if arr.ndim == 2:
        # single image
        out = median_filter(arr, size=window_size)

    elif arr.ndim == 3:
        # multiple images
        N = arr.shape[0]
        out = np.empty_like(arr)
        for i in range(N):
            out[i] = median_filter(arr[i], size=window_size)

    else:
        raise ValueError("images must be (H,W) or (N,H,W).")

    # ---------- save window size ----------
    if save:
        session_state.processing_info['median_window'] = int(window_size)

    return out
#%% Morphological cleaning
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_opening, binary_closing

def clean_beam_array(resid, thr_ratio=0.01, min_size=300,
                     do_open=True, do_close=True):
    """
    Morphologically clean beam images (2D or 3D) using
    threshold + size filter + optional opening/closing.

    Parameters
    ----------
    resid : array
        Background-subtracted image(s).
        Shape (H, W) or (Nshot, H, W).
    thr_ratio : float
        Threshold = thr_ratio * max(resid_per_shot).
    min_size : int
        Remove connected regions smaller than this many pixels (per shot).
    do_open : bool
        Whether to perform binary opening (remove thin noise).
    do_close : bool
        Whether to perform binary closing (fill small holes).

    Returns
    -------
    cleaned : array
        Same shape as resid, but with non-beam pixels set to 0.
        (Beam intensities are unchanged where kept.)
    """

    resid = np.asarray(resid, dtype=np.float32)

    # ---------------- single 2D image ----------------
    if resid.ndim == 2:
        img = resid
        max_val = img.max()
        if max_val <= 0:
            # nothing there
            return np.zeros_like(img)

        thr = thr_ratio * max_val
        mask = img > thr

        # size filter (2D)
        if min_size > 1:
            mask = remove_small_objects(mask, min_size=min_size)

        if do_open:
            mask = binary_opening(mask, iterations=1)
        if do_close:
            mask = binary_closing(mask, iterations=1)

        cleaned = img * mask
        return cleaned

    # ---------------- stack (N, H, W) ----------------
    elif resid.ndim == 3:
        N, H, W = resid.shape
        cleaned = np.zeros_like(resid, dtype=np.float32)

        for i in range(N):
            img = resid[i]
            max_val = img.max()
            if max_val <= 0:
                # leave this frame as zeros
                continue

            thr = thr_ratio * max_val
            mask = img > thr

            # size filter per frame (2D)
            if min_size > 1:
                mask = remove_small_objects(mask, min_size=min_size)

            if do_open:
                mask = binary_opening(mask, iterations=1)
            if do_close:
                mask = binary_closing(mask, iterations=1)

            cleaned[i] = img * mask

        return cleaned

    else:
        raise ValueError("clean_beam_array expects 2D (H,W) or 3D (N,H,W) array.")