"""
Collection of script that GH may use for processing

@author: Gwanghui

rotate_one_image(img, angle_deg=None)
rotate_all_image(img, angle_deg=None,overwrite=True)
    img: original image array
    angle_deg: rotation angle in deg. if this is None or [], it load data from session_state.
    overwrite: if this is true, angle used in this function will be uploaded to the session_state.

ellipse_info = ellipse_manual(ax, line_color="orange", line_style="--", line_width=1.5,)    
    manually drawing ellipse to on the canvas. You must open any type of canvas and provide its axes info.  
    ax: pyplot figure's axis info.  
    ellipse_info: dictionary including center_x, center_y, width, and height of ellipse.  

rect_info = rectangle_manual(ax= ax, line_color="orange", line_style="--", line_width=1.5)
    manually drawing rectangle to on the canvas. You must open any type of canvas and provide its axes info.  
    ax: pyplot figure's axis info.  
    rect_info: dictionary including center_x, center_y, width, and height of rectangle.  

conversion_yag(yag=50e-3,ellipse_info=None,overwrite=True)
    using ellipse info update calibration factor and fiducial
    yag: size of the yag in meter
    ellipse_info: ellipse dictionary. if it is None, the fuction will try to get ellipse_info from session_state.
    overwrite: if this is False, then the function will try to get info from sesson_state. Otherwise, it update the session_state.
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






