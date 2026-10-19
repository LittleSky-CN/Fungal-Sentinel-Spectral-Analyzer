# gui/theme.py
"""FSSA Neon-Pop theme, aligned with the Android concept art. Cosmetic only."""
import os, ctypes, tkinter as tk
from tkinter import ttk, font as tkfont

# ---------------- Palette (single source of truth) ----------------
BLACK        = "#000000"
MAGENTA      = "#E6007E"
MAGENTA_DARK = "#6E0037"
GREEN        = "#3CF000"
YELLOW       = "#F0E400"
CREAM        = "#F6F6CE"
CYAN         = "#00E5EE"
INK          = "#000000"   # text on neon surfaces
CONSOLE      = "#39FF14"   # log text

# Matplotlib data-color mapping (scientific meaning preserved)
MPL = {"std": "#00E5EE", "fit": "#E6007E", "sample": "#F0E400",
       "lod": "#FF9F1C", "loq": "#B14AFF", "grid": MAGENTA_DARK}

_PIXEL_FAMILY = None

def register_pixel_font(ttf_path=None, family=None):
    """Register a bundled TTF at runtime (Windows). Falls back silently."""
    global _PIXEL_FAMILY
    if family and family in set(tkfont.families()):
        _PIXEL_FAMILY = family; return _PIXEL_FAMILY
    if ttf_path and os.path.exists(ttf_path):
        try:
            ctypes.windll.gdi32.AddFontResourceW(ttf_path)
            HWND_BROADCAST, WM_FONTCHANGE = 0xFFFF, 0x001D
            res = ctypes.c_long()
            ctypes.windll.user32.SendMessageTimeoutW(
                HWND_BROADCAST, WM_FONTCHANGE, 0, 0, 0x0002, 1000, ctypes.byref(res))
            if family and family in set(tkfont.families()):
                _PIXEL_FAMILY = family
        except Exception:
            pass
    return _PIXEL_FAMILY

def H(size=10, bold=True):
    """Headline/UI font: pixel font if available, else bold Arial."""
    return (_PIXEL_FAMILY or "Arial", size, "bold" if bold else "normal")

MONO = ("Consolas", 10)   # logs & numeric panels: readability first

# ---------------- Global ttk theme ----------------
def apply_theme(root, pixel_ttf=None, pixel_family=None):
    register_pixel_font(pixel_ttf, pixel_family)
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=BLACK, foreground=CREAM,
                    fieldbackground=CREAM, borderwidth=0, font=H(10))
    style.configure("TFrame", background=BLACK)
    style.configure("Card.TFrame", background=CREAM)

    style.configure("TLabel", background=BLACK, foreground=CREAM, font=H(10))
    style.configure("Card.TLabel", background=CREAM, foreground=INK, font=H(10))
    style.configure("Title.TLabel", background=BLACK, foreground=YELLOW, font=H(14))
    style.configure("Status.TLabel", background=BLACK, foreground=GREEN, font=H(9))

    # Buttons: pale default; named variants for hierarchy
    style.configure("TButton", font=H(10), padding=(10, 6), borderwidth=0,
                    relief="flat", background=CREAM, foreground=INK)
    style.map("TButton", background=[("active", YELLOW), ("pressed", MAGENTA_DARK)])
    style.configure("Magenta.TButton", background=MAGENTA, foreground=INK, font=H(11))
    style.map("Magenta.TButton", background=[("active", GREEN), ("pressed", MAGENTA_DARK)])
    style.configure("Green.TButton", background=GREEN, foreground=INK)
    style.map("Green.TButton", background=[("active", YELLOW)])
    style.configure("Yellow.TButton", background=YELLOW, foreground=INK)
    style.map("Yellow.TButton", background=[("active", CREAM)])
    style.configure("Pale.TButton", background=CREAM, foreground=INK)
    style.map("Pale.TButton", background=[("active", YELLOW)])

    # Inputs: cream field with cyan edge (TIPS-card look)
    style.configure("TEntry", fieldbackground=CREAM, foreground=INK, insertcolor=INK,
                    lightcolor=CYAN, darkcolor=CYAN, borderwidth=2, padding=4)
    style.configure("TCombobox", fieldbackground=CREAM, foreground=INK,
                    arrowcolor=MAGENTA, lightcolor=CYAN, darkcolor=CYAN,
                    borderwidth=2, padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", CREAM)],
              foreground=[("readonly", INK)])

    # LabelFrame = cream card with cyan border
    style.configure("TLabelframe", background=CREAM, borderwidth=2,
                    relief="solid", bordercolor=CYAN)
    style.configure("TLabelframe.Label", background=CREAM, foreground=INK, font=H(10))

    # Radiobuttons (mode selector): neon indicator on black / on card
    style.configure("TRadiobutton", background=BLACK, foreground=CREAM,
                    font=H(10), indicatormargin=4)
    style.map("TRadiobutton", indicatorcolor=[("selected", GREEN), ("!selected", MAGENTA_DARK)],
              foreground=[("active", GREEN)])
    style.configure("Card.TRadiobutton", background=CREAM, foreground=INK, font=H(10))
    style.map("Card.TRadiobutton",
              indicatorcolor=[("selected", GREEN), ("!selected", MAGENTA_DARK)])

    # Notebook tabs: magenta when selected
    style.configure("TNotebook", background=BLACK, borderwidth=0)
    style.configure("TNotebook.Tab", background=BLACK, foreground=CREAM,
                    font=H(10), padding=(14, 8), borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", MAGENTA)],
              foreground=[("selected", INK)])

    # Scrollbars & separators
    style.configure("Vertical.TScrollbar", background=MAGENTA, troughcolor=BLACK,
                    borderwidth=0, arrowcolor=CREAM)
    style.configure("Horizontal.TScrollbar", background=MAGENTA, troughcolor=BLACK,
                    borderwidth=0, arrowcolor=CREAM)
    style.configure("Horizontal.TSeparator", background=CYAN)

    # [new] plain tk windows do not follow ttk styles, so their background must be set explicitly
    try:
        root.configure(background=BLACK)
    except tk.TclError:
        pass
    return style

def tint(window, color=None):
    """[new] Apply the base background to tk.Toplevel / tk.Tk windows (harmless when the theme is off)."""
    try:
        window.configure(background=color or BLACK)
    except Exception:
        pass




# ---------------- Matplotlib dark theme ----------------
def style_figure(fig, grid=True):
    """Cosmetic only: dark canvas, neon axes. Data colors untouched."""
    fig.patch.set_facecolor(BLACK)
    for ax in fig.axes:
        ax.set_facecolor(BLACK)
        for s in ax.spines.values():
            s.set_color(CYAN)
        ax.tick_params(colors=CREAM, which="both")
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            t.set_color(CREAM)
            t.set_fontweight("bold")
        if grid:
            ax.grid(color=MPL["grid"], linestyle=":", alpha=0.9)
        leg = ax.get_legend()
        if leg is not None:
            # [fix] Legend has no set_facecolor; its frame patch must be modified instead
            frame = leg.get_frame()
            frame.set_facecolor(BLACK)
            frame.set_edgecolor(GREEN)
            frame.set_alpha(0.85)
            for txt in leg.get_texts():
                txt.set_color(CREAM)
    return fig

# ---------------- Signature decor: parallelogram STEP banner ----------------
class StepBanner(tk.Canvas):
    """Green slanted banner from the Android 'STEP:' page. Pure decoration."""
    def __init__(self, master, height=44, **kw):
        super().__init__(master, height=height, bg=BLACK, highlightthickness=0, **kw)
        self._h, self._text = height, "STEP:"
        self.bind("<Configure>", self._redraw)

    def set_text(self, text):
        self._text = text; self._redraw()

    def _redraw(self, e=None):
        self.delete("all")
        w, h, sl = (self.winfo_width() or 600), self._h, 26
        self.create_polygon(sl, 0, w, 0, w - sl, h, 0, h, fill=GREEN, outline=GREEN)
        self.create_text(sl + 14, h // 2, text=self._text, anchor="w",
                         fill=INK, font=H(12))