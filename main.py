# main.py
"""FSSA v1.4.2 Main Entry Point."""
import tkinter as tk
from tkinter import ttk
from gui.main_window import FSSA_GUI

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

class ModeSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("FSSA v1.4.2 - Analysis Mode Selection")
        self.root.geometry("600x650")
        self.root.resizable(True, True)
        self.root.eval('tk::PlaceWindow . center')

        ttk.Label(root, text="Fungal Sentinel Spectral Analyzer", font=("Arial", 16, "bold")).pack(pady=(30, 10))
        ttk.Label(root, text="Please select the analysis mode for this session:", font=("Arial", 11)).pack(pady=(0, 20))

        mode_frame = ttk.LabelFrame(root, text="Application Mode", padding=20)
        mode_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)

        self.mode_var = tk.StringVar(value="lod_iupac")
        
        ttk.Radiobutton(
            mode_frame, 
            text="Fluorescence Quantification (LoD/LoQ)\n- IUPAC Standard Limit Analysis\n- Weighted Regression Support",
            variable=self.mode_var, value="lod_iupac"
        ).pack(anchor=tk.W, pady=10)
        
        ttk.Radiobutton(
            mode_frame, 
            text="Standard Fluorescence (Legacy)\n- Fast OLS linear regression\n- Routine sample testing",
            variable=self.mode_var, value="legacy"
        ).pack(anchor=tk.W, pady=10)

        ttk.Separator(mode_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Radiobutton(
            mode_frame, 
            text="Turbidity Measurement Mode [NEW]\n- Dedicated UI for scattering analysis\n- Microplate-style spectral reporting",
            variable=self.mode_var, value="turbidity"
        ).pack(anchor=tk.W, pady=10)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=30, pady=25)
        ttk.Button(btn_frame, text="Exit", command=root.destroy, width=12).pack(side=tk.RIGHT, padx=(15, 0))
        ttk.Button(btn_frame, text="Start Analysis", command=self.launch_main, width=15).pack(side=tk.RIGHT)

    def launch_main(self):
        selected_mode = self.mode_var.get()
        self.root.destroy()
        main_root = tk.Tk()
        app = FSSA_GUI(main_root, analysis_mode=selected_mode)
        main_root.mainloop()

if __name__ == "__main__":
    selector_root = tk.Tk()
    selector_root.withdraw() 
    app = ModeSelector(selector_root)
    selector_root.deiconify()
    selector_root.mainloop()