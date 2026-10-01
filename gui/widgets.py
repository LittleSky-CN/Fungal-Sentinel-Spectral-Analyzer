# gui/widgets.py
"""Reusable GUI components."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List

class MultiFileSelector(ttk.Frame):
    def __init__(self, master, title, max_files=5, **kwargs):
        super().__init__(master, **kwargs)
        self.max_files = max_files
        self.files = []
        
        ttk.Label(self, text=title, font=("Arial", 9, "bold")).pack(anchor=tk.W)
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_frame, text=f"Add (Max {max_files})", width=14, command=self.add_files).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Clear", width=6, command=self.clear_files).pack(side=tk.LEFT, padx=5)
        
        self.listbox = tk.Listbox(self, height=3, font=("Consolas", 8), selectmode=tk.EXTENDED)
        self.listbox.pack(fill=tk.X, expand=True)
        
        scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.listbox.xview)
        self.listbox.config(xscrollcommand=scrollbar.set)
        scrollbar.pack(fill=tk.X)

    def add_files(self):
        paths = filedialog.askopenfilenames(title="Select DNG Files", filetypes=[("DNG", "*.dng")])
        if not paths: return
        remaining = self.max_files - len(self.files)
        if remaining <= 0:
            messagebox.showinfo("Limit Reached", f"Maximum {self.max_files} files allowed.")
            return
        for p in paths[:remaining]:
            self.files.append(p)
            self.listbox.insert(tk.END, Path(p).name)

    def clear_files(self):
        self.files = []
        self.listbox.delete(0, tk.END)

    def get_files(self) -> List[str]:
        return self.files