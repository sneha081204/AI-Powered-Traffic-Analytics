"""
ViolaWatch — Desktop GUI Application
Built with Tkinter + CustomTkinter for a modern dark-theme interface
"""

import sys
import os
import base64
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

try:
    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    HAS_CTK = True

except ImportError:
    HAS_CTK = False
    print("[GUI] customtkinter not found, using standard tkinter")


from core.detector import ViolationDetector
from database.db_manager import DatabaseManager
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────

DARK = "#0a0e1a"
DARKER = "#060910"
PANEL = "#0d1424"
BORDER = "#1a2840"
ACCENT = "#00b4d8"
RED = "#e63946"
AMBER = "#f4a261"
GREEN = "#2ec4b6"
TEXT = "#c8d8e8"
TEXT2 = "#6a8ba5"
WHITE = "#e8f4f8"

FONT_MONO = ("Consolas", 10)
FONT_HEAD = ("Segoe UI", 11, "bold")
FONT_BIG = ("Segoe UI", 22, "bold")


class ViolaWatchApp:

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self):

        self.db = DatabaseManager()

        self.detector = ViolationDetector(
            db=self.db,
            on_violation=self._on_violation_callback
        )

        self.root = tk.Tk()

        self._page = 1
        self._total_pages = 1
        self._selected_id = None

        # -------------------------------------------------------------
        # VIDEO STATE
        # -------------------------------------------------------------

        self._video_display_active = False
        self._video_file_processing = False

        self.snap_img_ref = None

        self._build_window()
        self._build_layout()
        self._load_stats()
        self._load_violations()

        self._refresh_loop()

    # =========================================================================
    # WINDOW
    # =========================================================================

    def _build_window(self):

        self.root.title(
            "-- AI Traffic Analytics"
        )

        self.root.geometry("1400x860")
        self.root.minsize(1100, 700)

        self.root.configure(bg=DARKER)

        try:
            self.root.iconbitmap("")
        except Exception:
            pass

    # =========================================================================
    # MAIN LAYOUT
    # =========================================================================

    def _build_layout(self):

        # ---------------------------------------------------------------------
        # TOP BAR
        # ---------------------------------------------------------------------

        topbar = tk.Frame(
            self.root,
            bg=DARK,
            height=52
        )

        topbar.pack(
            fill="x",
            side="top"
        )

        topbar.pack_propagate(False)

        tk.Label(
            topbar,
            text="🎯 Traqvexa",
            bg=DARK,
            fg=WHITE,
            font=("Segoe UI", 16, "bold")
        ).pack(
            side="left",
            padx=20,
            pady=14
        )

        tk.Label(
            topbar,
            text=" --AI Traffic Analytics ",
            bg=DARK,
            fg=TEXT2,
            font=("Segoe UI", 10)
        ).pack(
            side="left"
        )

        self.live_badge = tk.Label(
            topbar,
            text="● STANDBY",
            bg=DARK,
            fg=TEXT2,
            font=FONT_MONO
        )

        self.live_badge.pack(
            side="right",
            padx=20
        )

        # ---------------------------------------------------------------------
        # MAIN BODY
        # ---------------------------------------------------------------------

        body = tk.Frame(
            self.root,
            bg=DARKER
        )

        body.pack(
            fill="both",
            expand=True
        )

        # ---------------------------------------------------------------------
        # LEFT SIDEBAR
        # ---------------------------------------------------------------------

        self.sidebar = tk.Frame(
            body,
            bg=DARK,
            width=230
        )

        self.sidebar.pack(
            side="left",
            fill="y"
        )

        self.sidebar.pack_propagate(False)

        # ---------------------------------------------------------------------
        # CENTER
        # ---------------------------------------------------------------------

        center = tk.Frame(
            body,
            bg=DARKER
        )

        center.pack(
            side="left",
            fill="both",
            expand=True
        )

        # ---------------------------------------------------------------------
        # RIGHT DETAIL PANEL
        # ---------------------------------------------------------------------

        self.detail_frame = tk.Frame(
            body,
            bg=DARK,
            width=360
        )

        self.detail_frame.pack(
            side="right",
            fill="y"
        )

        self.detail_frame.pack_propagate(False)

        self._build_sidebar()
        self._build_center(center)
        self._build_detail_panel()

    # =========================================================================
    # SIDEBAR
    # =========================================================================

    def _build_sidebar(self):

        s = self.sidebar

        pad = {
            "padx": 12,
            "pady": 4
        }

        # ---------------------------------------------------------------------
        # STATISTICS
        # ---------------------------------------------------------------------

        tk.Label(
            s,
            text="STATISTICS",
            bg=DARK,
            fg=TEXT2,
            font=("Consolas", 9)
        ).pack(
            anchor="w",
            padx=12,
            pady=(14, 4)
        )

        self.stat_total_var = tk.StringVar(value="—")
        self.stat_today_var = tk.StringVar(value="—")
        self.stat_helmet_var = tk.StringVar(value="—")
        self.stat_belt_var = tk.StringVar(value="—")

        for label, var, color in [
            ("Total Violations", self.stat_total_var, ACCENT),
            ("Today", self.stat_today_var, RED),
            ("No Helmet", self.stat_helmet_var, RED),
            ("No Seatbelt", self.stat_belt_var, AMBER),
        ]:

            card = tk.Frame(
                s,
                bg=PANEL,
                highlightbackground=BORDER,
                highlightthickness=1
            )

            card.pack(
                fill="x",
                **pad
            )

            tk.Label(
                card,
                text=label,
                bg=PANEL,
                fg=TEXT2,
                font=("Segoe UI", 9)
            ).pack(
                anchor="w",
                padx=8,
                pady=(6, 0)
            )

            tk.Label(
                card,
                textvariable=var,
                bg=PANEL,
                fg=color,
                font=("Segoe UI", 24, "bold")
            ).pack(
                anchor="w",
                padx=8,
                pady=(0, 6)
            )

        tk.Frame(
            s,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            padx=12,
            pady=8
        )

        # ---------------------------------------------------------------------
        # CAMERA
        # ---------------------------------------------------------------------

        tk.Label(
            s,
            text="CAMERA",
            bg=DARK,
            fg=TEXT2,
            font=("Consolas", 9)
        ).pack(
            anchor="w",
            padx=12,
            pady=(0, 4)
        )

        src_frame = tk.Frame(
            s,
            bg=DARK
        )

        src_frame.pack(
            fill="x",
            padx=12,
            pady=(0, 6)
        )

        tk.Label(
            src_frame,
            text="Source:",
            bg=DARK,
            fg=TEXT2,
            font=("Segoe UI", 9)
        ).pack(
            anchor="w"
        )

        self.cam_source_var = tk.StringVar(value="0")

        src_entry = tk.Entry(
            src_frame,
            textvariable=self.cam_source_var,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=FONT_MONO,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        src_entry.pack(
            fill="x",
            ipady=5
        )

        self.btn_start = tk.Button(
            s,
            text="▶  START DETECTION",
            bg="#0d3d1a",
            fg=GREEN,
            activebackground="#0a2e14",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            command=self._start_camera
        )

        self.btn_start.pack(
            fill="x",
            padx=12,
            pady=2,
            ipady=7
        )

        self.btn_stop = tk.Button(
            s,
            text="■  STOP",
            bg="#3d0d0d",
            fg=RED,
            activebackground="#2e0a0a",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            command=self._stop_detection,
            state="disabled"
        )

        self.btn_stop.pack(
            fill="x",
            padx=12,
            pady=2,
            ipady=7
        )

        tk.Frame(
            s,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            padx=12,
            pady=8
        )

        # ---------------------------------------------------------------------
        # VIDEO UPLOAD
        # ---------------------------------------------------------------------

        tk.Label(
            s,
            text="VIDEO UPLOAD (TEST)",
            bg=DARK,
            fg=TEXT2,
            font=("Consolas", 9)
        ).pack(
            anchor="w",
            padx=12,
            pady=(0, 4)
        )

        tk.Button(
            s,
            text="📂  Upload Video File",
            bg="#1a1a3d",
            fg=ACCENT,
            activebackground="#14142e",
            font=("Segoe UI", 10),
            relief="flat",
            cursor="hand2",
            command=self._upload_video
        ).pack(
            fill="x",
            padx=12,
            pady=2,
            ipady=7
        )

        self.progress_var = tk.DoubleVar(value=0)

        self.progress_label = tk.Label(
            s,
            text="",
            bg=DARK,
            fg=TEXT2,
            font=("Consolas", 9)
        )

        self.progress_label.pack(
            anchor="w",
            padx=12
        )

        self.progress_bar = ttk.Progressbar(
            s,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            length=200
        )

        self.progress_bar.pack(
            fill="x",
            padx=12,
            pady=(2, 8)
        )

        tk.Frame(
            s,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            padx=12,
            pady=4
        )

        # ---------------------------------------------------------------------
        # FILTERS
        # ---------------------------------------------------------------------

        tk.Label(
            s,
            text="FILTERS",
            bg=DARK,
            fg=TEXT2,
            font=("Consolas", 9)
        ).pack(
            anchor="w",
            padx=12
        )

        self.filter_type_var = tk.StringVar(value="All")

        tk.Label(
            s,
            text="Type:",
            bg=DARK,
            fg=TEXT2,
            font=("Segoe UI", 9)
        ).pack(
            anchor="w",
            padx=12
        )

        self.type_combo = ttk.Combobox(
            s,
            textvariable=self.filter_type_var,
            values=[
                "All",
                "No Helmet",
                "No Seatbelt"
            ],
            state="readonly",
            font=("Segoe UI", 9)
        )

        self.type_combo.pack(
            fill="x",
            padx=12,
            pady=(0, 6)
        )

        self.type_combo.bind(
            "<<ComboboxSelected>>",
            lambda _: self._load_violations()
        )

        self.filter_plate_var = tk.StringVar()

        tk.Label(
            s,
            text="Plate:",
            bg=DARK,
            fg=TEXT2,
            font=("Segoe UI", 9)
        ).pack(
            anchor="w",
            padx=12
        )

        plate_entry = tk.Entry(
            s,
            textvariable=self.filter_plate_var,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=FONT_MONO,
            highlightbackground=BORDER,
            highlightthickness=1
        )

        plate_entry.pack(
            fill="x",
            padx=12,
            ipady=5
        )

        plate_entry.bind(
            "<Return>",
            lambda _: self._load_violations()
        )

        tk.Button(
            s,
            text="Apply Filters",
            bg=PANEL,
            fg=TEXT,
            relief="flat",
            cursor="hand2",
            command=self._load_violations
        ).pack(
            fill="x",
            padx=12,
            pady=(6, 2),
            ipady=5
        )

        tk.Button(
            s,
            text="Clear",
            bg=DARKER,
            fg=TEXT2,
            relief="flat",
            cursor="hand2",
            command=self._clear_filters
        ).pack(
            fill="x",
            padx=12,
            pady=2,
            ipady=4
        )

        tk.Frame(
            s,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            padx=12,
            pady=8
        )

        tk.Button(
            s,
            text="🎲 Load Demo Data",
            bg=DARKER,
            fg=TEXT2,
            relief="flat",
            cursor="hand2",
            command=self._seed_demo
        ).pack(
            fill="x",
            padx=12,
            ipady=4
        )

    # =========================================================================
    # CENTER
    # =========================================================================

    def _build_center(self, parent):

        # ---------------------------------------------------------------------
        # VIDEO FEED
        # ---------------------------------------------------------------------

        feed_frame = tk.Frame(
            parent,
            bg=DARKER,
            height=450
        )

        feed_frame.pack(
            fill="x",
            padx=8,
            pady=(8, 4)
        )

        feed_frame.pack_propagate(False)

        cam_border = tk.Frame(
            feed_frame,
            bg=BORDER,
            height=440
        )

        cam_border.pack(
            fill="both",
            expand=True
        )

        cam_border.pack_propagate(False)

        self.cam_label = tk.Label(
            cam_border,
            bg="#060a10",
            fg=TEXT2,
            text="📷  NO FEED — Click START DETECTION or Upload a video",
            font=("Segoe UI", 11),
            anchor="center",
            justify="center"
        )

        self.cam_label.pack(
            fill="both",
            expand=True,
            padx=2,
            pady=2
        )

        # ---------------------------------------------------------------------
        # VIOLATIONS TABLE
        # ---------------------------------------------------------------------

        table_frame = tk.Frame(
            parent,
            bg=DARKER
        )

        table_frame.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=(0, 8)
        )

        hdr = tk.Frame(
            table_frame,
            bg=PANEL
        )

        hdr.pack(
            fill="x"
        )

        tk.Label(
            hdr,
            text="VIOLATIONS LOG",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 11, "bold")
        ).pack(
            side="left",
            padx=12,
            pady=8
        )

        self.count_label = tk.Label(
            hdr,
            text="0 records",
            bg=PANEL,
            fg=ACCENT,
            font=FONT_MONO
        )

        self.count_label.pack(
            side="left",
            padx=4
        )

        tk.Button(
            hdr,
            text="↻ Refresh",
            bg=PANEL,
            fg=ACCENT,
            relief="flat",
            cursor="hand2",
            command=self._load_violations
        ).pack(
            side="right",
            padx=12
        )

        style = ttk.Style()

        style.theme_use("clam")

        style.configure(
            "VW.Treeview",
            background=PANEL,
            foreground=TEXT,
            fieldbackground=PANEL,
            rowheight=28,
            font=("Segoe UI", 9),
            borderwidth=0
        )

        style.configure(
            "VW.Treeview.Heading",
            background=DARKER,
            foreground=TEXT2,
            font=("Consolas", 9, "bold"),
            borderwidth=0,
            relief="flat"
        )

        style.map(
            "VW.Treeview",
            background=[
                ("selected", "#1a2840")
            ],
            foreground=[
                ("selected", ACCENT)
            ]
        )

        cols = (
            "id",
            "type",
            "plate",
            "location",
            "time",
            "conf",
            "status"
        )

        self.tree = ttk.Treeview(
            table_frame,
            columns=cols,
            show="headings",
            style="VW.Treeview",
            selectmode="browse"
        )

        widths = {
            "id": 50,
            "type": 140,
            "plate": 130,
            "location": 160,
            "time": 145,
            "conf": 70,
            "status": 90
        }

        for c in cols:

            self.tree.heading(
                c,
                text=c.upper()
            )

            self.tree.column(
                c,
                width=widths[c],
                minwidth=40,
                anchor="w"
            )

        vsb = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.tree.yview
        )

        self.tree.configure(
            yscroll=vsb.set
        )

        self.tree.pack(
            side="left",
            fill="both",
            expand=True
        )

        vsb.pack(
            side="right",
            fill="y"
        )

        self.tree.tag_configure(
            "helmet",
            foreground="#ff6b6b"
        )

        self.tree.tag_configure(
            "seatbelt",
            foreground="#ffa94d"
        )

        self.tree.tag_configure(
            "reviewed",
            foreground=TEXT2
        )

        self.tree.bind(
            "<<TreeviewSelect>>",
            self._on_select
        )

        # ---------------------------------------------------------------------
        # PAGINATION
        # ---------------------------------------------------------------------

        pg_frame = tk.Frame(
            parent,
            bg=PANEL
        )

        pg_frame.pack(
            fill="x",
            padx=8,
            pady=(0, 4)
        )

        self.prev_btn = tk.Button(
            pg_frame,
            text="← Prev",
            bg=PANEL,
            fg=TEXT,
            relief="flat",
            cursor="hand2",
            command=self._prev_page,
            state="disabled"
        )

        self.prev_btn.pack(
            side="left",
            padx=8,
            pady=4
        )

        self.page_label = tk.Label(
            pg_frame,
            text="Page 1 / 1",
            bg=PANEL,
            fg=TEXT2,
            font=FONT_MONO
        )

        self.page_label.pack(
            side="left"
        )

        self.next_btn = tk.Button(
            pg_frame,
            text="Next →",
            bg=PANEL,
            fg=TEXT,
            relief="flat",
            cursor="hand2",
            command=self._next_page,
            state="disabled"
        )

        self.next_btn.pack(
            side="left",
            padx=8
        )

    # =========================================================================
    # DETAIL PANEL
    # =========================================================================

    def _build_detail_panel(self):

        d = self.detail_frame

        tk.Label(
            d,
            text="VIOLATION DETAIL",
            bg=DARK,
            fg=TEXT2,
            font=("Consolas", 9)
        ).pack(
            anchor="w",
            padx=12,
            pady=(14, 6)
        )

        tk.Frame(
            d,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            padx=12,
            pady=(0, 8)
        )

        # ---------------------------------------------------------------------
        # SNAPSHOT
        # ---------------------------------------------------------------------

        self.snap_frame = tk.Frame(
            d,
            bg="#060a10",
            width=336,
            height=220
        )

        self.snap_frame.pack(
            fill="x",
            padx=12,
            pady=(0, 10)
        )

        self.snap_frame.pack_propagate(False)

        self.snap_label = tk.Label(
            self.snap_frame,
            bg="#060a10",
            text="No snapshot",
            fg=TEXT2,
            font=("Segoe UI", 10),
            anchor="center",
            justify="center"
        )

        self.snap_label.pack(
            fill="both",
            expand=True
        )

        self.snap_img_ref = None

        # ---------------------------------------------------------------------
        # DETAIL FIELDS
        # ---------------------------------------------------------------------

        self.detail_vars = {}

        fields = [
            ("ID", "id"),
            ("Violation", "violation_type"),
            ("Plate", "plate_number"),
            ("Confidence", "confidence"),
            ("Timestamp", "timestamp"),
            ("Location", "location"),
            ("Source", "source_type"),
        ]

        for label, key in fields:

            row = tk.Frame(
                d,
                bg=DARK
            )

            row.pack(
                fill="x",
                padx=12,
                pady=2
            )

            tk.Label(
                row,
                text=label + ":",
                bg=DARK,
                fg=TEXT2,
                font=("Segoe UI", 9),
                width=10,
                anchor="w"
            ).pack(
                side="left"
            )

            var = tk.StringVar(
                value="—"
            )

            self.detail_vars[key] = var

            tk.Label(
                row,
                textvariable=var,
                bg=DARK,
                fg=TEXT,
                font=("Segoe UI", 9, "bold"),
                anchor="w",
                wraplength=210
            ).pack(
                side="left",
                fill="x"
            )

        tk.Frame(
            d,
            bg=BORDER,
            height=1
        ).pack(
            fill="x",
            padx=12,
            pady=10
        )

        # ---------------------------------------------------------------------
        # STATUS
        # ---------------------------------------------------------------------

        tk.Label(
            d,
            text="STATUS",
            bg=DARK,
            fg=TEXT2,
            font=("Consolas", 9)
        ).pack(
            anchor="w",
            padx=12,
            pady=(0, 4)
        )

        self.status_var = tk.StringVar(
            value="pending"
        )

        for status_value, lbl in [
            ("pending", "⏳ Pending"),
            ("reviewed", "✅ Reviewed"),
            ("dismissed", "❌ Dismissed")
        ]:

            tk.Radiobutton(
                d,
                text=lbl,
                variable=self.status_var,
                value=status_value,
                bg=DARK,
                fg=TEXT,
                selectcolor=PANEL,
                activebackground=DARK,
                font=("Segoe UI", 9)
            ).pack(
                anchor="w",
                padx=16
            )

        tk.Button(
            d,
            text="💾 Save Status",
            bg="#0d2f3d",
            fg=ACCENT,
            relief="flat",
            cursor="hand2",
            command=self._save_status
        ).pack(
            fill="x",
            padx=12,
            pady=(8, 4),
            ipady=6
        )

        tk.Button(
            d,
            text="🗑 Delete Record",
            bg="#3d0d0d",
            fg=RED,
            relief="flat",
            cursor="hand2",
            command=self._delete_violation
        ).pack(
            fill="x",
            padx=12,
            pady=2,
            ipady=6
        )

        tk.Button(
            d,
            text="📤 Export CSV",
            bg=DARKER,
            fg=TEXT2,
            relief="flat",
            cursor="hand2",
            command=self._export_csv
        ).pack(
            fill="x",
            padx=12,
            pady=2,
            ipady=5
        )

    # =========================================================================
    # CAMERA START
    # =========================================================================

    def _start_camera(self):

        # If a video is currently processing, don't start camera.
        if self._video_file_processing:

            messagebox.showwarning(
                "Video Processing",
                "Stop the current video processing first."
            )

            return

        src = self.cam_source_var.get().strip()

        source = int(src) if src.isdigit() else src

        try:

            self.detector.start_live(source)

            self.btn_start.config(
                state="disabled"
            )

            self.btn_stop.config(
                state="normal"
            )

            self.live_badge.config(
                text="● LIVE",
                fg=GREEN
            )

            self._start_frame_update()

        except Exception as e:

            messagebox.showerror(
                "Camera Error",
                str(e)
            )

    # =========================================================================
    # STOP DETECTION
    # =========================================================================

    def _stop_detection(self):

        print("[GUI] STOP button pressed")

        # -------------------------------------------------------------
        # Tell detector thread to stop
        # -------------------------------------------------------------

        self.detector.stop()

        # -------------------------------------------------------------
        # Update GUI immediately
        # -------------------------------------------------------------

        self._video_display_active = False
        self._video_file_processing = False

        self.btn_start.config(
            state="normal"
        )

        self.btn_stop.config(
            state="disabled"
        )

        self.live_badge.config(
            text="● STOPPED",
            fg=RED
        )

        self.progress_label.config(
            text="Stopped by user"
        )

        # -------------------------------------------------------------
        # Keep last frame visible, but remove running state
        # -------------------------------------------------------------

        print("[GUI] Detection stopped")

    # Keep old method name available in case another part of the
    # project calls _stop_camera().
    def _stop_camera(self):

        self._stop_detection()

    # =========================================================================
    # CAMERA FRAME DISPLAY
    # =========================================================================

    def _start_frame_update(self):

        def _update():

            if not self.detector.is_running:

                return

            b64 = self.detector.get_frame_b64()

            if b64:

                try:

                    data = base64.b64decode(
                        b64
                    )

                    img = Image.open(
                        BytesIO(data)
                    ).convert("RGB")

                    max_width = max(
                        self.cam_label.winfo_width() - 10,
                        640
                    )

                    max_height = 430

                    img.thumbnail(
                        (
                            max_width,
                            max_height
                        ),
                        Image.LANCZOS
                    )

                    photo = ImageTk.PhotoImage(
                        img
                    )

                    self.cam_label.config(
                        image=photo,
                        text=""
                    )

                    self.cam_label._img = photo

                    fps = self.detector.stats.get(
                        "fps",
                        0
                    )

                    total = self.detector.stats.get(
                        "total",
                        0
                    )

                    self.live_badge.config(
                        text=(
                            f"● LIVE  "
                            f"{fps} FPS  "
                            f"Violations: {total}"
                        ),
                        fg=GREEN
                    )

                except Exception as e:

                    print(
                        "[GUI] Camera display error:",
                        e
                    )

            if self.detector.is_running:

                self.root.after(
                    120,
                    _update
                )

        self.root.after(
            120,
            _update
        )

    # =========================================================================
    # VIDEO UPLOAD
    # =========================================================================

    def _upload_video(self):

        # Don't start another video while one is running.
        if self._video_file_processing:

            messagebox.showwarning(
                "Video Processing",
                "A video is already being processed."
            )

            return

        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                (
                    "Video files",
                    "*.mp4 *.avi *.mov *.mkv *.wmv"
                ),
                (
                    "All files",
                    "*.*"
                )
            ]
        )

        if not path:
            return

        try:

            # ---------------------------------------------------------
            # Reset detector state
            # ---------------------------------------------------------

            self.detector.stop()

            self._video_file_processing = True
            self._video_display_active = True

            self.progress_var.set(0)

            self.progress_label.config(
                text=(
                    f"Processing: "
                    f"{os.path.basename(path)}"
                )
            )

            self.live_badge.config(
                text="● VIDEO PROCESSING",
                fg=ACCENT
            )

            self.btn_start.config(
                state="disabled"
            )

            self.btn_stop.config(
                state="normal"
            )

            # ---------------------------------------------------------
            # Create database job
            # ---------------------------------------------------------

            job_id = self.db.create_job(
                os.path.basename(path)
            )

            # ---------------------------------------------------------
            # Progress callback
            # ---------------------------------------------------------

            def on_progress(
                pct,
                frames,
                viols
            ):

                try:

                    self.root.after(
                        0,
                        lambda p=pct,
                               f=frames,
                               v=viols:
                        self._update_video_progress(
                            p,
                            f,
                            v
                        )
                    )

                except Exception as e:

                    print(
                        "[GUI] Progress callback error:",
                        e
                    )

            # ---------------------------------------------------------
            # DONE callback
            # ---------------------------------------------------------

            def on_done(
                viols,
                err
            ):

                def finish_ui():

                    self._video_display_active = False
                    self._video_file_processing = False

                    self.btn_start.config(
                        state="normal"
                    )

                    self.btn_stop.config(
                        state="disabled"
                    )

                    if err:

                        self.live_badge.config(
                            text="● STOPPED",
                            fg=RED
                        )

                        self.progress_label.config(
                            text=str(err)
                        )

                        # Don't show popup for normal user stop.
                        if str(err) != "Stopped by user":

                            messagebox.showerror(
                                "Video Processing Error",
                                str(err)
                            )

                    else:

                        self.progress_var.set(
                            100
                        )

                        self.progress_label.config(
                            text=(
                                f"Done! Found "
                                f"{viols} violations"
                            )
                        )

                        self.live_badge.config(
                            text="● VIDEO COMPLETE",
                            fg=GREEN
                        )

                        self._load_violations()
                        self._load_stats()

                        messagebox.showinfo(
                            "Processing Complete",
                            (
                                "Video analysis complete.\n"
                                f"Found {viols} violations."
                            )
                        )

                try:

                    self.root.after(
                        0,
                        finish_ui
                    )

                except Exception as e:

                    print(
                        "[GUI] Done callback error:",
                        e
                    )

            # ---------------------------------------------------------
            # Start detector thread
            # ---------------------------------------------------------

            self.detector.process_video_file(
                path,
                job_id,
                progress_cb=on_progress,
                done_cb=on_done
            )

            # ---------------------------------------------------------
            # Start GUI video display
            # ---------------------------------------------------------

            self._start_video_frame_update()

        except Exception as e:

            self._video_display_active = False
            self._video_file_processing = False

            self.btn_start.config(
                state="normal"
            )

            self.btn_stop.config(
                state="disabled"
            )

            messagebox.showerror(
                "Video Error",
                str(e)
            )

    # =========================================================================
    # VIDEO PROGRESS
    # =========================================================================

    def _update_video_progress(
        self,
        pct,
        frames,
        viols
    ):

        try:

            self.progress_var.set(
                float(pct)
            )

            self.progress_label.config(
                text=(
                    f"{pct}% — "
                    f"{frames} frames — "
                    f"{viols} violations"
                )
            )

        except Exception as e:

            print(
                "[GUI] Progress update error:",
                e
            )

    # =========================================================================
    # VIDEO FRAME DISPLAY
    # =========================================================================

    def _start_video_frame_update(self):

        def update():

            try:

                b64 = self.detector.get_frame_b64()

                if b64:

                    data = base64.b64decode(
                        b64
                    )

                    img = Image.open(
                        BytesIO(data)
                    ).convert("RGB")

                    available_width = max(
                        self.cam_label.winfo_width() - 10,
                        640
                    )

                    available_height = 430

                    img.thumbnail(
                        (
                            available_width,
                            available_height
                        ),
                        Image.LANCZOS
                    )

                    photo = ImageTk.PhotoImage(
                        img
                    )

                    self.cam_label.config(
                        image=photo,
                        text="",
                        anchor="center"
                    )

                    self.cam_label._img = photo

                # -----------------------------------------------------
                # Continue ONLY if detector is actually running OR
                # video processing is still active.
                # -----------------------------------------------------

                if (
                    self.detector.is_running
                    or self._video_file_processing
                ):

                    self.root.after(
                        100,
                        update
                    )

            except Exception as e:

                print(
                    "[GUI] Video display error:",
                    e
                )

                if (
                    self.detector.is_running
                    or self._video_file_processing
                ):

                    self.root.after(
                        200,
                        update
                    )

        self.root.after(
            100,
            update
        )

    # =========================================================================
    # VIOLATIONS TABLE
    # =========================================================================

    def _load_violations(self):

        vtype = self.filter_type_var.get()

        plate = self.filter_plate_var.get().strip()

        type_map = {
            "No Helmet": "no_helmet",
            "No Seatbelt": "no_seatbelt",
            "All": None
        }

        vtype_key = type_map.get(
            vtype
        )

        limit = 50

        offset = (
            getattr(
                self,
                "_page",
                1
            ) - 1
        ) * limit

        records = self.db.get_violations(
            limit=limit,
            offset=offset,
            violation_type=vtype_key,
            plate=plate or None
        )

        total = self.db.count_violations(
            violation_type=vtype_key,
            plate=plate or None
        )

        pages = max(
            1,
            (total + limit - 1) // limit
        )

        page = getattr(
            self,
            "_page",
            1
        )

        if page > pages:

            page = pages

            self._page = page

        self.page_label.config(
            text=f"Page {page} / {pages}"
        )

        self.prev_btn.config(
            state=(
                "normal"
                if page > 1
                else "disabled"
            )
        )

        self.next_btn.config(
            state=(
                "normal"
                if page < pages
                else "disabled"
            )
        )

        self._total_pages = pages

        self.count_label.config(
            text=f"{total} records"
        )

        self.tree.delete(
            *self.tree.get_children()
        )

        for r in records:

            ts = str(
                r.get(
                    "timestamp",
                    ""
                )
            )[:19]

            try:

                conf = (
                    f"{int(float(r.get('confidence', 0)) * 100)}%"
                )

            except Exception:

                conf = "0%"

            violation_type = r.get(
                "violation_type",
                ""
            )

            tag = (
                "helmet"
                if "helmet" in violation_type
                else "seatbelt"
            )

            if r.get("status") == "reviewed":

                tag = "reviewed"

            vtype_disp = (
                "🪖 No Helmet"
                if "helmet" in violation_type
                else "🚗 No Seatbelt"
            )

            self.tree.insert(
                "",
                "end",
                iid=str(
                    r["id"]
                ),
                values=(
                    r["id"],
                    vtype_disp,
                    r.get(
                        "plate_number",
                        "?"
                    ),
                    r.get(
                        "location",
                        ""
                    ),
                    ts,
                    conf,
                    r.get(
                        "status",
                        "pending"
                    )
                ),
                tags=(tag,)
            )

    # =========================================================================
    # SELECT VIOLATION
    # =========================================================================

    def _on_select(self, _event=None):

        sel = self.tree.selection()

        if not sel:
            return

        try:

            vid = int(
                sel[0]
            )

        except Exception:

            return

        self._selected_id = vid

        records = self.db.get_violations(
            limit=10000
        )

        selected_record = None

        for record in records:

            try:

                if int(
                    record.get("id")
                ) == vid:

                    selected_record = record
                    break

            except Exception:

                continue

        if selected_record is None:

            print(
                "[GUI] Selected violation not found:",
                vid
            )

            return

        r = selected_record

        violation_type = str(
            r.get(
                "violation_type",
                ""
            )
        )

        mapping = {

            "id":
                str(
                    r.get(
                        "id",
                        ""
                    )
                ),

            "violation_type":
                (
                    "🪖 No Helmet"
                    if "helmet" in violation_type
                    else "🚗 No Seatbelt"
                ),

            "plate_number":
                r.get(
                    "plate_number",
                    "UNKNOWN"
                ),

            "confidence":
                f"{int(float(r.get('confidence', 0)) * 100)}%",

            "timestamp":
                str(
                    r.get(
                        "timestamp",
                        ""
                    )
                )[:19],

            "location":
                r.get(
                    "location",
                    ""
                ),

            "source_type":
                r.get(
                    "source_type",
                    "video"
                ),
        }

        for k, v in mapping.items():

            if k in self.detail_vars:

                self.detail_vars[k].set(
                    str(v)
                )

        self.status_var.set(
            r.get(
                "status",
                "pending"
            )
        )

        # ---------------------------------------------------------------------
        # SNAPSHOT
        # ---------------------------------------------------------------------

        snap = r.get(
            "snapshot_path",
            ""
        )

        print(
            "[GUI] Snapshot value from DB:",
            snap
        )

        if not snap:

            self.snap_label.config(
                image="",
                text="No snapshot"
            )

            self.snap_img_ref = None

            return

        snap = str(snap)

        possible_paths = []

        if os.path.isabs(snap):

            possible_paths.append(
                snap
            )

        else:

            possible_paths.append(
                os.path.abspath(snap)
            )

            possible_paths.append(
                os.path.abspath(
                    os.path.join(
                        str(cfg.SNAPSHOT_DIR),
                        snap
                    )
                )
            )

            possible_paths.append(
                os.path.abspath(
                    os.path.join(
                        str(cfg.SNAPSHOT_DIR),
                        os.path.basename(snap)
                    )
                )
            )

        snap_path = None

        for candidate in possible_paths:

            print(
                "[GUI] Checking snapshot:",
                candidate
            )

            if os.path.exists(candidate):

                snap_path = candidate
                break

        if snap_path is None:

            print(
                "[GUI] Snapshot file NOT FOUND"
            )

            self.snap_label.config(
                image="",
                text="Snapshot not found"
            )

            self.snap_img_ref = None

            return

        try:

            img = Image.open(
                snap_path
            ).convert("RGB")

            print(
                "[GUI] Snapshot original size:",
                img.width,
                "x",
                img.height
            )

            max_width = 336
            max_height = 220

            img.thumbnail(
                (
                    max_width,
                    max_height
                ),
                Image.LANCZOS
            )

            photo = ImageTk.PhotoImage(
                img
            )

            self.snap_label.config(
                image=photo,
                text=""
            )

            self.snap_img_ref = photo

            print(
                "[GUI] Snapshot displayed successfully:",
                img.width,
                "x",
                img.height
            )

        except Exception as e:

            print(
                "[GUI] Snapshot loading error:",
                e
            )

            self.snap_label.config(
                image="",
                text="Snapshot unavailable"
            )

            self.snap_img_ref = None

    # =========================================================================
    # PAGINATION
    # =========================================================================

    def _prev_page(self):

        self._page = max(
            1,
            getattr(
                self,
                "_page",
                1
            ) - 1
        )

        self._load_violations()

    def _next_page(self):

        tp = getattr(
            self,
            "_total_pages",
            1
        )

        self._page = min(
            tp,
            getattr(
                self,
                "_page",
                1
            ) + 1
        )

        self._load_violations()

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def _load_stats(self):

        s = self.db.get_stats()

        self.stat_total_var.set(
            str(
                s.get(
                    "total",
                    0
                )
            )
        )

        self.stat_today_var.set(
            str(
                s.get(
                    "today",
                    0
                )
            )
        )

        bt = s.get(
            "by_type",
            {}
        )

        self.stat_helmet_var.set(
            str(
                bt.get(
                    "no_helmet",
                    0
                )
            )
        )

        self.stat_belt_var.set(
            str(
                bt.get(
                    "no_seatbelt",
                    0
                )
            )
        )

    # =========================================================================
    # SAVE STATUS
    # =========================================================================

    def _save_status(self):

        if self._selected_id is None:

            messagebox.showwarning(
                "No selection",
                "Select a violation first"
            )

            return

        self.db.update_violation(
            self._selected_id,
            self.status_var.get()
        )

        self._load_violations()

    # =========================================================================
    # DELETE VIOLATION
    # =========================================================================

    def _delete_violation(self):

        if self._selected_id is None:

            return

        if messagebox.askyesno(
            "Confirm",
            "Delete this violation record?"
        ):

            self.db.delete_violation(
                self._selected_id
            )

            self._selected_id = None

            for k in self.detail_vars:

                self.detail_vars[k].set(
                    "—"
                )

            self.snap_label.config(
                image="",
                text="No snapshot"
            )

            self.snap_img_ref = None

            self._load_violations()
            self._load_stats()

    # =========================================================================
    # EXPORT CSV
    # =========================================================================

    def _export_csv(self):

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV", "*.csv")
            ],
            title="Export violations"
        )

        if not path:

            return

        import csv

        records = self.db.get_violations(
            limit=10000
        )

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            if records:

                w = csv.DictWriter(
                    f,
                    fieldnames=records[0].keys()
                )

                w.writeheader()
                w.writerows(records)

        messagebox.showinfo(
            "Exported",
            f"Saved {len(records)} records to {path}"
        )

    # =========================================================================
    # CLEAR FILTERS
    # =========================================================================

    def _clear_filters(self):

        self.filter_type_var.set(
            "All"
        )

        self.filter_plate_var.set(
            ""
        )

        self._page = 1

        self._load_violations()

    # =========================================================================
    # DEMO DATA
    # =========================================================================

    def _seed_demo(self):

        self.db.seed_demo()

        self._load_stats()
        self._load_violations()

    # =========================================================================
    # CALLBACK
    # =========================================================================

    def _on_violation_callback(
        self,
        v,
        snap_path
    ):

        try:

            self.root.after(
                0,
                self._load_violations
            )

            self.root.after(
                0,
                self._load_stats
            )

        except Exception as e:

            print(
                "[GUI] Violation callback error:",
                e
            )

    # =========================================================================
    # REFRESH
    # =========================================================================

    def _refresh_loop(self):

        try:

            self._load_stats()

        except Exception as e:

            print(
                "[GUI] Refresh error:",
                e
            )

        self.root.after(
            5000,
            self._refresh_loop
        )

    # =========================================================================
    # RUN
    # =========================================================================

    def run(self):

        self.root.mainloop()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    app = ViolaWatchApp()

    app.run()