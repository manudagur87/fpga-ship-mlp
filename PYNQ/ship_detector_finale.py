#!/usr/bin/env python3
"""
Ship Detection - EE4218 Finale (7 Coprocessors + SOFT)
AXI Timer + Python Timer for each mode.

Hardware map (from finale.hwh):
  DMA0 -> hdl_optimized_0    (myip_v1_0_v4_opt.v)  — HDL V4 Opt, dual MLP
  DMA1 -> mlp_A_packed_signed_0 (v2)                — HLS A Packed v2 (optimized)
  DMA2 -> DMA_Packed_0       (myip_v1_0_v4.v)      — HDL V4, single MLP
  DMA3 -> mlp_A_packed_signed_1                     — HLS A Packed (original)
  DMA4 -> mlp_B_packed_signed_0                     — HLS B Packed (AXI-Lite wt)
  DMA5 -> mlp_A_unpacked_signed_0                   — HLS A Unpacked
  DMA6 -> hdl_V3_0           (myip_v1_0_v3.v)      — HDL V3 (AXI-Lite wt)
  Timer: axi_timer_0 @ 0xA0070000
  hdl_V3_0 AXI-Lite @ 0xA0080000
  mlp_B_packed_signed_0 AXI-Lite @ 0xA0090000
"""

import os
import sys
import time
import threading
import queue
import csv
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)

from PIL import Image, ImageTk

try:
    from pynq import Overlay, allocate, MMIO
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(SCRIPT_DIR, "finale.bit")):
    # PYNQ deployment: everything in same folder
    WEIGHTS_DIR    = os.path.join(SCRIPT_DIR, "data", "weights")
    RANGES_PATH    = os.path.join(SCRIPT_DIR, "data", "feature_ranges.csv")
    SCENES_DIR     = os.path.join(SCRIPT_DIR, "data", "scenes")
    BITSTREAM_PATH = os.path.join(SCRIPT_DIR, "finale.bit")
else:
    # Dev machine: project-relative paths
    PROJECT_ROOT   = os.path.join(SCRIPT_DIR, "..", "..")
    WEIGHTS_DIR    = os.path.join(PROJECT_ROOT, "training", "weights", "extra", "mlp_7_2_1")
    RANGES_PATH    = os.path.join(PROJECT_ROOT, "training", "weights", "extra", "feature_ranges.csv")
    SCENES_DIR     = os.path.join(PROJECT_ROOT, "dataset", "scenes")
    BITSTREAM_PATH = os.path.join(PROJECT_ROOT, "finale_xsa", "finale.bit")

WINDOW_SIZE = 80
STRIDE = 40
HID_SHIFT = 5
N_HIDDEN = 2

# ---------------------------------------------------------------------------
# AXI Timer registers (pg079)
# ---------------------------------------------------------------------------
AXI_TIMER_BASE = 0xA0070000
AXI_TIMER_RANGE = 0x10000
TIMER_CLK_HZ = 99_999_001  # pl_clk0

TCSR0 = 0x00
TLR0  = 0x04
TCR0  = 0x08

TCSR_LOAD  = 1 << 5
TCSR_ENT   = 1 << 7


class AxiTimer:
    """Simple AXI Timer wrapper using MMIO. Count-up mode, 32-bit."""

    def __init__(self):
        self.mmio = None
        if PYNQ_AVAILABLE:
            try:
                self.mmio = MMIO(AXI_TIMER_BASE, AXI_TIMER_RANGE)
                self._init_timer()
            except Exception as e:
                print(f"Warning: Could not init AXI timer: {e}")
                self.mmio = None

    def _init_timer(self):
        self.mmio.write(TCSR0, 0)
        self.mmio.write(TLR0, 0)
        self.mmio.write(TCSR0, TCSR_LOAD)
        self.mmio.write(TCSR0, 0)

    def start(self):
        if not self.mmio:
            return
        self.mmio.write(TCSR0, 0)
        self.mmio.write(TLR0, 0)
        self.mmio.write(TCSR0, TCSR_LOAD)
        self.mmio.write(TCSR0, TCSR_ENT)

    def stop(self):
        if not self.mmio:
            return 0
        count = self.mmio.read(TCR0)
        self.mmio.write(TCSR0, 0)
        return count

    def ticks_to_us(self, ticks):
        return ticks / (TIMER_CLK_HZ / 1_000_000)

    def ticks_to_ms(self, ticks):
        return ticks / (TIMER_CLK_HZ / 1_000)

    @property
    def available(self):
        return self.mmio is not None


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------
ALL_MODES = [
    "SOFT",
    "HDL_V4_OPT (dual MLP)",
    "HLS_A_packed_v2 (opt)",
    "HDL_V4 (packed+dma)",
    "HLS_A_packed",
    "HLS_B_packed",
    "HLS_A_unpacked",
    "HDL_V3 (packed+axilite)",
]

MODE_CONFIG = {
    "HDL_V4_OPT (dual MLP)":    {"dma": "axi_dma_0", "axilite": None,                    "packed_in": True,  "packed_out": True,  "protocol": "hdl_v4"},
    "HLS_A_packed_v2 (opt)":    {"dma": "axi_dma_1", "axilite": None,                    "packed_in": True,  "packed_out": True,  "protocol": "hls_a_packed"},
    "HDL_V4 (packed+dma)":      {"dma": "axi_dma_2", "axilite": None,                    "packed_in": True,  "packed_out": True,  "protocol": "hdl_v4"},
    "HLS_A_packed":             {"dma": "axi_dma_3", "axilite": None,                    "packed_in": True,  "packed_out": True,  "protocol": "hls_a_packed"},
    "HLS_B_packed":             {"dma": "axi_dma_4", "axilite": "mlp_B_packed_signed_0", "packed_in": True,  "packed_out": True,  "protocol": "hls_b_packed"},
    "HLS_A_unpacked":           {"dma": "axi_dma_5", "axilite": None,                    "packed_in": False, "packed_out": False, "protocol": "hls_a_unpacked"},
    "HDL_V3 (packed+axilite)":  {"dma": "axi_dma_6", "axilite": "hdl_V3_0",              "packed_in": True,  "packed_out": True,  "protocol": "hdl_v3"},
}

# HLS B AXI-Lite register offsets (from Vitis HLS synthesis)
HLS_B_AP_CTRL     = 0x000
HLS_B_BATCH_SIZE  = 0x010
HLS_B_W_OUT_BASE  = 0x020
HLS_B_W_HID_BASE  = 0x040
HLS_B_LUT_BASE    = 0x400

# HDL V3 AXI-Lite register offsets
HDL_V3_W_HID_BASE = 0x000
HDL_V3_W_OUT_BASE = 0x040
HDL_V3_LUT_BASE   = 0x100


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image_bgr)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    g_safe = np.where(g == 0, 1, g).astype(np.float32)
    rg_ratio = np.mean(r.astype(np.float32) / g_safe)
    bg_ratio = np.mean(b.astype(np.float32) / g_safe)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    total_edge_energy = np.sum(np.abs(sobel_h)) + np.sum(np.abs(sobel_v))
    h_edge_ratio = np.sum(np.abs(sobel_h)) / total_edge_energy if total_edge_energy > 0 else 0.5
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_contrast = np.var(laplacian)
    return [mean_intensity, std_intensity, rg_ratio, bg_ratio,
            edge_density, h_edge_ratio, texture_contrast]


def load_weights():
    w_hid = []
    with open(os.path.join(WEIGHTS_DIR, "w_hid.csv")) as f:
        for line in f:
            w_hid.append([int(x) for x in line.strip().split(",")])
    w_hid = np.array(w_hid, dtype=np.int32)
    w_out = []
    with open(os.path.join(WEIGHTS_DIR, "w_out.csv")) as f:
        for line in f:
            w_out.append(int(line.strip()))
    w_out = np.array(w_out, dtype=np.int32)
    sigmoid_lut = []
    with open(os.path.join(WEIGHTS_DIR, "sigmoid.csv")) as f:
        for line in f:
            sigmoid_lut.append(int(line.strip()))
    sigmoid_lut = np.array(sigmoid_lut, dtype=np.int32)
    feat_min, feat_max = [], []
    with open(RANGES_PATH) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            feat_min.append(float(row[1]))
            feat_max.append(float(row[2]))
    return {"w_hid": w_hid, "w_out": w_out, "sigmoid_lut": sigmoid_lut,
            "feat_min": np.array(feat_min), "feat_max": np.array(feat_max)}


def normalize_feature_vector(raw_feats, feat_min, feat_max):
    raw = np.array(raw_feats, dtype=np.float64)
    normed = (raw - feat_min) / (feat_max - feat_min) * 255.0
    return np.clip(np.round(normed), 0, 255).astype(np.int32)


def mlp_predict(x_uint8, w_hid, w_out, sigmoid_lut):
    h = np.zeros(N_HIDDEN, dtype=np.int32)
    for j in range(N_HIDDEN):
        acc = int(w_hid[0, j]) * 255
        for i in range(7):
            acc += int(w_hid[i + 1, j]) * int(x_uint8[i])
        sig_idx = max(0, min(255, (acc >> HID_SHIFT) + 128))
        h[j] = sigmoid_lut[sig_idx]
    acc = int(w_out[0]) * 255
    for j in range(N_HIDDEN):
        acc += int(w_out[j + 1]) * int(h[j])
    return (1 if acc > 0 else 0), acc


# ---------------------------------------------------------------------------
# FPGA helpers
# ---------------------------------------------------------------------------

def _write_hls_b_weights(ip, w_hid, w_out, sigmoid_lut):
    """Write weights to HLS B variant via AXI-Lite. HLS uses 32-bit signed int."""
    for i in range(16):
        r, c = divmod(i, 2)
        ip.write(HLS_B_W_HID_BASE + i * 4, int(w_hid[r, c]) & 0xFFFFFFFF)
    for i in range(3):
        ip.write(HLS_B_W_OUT_BASE + i * 4, int(w_out[i]) & 0xFFFFFFFF)
    for i in range(256):
        ip.write(HLS_B_LUT_BASE + i * 4, int(sigmoid_lut[i]))


def _write_hdl_v3_weights(ip, w_hid, w_out, sigmoid_lut):
    """Write weights to HDL V3 via AXI-Lite. HDL stores low 8 bits only."""
    for i in range(16):
        r, c = divmod(i, 2)
        ip.write(HDL_V3_W_HID_BASE + i * 4, int(w_hid[r, c]) & 0xFF)
    for i in range(3):
        ip.write(HDL_V3_W_OUT_BASE + i * 4, int(w_out[i]) & 0xFF)
    for i in range(256):
        ip.write(HDL_V3_LUT_BASE + i * 4, int(sigmoid_lut[i]) & 0xFF)


def _build_weight_stream(w_hid, w_out, sigmoid_lut):
    """Build 275-word weight stream: 16 w_hid + 3 w_out + 256 LUT (row-major flat)."""
    words = []
    for r in range(8):
        for c in range(2):
            words.append(int(w_hid[r, c]) & 0xFFFFFFFF)
    for i in range(3):
        words.append(int(w_out[i]) & 0xFFFFFFFF)
    for i in range(256):
        words.append(int(sigmoid_lut[i]) & 0xFFFFFFFF)
    return words


def _pack_features(X, n_samples):
    """Pack 7 features into 2 words per sample (4+3 bytes)."""
    words = []
    for s in range(n_samples):
        w0 = (int(X[s,3])<<24)|(int(X[s,2])<<16)|(int(X[s,1])<<8)|int(X[s,0])
        w1 = (int(X[s,6])<<16)|(int(X[s,5])<<8)|int(X[s,4])
        words.append(w0 & 0xFFFFFFFF)
        words.append(w1 & 0xFFFFFFFF)
    return words


def _unpack_features(X, n_samples):
    """Unpack: 1 feature per word, 7 words per sample."""
    words = []
    for s in range(n_samples):
        for i in range(7):
            words.append(int(X[s, i]) & 0xFFFFFFFF)
    return words


def _unpack_results_packed(out_buf, n_samples):
    results = []
    for w in range(math.ceil(n_samples / 4)):
        word = int(out_buf[w])
        for slot in range(4):
            if len(results) >= n_samples:
                break
            results.append((word >> (slot * 8)) & 0xFF)
    return results


def _unpack_results_unpacked(out_buf, n_samples):
    return [int(out_buf[i]) & 0xFF for i in range(n_samples)]


def fpga_inference(overlay, mode, X, n_samples, weights, axi_timer):
    """
    Run inference on FPGA. Returns (predictions, axi_ticks).
    """
    cfg = MODE_CONFIG[mode]
    dma = getattr(overlay, cfg["dma"])
    protocol = cfg["protocol"]
    w_hid, w_out, sigmoid_lut = weights["w_hid"], weights["w_out"], weights["sigmoid_lut"]

    axi_ticks = 0

    if protocol == "hdl_v4":
        # Single DMA transfer: 275 weight words + packed features
        # Used by both HDL V4 (single MLP) and HDL V4 Opt (dual MLP)
        weight_words = _build_weight_stream(w_hid, w_out, sigmoid_lut)
        data_words = _pack_features(X, n_samples)
        in_words = weight_words + data_words
        out_size = math.ceil(n_samples / 4)

    elif protocol == "hdl_v3":
        # Weights via AXI-Lite, only packed features via DMA
        ip = getattr(overlay, cfg["axilite"])
        _write_hdl_v3_weights(ip, w_hid, w_out, sigmoid_lut)
        in_words = _pack_features(X, n_samples)
        out_size = math.ceil(n_samples / 4)

    elif protocol == "hls_a_packed":
        # Two-phase: weight load (mode=0x00 + 275 words, expect 1 ack),
        # then inference (mode=0x01 + batch_size + packed features)
        wl_words = [0x00] + _build_weight_stream(w_hid, w_out, sigmoid_lut)
        wl_in = allocate(shape=(len(wl_words),), dtype=np.uint32)
        wl_out = allocate(shape=(1,), dtype=np.uint32)
        for i, v in enumerate(wl_words): wl_in[i] = v
        dma.recvchannel.transfer(wl_out)
        dma.sendchannel.transfer(wl_in)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        del wl_in, wl_out
        data_words = [0x01, n_samples] + _pack_features(X, n_samples)
        in_words = data_words
        out_size = math.ceil(n_samples / 4)

    elif protocol == "hls_a_unpacked":
        # Two-phase: weight load, then inference with unpacked features
        wl_words = [0x00] + _build_weight_stream(w_hid, w_out, sigmoid_lut)
        wl_in = allocate(shape=(len(wl_words),), dtype=np.uint32)
        wl_out = allocate(shape=(1,), dtype=np.uint32)
        for i, v in enumerate(wl_words): wl_in[i] = v
        dma.recvchannel.transfer(wl_out)
        dma.sendchannel.transfer(wl_in)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        del wl_in, wl_out
        data_words = [0x01, n_samples] + _unpack_features(X, n_samples)
        in_words = data_words
        out_size = n_samples

    elif protocol == "hls_b_packed":
        # Weights via AXI-Lite, set batch_size, ap_start, then DMA packed features
        ip = getattr(overlay, cfg["axilite"])
        _write_hls_b_weights(ip, w_hid, w_out, sigmoid_lut)
        ip.write(HLS_B_BATCH_SIZE, n_samples)
        ip.write(HLS_B_AP_CTRL, 0x01)  # ap_start
        in_words = _pack_features(X, n_samples)
        out_size = math.ceil(n_samples / 4)

    # --- DMA transfer with AXI timer ---
    in_buf = allocate(shape=(len(in_words),), dtype=np.uint32)
    out_buf = allocate(shape=(out_size,), dtype=np.uint32)
    for i, v in enumerate(in_words):
        in_buf[i] = v

    axi_timer.start()

    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.transfer(in_buf)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    axi_ticks = axi_timer.stop()

    if cfg["packed_out"]:
        results = _unpack_results_packed(out_buf, n_samples)
    else:
        results = _unpack_results_unpacked(out_buf, n_samples)

    del in_buf, out_buf
    return results, axi_ticks


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class ShipDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ship Detection (Finale) - EE4218")
        self.root.geometry("1200x800")
        self.root.minsize(900, 650)

        self.image_bgr = None
        self.image_path = None
        self.display_scale = 1.0
        self.tk_image = None
        self.canvas_image_id = None
        self.detection_rects = []
        self.result_queue = queue.Queue()
        self.running = False
        self.overlay = None
        self.axi_timer = None  # created AFTER overlay load
        self.timing_history = {}

        self.weights = None
        try:
            self.weights = load_weights()
        except Exception as e:
            messagebox.showerror("Weight Loading Error", f"Could not load weights:\n{e}")

        # Load overlay FIRST, then create AXI timer (MMIO requires bitstream loaded)
        if PYNQ_AVAILABLE and os.path.exists(BITSTREAM_PATH):
            try:
                self.overlay = Overlay(BITSTREAM_PATH)
                self.axi_timer = AxiTimer()
            except Exception as e:
                print(f"Warning: Could not load overlay: {e}")

        if self.axi_timer is None:
            # Dummy timer (all methods return 0/False)
            self.axi_timer = AxiTimer.__new__(AxiTimer)
            self.axi_timer.mmio = None

        self._build_gui()
        self._start_queue_checker()
        self.root.after(100, self._auto_load_scene)

    def _build_gui(self):
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        ttk.Button(toolbar, text="Load Image", command=self._load_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Run Detection", command=self._run_detection).pack(side=tk.LEFT, padx=3)
        ttk.Label(toolbar, text="  Mode:").pack(side=tk.LEFT, padx=(10, 2))
        self.mode_var = tk.StringVar(value="SOFT")
        available = ["SOFT"] + ([m for m in ALL_MODES if m != "SOFT"] if self.overlay else [])
        self.mode_combo = ttk.Combobox(toolbar, textvariable=self.mode_var,
                                       values=available, state="readonly", width=26)
        self.mode_combo.pack(side=tk.LEFT, padx=3)

        timer_status = "AXI Timer: OK" if self.axi_timer.available else "AXI Timer: N/A"
        ttk.Label(toolbar, text=f"  [{timer_status}]",
                  foreground="green" if self.axi_timer.available else "red").pack(side=tk.LEFT, padx=5)

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas_frame = ttk.Frame(main_pane)
        main_pane.add(canvas_frame, weight=3)
        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", highlightthickness=0)
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        results_frame = ttk.LabelFrame(main_pane, text="Results", padding=10)
        main_pane.add(results_frame, weight=1)

        self.lbl_patches = ttk.Label(results_frame, text="Patches: --")
        self.lbl_patches.pack(anchor=tk.W, pady=2)
        self.lbl_ships = ttk.Label(results_frame, text="Ships detected: --")
        self.lbl_ships.pack(anchor=tk.W, pady=2)

        ttk.Separator(results_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(results_frame, text="Current Run", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
        self.lbl_feat_time = ttk.Label(results_frame, text="Feature extraction: --")
        self.lbl_feat_time.pack(anchor=tk.W, pady=2)
        self.lbl_py_time = ttk.Label(results_frame, text="Python timer (MLP): --")
        self.lbl_py_time.pack(anchor=tk.W, pady=2)
        self.lbl_axi_time = ttk.Label(results_frame, text="AXI timer (HW): --")
        self.lbl_axi_time.pack(anchor=tk.W, pady=2)
        self.lbl_total_time = ttk.Label(results_frame, text="Total (feat+py_mlp): --")
        self.lbl_total_time.pack(anchor=tk.W, pady=2)

        ttk.Separator(results_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(results_frame, text="Timing Comparison", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)

        cols = ("mode", "py_ms", "axi_ms", "axi_us", "samples")
        self.timing_tree = ttk.Treeview(results_frame, columns=cols, show="headings", height=9)
        self.timing_tree.heading("mode", text="Mode")
        self.timing_tree.heading("py_ms", text="Py (ms)")
        self.timing_tree.heading("axi_ms", text="AXI (ms)")
        self.timing_tree.heading("axi_us", text="AXI (us)")
        self.timing_tree.heading("samples", text="N")
        self.timing_tree.column("mode", width=180)
        self.timing_tree.column("py_ms", width=65, anchor=tk.E)
        self.timing_tree.column("axi_ms", width=65, anchor=tk.E)
        self.timing_tree.column("axi_us", width=70, anchor=tk.E)
        self.timing_tree.column("samples", width=45, anchor=tk.E)
        self.timing_tree.pack(fill=tk.X, pady=2)

        ttk.Separator(results_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        self.lbl_image_info = ttk.Label(results_frame, text="Image: none loaded", wraplength=280)
        self.lbl_image_info.pack(anchor=tk.W, pady=2)

        bottom = ttk.Frame(self.root, padding=5)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)
        self.progress = ttk.Progressbar(bottom, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 10))
        self.lbl_status = ttk.Label(bottom, text="Ready", width=50, anchor=tk.W)
        self.lbl_status.pack(side=tk.RIGHT)

    def _start_queue_checker(self):
        try:
            while True:
                msg = self.result_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.root.after(50, self._start_queue_checker)

    def _handle_message(self, msg):
        kind = msg[0]
        if kind == "status":
            self.lbl_status.config(text=msg[1])
        elif kind == "progress":
            self.progress["value"] = msg[1]
        elif kind == "done":
            self._show_results(msg[1])
            self.running = False
        elif kind == "error":
            messagebox.showerror("Detection Error", msg[1])
            self.running = False
            self.lbl_status.config(text="Error")

    def _auto_load_scene(self):
        scenes_dir = os.path.normpath(SCENES_DIR)
        if os.path.isdir(scenes_dir):
            files = sorted([f for f in os.listdir(scenes_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))])
            if files:
                self._load_image_file(os.path.join(scenes_dir, files[0]))

    def _load_image(self):
        init_dir = os.path.normpath(SCENES_DIR) if os.path.isdir(SCENES_DIR) else "."
        path = filedialog.askopenfilename(title="Select scene", initialdir=init_dir,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All", "*.*")])
        if path:
            self._load_image_file(path)

    def _load_image_file(self, path):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not load: {path}")
            return
        self.image_bgr = img
        self.image_path = path
        h, w = img.shape[:2]
        if h < WINDOW_SIZE or w < WINDOW_SIZE:
            messagebox.showwarning("Too Small", f"{w}x{h}, min {WINDOW_SIZE}x{WINDOW_SIZE}")
            return
        for rid in self.detection_rects:
            self.canvas.delete(rid)
        self.detection_rects.clear()
        self._display_image(img)
        n_x = (w - WINDOW_SIZE) // STRIDE + 1
        n_y = (h - WINDOW_SIZE) // STRIDE + 1
        self.lbl_image_info.config(text=f"Image: {os.path.basename(path)}\n{w}x{h}, {n_x*n_y} patches")
        self.lbl_patches.config(text=f"Patches: {n_x*n_y}")
        self.lbl_ships.config(text="Ships detected: --")
        self.lbl_status.config(text="Image loaded. Press Run Detection.")
        self.progress["value"] = 0

    def _display_image(self, img_bgr):
        h, w = img_bgr.shape[:2]
        self.canvas.update_idletasks()
        cw, ch = max(self.canvas.winfo_width(), 400), max(self.canvas.winfo_height(), 400)
        scale = min(cw / w, ch / h, 1.0)
        self.display_scale = scale
        nw, nh = int(w * scale), int(h * scale)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((nw, nh), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, nw, nh))

    def _run_detection(self):
        if self.running:
            return
        if self.image_bgr is None:
            messagebox.showwarning("No Image", "Load an image first.")
            return
        if self.weights is None:
            messagebox.showerror("No Weights", "Weights not loaded.")
            return
        mode = self.mode_var.get()
        if mode != "SOFT" and not self.overlay:
            messagebox.showinfo("No FPGA", "No overlay loaded. Use SOFT mode.")
            return
        self.running = True
        for rid in self.detection_rects:
            self.canvas.delete(rid)
        self.detection_rects.clear()
        self._display_image(self.image_bgr)
        t = threading.Thread(target=self._detection_worker, args=(mode,), daemon=True)
        t.start()

    def _detection_worker(self, mode):
        try:
            q = self.result_queue
            img = self.image_bgr
            h, w = img.shape[:2]
            weights = self.weights

            n_x = (w - WINDOW_SIZE) // STRIDE + 1
            n_y = (h - WINDOW_SIZE) // STRIDE + 1
            total_patches = n_x * n_y

            q.put(("status", f"Extracting features ({mode})..."))
            q.put(("progress", 5))

            t_feat_start = time.perf_counter()
            X_all = np.zeros((total_patches, 7), dtype=np.int32)
            patch_coords = []
            idx = 0
            for iy in range(n_y):
                for ix in range(n_x):
                    px, py = ix * STRIDE, iy * STRIDE
                    patch = img[py:py+WINDOW_SIZE, px:px+WINDOW_SIZE]
                    raw = extract_features(patch)
                    normed = normalize_feature_vector(raw, weights["feat_min"], weights["feat_max"])
                    X_all[idx] = normed
                    patch_coords.append((px, py))
                    idx += 1
                    if idx % 50 == 0:
                        q.put(("progress", 5 + int(idx / total_patches * 70)))
                        q.put(("status", f"Features ({idx}/{total_patches})..."))
            t_feat_end = time.perf_counter()
            feat_time = t_feat_end - t_feat_start

            q.put(("status", f"MLP inference ({mode})..."))
            q.put(("progress", 80))

            total_axi_ticks = 0
            t_mlp_start = time.perf_counter()

            if mode == "SOFT":
                predictions = []
                w_hid, w_out, slut = weights["w_hid"], weights["w_out"], weights["sigmoid_lut"]
                for idx in range(total_patches):
                    pred, _ = mlp_predict(X_all[idx], w_hid, w_out, slut)
                    predictions.append(pred)
                    if idx % 200 == 0:
                        q.put(("progress", 80 + int(idx / total_patches * 18)))
            else:
                MAX_BATCH = 64
                predictions = []
                for start in range(0, total_patches, MAX_BATCH):
                    end = min(start + MAX_BATCH, total_patches)
                    chunk = X_all[start:end]
                    chunk_results, ticks = fpga_inference(
                        self.overlay, mode, chunk, end - start, weights, self.axi_timer)
                    predictions.extend(chunk_results)
                    total_axi_ticks += ticks
                    q.put(("progress", 80 + int(end / total_patches * 18)))

            t_mlp_end = time.perf_counter()
            py_mlp_time = t_mlp_end - t_mlp_start
            total_time = feat_time + py_mlp_time

            axi_us = self.axi_timer.ticks_to_us(total_axi_ticks) if total_axi_ticks > 0 else 0
            axi_ms = axi_us / 1000.0

            detections = [(px, py) for i, (px, py) in enumerate(patch_coords)
                          if predictions[i] == 1]

            q.put(("progress", 100))
            q.put(("status", "Done"))
            q.put(("done", {
                "mode": mode,
                "detections": detections,
                "total_patches": total_patches,
                "n_ships": len(detections),
                "feat_time": feat_time,
                "py_mlp_ms": py_mlp_time * 1000,
                "axi_ms": axi_ms,
                "axi_us": axi_us,
                "total_time": total_time,
            }))
        except Exception as e:
            import traceback
            q.put(("error", f"{e}\n\n{traceback.format_exc()}"))

    def _show_results(self, r):
        for rid in self.detection_rects:
            self.canvas.delete(rid)
        self.detection_rects.clear()
        self._display_image(self.image_bgr)
        scale = self.display_scale
        for (px, py) in r["detections"]:
            dx, dy = px * scale, py * scale
            dw, dh = WINDOW_SIZE * scale, WINDOW_SIZE * scale
            rid = self.canvas.create_rectangle(dx, dy, dx+dw, dy+dh, outline="#00ff00", width=2)
            self.detection_rects.append(rid)

        self.lbl_patches.config(text=f"Patches: {r['total_patches']}")
        self.lbl_ships.config(text=f"Ships detected: {r['n_ships']}")
        self.lbl_feat_time.config(text=f"Feature extraction: {r['feat_time']:.2f}s")
        self.lbl_py_time.config(text=f"Python timer (MLP): {r['py_mlp_ms']:.2f} ms")
        if r["axi_us"] > 0:
            self.lbl_axi_time.config(text=f"AXI timer (HW): {r['axi_us']:.1f} us ({r['axi_ms']:.3f} ms)")
        else:
            self.lbl_axi_time.config(text="AXI timer (HW): N/A (SOFT mode)")
        self.lbl_total_time.config(text=f"Total: {r['total_time']:.2f}s")
        self.lbl_status.config(text=f"Done ({r['mode']}) - {r['n_ships']} ships")

        mode = r["mode"]
        self.timing_history[mode] = r
        for item in self.timing_tree.get_children():
            self.timing_tree.delete(item)
        for m, data in sorted(self.timing_history.items()):
            py_ms = f"{data['py_mlp_ms']:.2f}"
            axi_ms = f"{data['axi_ms']:.3f}" if data['axi_us'] > 0 else "--"
            axi_us = f"{data['axi_us']:.1f}" if data['axi_us'] > 0 else "--"
            n = str(data['total_patches'])
            self.timing_tree.insert("", tk.END, values=(m, py_ms, axi_ms, axi_us, n))


def main():
    root = tk.Tk()
    app = ShipDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
