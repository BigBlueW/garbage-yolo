"""Desktop GUI for testing the current detector and grasp-order model on images."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = {
    0: "Plastic",
    1: "Metal",
    2: "Paper",
    3: "General Waste",
}


class ImageTestGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("垃圾 OBB 與夾取順序測試工具")
        self.root.geometry("1380x900")
        self.root.minsize(1000, 700)

        self.image_path: Path | None = None
        self.folder_images: list[Path] = []
        self.folder_index = -1
        self.photo: ImageTk.PhotoImage | None = None
        self.running = False
        self.last_output: Path | None = None

        self.detector_var = tk.StringVar(value=str(PROJECT_ROOT / "best_incremental_v2.pt"))
        self.order_var = tk.StringVar(value=str(PROJECT_ROOT / "order_runs" / "order_model_grouped_v3_best.pt"))
        self.det_conf_var = tk.DoubleVar(value=0.55)
        self.relation_conf_var = tk.DoubleVar(value=0.80)
        self.risk_floor_var = tk.DoubleVar(value=0.45)
        self.device_var = tk.StringVar(value="cuda")
        self.show_uncertain_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="請選擇圖片，或直接使用自動載入的最新照片。")
        self.file_var = tk.StringVar(value="尚未選擇圖片")

        self._build_ui()
        self.root.after(100, self._load_initial_image)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill=tk.X)

        row1 = ttk.Frame(controls)
        row1.pack(fill=tk.X, pady=(0, 7))
        ttk.Button(row1, text="選擇圖片", command=self.choose_image).pack(side=tk.LEFT)
        ttk.Button(row1, text="上一張", command=lambda: self.navigate(-1)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(row1, text="下一張", command=lambda: self.navigate(1)).pack(side=tk.LEFT, padx=(6, 0))
        self.run_button = ttk.Button(row1, text="開始辨識", command=self.run_prediction)
        self.run_button.pack(side=tk.LEFT, padx=(14, 0))
        ttk.Button(row1, text="開啟結果資料夾", command=self.open_output_folder).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(row1, textvariable=self.file_var).pack(side=tk.LEFT, padx=14, fill=tk.X, expand=True)

        models = ttk.LabelFrame(controls, text="模型", padding=8)
        models.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(models, text="OBB：").grid(row=0, column=0, sticky="w")
        ttk.Entry(models, textvariable=self.detector_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(models, text="瀏覽", command=lambda: self.choose_model(self.detector_var)).grid(row=0, column=2)
        ttk.Label(models, text="順序：").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(models, textvariable=self.order_var).grid(row=1, column=1, sticky="ew", padx=5, pady=(5, 0))
        ttk.Button(models, text="瀏覽", command=lambda: self.choose_model(self.order_var)).grid(row=1, column=2, pady=(5, 0))
        models.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(controls, text="測試設定", padding=8)
        settings.pack(fill=tk.X)
        ttk.Label(settings, text="偵測門檻").grid(row=0, column=0, sticky="w")
        ttk.Scale(settings, from_=0.20, to=0.90, variable=self.det_conf_var, command=self._update_values).grid(row=0, column=1, sticky="ew", padx=5)
        self.det_value = ttk.Label(settings, width=5)
        self.det_value.grid(row=0, column=2)
        ttk.Label(settings, text="順序門檻").grid(row=0, column=3, sticky="w", padx=(16, 0))
        ttk.Scale(settings, from_=0.50, to=0.95, variable=self.relation_conf_var, command=self._update_values).grid(row=0, column=4, sticky="ew", padx=5)
        self.rel_value = ttk.Label(settings, width=5)
        self.rel_value.grid(row=0, column=5)
        ttk.Label(settings, text="風險門檻").grid(row=0, column=6, sticky="w", padx=(16, 0))
        ttk.Scale(settings, from_=0.20, to=0.80, variable=self.risk_floor_var, command=self._update_values).grid(row=0, column=7, sticky="ew", padx=5)
        self.risk_value = ttk.Label(settings, width=5)
        self.risk_value.grid(row=0, column=8)
        ttk.Label(settings, text="裝置").grid(row=0, column=9, padx=(16, 4))
        ttk.Combobox(settings, textvariable=self.device_var, values=("cuda", "auto", "cpu"), width=7, state="readonly").grid(row=0, column=10)
        ttk.Checkbutton(settings, text="顯示灰色不確定線", variable=self.show_uncertain_var).grid(row=0, column=11, padx=(14, 0))
        for column in (1, 4, 7):
            settings.columnconfigure(column, weight=1)
        self._update_values()

        content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        image_frame = ttk.LabelFrame(content, text="圖片預覽", padding=4)
        self.canvas = tk.Canvas(image_frame, background="#202020", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._redraw_image())
        content.add(image_frame, weight=4)

        result_frame = ttk.LabelFrame(content, text="完整夾取順序", padding=6)
        self.tree = ttk.Treeview(result_frame, columns=("rank", "class", "layer", "score", "risk"), show="headings", height=18)
        headings = {
            "rank": ("順序", 55),
            "class": ("類別", 120),
            "layer": ("層級", 55),
            "score": ("分數", 70),
            "risk": ("風險", 70),
        }
        for key, (label, width) in headings.items():
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.summary = tk.Text(result_frame, height=12, width=45, wrap=tk.WORD, state=tk.DISABLED)
        self.summary.pack(fill=tk.X, pady=(8, 0))
        content.add(result_frame, weight=1)

        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=5)
        status.pack(fill=tk.X, padx=10, pady=(0, 10))

    def _update_values(self, _value: str | None = None) -> None:
        self.det_value.configure(text=f"{self.det_conf_var.get():.2f}")
        self.rel_value.configure(text=f"{self.relation_conf_var.get():.2f}")
        self.risk_value.configure(text=f"{self.risk_floor_var.get():.2f}")

    def _load_initial_image(self) -> None:
        candidates: list[Path] = []
        for folder in (
            PROJECT_ROOT / "capture_inbox",
            PROJECT_ROOT / "robotic-garbage-obb-detector-v2" / "test" / "images",
        ):
            if folder.is_dir():
                candidates.extend(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
            if candidates:
                break
        if candidates:
            self.set_image(max(candidates, key=lambda path: path.stat().st_mtime))

    def choose_image(self) -> None:
        initial = self.image_path.parent if self.image_path else PROJECT_ROOT / "capture_inbox"
        path = filedialog.askopenfilename(
            title="選擇要測試的圖片",
            initialdir=str(initial if initial.exists() else PROJECT_ROOT),
            filetypes=(("圖片", "*.jpg *.jpeg *.png *.webp *.bmp"), ("所有檔案", "*.*")),
        )
        if path:
            self.set_image(Path(path))

    def choose_model(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="選擇模型權重",
            initialdir=str(PROJECT_ROOT),
            filetypes=(("PyTorch 權重", "*.pt"), ("所有檔案", "*.*")),
        )
        if path:
            variable.set(path)

    def set_image(self, path: Path) -> None:
        path = path.resolve()
        self.image_path = path
        self.folder_images = sorted(
            (item for item in path.parent.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda item: item.name.lower(),
        )
        try:
            self.folder_index = self.folder_images.index(path)
        except ValueError:
            self.folder_index = -1
        self.file_var.set(f"{path.name}  ({self.folder_index + 1}/{len(self.folder_images)})")
        self.last_output = None
        self._show_image(path)
        self._clear_results()
        self.status_var.set("已載入圖片；按『開始辨識』。")

    def navigate(self, delta: int) -> None:
        if self.running or not self.folder_images or self.folder_index < 0:
            return
        new_index = max(0, min(len(self.folder_images) - 1, self.folder_index + delta))
        self.set_image(self.folder_images[new_index])

    def _show_image(self, path: Path) -> None:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("無法開啟圖片", str(exc))
            return
        self._source_pil = image
        self._redraw_image()

    def _redraw_image(self) -> None:
        image = getattr(self, "_source_pil", None)
        if image is None:
            return
        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 100)
        scale = min(width / image.width, height / image.height)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resized = image.resize(size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self.photo, anchor=tk.CENTER)

    def _clear_results(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._set_summary("")

    def _set_summary(self, text: str) -> None:
        self.summary.configure(state=tk.NORMAL)
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, text)
        self.summary.configure(state=tk.DISABLED)

    def run_prediction(self) -> None:
        if self.running:
            return
        if self.image_path is None:
            messagebox.showwarning("尚未選圖", "請先選擇一張圖片。")
            return
        detector = Path(self.detector_var.get())
        order_model = Path(self.order_var.get())
        missing = [str(path) for path in (detector, order_model) if not path.is_file()]
        if missing:
            messagebox.showerror("找不到模型", "\n".join(missing))
            return

        output_dir = PROJECT_ROOT / "order_runs" / "gui_tests"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = output_dir / f"{self.image_path.stem}_{stamp}.jpg"
        json_path = output.with_suffix(".json")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "predict_grasp_sequence.py"),
            "--image", str(self.image_path),
            "--detector", str(detector),
            "--order-model", str(order_model),
            "--det-conf", f"{self.det_conf_var.get():.3f}",
            "--relation-conf", f"{self.relation_conf_var.get():.3f}",
            "--uncertain-risk-floor", f"{self.risk_floor_var.get():.3f}",
            "--device", self.device_var.get(),
            "--output", str(output),
            "--json", str(json_path),
        ]
        if self.show_uncertain_var.get():
            command.append("--show-uncertain")

        self.running = True
        self.run_button.configure(state=tk.DISABLED)
        self.status_var.set("正在使用 GPU 辨識，請稍候……")
        threading.Thread(
            target=self._prediction_worker,
            args=(command, output, json_path),
            daemon=True,
        ).start()

    def _prediction_worker(self, command: list[str], output: Path, json_path: Path) -> None:
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
        self.root.after(0, self._prediction_finished, result, output, json_path)

    def _prediction_finished(self, result: subprocess.CompletedProcess[str], output: Path, json_path: Path) -> None:
        self.running = False
        self.run_button.configure(state=tk.NORMAL)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "未知錯誤").strip()
            if "No objects were found" in detail:
                self.last_output = None
                self._clear_results()
                if self.image_path is not None:
                    self._show_image(self.image_path)
                self._set_summary(
                    "未偵測到物品\n\n"
                    "桌面可能已清空，不需要夾取。\n"
                    "如果畫面其實有物品，可降低偵測門檻後再試一次。"
                )
                self.status_var.set("未偵測到物品：桌面已清空，不需要夾取。")
                return
            self.status_var.set("辨識失敗。")
            messagebox.showerror("辨識失敗", detail[-3000:])
            return
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self._populate_results(payload)
            self._show_image(output)
        except Exception as exc:
            self.status_var.set("結果已產生，但讀取結果時發生錯誤。")
            messagebox.showerror("結果讀取失敗", str(exc))
            return
        self.last_output = output
        count = len(payload.get("sequence", []))
        uncertain = len(payload.get("uncertain_pairs", []))
        self.status_var.set(f"完成：偵測到 {count} 個物品，不確定配對 {uncertain} 組。結果已另存，不會修改原圖。")

    def _populate_results(self, payload: dict) -> None:
        self._clear_results()
        for item in payload.get("sequence", []):
            class_name = CLASS_NAMES.get(int(item["class_id"]), str(item["class_id"]))
            self.tree.insert("", tk.END, values=(
                item["rank"],
                class_name,
                item["layer"],
                f"{float(item['score']):.3f}",
                f"{float(item.get('uncertainty_risk', 0.0)):.3f}",
            ))
        relations = payload.get("relations", [])
        uncertain = payload.get("uncertain_pairs", [])
        first = payload.get("sequence", [{}])[0] if payload.get("sequence") else {}
        first_class = CLASS_NAMES.get(int(first.get("class_id", -1)), str(first.get("class_id", "-")))
        lines = [
            f"下一個應夾：順序 1（物件 #{payload.get('next_object_id', '-')}，{first_class}）",
            f"正式箭頭：{len(relations)} 條",
            f"不確定配對：{len(uncertain)} 組",
            "",
            "黃線方向代表：先夾 → 後夾",
            "同一層沒有強制先後，數字只是安全性排序。",
        ]
        self._set_summary("\n".join(lines))

    def open_output_folder(self) -> None:
        folder = PROJECT_ROOT / "order_runs" / "gui_tests"
        folder.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(folder)])


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    ImageTestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
