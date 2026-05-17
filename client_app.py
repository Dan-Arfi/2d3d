#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import json
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)
import websockets

from protocol import make_message

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEngineSettings
except Exception:  # noqa: BLE001
    QWebEngineView = None
    QWebEngineSettings = None


SERVER_URL = "ws://127.0.0.1:8765"


@dataclass
class UiEvent:
    typ: str
    payload: dict


class SocketWorker:
    def __init__(self, out_queue: Queue[UiEvent]) -> None:
        self.out_queue = out_queue
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect_and_listen())

    async def _connect_and_listen(self) -> None:
        async with websockets.connect(SERVER_URL, max_size=10_000_000) as ws:
            self.ws = ws
            self.out_queue.put(UiEvent("connected", {}))
            async for raw in ws:
                data = json.loads(raw)
                self.out_queue.put(UiEvent("message", data))

    def send(self, msg: str) -> None:
        if self.loop is None or self.ws is None:
            return
        asyncio.run_coroutine_threadsafe(self.ws.send(msg), self.loop)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3D Reconstruction Client App")
        self.resize(1320, 900)
        self.setMinimumSize(980, 680)
        self.queue: Queue[UiEvent] = Queue()
        self.worker = SocketWorker(self.queue)
        self.frame_labels: list[QLabel] = []
        self.current_model_id = ""
        self._last_result_html_path: str | None = None
        self.current_user: dict | None = None

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack)

        self._build_auth_page()
        self._build_library_page()
        self._build_editor_page()
        self._build_history_page()

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_queue)
        self.poll_timer.start(100)

        self.worker.start()
        self.stack.setCurrentIndex(0)
        self.auth_page_status.setText(f"Connecting to {SERVER_URL} ...")

    def _build_auth_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()

        box = QGroupBox("Authentication Required")
        box_l = QVBoxLayout(box)
        box_l.addWidget(QLabel("Please login or register to continue."))

        self.auth_user = QLineEdit()
        self.auth_user.setPlaceholderText("username")
        box_l.addWidget(self.auth_user)

        self.auth_pass = QLineEdit()
        self.auth_pass.setPlaceholderText("password")
        self.auth_pass.setEchoMode(QLineEdit.EchoMode.Password)
        box_l.addWidget(self.auth_pass)

        btn_row = QHBoxLayout()
        self.login_btn = QPushButton("Login")
        self.login_btn.clicked.connect(self.login)
        btn_row.addWidget(self.login_btn)
        self.register_btn = QPushButton("Register")
        self.register_btn.clicked.connect(self.register)
        btn_row.addWidget(self.register_btn)
        box_l.addLayout(btn_row)

        self.auth_page_status = QLabel("Not logged in")
        box_l.addWidget(self.auth_page_status)
        layout.addWidget(box)
        layout.addStretch()

        self.stack.addWidget(page)

    def _build_library_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)

        header = QGroupBox("Library")
        header_l = QHBoxLayout(header)
        layout.addWidget(header)

        self.split_combo = QComboBox()
        self.split_combo.addItems(["val", "train"])
        self.split_combo.currentTextChanged.connect(self.refresh_library)
        header_l.addWidget(QLabel("Split"))
        header_l.addWidget(self.split_combo)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search model id...")
        self.search_box.returnPressed.connect(self.refresh_library)
        header_l.addWidget(QLabel("Filter"))
        header_l.addWidget(self.search_box)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_library)
        header_l.addWidget(refresh_btn)

        header_l.addStretch()
        self.logout_btn = QPushButton("Logout")
        self.logout_btn.clicked.connect(self.logout)
        header_l.addWidget(self.logout_btn)
        self.history_btn = QPushButton("History")
        self.history_btn.clicked.connect(self.go_history)
        header_l.addWidget(self.history_btn)

        self.library_list = QListWidget()
        self.library_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.library_list.setIconSize(QPixmap(128, 128).size())
        self.library_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.library_list.setGridSize(QPixmap(170, 170).size())
        self.library_list.setMovement(QListWidget.Movement.Static)
        self.library_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.library_list.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.library_list.setSpacing(8)
        self.library_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.library_list.itemDoubleClicked.connect(self.open_model_from_library)
        layout.addWidget(self.library_list)

        self.library_status = QLabel("Loading library...")
        layout.addWidget(self.library_status)
        self.auth_status = QLabel("Not logged in")
        layout.addWidget(self.auth_status)

        self.stack.addWidget(page)

    def _build_editor_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)

        top = QGroupBox("Selected Model")
        top_l = QHBoxLayout(top)
        layout.addWidget(top)

        self.back_btn = QPushButton("Back To Library")
        self.back_btn.clicked.connect(self.go_library)
        top_l.addWidget(self.back_btn)

        self.model_label = QLabel("No model selected")
        top_l.addWidget(self.model_label)
        top_l.addStretch()

        self.view_mode = QComboBox()
        self.view_mode.addItems(["mask_edge", "regular", "binary", "compare"])
        top_l.addWidget(QLabel("View mode"))
        top_l.addWidget(self.view_mode)

        self.load_btn = QPushButton("Load Views")
        self.load_btn.clicked.connect(self.load_views)
        top_l.addWidget(self.load_btn)

        infer_box = QGroupBox("Inference")
        infer_l = QHBoxLayout(infer_box)
        layout.addWidget(infer_box)

        self.res_spin = QSpinBox()
        self.res_spin.setRange(32, 96)
        self.res_spin.setSingleStep(16)
        self.res_spin.setValue(64)
        infer_l.addWidget(QLabel("Resolution"))
        infer_l.addWidget(self.res_spin)

        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.3, 3.0)
        self.thresh_spin.setSingleStep(0.1)
        self.thresh_spin.setValue(1.0)
        infer_l.addWidget(QLabel("Threshold mult"))
        infer_l.addWidget(self.thresh_spin)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["height", "confidence"])
        infer_l.addWidget(QLabel("Color mode"))
        infer_l.addWidget(self.color_combo)

        self.show_gt = QCheckBox("Show GT")
        self.show_gt.setChecked(True)
        infer_l.addWidget(self.show_gt)

        self.run_btn = QPushButton("Run Reconstruction")
        self.run_btn.clicked.connect(self.run_infer)
        infer_l.addWidget(self.run_btn)

        grid_box = QGroupBox("Views")
        grid = QGridLayout(grid_box)
        layout.addWidget(grid_box)
        for i in range(12):
            lbl = QLabel("No frame")
            lbl.setFixedSize(170, 170)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("border:1px solid #444; background:#111; color:#aaa;")
            lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            grid.addWidget(lbl, i // 6, i % 6)
            self.frame_labels.append(lbl)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log)

        self.preview = None
        if QWebEngineView is not None:
            self.preview = QWebEngineView()
            self.preview.setMinimumHeight(0)
            self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.preview.loadFinished.connect(self.on_preview_load_finished)
            if QWebEngineSettings is not None:
                self.preview.page().settings().setAttribute(
                    QWebEngineSettings.WebAttribute.ShowScrollBars, False
                )
            layout.addWidget(self.preview)
            self.preview.setFixedHeight(800)
        else:
            self.append_log("Qt WebEngine not available. Will open results in browser.")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)

    def append_log(self, text: str) -> None:
        if hasattr(self, "log"):
            self.log.append(text)

    def _update_auth_ui(self) -> None:
        if self.current_user:
            text = f"מחובר כ: {self.current_user['username']}"
        else:
            text = "לא מחובר"
        self.auth_status.setText(text)
        self.auth_page_status.setText(text)

    def login(self) -> None:
        username = self.auth_user.text().strip()
        password = self.auth_pass.text()
        self.worker.send(make_message("login", username=username, password=password))

    def register(self) -> None:
        username = self.auth_user.text().strip()
        password = self.auth_pass.text()
        self.worker.send(make_message("register", username=username, password=password))

    def logout(self) -> None:
        self.worker.send(make_message("logout"))

    def refresh_library(self) -> None:
        split = self.split_combo.currentText()
        search = self.search_box.text().strip()
        self.library_status.setText("Loading...")
        self.worker.send(make_message("library", split=split, search=search))

    def open_model_from_library(self, item: QListWidgetItem) -> None:
        model_id = item.data(Qt.ItemDataRole.UserRole)
        if not model_id:
            return
        self.current_model_id = model_id
        self.model_label.setText(f"Model: {model_id}")
        self.stack.setCurrentIndex(2)
        self.load_views()

    def go_library(self) -> None:
        self.stack.setCurrentIndex(1)

    def go_history(self) -> None:
        self.stack.setCurrentIndex(3)
        self.history_status.setText("Loading history...")
        self.worker.send(make_message("history"))

    def load_views(self) -> None:
        if not self.current_model_id:
            return
        self.append_log(f"Loading views for {self.current_model_id}")
        self.worker.send(
            make_message("load_views", model_id=self.current_model_id, view_mode=self.view_mode.currentText())
        )

    def run_infer(self) -> None:
        if not self.current_model_id:
            return
        self.append_log(f"Running inference for {self.current_model_id}")
        self.worker.send(
            make_message(
                "infer",
                model_id=self.current_model_id,
                resolution=int(self.res_spin.value()),
                threshold_mult=float(self.thresh_spin.value()),
                color_mode=self.color_combo.currentText(),
                show_gt=bool(self.show_gt.isChecked()),
            )
        )

    def poll_queue(self) -> None:
        while True:
            try:
                ev = self.queue.get_nowait()
            except Empty:
                break
            if ev.typ == "connected":
                self.library_status.setText("Connected.")
                self._update_auth_ui()
                self.stack.setCurrentIndex(0)
                continue
            if ev.typ == "message":
                self.on_message(ev.payload)

    def on_message(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "status":
            status_msg = msg.get("message", "status")
            self.library_status.setText(status_msg)
            self.auth_status.setText(status_msg)
            self.auth_page_status.setText(status_msg)
        elif t == "library":
            items = msg.get("items", [])
            self.library_list.clear()
            for entry in items:
                model_id = entry.get("model_id", "")
                thumb = entry.get("thumb", "")
                if not model_id or not thumb:
                    continue
                raw = base64.b64decode(thumb)
                pix = QPixmap()
                pix.loadFromData(raw, "PNG")
                list_item = QListWidgetItem(QIcon(pix), model_id[:16])
                list_item.setData(Qt.ItemDataRole.UserRole, model_id)
                self.library_list.addItem(list_item)
            self.library_status.setText(f"Loaded {self.library_list.count()} models.")
        elif t == "views":
            frames = msg.get("frames", [])
            for i, lbl in enumerate(self.frame_labels):
                if i < len(frames):
                    raw = base64.b64decode(frames[i])
                    pm = QPixmap()
                    pm.loadFromData(raw, "PNG")
                    lbl.setPixmap(
                        pm.scaled(
                            lbl.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                else:
                    lbl.setText("No frame")
                    lbl.setPixmap(QPixmap())
            self.append_log(f"Received {len(frames)} frames.")
        elif t == "progress":
            self.append_log(f"Progress {msg.get('pct', 0)}% - {msg.get('message', '')}")
        elif t == "result":
            html_path = msg.get("html_path", "")
            pred_points = msg.get("pred_points", 0)
            gt_voxels = msg.get("gt_voxels", 0)
            self.append_log(f"Done. pred_points={pred_points}, gt_voxels={gt_voxels}")
            if html_path:
                if self.preview is not None:
                    self._last_result_html_path = html_path
                    self.preview.setUrl(QUrl.fromLocalFile(html_path))
                    self.append_log(f"Rendered in-app: {html_path}")
                else:
                    self.append_log(f"Opening: {html_path}")
                    webbrowser.open(f"file://{html_path}")
        elif t == "error":
            self.append_log(f"ERROR: {msg.get('message', 'unknown')}")
            self.history_status.setText(msg.get("message", "Error"))
        elif t == "auth_result":
            ok = bool(msg.get("ok", False))
            action = str(msg.get("action", "auth"))
            text = str(msg.get("message", ""))
            if action == "login" and ok:
                self.current_user = msg.get("user", None)
            elif action == "logout" and ok:
                self.current_user = None
            self._update_auth_ui()
            self.library_status.setText(text)
            self.auth_status.setText(text)
            self.auth_page_status.setText(text)
            if action == "login" and ok:
                self.go_library()
                self.refresh_library()
            if action == "logout" and ok:
                self.stack.setCurrentIndex(0)
        elif t == "history_result":
            ok = bool(msg.get("ok", False))
            rows = msg.get("rows", [])
            self.history_table.setRowCount(0)
            if ok:
                self.history_status.setText(str(msg.get("message", "ok")))
                for r in rows:
                    row_idx = self.history_table.rowCount()
                    self.history_table.insertRow(row_idx)
                    vals = [
                        str(r.get("created_at", "")),
                        str(r.get("model_id", "")),
                        str(r.get("resolution", "")),
                        f"{float(r.get('threshold_mult', 0)):.2f}" if r.get("threshold_mult") is not None else "",
                        str(r.get("pred_points", "")),
                        str(r.get("gt_voxels", "")),
                    ]
                    for ci, v in enumerate(vals):
                        self.history_table.setItem(row_idx, ci, QTableWidgetItem(v))
            else:
                self.history_status.setText(str(msg.get("message", "history error")))

    def on_preview_load_finished(self, ok: bool) -> None:
        if self.preview is None:
            return
        if ok:
            # Remove inner web scrollbars and resize the web view to full document height.
            self.preview.page().runJavaScript(
                "document.documentElement.style.overflow='hidden';"
                "document.body.style.overflow='hidden';"
            )
            self.preview.page().runJavaScript(
                "Math.max("
                "document.body.scrollHeight,"
                "document.documentElement.scrollHeight,"
                "document.body.offsetHeight,"
                "document.documentElement.offsetHeight"
                ");",
                self._apply_preview_height,
            )
            return
        if not self._last_result_html_path:
            return
        try:
            html_path = Path(self._last_result_html_path)
            html = html_path.read_text(encoding="utf-8")
            self.preview.setHtml(html, baseUrl=QUrl.fromLocalFile(str(html_path.parent) + "/"))
            self.append_log("Preview URL load failed, used inline HTML fallback.")
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"Preview fallback failed: {exc}")

    def _apply_preview_height(self, raw_height: object) -> None:
        if self.preview is None:
            return
        try:
            h = int(float(raw_height))
        except Exception:  # noqa: BLE001
            h = 900
        h = max(700, h + 24)
        self.preview.setFixedHeight(h)

    def _build_history_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)

        top = QGroupBox("History")
        top_l = QHBoxLayout(top)
        layout.addWidget(top)
        back = QPushButton("Back To Library")
        back.clicked.connect(self.go_library)
        top_l.addWidget(back)
        refresh = QPushButton("Refresh History")
        refresh.clicked.connect(lambda: self.worker.send(make_message("history")))
        top_l.addWidget(refresh)
        top_l.addStretch()

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(
            ["Time", "Model", "Resolution", "Threshold", "Pred Points", "GT Voxels"]
        )
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.history_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.history_table)

        self.history_status = QLabel("History page")
        layout.addWidget(self.history_status)
        self.stack.addWidget(page)


def main() -> None:
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
