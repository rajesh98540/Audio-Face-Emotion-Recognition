"""Desktop interface for the SkillBarter emotion-recognition demo."""

from __future__ import annotations

import queue
import shutil
import threading
import traceback
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Button, Canvas, Entry, Frame, Label, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk

import source.config as config
from source.face_emotion_utils import predict as face_predict
from source.audio_analysis_utils import predict as audio_predict


ROOT = Path(config.MAIN_PATH)
INPUT_DIR = Path(config.INPUT_FOLDER_PATH)
OUTPUT_DIR = Path(config.OUTPUT_FOLDER_PATH)


class EmotionApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Emotion Recognition Studio")
        self.root.geometry("1020x700")
        self.root.minsize(860, 600)
        self.root.configure(bg="#f4f7fb")

        self.file_path = StringVar()
        self.mode = StringVar(value="Face image")
        self.status = StringVar(value="Choose an image to begin.")
        self.result = StringVar(value="No prediction yet")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.preview: ImageTk.PhotoImage | None = None
        self.running = False

        self._build_style()
        self._build_layout()
        self.root.after(100, self._drain_events)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Title.TLabel", background="#f4f7fb", foreground="#17233c", font=("Segoe UI", 24, "bold"))
        style.configure("Subtitle.TLabel", background="#f4f7fb", foreground="#64748b", font=("Segoe UI", 10))
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#17233c", font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background="#ffffff", foreground="#475569", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), foreground="#ffffff", background="#2563eb")
        style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")])

    def _build_layout(self) -> None:
        header = Frame(self.root, bg="#f4f7fb")
        header.pack(fill=X, padx=32, pady=(26, 18))
        ttk.Label(header, text="Emotion Recognition Studio", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Analyze facial expressions from the bundled deep-learning models.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(5, 0))

        content = Frame(self.root, bg="#f4f7fb")
        content.pack(fill=BOTH, expand=True, padx=32, pady=(0, 25))

        controls = ttk.Frame(content, style="Card.TFrame", padding=22)
        controls.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 12))
        ttk.Label(controls, text="Run a prediction", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            controls,
            text="Select a mode and choose a file. Face image mode produces the emotion and confidence immediately.",
            style="Body.TLabel",
            wraplength=420,
        ).pack(anchor="w", pady=(8, 20))

        ttk.Label(controls, text="Analysis mode", style="Body.TLabel").pack(anchor="w")
        mode_box = ttk.Combobox(
            controls,
            textvariable=self.mode,
            values=("Face image", "Audio"),
            state="readonly",
        )
        mode_box.pack(fill=X, pady=(6, 14))
        mode_box.bind("<<ComboboxSelected>>", lambda _event: self._mode_changed())

        ttk.Label(controls, text="Selected file", style="Body.TLabel").pack(anchor="w")
        file_row = Frame(controls, bg="#ffffff")
        file_row.pack(fill=X, pady=(6, 14))
        self.file_entry = ttk.Entry(file_row, textvariable=self.file_path)
        self.file_entry.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(file_row, text="Browse", command=self._browse).pack(side=RIGHT, padx=(8, 0))

        button_row = Frame(controls, bg="#ffffff")
        button_row.pack(fill=X, pady=(5, 20))
        self.run_button = ttk.Button(button_row, text="Run prediction", style="Accent.TButton", command=self._start)
        self.run_button.pack(side=LEFT)
        ttk.Button(button_row, text="Clear", command=self._clear).pack(side=LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Refresh", command=self._refresh).pack(side=LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Open outputs", command=self._open_outputs).pack(side=RIGHT)

        ttk.Label(controls, text="Result", style="Body.TLabel").pack(anchor="w")
        self.result_label = Label(
            controls,
            textvariable=self.result,
            bg="#eff6ff",
            fg="#1e3a8a",
            font=("Segoe UI", 17, "bold"),
            anchor="w",
            padx=14,
            pady=13,
        )
        self.result_label.pack(fill=X, pady=(6, 14))

        ttk.Label(controls, text="Activity", style="Body.TLabel").pack(anchor="w")
        self.log = ScrolledText(controls, height=10, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.pack(fill=BOTH, expand=True)

        preview_card = ttk.Frame(content, style="Card.TFrame", padding=18)
        preview_card.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))
        ttk.Label(preview_card, text="Preview", style="CardTitle.TLabel").pack(anchor="w")
        self.preview_canvas = Canvas(preview_card, bg="#e8eef7", highlightthickness=0)
        self.preview_canvas.pack(fill=BOTH, expand=True, pady=(12, 12))
        ttk.Label(preview_card, textvariable=self.status, style="Body.TLabel", wraplength=400).pack(anchor="w")

        self._mode_changed()

    def _mode_changed(self) -> None:
        mode = self.mode.get()
        if mode == "Face image":
            self.status.set("Supported formats: PNG, JPG, JPEG, BMP.")
        else:
            self.status.set("Supported formats: WAV, MP3, M4A. Audio must be placed in the project input folder.")

    def _browse(self) -> None:
        patterns = {
            "Face image": [("Images", "*.png *.jpg *.jpeg *.bmp")],
            "Audio": [("Audio", "*.wav *.mp3 *.m4a")],
        }
        selected = filedialog.askopenfilename(
            title=f"Choose {self.mode.get().lower()}",
            filetypes=patterns[self.mode.get()] + [("All files", "*.*")],
        )
        if selected:
            self.file_path.set(selected)
            self._show_preview(Path(selected))

    def _show_preview(self, path: Path) -> None:
        if self.mode.get() == "Audio":
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                300, 240, text=f"Audio file\n{path.name}", fill="#475569", font=("Segoe UI", 16), justify="center"
            )
            return
        try:
            image = Image.open(path)
            image.thumbnail((520, 460))
            self.preview = ImageTk.PhotoImage(image)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(300, 240, image=self.preview)
        except Exception:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(300, 240, text=f"Video file\n{path.name}", fill="#475569", font=("Segoe UI", 16))

    def _start(self) -> None:
        if self.running:
            return
        path = Path(self.file_path.get().strip().strip('"'))
        if not path.is_file():
            messagebox.showwarning("Choose a file", "Please select a valid input file first.")
            return

        self.running = True
        self.run_button.configure(state="disabled")
        self.result.set("Running...")
        self.status.set("The model is processing your file. The interface will remain responsive.")
        self._log(f"Starting {self.mode.get()} analysis for {path.name}")
        threading.Thread(target=self._worker, args=(path, self.mode.get()), daemon=True).start()

    def _worker(self, path: Path, mode: str) -> None:
        try:
            if mode == "Face image":
                result = face_predict.predict(str(path), imshow=False, grad_cam=False, verbose=False)
                self.events.put(("result", f"{result[0]}  |  {result[2][result[1]] * 100:.2f}% confidence"))
                self.events.put(("log", f"Prediction complete: {result[0]}"))
            else:
                INPUT_DIR.mkdir(parents=True, exist_ok=True)
                destination = INPUT_DIR / path.name
                if path.resolve() != destination.resolve():
                    shutil.copy2(path, destination)
                result = audio_predict.predict(path.name)
                self.events.put(("result", f"{result[0]}  |  {result[1] * 100:.2f}% confidence"))
                self.events.put(("log", f"Prediction complete: {result[0]}"))
        except Exception as exc:
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))
            self.events.put(("log", traceback.format_exc()))
        finally:
            self.events.put(("done", None))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "result":
                    self.result.set(str(value))
                    self.status.set("Prediction finished.")
                elif kind == "error":
                    self.result.set("Prediction failed")
                    self.status.set("See Activity for the error details.")
                    messagebox.showerror("Prediction error", str(value))
                elif kind == "log":
                    self._log(str(value))
                elif kind == "done":
                    self.running = False
                    self.run_button.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(END, text.rstrip() + "\n")
        self.log.see(END)
        self.log.configure(state="disabled")

    def _clear(self) -> None:
        if self.running:
            messagebox.showinfo(
                "Prediction in progress",
                "Please wait for the current prediction to finish before clearing the form.",
            )
            return

        self.file_path.set("")
        self.result.set("No prediction yet")
        self.status.set("Choose an image to begin.")
        self.preview_canvas.delete("all")
        self.run_button.configure(state="normal")
        self._log("Cleared.")

    def _refresh(self) -> None:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        selected = Path(self.file_path.get().strip().strip('"'))
        if selected.is_file():
            self._show_preview(selected)
            self.status.set(f"Refreshed input files. Selected: {selected.name}")
            self._log(f"Refreshed input files ({len(list(INPUT_DIR.iterdir()))} entries).")
            return

        self.file_path.set("")
        self.result.set("No prediction yet")
        self.preview_canvas.delete("all")
        self.status.set("Input files refreshed. Choose a file to begin.")
        self._log(f"Refreshed input files ({len(list(INPUT_DIR.iterdir()))} entries).")

    def _open_outputs(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import os
            os.startfile(str(OUTPUT_DIR))
        except OSError as exc:
            messagebox.showerror("Could not open outputs", str(exc))


if __name__ == "__main__":
    root = Tk()
    EmotionApp(root)
    root.mainloop()
