import os
import subprocess
import tempfile
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from ttkthemes import ThemedTk
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from faster_whisper import WhisperModel

# ---------------- Configuration ----------------

# Silence Detection
MIN_SILENCE_LEN_MS = 300   # Minimum silence duration (ms)
SILENCE_THRESH_DB = -45    # Threshold for silence (dB)
SPEECH_PAD_MS = 100        # Padding around speech chunks
SPEECH_GROUP_GAP_MS = 200  # Gap threshold to split speech groups

# Music Detection
MUSIC_MIN_LEN_SEC = 2.0

# Subtitle Formatting
MAX_CHARS_PER_LINE = 60
MAX_LINES_PER_SUB = 2

# Whisper Config
WHISPER_MODEL = "large-v3"
if torch.cuda.is_available():
    COMPUTE_DEVICE = "cuda"
else:
    messagebox.showerror("CUDA Error", "CUDA not available. Please run on a system with GPU support. For Nvidia Download and Install CuDnn")
    raise SystemExit(1)


# ---------------- Helper Functions ----------------

def format_timestamp(seconds):
    """Convert seconds to SRT time format."""
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def is_music(y, sr):
    """Heuristic to check if a segment is likely music."""
    if len(y) < 2048:
        return False
    y_harmonic, _ = librosa.effects.hpss(y)
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y_harmonic))
    onset_env = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    return spectral_flatness < 0.05 and len(onset_env) > 5


# ---------------- Core Processing ----------------

class SubtitleProcessor:
    def __init__(self, video_path, progress_queue, whisper_model):
        self.video_path = video_path
        self.progress_queue = progress_queue
        self.whisper_model = whisper_model
        self.base_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_srt_path = os.path.join(os.path.dirname(video_path), self.base_name + ".srt")
        self.temp_audio_file = None

    def update_progress(self, message, value=None):
        self.progress_queue.put(("progress", (self.video_path, message, value)))

    def run(self):
        try:
            if os.path.exists(self.output_srt_path):
                self.update_progress("Skipping (SRT already exists)", 100)
                return

            # Step 1: Extract Audio
            self.update_progress("Extracting audio...", 5)
            self.temp_audio_file = self.extract_audio()

            # Step 2: Analyze
            self.update_progress("Analyzing audio...", 20)
            audio = AudioSegment.from_wav(self.temp_audio_file)
            y, sr = librosa.load(self.temp_audio_file, sr=None)
            speech_groups, music_regions = self.analyze_audio_regions(audio, y, sr)

            if not speech_groups:
                self.write_srt_file([], music_regions)
                self.update_progress("No speech detected.", 100)
                return

            # Step 3: Whisper Transcription
            self.update_progress("Transcribing speech...", 40)
            all_realigned_segments = []

            for i, group in enumerate(speech_groups):
                self.update_progress(f"Transcribing {i + 1}/{len(speech_groups)}",
                                     40 + int(50 * (i + 1) / len(speech_groups)))

                # Combine audio
                group_audio = sum(audio[start:end] for start, end in group)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    speech_audio_path = f.name
                group_audio.export(speech_audio_path, format="wav")

                segments, _ = self.whisper_model.transcribe(speech_audio_path, word_timestamps=True)

                realigned = self.realign_whisper_timestamps(list(segments), group)
                all_realigned_segments.extend(realigned)
                os.remove(speech_audio_path)

            # Step 4: Write Subtitles
            self.update_progress("Writing subtitles...", 95)
            self.write_srt_file(all_realigned_segments, music_regions)
            self.update_progress("Done ✅", 100)

        except Exception as e:
            self.update_progress(f"Error: {e}", -1)
        finally:
            if self.temp_audio_file and os.path.exists(self.temp_audio_file):
                os.remove(self.temp_audio_file)

    def extract_audio(self):
        temp_dir = tempfile.gettempdir()
        temp_audio_path = os.path.join(temp_dir, self.base_name + ".wav")
        command = [
            'ffmpeg', '-i', self.video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', '-y', temp_audio_path
        ]
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.run(command, check=True, capture_output=True, text=True, startupinfo=startupinfo)
        return temp_audio_path

    def analyze_audio_regions(self, audio, y, sr):
        non_silent_chunks = detect_nonsilent(
            audio, min_silence_len=MIN_SILENCE_LEN_MS,
            silence_thresh=SILENCE_THRESH_DB, seek_step=1
        )
        speech_regions, music_regions = [], []

        for start_ms, end_ms in non_silent_chunks:
            start_ms = max(0, start_ms - SPEECH_PAD_MS)
            end_ms = min(len(audio), end_ms + SPEECH_PAD_MS)

            duration = (end_ms - start_ms) / 1000.0
            start_sample = int(start_ms * sr / 1000)
            end_sample = int(end_ms * sr / 1000)
            chunk = y[start_sample:end_sample]

            if is_music(chunk, sr) and duration >= MUSIC_MIN_LEN_SEC:
                rms_energy = librosa.feature.rms(y=chunk)
                if np.mean(rms_energy) < 0.01:
                    music_regions.append((start_ms / 1000.0, end_ms / 1000.0))
                else:
                    speech_regions.append((start_ms, end_ms))
            else:
                speech_regions.append((start_ms, end_ms))

        if not speech_regions:
            return [], music_regions

        speech_regions.sort(key=lambda x: x[0])
        grouped_speech_regions, current_group = [], [speech_regions[0]]

        for i in range(1, len(speech_regions)):
            gap = speech_regions[i][0] - current_group[-1][1]
            if gap < SPEECH_GROUP_GAP_MS:
                current_group.append(speech_regions[i])
            else:
                grouped_speech_regions.append(current_group)
                current_group = [speech_regions[i]]
        grouped_speech_regions.append(current_group)

        return grouped_speech_regions, music_regions

    def realign_whisper_timestamps(self, segments, speech_regions):
        realigned = []
        durations = [(end - start) / 1000.0 for start, end in speech_regions]
        cumulative = np.cumsum([0] + durations)

        for seg in segments:
            if not seg.text.strip():
                continue
            start, end = seg.start, seg.end
            idx = np.searchsorted(cumulative, start, side='right') - 1
            if idx < 0 or idx >= len(speech_regions):
                continue
            offset_start = start - cumulative[idx]
            offset_end = end - cumulative[idx]
            orig_start = (speech_regions[idx][0] / 1000.0) + offset_start
            orig_end = (speech_regions[idx][0] / 1000.0) + offset_end
            realigned.append({'text': seg.text, 'start': orig_start, 'end': orig_end, 'words': seg.words})
        return realigned

    def write_srt_file(self, segments, music_regions):
        items = []
        for seg in segments:
            items.append({'start': seg['start'], 'end': seg['end'], 'text': seg['text'].strip()})
        for s, e in music_regions:
            items.append({'start': s, 'end': e, 'text': '♪'})
        items.sort(key=lambda x: x['start'])

        with open(self.output_srt_path, 'w', encoding='utf-8') as f:
            count = 1
            for item in items:
                if not item['text']:
                    continue
                words = item['text'].split()
                lines, current_line = [], ""

                if item['text'] == '♪':
                    lines = ['♪']
                else:
                    for word in words:
                        if len(current_line) + len(word) + 1 > MAX_CHARS_PER_LINE:
                            lines.append(current_line)
                            current_line = ""
                        current_line += f" {word}"
                    lines.append(current_line.strip())

                final_lines, temp_line = [], ""
                for i, line in enumerate(lines):
                    temp_line += " " + line if temp_line else line
                    if (i + 1) % MAX_LINES_PER_SUB == 0 or (i + 1) == len(lines):
                        final_lines.append(temp_line)
                        temp_line = ""

                for line_group in final_lines:
                    f.write(f"{count}\n")
                    f.write(f"{format_timestamp(item['start'])} --> {format_timestamp(item['end'])}\n")
                    f.write(line_group.strip() + "\n\n")
                    count += 1


# ---------------- GUI ----------------

class SubtitleGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Subtitle Generator (GPU) by Wakkyb")
        self.root.geometry("900x650")
        self.file_list = []
        self.progress_queue = queue.Queue()
        self.processing_thread = None

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Controls
        ctrl = ttk.Frame(main_frame)
        ctrl.pack(fill=tk.X, pady=8)
        ttk.Label(ctrl, text="Select videos to generate subtitles:", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=5)
        self.file_button = ttk.Button(ctrl, text="📂 Files", command=self.select_files)
        self.file_button.pack(side=tk.LEFT, padx=5)
        self.folder_button = ttk.Button(ctrl, text="🗂 Folder", command=self.select_folder)
        self.folder_button.pack(side=tk.LEFT, padx=5)

        # Status list
        list_frame = ttk.LabelFrame(main_frame, text="Processing Status", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.canvas = tk.Canvas(list_frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.file_widgets = {}

        # Overall progress
        self.overall_progress = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate', length=400)
        self.overall_progress.pack(fill=tk.X, pady=5)
        self.check_queue()

    def select_files(self):
        ftypes = [("Video Files", "*.mp4 *.mkv *.avi *.mov *.webm"), ("All files", "*.*")]
        fnames = filedialog.askopenfilenames(title="Select Video Files", filetypes=ftypes)
        if fnames:
            self.start_processing(list(fnames))

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Video Folder")
        if folder:
            exts = ('.mp4', '.mkv', '.avi', '.mov', '.webm')
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
            if not files:
                messagebox.showinfo("No Videos Found", "No supported video files in folder.")
                return
            self.start_processing(files)

    def start_processing(self, files):
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("In Progress", "A batch is already running.")
            return

        self.toggle_buttons(False)
        self.file_list = files
        self.cleanup_ui()
        self.overall_progress['maximum'] = len(files)
        self.overall_progress['value'] = 0

        for fp in files:
            base = os.path.basename(fp)
            row = ttk.Frame(self.scrollable_frame)
            row.pack(fill=tk.X, padx=5, pady=4)
            row.columnconfigure(0, weight=3)
            row.columnconfigure(1, weight=2)
            lbl = ttk.Label(row, text=f"{base}: Waiting...", anchor="w")
            lbl.grid(row=0, column=0, sticky="ew", padx=5)
            pbar = ttk.Progressbar(row, orient='horizontal', mode='determinate')
            pbar.grid(row=0, column=1, sticky="ew", padx=5)
            self.file_widgets[fp] = {'label': lbl, 'pbar': pbar}

        self.processing_thread = threading.Thread(target=self.process_files_worker, daemon=True)
        self.processing_thread.start()

    def process_files_worker(self):
        try:
            whisper_model = WhisperModel(WHISPER_MODEL, device=COMPUTE_DEVICE, compute_type="auto")
            for i, fp in enumerate(self.file_list):
                proc = SubtitleProcessor(fp, self.progress_queue, whisper_model)
                proc.run()
                self.progress_queue.put(("file_done", (fp, i + 1)))
            self.progress_queue.put(("all_done", None))
        except Exception as e:
            self.progress_queue.put(("progress", ("GLOBAL", f"Fatal error: {e}", -1)))
            self.progress_queue.put(("all_done", None))

    def check_queue(self):
        try:
            while True:
                mtype, data = self.progress_queue.get_nowait()
                if mtype == "progress":
                    fp, msg, val = data
                    if fp in self.file_widgets:
                        w = self.file_widgets[fp]
                        w['label'].config(text=f"{os.path.basename(fp)}: {msg}")
                        if val is not None:
                            if val == -1:
                                w['pbar']['style'] = 'red.Horizontal.TProgressbar'
                                w['pbar']['value'] = 100
                            else:
                                w['pbar']['value'] = val
                                w['pbar']['style'] = 'green.Horizontal.TProgressbar'
                elif mtype == "file_done":
                    _, count = data
                    self.overall_progress['value'] = count
                elif mtype == "all_done":
                    self.toggle_buttons(True)
                    messagebox.showinfo("Complete", "All files processed.")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)

    def cleanup_ui(self):
        for w in self.scrollable_frame.winfo_children():
            w.destroy()
        self.file_widgets.clear()

    def toggle_buttons(self, state):
        status = tk.NORMAL if state else tk.DISABLED
        self.file_button.config(state=status)
        self.folder_button.config(state=status)


def main():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True,
                       startupinfo=(subprocess.STARTUPINFO(
                           dwFlags=subprocess.STARTF_USESHOWWINDOW) if os.name == 'nt' else None))
    except (subprocess.CalledProcessError, FileNotFoundError):
        messagebox.showerror("Dependency Error", "FFmpeg not found in PATH.")
        return

    root = ThemedTk(theme="equilux")  # modern dark theme
    style = ttk.Style(root)

    style.configure("green.Horizontal.TProgressbar", troughcolor="#2e2e2e", background="#4CAF50")
    style.configure("red.Horizontal.TProgressbar", troughcolor="#2e2e2e", background="#F44336")

    app = SubtitleGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
