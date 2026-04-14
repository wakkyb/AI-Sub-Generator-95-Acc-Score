
# 🎬 GPU Subtitle Generator — Quickstart Guide

> **Automatically create subtitles for your videos using your NVIDIA GPU!**
> Fast • Accurate • Simple — powered by **Whisper AI** and **Faster-Whisper**

---

## 💡 What This App Does

This app listens to your video’s audio, detects where people speak, and generates **accurate subtitles (SRT files)** automatically.
It runs completely on your computer using your **NVIDIA GPU** for **super-fast transcription** — no internet required.

🟢 **You can:**
✅ Add subtitles to movies, lectures, YouTube videos, interviews, etc.
✅ Get perfect timing and speech/music separation
✅ Create standard `.srt` subtitle files instantly

🔴 **You need:**
An **NVIDIA GPU** — the app will not run on CPU.

---

## 🧰 What You Need Before Starting

| Requirement                      | Description                     | Download / Check                                                                |
| -------------------------------- | ------------------------------- | ------------------------------------------------------------------------------- |
| 🖥️ **NVIDIA GPU**               | Required for GPU acceleration   | [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)                    |
| ⚙️ **CUDA Toolkit 12.1**         | Lets your GPU run AI code       | [CUDA 12.1 Download](https://developer.nvidia.com/cuda-12-1-0-download-archive) |
| ⚡ **cuDNN 8.9 for CUDA 12.x**    | Deep learning performance boost | [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)                 |
| 🐍 **Python 3.9 or later**       | Required to run the program     | [Python.org](https://www.python.org/downloads/)                                 |
| 🎵 **FFmpeg**                    | Extracts audio from videos      | [FFmpeg.org](https://ffmpeg.org/download.html)                                  |
| 🧩 **Required Python libraries** | Used by the program             | Installed via `pip` (shown below)                                               |

---

## 🪜 Step-by-Step Installation Guide

### 🥇 Step 1: Check Your GPU Driver

1. Make sure your NVIDIA drivers are up to date.
2. Open **---Command Prompt---** and type:

   ---Command Prompt---
   
   ```
   nvidia-smi

If you see driver information — you’re good to go ✅

---

### 🥈 Step 2: Install CUDA Toolkit 12.1

1. Go to the official NVIDIA download page:
   👉 [CUDA 12.1 Download](https://developer.nvidia.com/cuda-12-1-0-download-archive)
2. Choose your operating system (Windows, Linux, etc.).
3. Download and run the **Local Installer**.
4. During setup, make sure these boxes are checked:

   * [x] CUDA Toolkit
   * [x] Runtime Libraries
   * [x] Developer Drivers

After installation, open **---Command Prompt---** and verify:

   ---Command Prompt---
   
         nvcc --version

You should see something like `release 12.1`.



### 🥉 Step 3: Install cuDNN (for CUDA 12.x)

1. Go to: 👉 [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)
2. Choose **cuDNN 8.9.x for CUDA 12.x**
3. Download the ZIP file for your system
4. Extract it and copy its contents into your CUDA folder (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\`)

**You’ll need to copy:**

* `bin` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`
* `lib` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib`
* `include` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include`

5. Restart your computer.

---

### 🧱 Step 4: Install Python (V 3.10.11)

* Download from [python.org/downloads](https://www.python.org/downloads/)
* During installation:

  * ✅ Check **“Add Python to PATH”**
  * Click **Install Now**

To verify Python:

---Command Prompt---

      python --version


### 🪄 Step 5: Install the Required Libraries

1. Open ---Command Prompt--- where you saved the subtitle generator script

2. Create a virtual environment (optional but recommended):

   ---Command Prompt---

  

   ```
   python -m venv venv
   venv\Scripts\activate

4. Install dependencies:

   ---Command Prompt---
      ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install faster-whisper numpy librosa pydub ttkthemes


5. Confirm GPU support:

   ---Command Prompt---
  
   ```
    python -c "import torch; print(torch.cuda.is_available())"

✅ If it says `True`, CUDA is ready!

---

### 🎞️ Step 6: Install FFmpeg

* Download from [FFmpeg.org](https://ffmpeg.org/download.html)
* Extract it (e.g., to `C:\ffmpeg`)
* Add its `bin` folder (e.g., `C:\ffmpeg\bin`) to your **PATH**

To verify:

---Command Prompt---
   
      ffmpeg -version
---

## 🚀 How to Run the App

1. Open ---Command Prompt--- in the folder containing your script
2. Run:

   ---Command Prompt---
  
   ```
    python subtitle_generator.py
3. The app window will open 🎉

🟢 **To use it:**

* Click **“File(s)…”** to pick one or more videos, or
* Click **“Folder…”** to process an entire folder

The program will:

1. Extract audio
2. Detect speech and silence
3. Transcribe speech with Whisper
4. Save subtitles next to your videos (e.g., `myvideo.srt`)

You’ll see progress bars for each file as they’re processed.

---

## 🧩 Common Problems & Fixes

| Problem                      | Fix                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| ❌ “CUDA not available”       | Check CUDA Toolkit and cuDNN installation. Verify with `torch.cuda.is_available()` |
| ⚠️ “ffmpeg not found”        | Add FFmpeg’s `bin` folder to PATH                                                  |
| 💤 App closes after one file | Ensure CUDA was detected — the script stops if no GPU found                        |
| 🔴 “Error: cuDNN not found”  | Make sure cuDNN 8.9 files are copied into your CUDA folder                         |
| 🟠 Subtitles are empty       | Check that the video actually has speech and clear audio                           |

---

## 💬 Tips

* Large videos → need more VRAM (GPU memory)
* For faster processing, close heavy GPU programs (like games)
* You can edit this program in any text editor like notepad (on windows). just open the .py file with a text editor using the "open with" option from mouse right click.
* You can also change the model version from large-v3 (line 32) --> small/base. A complete list of models can be found @ https://huggingface.co/collections/Systran/faster-whisper-6867ecec0e757ee14896e2d3
* The `.srt` files work in any media player (VLC, YouTube, etc.)

---

## 🔗 Helpful Links

* 🔹 [CUDA 12.1 Toolkit](https://developer.nvidia.com/cuda-12-1-0-download-archive)
* 🔹 [cuDNN 8.9 Archive](https://developer.nvidia.com/rdp/cudnn-archive)
* 🔹 [FFmpeg Downloads](https://ffmpeg.org/download.html)
* 🔹 [Python Downloads](https://www.python.org/downloads/)
* 🔹 [Faster-Whisper GitHub](https://github.com/SYSTRAN/faster-whisper)

---

## ❤️ Credits

Developed by Wakkyb (devwakky@gmail.com) with 🧠 Whisper AI + ⚡ CUDA GPU acceleration
Built for anyone who wants **fast, offline, free, accurate subtitles**.

---


