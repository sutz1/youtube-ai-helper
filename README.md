# 🎙️ YouTube Podcast Assistant

This Python application transcribes podcast audio (focused on *The Finals* video game), generates YouTube-ready titles, descriptions, and timestamps, and saves them for upload — all locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [Ollama](https://ollama.com/).

## 🚀 Features

- 🎧 Transcribes long podcast episodes using `faster-whisper` (GPU-accelerated Whisper model).
- 🧠 Summarizes and chunks transcriptions into manageable parts.
- ✍️ Generates:
  - A **clickable title**
  - A **YouTube description** (4–6 sentences)
  - **Timestamps** with labels
- ⚡ Runs locally via [Ollama](https://ollama.com/) using models like `llama3.2`.

---

## 🛠 Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (e.g., RTX 3080 Ti)
- [Ollama](https://ollama.com/download) installed and running
- Models:
  - `faster-whisper` (large-v3)
  - Ollama model: `llama3.2` or similar

### Python Dependencies

Install them with:

```bash
pip install -r requirements.txt
```

**`requirements.txt` Example:**
```txt
faster-whisper
ollama
```

---

## 📁 Project Structure

```
.
├── src/
│   └── app.py
│   └── prompts.py
│   └── your_audio_file.mp3
├── .gitignore
├── transcription_output.txt
├── chunked_outputs.txt
├── final_youtube_summary.txt
└── README.md
```

---

## 📦 prompts.py

Create a `prompts.py` file in the root directory:

```python
# prompts.py
previous_description = """
In this episode, we break down the latest update for The Finals, discuss balance changes, and speculate on future content. Featuring commentary from top creators in the scene.
"""
```

This gives your prompt a consistent tone without reusing content.

---

## 🧪 How to Run

First, it is recommended to work in a virtual environment to avoid conflicts.
Create a virtual environment and install dependencies:

Windows:
```Powershell
python3 -m venv .venv
 .\.venv\Scripts\activate
pip install -r requirements.txt
```
Mac/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



Place your `.mp3` file in the `src/` directory and update the filename in `main.py`:

```python
AUDIO_FILE_PATH = r"<your relative audio file path here>"
```

Then run the script:

```bash
python .\src\app.py
```

---

## 🧠 Model Info

### Transcription
Uses `faster-whisper` with the `"large-v3"` model on GPU (`float16`).

### Language Model
Use Ollama with a local model like:

```bash
ollama run llama3.2
```

You can change the model in `generate_from_chunks()` and `merge_chunk_summaries()` functions.

---

## 📤 Output Files

- `transcription_output.txt`: Full transcript with timestamps
- `chunked_outputs.txt`: Per-chunk generation logs (useful for review/debug)
- `final_youtube_summary.txt`: Clean final output (title, description, timestamps)

---

## 🧩 Future Enhancements

- CLI support for file input
- Better formatting of timestamps
- GUI with file drag-and-drop
- Automatic YouTube upload integration

---

## 🧑‍💻 Author

Created by a developer passionate about gaming and automation. Uses local models for full control and privacy.
---