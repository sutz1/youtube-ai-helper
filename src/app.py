import os
import sys
from pathlib import Path
from faster_whisper import WhisperModel, BatchedInferencePipeline
from prompts import previous_description
import ollama

# ----------------- CUDA PATH SETUP (for Windows + GPU use) -----------------
def set_cuda_paths():
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path = nvidia_base_path / 'cuda_runtime' / 'bin'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    paths_to_add = [str(cuda_path), str(cublas_path), str(cudnn_path)]
    env_vars = ['CUDA_PATH', 'CUDA_PATH_V12_4', 'PATH']
    
    for env_var in env_vars:
        current_value = os.environ.get(env_var, '')
        new_value = os.pathsep.join(paths_to_add + [current_value] if current_value else paths_to_add)
        os.environ[env_var] = new_value

# ----------------- TRANSCRIPTION -----------------
def transcribe_audio(audio_path, output_path="transcription_output.txt", model_size="large-v3"):
    set_cuda_paths()
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        batched_model = BatchedInferencePipeline(model=model)
        segments, info = batched_model.transcribe(audio_path, batch_size=16)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Detected language: {}\n".format(info.language))
            f.write("Language probability: {}\n\n".format(info.language_probability))
            for segment in segments:
                f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
        
        print(f"✅ Transcription completed. Output saved to {output_path}")
    except Exception as e:
        print(f"❌ Error during transcription: {e}")

# ----------------- CHUNKING -----------------
def split_text_into_chunks(text, max_words=1000):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ----------------- OLLAMA GENERATION -----------------
def generate_from_chunks(full_transcript_text, model="gemma3:4b"):
    chunks = split_text_into_chunks(full_transcript_text)
    all_outputs = []

    for i, chunk in enumerate(chunks):
        print(f"🔍 Processing chunk {i+1}/{len(chunks)}...")

        prompt = f"""
You are a YouTube assistant. A new episode of a podcast about the video game The Finals has been transcribed.
Use the text below to create the following:
- A brief summary of what this chunk was about.
- Provide **timestamps** for each segment of the podcast in minute:second format (e.g. 00:01) with short labels.
⚠️ Do NOT reuse anything from this example — it is only for tone reference:
"{previous_description}"

--- Begin Transcript Chunk ---
{chunk}
--- End Transcript Chunk ---
"""

        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
        )

        content = response['message']['content']
        all_outputs.append((i + 1, content))
    
    return all_outputs

# ----------------- OPTIONAL FINAL SUMMARY -----------------
def merge_chunk_summaries(chunks_outputs, model="gemma3:4b"):
    combined = "\n\n".join([f"Chunk {i}:\n{output}" for i, output in chunks_outputs])

    final_prompt = f"""
Based on the following chunked summaries of a podcast, please combine them into a single:
1. YouTube **Title**
2. **Description**
3. **timestamps** with short labels in minute :second format (e.g. 00:01) for each segment of the podcast. The time stamps should continue off the previous ones provided.

Here are the chunks:
{combined}
"""

    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': final_prompt}],
        stream=False,
    )

    return response['message']['content']

# ----------------- MAIN PIPELINE -----------------
def run_pipeline(audio_path):
    print("🗣️ Starting transcription...")
    """ transcribe_audio(audio_path) """
    
    print(f"📄 Transcription complete. Reading from transcription_out.txt")
    # Step 2: Read and chunk
    with open("transcription_output.txt", "r", encoding="utf-8") as f:
        transcript_text = f.read()
        print("📜 Transcript loaded.")

    print("🔍 Splitting transcript into chunks...")
    chunk_outputs = generate_from_chunks(transcript_text)

    print("📝 Saving chunk outputs...")
    with open("chunked_outputs.txt", "w", encoding="utf-8") as f:
        for i, output in chunk_outputs:
            f.write(f"\n--- CHUNK {i} ---\n\n{output}\n")

    print("🧠 Generating final summary...")
    final_summary = merge_chunk_summaries(chunk_outputs)

    with open("final_youtube_summary.txt", "w", encoding="utf-8") as f:
        f.write(final_summary)
    
    print("✅ All steps complete. Outputs saved to 'chunked_outputs.txt' and 'final_youtube_summary.txt'")

# ----------------- ENTRYPOINT -----------------
if __name__ == "__main__":
    AUDIO_FILE_PATH = r"src\Is The Finals the next big ESPORT_ Feat_ @WaltzCasts _ Around the Arena EP 4 _ The Finals Podcast.mp3"
    run_pipeline(AUDIO_FILE_PATH)