# Sbobbinator

**Automated High-Quality Audio Transcription & Revision Tool**

---

## About

**"Sbobbinare"** is a playful Italian word university students use to describe the tedious task of listening to lecture recordings and writing down everything by hand to create clear notes, especially when no textbook is available. Sbobbinator makes this process easy and automatic. Just provide an audio file or a YouTube link—Sbobbinator transcribes it and turns it into polished, textbook-quality notes using LLMs.  

Originally designed to help students automatically transcribe and revise university lecture recordings, Sbobbinator can work with any audio content—lectures, podcasts, interviews, and more. You can also easily adjust the prompt used by the language model to suit any specific transcription needs you have.

---

## How It Works

Sbobbinator leverages:

- **Whisper/WhisperX** or **AssemblyAI** for audio transcription.
- **OpenAI GPT** to rewrite transcripts with improved grammar, style, and clarity, turning them into polished notes.
- **yt-dlp** to download and extract audio from YouTube videos.

The tool performs these steps:

1. **Audio Extraction:**  
   Downloads audio from YouTube videos (to an `./audio/` folder) or processes local audio files and direct audio URLs.
2. **Transcription:**  
   Uses Whisper, WhisperX, or AssemblyAI to convert the audio into text.
3. **Chunking:**  
   Splits the transcript into manageable chunks based on a target word count.
4. **Revision:**  
   Revises each chunk with the chosen LLM model, optionally including a running summary to provide additional context for better continuity.
5. **Final Output:**  
   Produces a clean, readable transcript stored as JSON and plain text files.

---

## Configuration & Setup

### Environment Setup
Create a file named `config.env` at the root of your repository with your API keys:

```bash
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Configuration File
Create a `config.yaml` file to customize the transcription and processing:

```yaml
# Transcription settings
transcription:
  service: "whisper"  # Options: "whisper", "whisperx", "assemblyai"
  whisper:
    model: "distil-large-v3"  # Options: "tiny", "base", "small", "medium", "large-v3", "distil-large-v3"
    batch_size: 2
    device: "auto"  # Options: "cpu", "cuda"
    language_code: "en"
    beam_size: 4
  assemblyai:
    language_code: "en"
    model: "nano"

# OpenAI settings
openai:
  model: "gpt-4o-mini"

# Chunking parameters
chunking:
  chunk_word_target: 500
  max_summary_words: 300
  enable_summary_summarization: true

# Environment file path
env_file: "./config.env"
```

---

## Transcription Options

### Whisper (Local)
Whisper is a free, open-source speech recognition model from OpenAI that runs locally on your machine:

- **Models**: tiny, base, small, medium, large-v3, distil-large-v3
- **Performance**: Varies by model size; larger models have better accuracy but require more resources
- **Cost**: Free (runs locally)

### AssemblyAI (Cloud)
AssemblyAI offers cloud-based transcription models:

- **Nano**: Around $0.12 per hour of audio.
- **Best**: Around $0.37 per hour of audio.

Upon signing up, you usually receive free credits to test the service without initial costs.

---

## Dependencies

You need to install these Python packages:

```bash
pip install -r requirements.txt
```

Additionally, ensure that you have **ffmpeg** installed (required by yt-dlp to extract audio).

## Usage Instructions

### Using the Standalone Script

```bash
python transcribe_audio.py path/to/audio.mp3 --job-name my_transcript
```

or with a YouTube URL:

```bash
python transcribe_audio.py "https://www.youtube.com/watch?v=example" --job-name youtube_transcript
```

Optional arguments:
- `--job-name`: Name for output files (default: "transcript")
- `--config`: Path to config file (default: "./config.yaml")

### Using the Jupyter Notebook
If you prefer using the notebook version:

1. **Setup the environment**  
   - Enter your API keys into `config.env`.
   - Adjust notebook settings to your preferences (LLM model, chunk size, etc.).

2. **Run the notebook**  
   - Provide the path or URL for your audio or YouTube video when prompted.
   - Let the notebook perform the transcription, chunking, and revision automatically.
   - The final revised transcript will be available in the generated JSON file (`transcript_<job_name>.json`).

---

## Output Files

Each transcription process produces:

- `transcript_<job_name>.json`: Contains complete data including:
  - `final_text`: Complete revised transcript text.
  - `running_summary`: Contextual summary used during chunk processing.
  - `audio_transcript`: Original unprocessed transcript.
  
- `transcript_<job_name>.txt`: Plain text version of the final transcript for easy reading.

---

## Customization

Feel free to adapt the provided system prompts in the script or notebook. Tailor Sbobbinator to your specific audio transcription and revision needs, whether for university lessons, podcasts, interviews, or conferences.

---

## License

Sbobbinator is open source—use it freely, adapt it, and share it with others!

Enjoy effortless, high-quality transcription with Sbobbinator—saving you valuable time and making studying easier!
