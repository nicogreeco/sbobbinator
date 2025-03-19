# Sbobbinator

**Automated High-Quality Audio Transcription & Revision Tool**

---

## About

**"Sbobbinare"** is a playful Italian word that university students use to describe the long, boring task of listening to lecture recordings and writing down everything by hand, creating clear and comprehensive notes for study purposes in asbsence of a textbook. Sbobbinator was built to make that process way easier. It automatically transcribes your audio—whether it’s a lecture, podcast, or any other recording—and cleans it up into clear, study-ready notes. You can even tweak it to suit different types of audio. Say goodbye to tedious manual transcriptions and hello to more time for studying!

---

## How It Works

Sbobbinator leverages:
- **AssemblyAI** for accurate audio transcription.
- **OpenAI GPT (or Deepseek)** to rewrite transcripts with improved grammar, style, and clarity, transforming them into a textbook-quality resource.
- **yt-dlp** to seamlessly extract audio from YouTube videos.

The pipeline follows these key steps:

1. **Audio Extraction:**  
   Extract audio from YouTube URLs or process local audio files and remote URLs.
2. **Transcription:**  
   Convert audio to text using AssemblyAI.
3. **Chunking:**  
   Divide the transcription into manageable segments based on a defined word count.
4. **Revision:**  
   Refine each chunk via a specified LLM, optionally providing a running summary for context, ensuring cohesive and high-quality output.
5. **Final Output:**  
   Generate a comprehensive and polished transcript saved as a JSON file.

---

## Configuration & Setup

Create a file named `config.env` in your repository root, containing your API keys:

```bash
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Key Configuration Parameters (in Notebook cell labeled `# 1. CONFIGURATION`):

- **Job Name:** Identifier for output JSON file (e.g., `transcript_<job_name>.json`).
- **LLM Model:** Choose between OpenAI models like `gpt-4o-mini` or Deepseek.
- **Chunking Parameters:**  
  - `CHUNK_WORD_TARGET`: Desired word count per chunk.
  - `MAX_SUMMARY_WORDS`: Maximum length of running summary.
- **Summary Flag:**  
  - `ENABLE_SUMMARY_SUMMARIZATION`: Set to `False` to disable running summaries, saving tokens and reducing costs.

---

## AssemblyAI Models and Pricing

AssemblyAI provides two efficient models:

- **Nano:** Approximately $0.12/hour of audio.
- **Best:** Approximately $0.37/hour of audio.

New users typically receive free credits upon signing up, making it easy to test both models without initial costs.

---

## Dependencies

Ensure these dependencies are installed:

```bash
pip install yt-dlp assemblyai openai python-dotenv
```

---

## Usage Instructions

1. **Setup Environment:**
   - Configure API keys in `config.env`.
   - Define your preferred settings in the notebook's configuration cell.

2. **Run Notebook:**
   - Enter a path or URL for your audio file or YouTube video when prompted.
   - Allow the notebook to complete transcription, chunking, and revision.
   - Find your polished transcript in the generated JSON file (`transcript_<job_name>.json`).

---

## Output

The output JSON includes:

- `final_text`: The full, revised transcript.
- `running_summary`: Contextual summary of the lesson content.

---

## Customization

Modify the provided system prompt within the notebook to adapt Sbobbinator for transcribing podcasts, interviews, conference talks, or other audio materials requiring a high-quality textual transcript.

---

## License

Sbobbinator is open source. Feel free to contribute, adapt, and distribute as needed.

Enjoy automatic, high-quality transcription with Sbobbinator—spend less time transcribing and more time studying effectively!