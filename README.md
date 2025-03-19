# Sbobbinator

**Automated High-Quality Audio Transcription & Revision Tool**

---

## About

**"Sbobbinare"** is a playful Italian word university students use to describe the tedious task of listening to lecture recordings and writing down everything by hand to create clear notes, especially when no textbook is available. Sbobbinator makes this process easy and automatic. Just provide an audio file or a YouTube link—Sbobbinator transcribes it and turns it into polished, textbook-quality notes using LLMs.  

Originally designed to help students automatically transcribe and revise university lecture recordings, Sbobbinator can work with any audio content—lectures, podcasts, interviews, and more. You can also easily adjust the prompt used by the language model to suit any specific transcription needs you have.

---

## How It Works

Sbobbinator leverages:

- **AssemblyAI** for audio transcription.
- **OpenAI GPT (or Deepseek)** to rewrite transcripts with improved grammar, style, and clarity, turning them into polished notes.
- **yt-dlp** to download and extract audio from YouTube videos.

The tool performs these steps:

1. **Audio Extraction:**  
   Downloads audio from YouTube videos (to an `./audio/` folder) or processes local audio files and direct audio URLs.
2. **Transcription:**  
   Uses AssemblyAI to convert the audio into text.
3. **Chunking:**  
   Splits the transcript into manageable chunks based on a target word count.
4. **Revision:**  
   Revises each chunk with the chosen LLM model, optionally including a running summary to provide additional context for better continuity.
5. **Final Output:**  
   Produces a clean, readable transcript stored as a JSON file.

---

## Configuration & Setup

Create a file named `config.env` at the root of your repository with your API keys:

```bash
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Key Configuration Parameters (in Notebook cell labeled `# 1. CONFIGURATION`):

- **Job Name:** Used to name the output JSON file (e.g., `transcript_<job_name>.json`).
- **LLM Model:** Select your preferred model, such as OpenAI's `gpt-4o-mini` or Deepseek.
- **Chunking Parameters:**  
  - `CHUNK_WORD_TARGET`: Target number of words per chunk.
  - `MAX_SUMMARY_WORDS`: Maximum number of words for the running summary.
- **Summary Flag:**  
  - `ENABLE_SUMMARY_SUMMARIZATION`: Set to `False` to avoid using running summaries, thus reducing input tokens and overall costs.

---

## AssemblyAI Models and Pricing

AssemblyAI offers two affordable transcription models:

- **Nano:** Around $0.12 per hour of audio.
- **Best:** Around $0.37 per hour of audio.

Upon signing up, you usually receive free credits to test the service without initial costs.

---

## Dependencies

You need to install these Python packages:

```bash
pip install yt-dlp assemblyai openai python-dotenv
```

Additionally, ensure that you have **ffmpeg** installed (required by yt-dlp to extract audio):

## Usage Instructions

1. **Setup the environment**  
   - Enter your API keys into `config.env`.
   - Adjust notebook settings to your preferences (LLM model, chunk size, etc.).

2. **Run the notebook**  
   - Provide the path or URL for your audio or YouTube video when prompted.
   - Let the notebook perform the transcription, chunking, and revision automatically.
   - The final revised transcript will be available in the generated JSON file (`transcript_<job_name>.json`).

---

## Output Example

Check out the included example file:

- [`transcript_Standford_University_Building_LLMs.json`](transcript_Standford_University_Building_LLMs.json)  
  This example is an automated transcript and revision of a Stanford University lecture: ["Building Large Language Models (LLMs)"](https://www.youtube.com/watch?v=9vM4p9NN0Ts&t=28s&pp=ygUHbWl0IGxsbQ%3D%3D).

Each output JSON includes:

- `final_text`: Complete revised transcript text.
- `running_summary`: Contextual summary used during chunk processing.

---

## Customization

Feel free to adapt the provided system prompt inside the notebook. Tailor Sbobbinator to your specific audio transcription and revision needs, whether for university lessons, podcasts, interviews, or conferences.

---

## License

Sbobbinator is open source—use it freely, adapt it, and share it with others!

Enjoy effortless, high-quality transcription with Sbobbinator—saving you valuable time and making studying easier!
