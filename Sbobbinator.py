import os
import time
import assemblyai as aai
import openai
from dotenv import load_dotenv

###############################################################################
# 1. CONFIGURATION
###############################################################################

# Load environment variables from config.env
load_dotenv('./config.env')

# Retrieve API keys from environment variables
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API keys
if not ASSEMBLYAI_API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY not found in config.env")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in config.env")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Define OpenAI model to use
OPENAI_MODEL = "gpt-4"  # Ensure this model is accessible with your API key

# Define chunking parameters
CHUNK_WORD_TARGET = 600  # Target words per chunk
MAX_SUMMARY_WORDS = 400  # Maximum words in running summary before summarization
ENABLE_SUMMARY_SUMMARIZATION = True  # Toggle for summary summarization

###############################################################################
# 2. ASSEMBLYAI TRANSCRIPTION
###############################################################################

def transcribe_audio_assemblyai(audio_url_or_path: str) -> str:
    """
    Transcribe the audio from the given URL or local file path using AssemblyAI.
    Returns the raw transcribed text.
    """
    # Set up AssemblyAI
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    transcriber = aai.Transcriber()

    # Start transcription
    transcript = transcriber.transcribe(audio_url_or_path)

    # Poll for completion
    while transcript.status not in ['completed', 'error']:
        print(f"Transcription status: {transcript.status}. Waiting...")
        time.sleep(5)  # Wait for 5 seconds before checking again
        transcript = transcriber.get_transcription(transcript.id)

    # Check for errors
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    return transcript.text

###############################################################################
# 3. CHUNKING THE TRANSCRIPT
###############################################################################

def chunk_text_by_paragraphs(text: str, chunk_word_target: int = 600) -> list:
    """
    Splits the full transcript into chunks, aiming for about `chunk_word_target`
    words each. Attempts to avoid splitting paragraphs in half.

    Returns a list of textual chunks (strings).
    """
    paragraphs = text.split("\n")
    chunks = []
    current_chunk_words = []
    current_word_count = 0

    for paragraph in paragraphs:
        # Trim leading/trailing whitespace
        p = paragraph.strip()
        if not p:
            # Skip empty paragraphs
            continue

        # Count words in the paragraph
        words_in_p = p.split()

        # If adding this paragraph would exceed the chunk size target,
        # start a new chunk first (unless the current chunk is empty).
        if current_word_count + len(words_in_p) > chunk_word_target and current_word_count != 0:
            # Close off the current chunk
            chunk_text = " ".join(current_chunk_words)
            chunks.append(chunk_text)
            # Reset
            current_chunk_words = []
            current_word_count = 0

        # Add the paragraph to the current chunk
        current_chunk_words.extend(words_in_p)
        current_word_count += len(words_in_p)

    # Any leftover words form the last chunk
    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        chunks.append(chunk_text)

    return chunks

###############################################################################
# 4. OPENAI REWRITING (PAGE-BY-PAGE)
###############################################################################

def rewrite_chunk_with_openai(chunk_text: str, model: str = OPENAI_MODEL, prev_summary: str = "") -> str:
    """
    Sends a chunk of text to OpenAI for rewriting in a 'professorial' register.

    Optionally includes `prev_summary` – a short summary of all previously
    processed chunks – as context for better continuity across chunks.

    Returns the revised chunk as a string.
    """
    # Build system prompt with instructions
    system_prompt = (
        "You are an expert in rewriting transcripts with a professorial register. "
        "You will receive fragments of a university lesson transcript generated "
        "from an audio recording. Your role is to correct grammar, punctuation, "
        "and spelling, fix words that may be misrecognized, remove filler words, "
        "and elevate the text to an academic standard. Output only the revised "
        "transcript text in plain text, without titles, markdown, or other formatting. "
        "Maintain context as if it were in medias res."
    )

    # Build user prompt with the chunk, plus the short summary of prior chunks
    # The summary is for context only; it helps the model keep track of earlier topics.
    if prev_summary:
        user_prompt = (
            f"Here is a short summary of what has come before:\n{prev_summary}\n\n"
            f"Now, rewrite the following chunk:\n\n{chunk_text}\n\n"
            "Output only the revised text. Do not add extra commentary or formatting."
        )
    else:
        user_prompt = (
            f"Now, rewrite the following chunk:\n\n{chunk_text}\n\n"
            "Output only the revised text. Do not add extra commentary or formatting."
        )

    # Call OpenAI ChatCompletion with error handling and retries
    for attempt in range(5):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # Keep temperature low for consistent rewriting
                max_tokens=1500,   # Enough tokens to handle rewriting a ~600-word chunk
            )
            revised_text = response.choices[0]["message"]["content"]
            return revised_text.strip()
        except openai.error.RateLimitError:
            wait_time = 2 ** attempt
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to rewrite chunk after multiple attempts.")

def summarize_text_with_openai(text: str, model: str = OPENAI_MODEL) -> str:
    """
    Summarizes the given text in a couple of sentences to maintain context
    for future rewriting chunks.
    """
    # Build system prompt for summarization
    system_prompt = (
        "You are a concise and precise summarizer. Summarize the following text "
        "in two sentences, focusing on the key ideas. Keep it short."
    )

    user_prompt = f"Text to summarize:\n{text}"

    # Call OpenAI ChatCompletion with error handling and retries
    for attempt in range(5):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            summary = response.choices[0]["message"]["content"]
            return summary.strip()
        except openai.error.RateLimitError:
            wait_time = 2 ** attempt
            print(f"Rate limit exceeded during summarization. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error during summarization: {e}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to summarize text after multiple attempts.")

def get_word_count(text: str) -> int:
    """
    Returns the word count of the given text.
    """
    return len(text.split())

###############################################################################
# 5. MAIN PROCESS
###############################################################################

def main():
    # 1) Transcribe audio
    # You can point to a local file or a remote URL. E.g.:
    # audio_source = "https://assembly.ai/path_to_your_audio_file.mp3"
    # or
    # audio_source = "./local_file.mp3"
    audio_source = input('Path to audio file: ')  # Replace with your audio file path or URL

    print("Transcribing audio... please wait.")
    try:
        full_transcript_text = transcribe_audio_assemblyai(audio_source)
    except RuntimeError as e:
        print(str(e))
        return
    print("Transcription complete.")

    # 2) Chunk the transcript
    print("Splitting transcript into chunks...")
    chunks = chunk_text_by_paragraphs(full_transcript_text, chunk_word_target=CHUNK_WORD_TARGET)
    print(f"Created {len(chunks)} chunks of ~{CHUNK_WORD_TARGET} words each.")

    # 3) For each chunk, rewrite with OpenAI
    final_rewritten_text = []
    running_summary = ""  # Will accumulate short summaries of prior chunks

    for i, chunk_text in enumerate(chunks, start=1):
        print(f"Rewriting chunk {i}/{len(chunks)}...")

        # Rewrite the chunk
        try:
            revised_text = rewrite_chunk_with_openai(
                chunk_text=chunk_text,
                model=OPENAI_MODEL,
                prev_summary=running_summary
            )
        except RuntimeError as e:
            print(f"Error rewriting chunk {i}: {str(e)}")
            continue  # Skip to the next chunk

        # Append the revised text to our final output
        final_rewritten_text.append(revised_text)

        # Summarize this revised chunk to update context
        try:
            chunk_summary = summarize_text_with_openai(revised_text, model=OPENAI_MODEL)
            print(f"Summary for chunk {i}: {chunk_summary}")
        except RuntimeError as e:
            print(f"Error summarizing chunk {i}: {str(e)}")
            chunk_summary = ""

        # Append new summary to the running summary
        # Check if summarization of the running summary is enabled
        if ENABLE_SUMMARY_SUMMARIZATION:
            running_summary += f" {chunk_summary}"
            # Check if running_summary exceeds MAX_SUMMARY_WORDS
            if get_word_count(running_summary) > MAX_SUMMARY_WORDS:
                print("Running summary exceeds maximum word limit. Summarizing the running summary...")
                try:
                    summarized_running_summary = summarize_text_with_openai(running_summary, model=OPENAI_MODEL)
                    running_summary = summarized_running_summary
                    print(f"Summarized running summary: {running_summary}")
                except RuntimeError as e:
                    print(f"Error summarizing running summary: {str(e)}")
                    # Optionally, you can reset the running_summary or keep it as is
        else:
            running_summary += f" {chunk_summary}"

    # 4) Output the final revised text
    print("\n=== FINAL REWRITTEN TRANSCRIPT ===\n")
    final_text = "\n\n".join(final_rewritten_text)
    print(final_text)

    # Optionally, save the final text to a file
    # Uncomment the lines below to enable saving to a text file
    
    try:
        with open("final_transcript.txt", "w", encoding="utf-8") as f:
            f.write(final_text)
        print("Final transcript saved to final_transcript.txt")
    except Exception as e:
        print(f"Error saving final transcript: {str(e)}")
    

    print("\nDone.")

if __name__ == "__main__":
    main()
