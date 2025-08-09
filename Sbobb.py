"""
Audio Transcription and Processing Script

This script transcribes audio files using either Whisper or AssemblyAI,
then processes the transcript by chunking it and rewriting it using OpenAI.
"""

import os
import time
import argparse
import yaml
import json
import codecs
import tempfile
import subprocess
from pathlib import Path

# Third-party imports
import yt_dlp
import assemblyai as aai
import whisperx
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pydub import AudioSegment
from dotenv import load_dotenv
from openai import OpenAI

###############################################################################
# AUDIO PREPROCESSING FUNCTIONS
###############################################################################

def get_audio_duration(audio_file_path: str) -> float:
    """Get the duration of an audio file in seconds."""
    try:
        audio = AudioSegment.from_file(audio_file_path)
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except Exception as e:
        raise Exception(f"Could not get audio duration: {e}")

def validate_audio_file(audio_file_path: str, max_duration: int = 7200, min_duration: int = 1) -> tuple:
    """Validate audio file format and duration. Returns (is_valid, message)"""
    import os
    from pathlib import Path
    
    audio_path = Path(audio_file_path)
    
    # Check if file exists
    if not audio_path.exists():
        return False, f"File not found: {audio_path}"
    
    # Check file extension
    supported_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.avi', '.mov']
    if audio_path.suffix.lower() not in supported_formats:
        return False, f"Unsupported format: {audio_path.suffix}"
    
    # Check duration
    try:
        duration = get_audio_duration(audio_file_path)
        
        if duration < min_duration:
            return False, f"Audio too short: {duration:.1f}s (minimum: {min_duration}s)"
        
        if duration > max_duration:
            return False, f"Audio too long: {duration:.1f}s (maximum: {max_duration/60:.0f} minutes)"
        
        return True, f"Valid audio file: {duration:.1f}s ({duration/60:.1f} minutes)"
        
    except Exception as e:
        return False, f"Error validating audio: {e}"

def normalize_audio_for_whisper(input_path: str) -> str:
    """Convert audio to optimal format for Whisper (16kHz, mono, WAV)"""
    try:
        # Create temporary file for normalized audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        print("Normalizing audio (16kHz, mono, WAV)...")
        
        # Try using ffmpeg first (more reliable)
        try:
            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-avoid_negative_ts', 'make_zero',
                '-loglevel', 'error',  # Suppress ffmpeg output
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")
            
            print("✓ Audio normalized with FFmpeg")
            return temp_path
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            # Fallback to pydub if ffmpeg fails
            print("FFmpeg not available, using pydub for normalization...")
            
            # Clean up failed ffmpeg attempt
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Create new temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            # Load and convert with pydub
            audio = AudioSegment.from_file(input_path)
            
            # Convert to mono and set sample rate
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(16000)  # 16kHz
            
            # Export as WAV
            audio.export(temp_path, format="wav")
            
            print("✓ Audio normalized with pydub")
            return temp_path
            
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        raise Exception(f"Audio normalization failed: {e}")

###############################################################################
# YOUTUBE DOWNLOAD FUNCTION
###############################################################################

def download_youtube_audio(youtube_url: str) -> str:
    """
    Downloads the best available audio from a YouTube video, converts it to MP3 (192 kbps),
    and saves it into ./audio/ with a proper file name.
    
    Returns the final path to the downloaded MP3 file.
    """

    # Ensure the output directory exists
    output_dir = './audio'
    os.makedirs(output_dir, exist_ok=True)

    # We'll store the final file in ./audio/<title>.mp3
    outtmpl = os.path.join(output_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",             # Download best-quality audio
        "outtmpl": outtmpl,                    # Save to ./audio/<title>.<ext>
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",         # ~192 kbps MP3
        }],
        "cookiesfrombrowser": ("firefox",),    # Use Firefox cookies if needed
        "quiet": True,                         # Suppress non-error messages
        "no_warnings": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        # Prepare the base filename (e.g. 'My Song.m4a' before post-processing)
        temp_path = ydl.prepare_filename(info)

    # Because of the FFmpegExtractAudio postprocessor, the final file is .mp3
    # We just replace the extension if needed:
    base, _ = os.path.splitext(temp_path)
    final_path = base + ".mp3"

    return final_path

###############################################################################
# TRANSCRIPTION FUNCTIONS
###############################################################################

def transcribe_audio_assemblyai(audio_url_or_path: str, language_code: str = "en", model: str='nano') -> aai.Transcript:
    """
    Transcribe the audio from a local file path using AssemblyAI.
    Returns the transcript object.
    """
    # Set up AssemblyAI
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if model == 'nano':
        config = aai.TranscriptionConfig(language_code=language_code, speech_model=aai.SpeechModel.nano)
    elif model=='slam':
        config = aai.TranscriptionConfig(language_code=language_code, speech_model=aai.SpeechModel.slam_1)
    
    transcriber = aai.Transcriber(config=config)

    print("Uploading file to AssemblyAI for transcription...")
    transcript = transcriber.transcribe(audio_url_or_path)

    # Poll for completion
    while transcript.status not in ['completed', 'error']:
        print(f"Transcription status: {transcript.status}. Waiting...")
        time.sleep(5)
        transcript = transcriber.get_transcription(transcript.id)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    return transcript

def transcribe_audio_whisper(audio_file_path: str, language_code: str = 'en', whisper_model: str = 'small', batch_size: int=1, beam_size: int=2, device: str = 'cpu') -> str:
    """
    Transcribe audio file using local Whisper with preprocessing.
    Returns the transcript text as a string.
    """
    print(f"Processing audio file: {os.path.basename(audio_file_path)}")
    
    # Step 1: Validate audio file
    print("Validating audio file...")
    # is_valid, message = validate_audio_file(audio_file_path)
    is_valid = True
    message = ''
    if not is_valid:
        raise RuntimeError(f"Audio validation failed: {message}")
    print(f"✓ {message}")
    
    # Step 2: Normalize audio for optimal Whisper performance
    normalized_path = None
    try:
        normalized_path = normalize_audio_for_whisper(audio_file_path)
        
        print("Loading local Whisper model...")

        # Load the whisper model (you can change 'base' to 'small', 'medium', 'large' for better accuracy)
        model = WhisperModel(whisper_model, device=device, compute_type="int8")
        model_batch = BatchedInferencePipeline(model=model)
        
        print("Transcribing with local Whisper...")
        segments, info = model_batch.transcribe(normalized_path, 
                                            batch_size=batch_size, 
                                            # beam_size=beam_size,
                                            language=language_code, 
                                            log_progress=True, 
                                            word_timestamps=False)
        
        segments = list(segments)  # The transcription will actually run here.
        result = [segment.text.strip() for segment in segments]
        
        print("✓ Transcription completed successfully")
        return ' '.join(result).strip()
        
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {str(e)}")
    
    finally:
        # Clean up temporary normalized file
        if normalized_path and os.path.exists(normalized_path):
            try:
                os.unlink(normalized_path)
                print("✓ Cleaned up temporary files")
            except:
                pass  # Ignore cleanup errors

def transcribe_audio_whisperX(audio_file_path: str, language_code: str = 'en', whisper_model: str = 'small', batch_size: int=1, beam_size: int = 1, device: str = 'cpu', compute_type:str='int8') -> str:
    """
    Transcribe audio file using local WhisperX with preprocessing.
    Returns the transcript text as a string.
    """
    print(f"Processing audio file: {os.path.basename(audio_file_path)}")
    
    # Step 1: Validate audio file
    print("Validating audio file...")
    # is_valid, message = validate_audio_file(audio_file_path)
    is_valid = True
    message = ''
    if not is_valid:
        raise RuntimeError(f"Audio validation failed: {message}")
    print(f"✓ {message}")
    
    # Step 2: Normalize audio for optimal Whisper performance
    normalized_path = None
    try:
        normalized_path = normalize_audio_for_whisper(audio_file_path)
        
        print("Loading local Whisper model...")

        # Load the whisper model (you can change 'base' to 'small', 'medium', 'large' for better accuracy)
        model = whisperx.load_model(whisper_model, device, compute_type=compute_type)
        audio = whisperx.load_audio(normalized_path)
        segments, lang = model.transcribe(audio, 
                                      batch_size=batch_size, 
                                      language=language_code, 
                                      # beam_size=beam_size, 
                                      print_progress=True).values()
        
        text = [segment['text'] for segment in segments]   
        print("✓ Transcription completed successfully")
        return ' '.join(text).strip()
        
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {str(e)}")
    
    finally:
        # Clean up temporary normalized file
        if normalized_path and os.path.exists(normalized_path):
            try:
                os.unlink(normalized_path)
                print("✓ Cleaned up temporary files")
            except:
                pass  # Ignore cleanup errors

def transcribe_audio(audio_source: str, service: str = "whisper", language_code: str = "en", model: str = 'small', batch_size: int = 1, beam_size: int = 1, device: str = 'cpu', compute_type:str = 'int8'):
    """
    Unified transcription function that handles both Whisper and AssemblyAI.
    Returns transcript text for consistency.
    """
    # Handle YouTube URLs by downloading first
    if "youtube.com" in audio_source or "youtu.be" in audio_source:
        print("Detected YouTube URL. Downloading audio locally...")
        audio_source = download_youtube_audio(audio_source)
        print(f"Local file path: {audio_source}")
    
    if service == "whisper":
        return transcribe_audio_whisper(audio_source, language_code, model, batch_size, beam_size, device)
    if service == "whisperx":
        return transcribe_audio_whisperX(audio_source, language_code, model, batch_size, beam_size, device, compute_type)
    elif service == "assemblyai":
        transcript_obj = transcribe_audio_assemblyai(audio_source, language_code, model)
        return transcript_obj.text
    else:
        raise ValueError(f"Unknown transcription service: {service}")

###############################################################################
# CHUNKING FUNCTIONS
###############################################################################

def chunk_text_by_sentences(transcript_text: str, chunk_word_target: int = 600) -> list:
    """
    Splits the transcript text into chunks based on sentences, aiming for about
    `chunk_word_target` words each. This works for plain text from Whisper.
    """
    import re
    
    # Split into sentences using basic punctuation
    sentences = re.split(r'[.!?]+', transcript_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # If adding this sentence exceeds the target and current chunk is not empty
        if (current_word_count + sentence_word_count) > chunk_word_target and current_chunk:
            chunk = ". ".join(current_chunk) + "."
            chunks.append(chunk)
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += sentence_word_count

    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunk = ". ".join(current_chunk) + "."
        chunks.append(chunk)

    return chunks

def chunk_text_by_paragraphs(transcript: aai.Transcript, chunk_word_target: int = 600) -> list:
    """
    Splits the AssemblyAI transcript into chunks based on its paragraphs.
    """
    paragraphs = transcript.get_paragraphs()
    chunks = []
    current_chunk = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_text = paragraph.text.strip()
        if not paragraph_text:
            continue

        paragraph_word_count = len(paragraph_text.split())

        if (current_word_count + paragraph_word_count) > chunk_word_target and current_chunk:
            chunk = "\n".join(current_chunk)
            chunks.append(chunk)
            current_chunk = []
            current_word_count = 0

        current_chunk.append(paragraph_text)
        current_word_count += paragraph_word_count

    if current_chunk:
        chunk = "\n".join(current_chunk)
        chunks.append(chunk)

    return chunks

###############################################################################
# OPENAI REWRITING FUNCTIONS
###############################################################################

def rewrite_chunk_with_openai(chunk_text: str, model: str, prev_summary: str = "", client=None) -> str:
    """
    Sends a chunk of text to OpenAI for rewriting in a 'professorial' register.

    Optionally includes `prev_summary` - a short summary of all previously
    processed chunks - as context for better continuity across chunks.

    Returns the revised chunk as a string.
    """
    # Build system prompt with instructions
    system_prompt = (
        "You are an expert in rewriting transcripts with a professorial register. "
        "You will receive fragments of an"
        "audio recording regarding a meeting of me and my supervisor, she explains "
        "things to me regarding bioinformatics, structural biology, network science, graph neural network, computer science."
        " Your role is to correct grammar, punctuation, "
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

    # Call OpenAI ChatCompletion using the client
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # Keep temperature low for consistent rewriting
        max_tokens=1500,   # Enough tokens to handle rewriting a ~600-word chunk
    )

    revised_text = response.choices[0].message.content
    return revised_text.strip()

def summarize_text_with_openai(text: str, model: str, client=None) -> str:
    """
    Summarizes the given text in a couple of sentences to maintain context
    for future rewriting chunks.
    """
    system_prompt = (
        "You are a concise and precise summarizer. Summarize the following text "
        "in one sentence, focusing on the key ideas. Keep it short. Do not referes "
        "to the text itself, just provide a single sentence that capture the kay ideas."
    )

    user_prompt = f"Text to summarize:\n{text}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )

    summary = response.choices[0].message.content
    return summary.strip()

def get_word_count(text: str) -> int:
    """
    Returns the word count of the given text.
    """
    return len(text.split())

###############################################################################
# MAIN FUNCTION
###############################################################################

def main():
    """Main function to orchestrate the audio processing workflow."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Audio transcription and processing script")
    parser.add_argument("audio_path", help="Path to audio file or YouTube URL")
    parser.add_argument("-j", "--job-name", help="Job name for output files", default="transcript")
    parser.add_argument("-c", "--config", help="Path to config file", default="./config.yaml")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment variables
    load_dotenv(config.get('env_file', './config.env'))
    
    # Retrieve API keys
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Validate API keys
    transcription_service = config['transcription']['service']
    if transcription_service == 'assemblyai' and not ASSEMBLYAI_API_KEY:
        raise ValueError("ASSEMBLYAI_API_KEY not found in config.env")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in config.env")
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Get chunking parameters
    CHUNK_WORD_TARGET = config['chunking']['chunk_word_target']
    MAX_SUMMARY_WORDS = config['chunking']['max_summary_words']
    ENABLE_SUMMARY_SUMMARIZATION = config['chunking']['enable_summary_summarization']
    
    # Get OpenAI model
    OPENAI_MODEL = config['openai']['model']
    
    print(f"Job name: {args.job_name}")
    print(f"Transcription service: {transcription_service}")
    print(f"OpenAI model: {OPENAI_MODEL}")
    
    # Transcribe the audio
    print("Transcribing audio... please wait.")
    
    transcription_config = config['transcription'][transcription_service]
    
    full_transcript_text = transcribe_audio(
        audio_source=args.audio_path,
        service=transcription_service,
        language_code=transcription_config.get('language_code', 'en'),
        model=transcription_config.get('model', 'small'),
        batch_size=transcription_config.get('batch_size', 1),
        beam_size=transcription_config.get('beam_size', 1),
        device=transcription_config.get('device', 'cpu'),
        compute_type='int8'
    )
    
    # Split transcript into chunks
    print("Splitting transcript into chunks...")
    
    if transcription_service == 'assemblyai':
        chunks = chunk_text_by_paragraphs(full_transcript_text, chunk_word_target=CHUNK_WORD_TARGET)
    else:  # whisper or whisperx
        chunks = chunk_text_by_sentences(full_transcript_text, chunk_word_target=CHUNK_WORD_TARGET)
    
    print(f"Created {len(chunks)} chunk(s) of ~{CHUNK_WORD_TARGET} words each.")
    
    # Rewrite each chunk with OpenAI
    final_rewritten_text = []
    running_summary = ""  # Will accumulate short summaries of prior chunks
    print("Rewriting the transcript...")
    
    for i, chunk_text in enumerate(chunks, start=1):
        # Rewrite the chunk
        try:
            revised_text = rewrite_chunk_with_openai(
                chunk_text=chunk_text,
                model=OPENAI_MODEL,
                prev_summary=running_summary,
                client=client
            )
        except Exception as e:
            print(f"Error rewriting chunk {i}: {str(e)}")
            continue  # Skip to the next chunk
    
        # Append the revised text to our final output
        final_rewritten_text.append(revised_text)
    
        # Summarize this revised chunk to update context
        try:
            chunk_summary = summarize_text_with_openai(revised_text, model=OPENAI_MODEL, client=client)
        except Exception as e:
            print(f"Error summarizing chunk {i}: {str(e)}")
            chunk_summary = ""
    
        # Append new summary to the running summary
        if ENABLE_SUMMARY_SUMMARIZATION:
            running_summary += f" {chunk_summary}"
            # Check if running_summary exceeds MAX_SUMMARY_WORDS
            if get_word_count(running_summary) > MAX_SUMMARY_WORDS:
                try:
                    summarized_running_summary = summarize_text_with_openai(running_summary, model=OPENAI_MODEL, client=client)
                    running_summary = summarized_running_summary
                except Exception as e:
                    print(f"Error summarizing running summary: {str(e)}")
        else:
            running_summary += f" {chunk_summary}"
    
    # Output the final revised text and running summary as a JSON file
    final_text = " ".join(final_rewritten_text)
    
    data = {
        "final_text": codecs.decode(final_text, "unicode_escape"),
        "running_summary": running_summary,
        "audio_transcript": " ".join(chunks)
    }
    
    output_filename = f"transcript_{args.job_name}.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Final transcript and running summary saved to {output_filename}")
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")
    
    # Also save as plain text for easier reading
    text_filename = f"transcript_{args.job_name}.txt"
    try:
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(data["final_text"])
        print(f"Plain text transcript saved to {text_filename}")
    except Exception as e:
        print(f"Error saving text file: {str(e)}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()