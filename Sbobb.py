#!/usr/bin/env python3
"""
Audio Transcription and Rewriting Script
Transcribes audio using Whisper or AssemblyAI, then rewrites it with OpenAI.
"""

import os
import sys
import json
import re
import time
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Audio processing imports
import yt_dlp
from pydub import AudioSegment
import subprocess
import tempfile

# Transcription imports
import assemblyai as aai
import whisperx
from faster_whisper import WhisperModel, BatchedInferencePipeline

# OpenAI import
from openai import OpenAI

# Environment loading
from dotenv import load_dotenv


class AudioTranscriber:
    """Main class for audio transcription and rewriting."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self._load_environment()
        self._validate_api_keys()
        self._init_openai_client()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_environment(self):
        """Load environment variables from config.env."""
        env_file = self.config.get('env_file', './config.env')
        load_dotenv(env_file)
        
        self.assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    def _validate_api_keys(self):
        """Validate required API keys based on transcription service."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in config.env")
            
        if self.config['transcription']['service'] == 'assemblyai' and not self.assemblyai_api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in config.env")
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        self.openai_client = OpenAI(api_key=self.openai_api_key)
    
    def download_youtube_audio(self, youtube_url: str) -> str:
        """Download audio from YouTube URL."""
        output_dir = './audio'
        os.makedirs(output_dir, exist_ok=True)
        
        outtmpl = os.path.join(output_dir, "%(title)s.%(ext)s")
        
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "cookiesfrombrowser": ("firefox",),
            "quiet": True,
            "no_warnings": True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            temp_path = ydl.prepare_filename(info)
        
        base, _ = os.path.splitext(temp_path)
        final_path = base + ".mp3"
        
        return final_path
    
    def get_audio_duration(self, audio_file_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            audio = AudioSegment.from_file(audio_file_path)
            return len(audio) / 1000.0
        except Exception as e:
            raise Exception(f"Could not get audio duration: {e}")
    
    def normalize_audio_for_whisper(self, input_path: str) -> str:
        """Convert audio to optimal format for Whisper (16kHz, mono, WAV)."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        print("Normalizing audio (16kHz, mono, WAV)...")
        
        try:
            # Try ffmpeg first
            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-avoid_negative_ts', 'make_zero',
                '-loglevel', 'error',
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")
            
            print("✓ Audio normalized with FFmpeg")
            return temp_path
            
        except Exception:
            # Fallback to pydub
            print("FFmpeg not available, using pydub for normalization...")
            
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(temp_path, format="wav")
            
            print("✓ Audio normalized with pydub")
            return temp_path
    
    def transcribe_audio_whisper(self, audio_file_path: str) -> str:
        """Transcribe using local Whisper."""
        config = self.config['transcription']['whisper']
        
        print(f"Processing audio file: {os.path.basename(audio_file_path)}")
        
        normalized_path = None
        try:
            normalized_path = self.normalize_audio_for_whisper(audio_file_path)
            
            print("Loading local Whisper model...")
            
            model = WhisperModel(
                config['model'], 
                device=config['device'], 
                compute_type="int8"
            )
            model_batch = BatchedInferencePipeline(model=model)
            
            print("Transcribing with local Whisper...")
            segments, info = model_batch.transcribe(
                normalized_path,
                batch_size=config['batch_size'],
                language=config['language_code'],
                log_progress=True,
                word_timestamps=False
            )
            
            segments = list(segments)
            result = [segment.text.strip() for segment in segments]
            
            print("✓ Transcription completed successfully")
            return ' '.join(result).strip()
            
        except Exception as e:
            raise RuntimeError(f"Whisper transcription failed: {str(e)}")
        
        finally:
            if normalized_path and os.path.exists(normalized_path):
                try:
                    os.unlink(normalized_path)
                    print("✓ Cleaned up temporary files")
                except:
                    pass
    
    def transcribe_audio_whisperx(self, audio_file_path: str) -> str:
        """Transcribe using WhisperX."""
        config = self.config['transcription']['whisper']
        
        print(f"Processing audio file: {os.path.basename(audio_file_path)}")
        
        normalized_path = None
        try:
            normalized_path = self.normalize_audio_for_whisper(audio_file_path)
            
            print("Loading WhisperX model...")
            
            model = whisperx.load_model(
                config['model'], 
                config['device'], 
                compute_type="int8"
            )
            audio = whisperx.load_audio(normalized_path)
            segments, lang = model.transcribe(
                audio,
                batch_size=config['batch_size'],
                language=config['language_code'],
                print_progress=True
            ).values()
            
            text = [segment['text'] for segment in segments]
            print("✓ Transcription completed successfully")
            return ' '.join(text).strip()
            
        except Exception as e:
            raise RuntimeError(f"WhisperX transcription failed: {str(e)}")
        
        finally:
            if normalized_path and os.path.exists(normalized_path):
                try:
                    os.unlink(normalized_path)
                    print("✓ Cleaned up temporary files")
                except:
                    pass
    
    def transcribe_audio_assemblyai(self, audio_url_or_path: str) -> str:
        """Transcribe using AssemblyAI."""
        config = self.config['transcription']['assemblyai']
        
        aai.settings.api_key = self.assemblyai_api_key
        
        if config['model'] == 'nano':
            aai_config = aai.TranscriptionConfig(
                language_code=config['language_code'],
                speech_model=aai.SpeechModel.nano
            )
        elif config['model'] == 'slam':
            aai_config = aai.TranscriptionConfig(
                language_code=config['language_code'],
                speech_model=aai.SpeechModel.slam_1
            )
        
        transcriber = aai.Transcriber(config=aai_config)
        
        print("Uploading file to AssemblyAI for transcription...")
        transcript = transcriber.transcribe(audio_url_or_path)
        
        while transcript.status not in ['completed', 'error']:
            print(f"Transcription status: {transcript.status}. Waiting...")
            time.sleep(5)
            transcript = transcriber.get_transcription(transcript.id)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Transcription failed: {transcript.error}")
        
        return transcript.text
    
    def transcribe_audio(self, audio_source: str, service_override: Optional[str] = None) -> str:
        """Main transcription method."""
        # Handle YouTube URLs
        if "youtube.com" in audio_source or "youtu.be" in audio_source:
            print("Detected YouTube URL. Downloading audio locally...")
            audio_source = self.download_youtube_audio(audio_source)
            print(f"Local file path: {audio_source}")
        
        service = service_override or self.config['transcription']['service']
        
        if service == "whisper":
            return self.transcribe_audio_whisper(audio_source)
        elif service == "whisperx":
            return self.transcribe_audio_whisperx(audio_source)
        elif service == "assemblyai":
            return self.transcribe_audio_assemblyai(audio_source)
        else:
            raise ValueError(f"Unknown transcription service: {service}")
    
    def chunk_text_by_sentences(self, transcript_text: str) -> List[str]:
        """Split text into chunks by sentences."""
        chunk_word_target = self.config['chunking']['chunk_word_target']
        
        sentences = re.split(r'[.!?]+', transcript_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            if (current_word_count + sentence_word_count) > chunk_word_target and current_chunk:
                chunk = ". ".join(current_chunk) + "."
                chunks.append(chunk)
                current_chunk = []
                current_word_count = 0
            
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
        
        if current_chunk:
            chunk = ". ".join(current_chunk) + "."
            chunks.append(chunk)
        
        return chunks
    
    def rewrite_chunk_with_openai(self, chunk_text: str, prev_summary: str = "") -> str:
        """Rewrite chunk using OpenAI."""
        system_prompt = (
            "You are an expert in rewriting transcripts with a professorial register. "
            "You will receive fragments of an audio recording regarding a meeting of me and my supervisor, "
            "she explains things to me regarding bioinformatics, structural biology, network science, "
            "graph neural network, computer science. Your role is to correct grammar, punctuation, "
            "and spelling, fix words that may be misrecognized, remove filler words, "
            "and elevate the text to an academic standard. Output only the revised "
            "transcript text in plain text, without titles, markdown, or other formatting. "
            "Maintain context as if it were in medias res. "
            "Maintain the language of the text, so if it is english, write it in "
            "english, if it is italian, write it in italian."
        )
        
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
        
        response = self.openai_client.chat.completions.create(
            model=self.config['openai']['model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        
        return response.choices[0].message.content.strip()
    
    def summarize_text_with_openai(self, text: str) -> str:
        """Summarize text using OpenAI."""
        system_prompt = (
            "You are a concise and precise summarizer. Summarize the following text "
            "in one sentence, focusing on the key ideas. Keep it short. Do not refer "
            "to the text itself, just provide a single sentence that capture the key ideas. "
            "Maintain the language of the text, so if it is english, write it in "
            "english, if it is italian, write it in italian."
        )
        
        user_prompt = f"Text to summarize:\n{text}"
        
        response = self.openai_client.chat.completions.create(
            model=self.config['openai']['model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        
        return response.choices[0].message.content.strip()
    
    def process_transcript(self, transcript_text: str) -> Tuple[str, str, List[str]]:
        """Process transcript through chunking and rewriting."""
        print("Splitting transcript into chunks...")
        chunks = self.chunk_text_by_sentences(transcript_text)
        print(f"Created {len(chunks)} chunk(s) of ~{self.config['chunking']['chunk_word_target']} words each.")
        
        final_rewritten_text = []
        running_summary = ""
        
        print("Rewriting the transcript...")
        
        for i, chunk_text in enumerate(chunks, start=1):
            # Rewrite chunk
            try:
                revised_text = self.rewrite_chunk_with_openai(
                    chunk_text=chunk_text,
                    prev_summary=running_summary
                )
            except Exception as e:
                print(f"Error rewriting chunk {i}: {str(e)}")
                continue
            
            final_rewritten_text.append(revised_text)
            
            # Summarize chunk
            try:
                chunk_summary = self.summarize_text_with_openai(revised_text)
            except Exception as e:
                print(f"Error summarizing chunk {i}: {str(e)}")
                chunk_summary = ""
            
            # Update running summary
            if self.config['chunking']['enable_summary_summarization']:
                running_summary += f" {chunk_summary}"
                
                # Check if running summary exceeds max words
                if len(running_summary.split()) > self.config['chunking']['max_summary_words']:
                    try:
                        running_summary = self.summarize_text_with_openai(running_summary)
                    except Exception as e:
                        print(f"Error summarizing running summary: {str(e)}")
            else:
                running_summary += f" {chunk_summary}"
        
        final_text = " ".join(final_rewritten_text)
        
        # Fix escape sequences
        final_text = re.sub(r'\\n', '\n', final_text)
        final_text = re.sub(r"\\'", "'", final_text)
        final_text = re.sub(r'\\"', '"', final_text)
        final_text = re.sub(r'\\t', '\t', final_text)
        
        return final_text, running_summary, chunks
    
    def save_outputs(self, job_name: str, final_text: str, running_summary: str, chunks: List[str]):
        """Save outputs to files."""
        # Save markdown file
        markdown_filename = f"./results/transcript_{job_name}.md"
        with open(markdown_filename, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"✓ Final transcript saved to {markdown_filename}")
        
        # Save JSON file
        json_data = {
            "running_summary": running_summary,
            "audio_transcript": " ".join(chunks)
        }
        
        json_filename = f"./results/transcript_{job_name}_metadata.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"✓ Metadata saved to {json_filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe and rewrite audio files using AI services."
    )
    parser.add_argument(
        "audio_source",
        help="Path to audio file or YouTube URL"
    )
    parser.add_argument(
        "job_name",
        help="Name for the output files"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--service",
        choices=["whisper", "whisperx", "assemblyai"],
        help="Override transcription service from config"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize transcriber
        transcriber = AudioTranscriber(args.config)
        
        # Transcribe audio
        print("Transcribing audio... please wait.")
        full_transcript_text = transcriber.transcribe_audio(
            args.audio_source,
            service_override=args.service
        )
        
        # Process transcript
        final_text, running_summary, chunks = transcriber.process_transcript(full_transcript_text)
        
        # Save outputs
        transcriber.save_outputs(args.job_name, final_text, running_summary, chunks)
        
        print("\nDone.")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()