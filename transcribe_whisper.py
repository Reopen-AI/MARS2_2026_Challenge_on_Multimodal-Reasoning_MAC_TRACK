import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


DEFAULT_MODEL = "openai/whisper-large-v3"
MAX_PARSE_RETRIES = 5
RETRY_TEMPERATURE = 0.3
RETRY_TEMPERATURE_STEP = 0.1
RETRY_REPETITION_PENALTY = 1.1
RETRY_REPETITION_PENALTY_STEP = 0.05
WHISPER_CHUNK_LENGTH_S = 30
TOKEN_LIMIT_MARGIN = 8
REPEAT_NGRAM_SIZE = 12
MIN_REPETITION_TOKENS = 64
MAX_REPEAT_COVERAGE = 0.8
MAX_NGRAM_REPEATS = 4


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def validate_text_quality(
    segments: list[dict], tokenizer, max_output_tokens: int
) -> None:
    token_ids = []
    for segment in segments:
        segment_tokens = tokenizer.encode(
            segment["text"],
            add_special_tokens=False,
        )
        token_ids.extend(segment_tokens)
        windows = max(
            1,
            math.ceil((segment["end"] - segment["start"]) / WHISPER_CHUNK_LENGTH_S),
        )
        token_limit = windows * max(1, max_output_tokens - TOKEN_LIMIT_MARGIN)
        if len(segment_tokens) >= token_limit:
            raise ValueError(
                f"Whisper output reached the token limit: "
                f"{len(segment_tokens)} >= {token_limit}"
            )

    if len(token_ids) < MIN_REPETITION_TOKENS:
        return
    ngrams = [
        tuple(token_ids[index : index + REPEAT_NGRAM_SIZE])
        for index in range(len(token_ids) - REPEAT_NGRAM_SIZE + 1)
    ]
    counts = Counter(ngrams)
    repeated = {ngram for ngram, count in counts.items() if count > 1}
    covered = bytearray(len(token_ids))
    for index, ngram in enumerate(ngrams):
        if ngram in repeated:
            covered[index : index + REPEAT_NGRAM_SIZE] = b"\x01" * REPEAT_NGRAM_SIZE
    repeat_coverage = sum(covered) / len(token_ids)
    max_repeats = max(counts.values(), default=0)
    if repeat_coverage >= MAX_REPEAT_COVERAGE or max_repeats >= MAX_NGRAM_REPEATS:
        raise ValueError(
            "Excessive repeated transcript text: "
            f"coverage={repeat_coverage:.0%}, max_ngram_repeats={max_repeats}"
        )


def format_transcript(result: dict, tokenizer, max_output_tokens: int) -> dict:
    """Convert a Transformers pipeline result to MARS-friendly segments."""
    segments = []
    timestamped_text = []
    for chunk in result.get("chunks", []):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        timestamp = chunk.get("timestamp")
        if (
            not isinstance(timestamp, (list, tuple))
            or len(timestamp) != 2
            or timestamp[0] is None
            or timestamp[1] is None
        ):
            raise ValueError(f"Missing timestamp for chunk: {text!r}")
        start, end = map(float, timestamp)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"Non-finite timestamp {start}-{end}")
        if start < 0 or end < start:
            raise ValueError(f"Invalid timestamp {start}-{end}")
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )
        timestamped_text.append(
            f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
        )

    if str(result.get("text", "")).strip() and not segments:
        raise ValueError("Whisper returned text but no valid timestamped segments")
    validate_text_quality(segments, tokenizer, max_output_tokens)

    return {
        "transcript": "\n".join(timestamped_text),
        "segments": segments,
    }


def transcribe_batch(
    asr,
    videos: list[Path],
    language: str | None,
    batch_size: int,
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
) -> list[dict]:
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    if temperature > 0:
        generate_kwargs.update(
            do_sample=True,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

    results = asr(
        [str(video) for video in videos],
        batch_size=batch_size,
        return_timestamps=True,
        generate_kwargs=generate_kwargs,
    )
    if len(results) != len(videos):
        raise RuntimeError(f"Expected {len(videos)} results, got {len(results)}")
    return results


def parse_with_retries(
    asr,
    videos: list[Path],
    language: str | None,
    batch_size: int,
) -> tuple[list[dict | None], list[Exception | None]]:
    """Parse a batch and rerun only samples with invalid transcripts."""
    raw_results = transcribe_batch(asr, videos, language, batch_size)
    max_output_tokens = int(
        getattr(asr.model.generation_config, "max_length", None)
        or asr.model.config.max_target_positions
    )
    transcripts: list[dict | None] = [None] * len(videos)
    errors: list[Exception | None] = [None] * len(videos)
    retry_indices = []

    for index, result in enumerate(raw_results):
        try:
            transcripts[index] = format_transcript(
                result,
                asr.tokenizer,
                max_output_tokens,
            )
        except Exception as error:
            errors[index] = error
            retry_indices.append(index)

    for attempt in range(1, MAX_PARSE_RETRIES + 1):
        if not retry_indices:
            break
        temperature = round(RETRY_TEMPERATURE + (attempt - 1) * RETRY_TEMPERATURE_STEP, 2)
        repetition_penalty = round(
            RETRY_REPETITION_PENALTY
            + (attempt - 1) * RETRY_REPETITION_PENALTY_STEP,
            2,
        )
        names = ", ".join(videos[index].stem for index in retry_indices)
        print(
            f"Parse retry {attempt}/{MAX_PARSE_RETRIES} ({names}): "
            f"temperature={temperature}, repetition_penalty={repetition_penalty}"
        )

        try:
            retry_results = transcribe_batch(
                asr,
                [videos[index] for index in retry_indices],
                language,
                batch_size,
                temperature,
                repetition_penalty,
            )
        except Exception as error:
            for index in retry_indices:
                errors[index] = error
            continue

        remaining = []
        for index, result in zip(retry_indices, retry_results):
            try:
                transcript = format_transcript(
                    result,
                    asr.tokenizer,
                    max_output_tokens,
                )
                transcript.update(
                    retried=True,
                    retries_attempted=attempt,
                    retry_temperature=temperature,
                    retry_repetition_penalty=repetition_penalty,
                )
                transcripts[index] = transcript
                errors[index] = None
            except Exception as error:
                errors[index] = error
                remaining.append(index)
        retry_indices = remaining

    return transcripts, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe MARS2 videos with Transformers Whisper large-v3"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Mars2Dataset/MAC_QA.jsonl"),
        help="QA JSONL used to select video IDs",
    )
    parser.add_argument(
        "--videos",
        type=Path,
        default=Path("Mars2Dataset/mars2_videos"),
    )
    parser.add_argument("--output", type=Path, default=Path("asr_mac.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--language", help="Language code such as en; auto by default")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of 30-second chunks inferred together; reduce if CUDA OOM",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    video_ids = list(dict.fromkeys(row["id"] for row in read_jsonl(args.input)))
    done = (
        {row["id"] for row in read_jsonl(args.output)}
        if args.output.exists()
        else set()
    )
    video_ids = [video_id for video_id in video_ids if video_id not in done]
    if args.limit is not None:
        video_ids = video_ids[: args.limit]
    if not video_ids:
        print("No unfinished videos.")
        return
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        device = "cuda:0"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(f"Loading {args.model} on {device}...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        torch_dtype=dtype,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = []
    failure_count = 0
    for index, video_id in enumerate(video_ids, 1):
        video = (args.videos / f"{video_id}.mp4").resolve()
        if not video.exists():
            print(f"[{index}/{len(video_ids)}] missing: {video}")
            failure_count += 1
        else:
            pending.append((index, video_id, video))

    with args.output.open("a", encoding="utf-8") as f:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            try:
                transcripts, errors = parse_with_retries(
                    asr,
                    [video for _, _, video in batch],
                    args.language,
                    args.batch_size,
                )
            except Exception as error:
                if "torchcodec" in str(error).lower() or "GLIBCXX_" in str(error):
                    raise RuntimeError(
                        "TorchCodec cannot load in this environment. It is not "
                        "needed here; run `python -m pip uninstall -y torchcodec` "
                        "and retry so Transformers uses FFmpeg instead."
                    ) from error
                ids = ", ".join(video_id for _, video_id, _ in batch)
                print(f"Batch failed ({ids}): {error}")
                failure_count += len(batch)
                continue

            for (index, video_id, _), transcript, error in zip(
                batch, transcripts, errors
            ):
                if transcript is None:
                    print(
                        f"[{index}/{len(video_ids)}] failed after "
                        f"{MAX_PARSE_RETRIES} retries: {error}"
                    )
                    failure_count += 1
                    continue
                result = {"id": video_id, **transcript}
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[{index}/{len(video_ids)}] {video_id}: "
                    f"{len(result['segments'])} segments"
                )

    print(f"Saved transcripts to {args.output}; failures: {failure_count}")


if __name__ == "__main__":
    main()
