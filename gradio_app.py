from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from web_app import process_image_file, process_video_file


@dataclass
class LocalUpload:
    source_path: str
    filename: str | None = None

    def __post_init__(self) -> None:
        if not self.filename:
            self.filename = Path(self.source_path).name

    def save(self, destination_path: str | os.PathLike[str]) -> None:
        shutil.copy2(self.source_path, destination_path)


def _format_records(records: list[dict]) -> str:
    if not records:
        return "No violation records were returned."

    lines = []
    for record in records:
        lines.append(
            f"- Plate: {record.get('plate_text', 'NOT_DETECTED')} | "
            f"Owner: {record.get('owner_name', 'Unknown')} | "
            f"Violations: {record.get('violation_count', 0)} | "
            f"Email: {'Sent' if record.get('email_sent') else 'Not sent'}"
        )
    return "\n".join(lines)


def _image_to_result(upload_path: str | None):
    if not upload_path:
        return None, "Upload or capture an image to start detection.", {}

    result = process_image_file(LocalUpload(upload_path))
    annotated = result.get("annotated_path")

    summary = result.get("summary", {})
    records = result.get("records", [])
    text = [
        f"Source: {result.get('source_name', '')}",
        f"Riders: {summary.get('total_riders', 0)} | Safe: {summary.get('safe_riders', 0)} | Violations: {summary.get('violations', 0)}",
        f"Plates: {', '.join(result.get('plates', [])) if result.get('plates') else 'None detected'}",
        _format_records(records),
    ]
    return annotated, "\n\n".join(text), result


def _video_to_result(upload_path: str | None, frame_stride: int, max_frames: int):
    if not upload_path:
        return "Upload or record a video to start detection.", {}

    result = process_video_file(LocalUpload(upload_path), frame_stride, max_frames)
    frames = result.get("frames", [])
    lines = [
        f"Source: {result.get('video_name', '')}",
        f"Processed frames: {result.get('processed_frames', 0)} | Violations: {result.get('violations', 0)}",
        f"Plates: {', '.join(result.get('plates', [])) if result.get('plates') else 'None detected'}",
    ]
    for frame in frames[:10]:
        lines.append(
            f"- {frame.get('frame_name')} | riders={frame.get('riders', 0)} | violations={frame.get('violations', 0)}"
        )
    return "\n".join(lines), result


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Helmet Violation System", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Helmet Violation System\n"
            "A public demo wrapper for helmet detection, plate OCR, cooldown-aware email logic, and annotated results."
        )

        with gr.Tab("Image detection"):
            image_input = gr.Image(sources=["upload", "webcam"], type="filepath", label="Upload or capture an image")
            run_image = gr.Button("Run image detection", variant="primary")
            image_output = gr.Image(label="Annotated output")
            image_summary = gr.Textbox(label="Summary", lines=10)
            image_json = gr.JSON(label="Raw result")

            run_image.click(
                fn=_image_to_result,
                inputs=[image_input],
                outputs=[image_output, image_summary, image_json],
            )

        with gr.Tab("Video detection"):
            video_input = gr.Video(sources=["upload", "webcam"], label="Upload or record a video")
            frame_stride = gr.Slider(5, 60, value=20, step=1, label="Process every Nth frame")
            max_frames = gr.Slider(5, 120, value=40, step=1, label="Max frames to process")
            run_video = gr.Button("Run video detection", variant="primary")
            video_summary = gr.Textbox(label="Summary", lines=10)
            video_json = gr.JSON(label="Raw result")

            run_video.click(
                fn=_video_to_result,
                inputs=[video_input, frame_stride, max_frames],
                outputs=[video_summary, video_json],
            )

        gr.Markdown(
            "## Sharing and deployment\n"
            "Push this repo to GitHub and create a Hugging Face Space from the repository.\n"
            "If the model weights are too large for the Space, host them externally and download them at startup."
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
