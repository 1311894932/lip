import argparse
import os
import subprocess
import time
import shutil
from dataclasses import dataclass
from datetime import datetime

# --- 兼容补丁:huggingface_hub 新版本移除了 HfFolder,gradio 4.44.0 仍需要它 ---
import huggingface_hub
if not hasattr(huggingface_hub, "HfFolder"):
    class HfFolder:
        @staticmethod
        def get_token():
            try:
                return huggingface_hub.get_token()
            except Exception:
                return None

        @staticmethod
        def save_token(token):
            try:
                huggingface_hub.login(token=token, add_to_git_credential=False)
            except Exception:
                pass

        @staticmethod
        def delete_token():
            pass

    huggingface_hub.HfFolder = HfFolder
# --- 补丁结束 ---


import gradio as gr
import torch
from PIL import Image

from diffsynth.pipelines.lip_sync import LipSyncPipeline, ModelConfig
from face_tracking import (
    prepare_face_selection_image,
    process_video_with_face_tracking,
    draw_face_bboxes_on_frame,
)
from utils import (
    blend_crop_video_with_ref,
    color_correction,
    get_total_length,
    make_pingpong_indices,
    paste_video_back,
    save_video_with_audio,
)

CLIP_NUM_FRAMES = 77
MOTION_NUM_FRAMES = 5

# Model weights on the shared storage
CHECKPOINTS_DIR = "/afs-mt/shanhai-ai/zhouyang/zy_projects/X-Dub/checkpoints"
DEFAULT_DIT_PATH = os.path.join(CHECKPOINTS_DIR, "X-Dub_model.safetensors")
DEFAULT_TEXT_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "models_t5_umt5-xxl-enc-bf16.safetensors")
DEFAULT_VAE_PATH = os.path.join(CHECKPOINTS_DIR, "Wan2.2_VAE.safetensors")
DEFAULT_TOKENIZER_PATH = os.path.join(CHECKPOINTS_DIR, "umt5-xxl")
DEFAULT_WHISPER_PATH = os.path.join(CHECKPOINTS_DIR, "whisper", "large-v2.pt")
DEFAULT_WAV2VEC_PATH = os.path.join(CHECKPOINTS_DIR, "wav2vec2-base-960h")

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

# Global pipeline - loaded once at startup
pipe = None

# Global state for face selection
_selected_video_path = None
_detected_bboxes = None
_current_output_dir = None  # Shared output directory for the current session
_video_info = None  # Store video info (frame count, width, height)
_video_has_audio = False  # Whether the uploaded reference video carries an audio stream
_use_video_audio = True  # Prefer the video's own audio track over a separate upload
_final_output_path = None  # Optional flat directory collecting every final output


def has_audio_stream(video_path):
    """Return True if the container carries at least one audio stream."""
    if not video_path or not os.path.exists(video_path):
        return False
    try:
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            result = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    video_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return bool(result.stdout.strip())
        # ffprobe missing: fall back to parsing the `ffmpeg -i` banner
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", video_path],
            capture_output=True,
            text=True,
            check=False,
        )
        return "Audio:" in result.stderr
    except Exception as e:
        print(f"[audio probe] failed on {video_path}: {e}")
        return False


def extract_audio_from_video(video_path, output_path):
    """Extract the video's audio track into the 16 kHz mono wav the pipeline expects."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            output_path,
        ],
        check=True,
    )
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Extracted audio track is empty: {video_path}")
    return output_path


def collect_final_output(src_path, run_name):
    """Copy a finished output into ``_final_output_path`` for later review.

    Files land flat in that directory, prefixed with the run name (which already
    carries a timestamp), so repeated runs stay distinguishable. Returns the
    destination path, or ``src_path`` when no collection directory is configured.
    """
    if _final_output_path is None or src_path is None or not os.path.exists(src_path):
        return src_path

    dst_path = os.path.join(_final_output_path, f"{run_name}_{os.path.basename(src_path)}")
    try:
        os.makedirs(_final_output_path, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        print(f"Collected final output -> {dst_path}")
    except Exception as e:
        # Never fail a finished generation just because the copy failed.
        print(f"WARNING: could not collect {src_path} into {_final_output_path}: {e}")
        return src_path
    return dst_path


def get_video_info(video_file):
    """Get video info (frame count, width, height, audio presence)."""
    global _video_info, _video_has_audio
    if video_file is None:
        _video_info = None
        _video_has_audio = False
        return "No video uploaded"

    try:
        import cv2
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return "Error: Could not open video"

        # Get frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Get width and height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Get FPS
        fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        _video_info = {
            "num_frames": frame_count,
            "width": width,
            "height": height,
            "fps": fps
        }
        _video_has_audio = has_audio_stream(video_file)
        audio_tag = "audio: yes (can be used directly)" if _video_has_audio else "audio: none"
        return f"Video: {frame_count} frames, {width}x{height}, {fps:.2f} fps, {audio_tag}"
    except Exception as e:
        return f"Error reading video: {str(e)}"


@dataclass
class PreprocessedLipSyncSample:
    raw_video: list[Image.Image]
    ref_video: list[Image.Image]
    bboxes: list[list[int]]
    audio_path: str


def smooth_transition_latent(previous_segment, current_segment):
    boundary_latent = 0.5 * (previous_segment[:, :, -1:, :, :] + current_segment[:, :, 1:2, :, :])
    previous_segment = previous_segment.clone()
    previous_segment[:, :, -1:, :, :] = boundary_latent
    return previous_segment, current_segment[:, :, 2:, :, :]


def load_model():
    """Load the model once at startup."""
    global pipe
    print("Loading model...")
    start_time = time.time()
    pipe = LipSyncPipeline().from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=DEFAULT_DIT_PATH, **vram_config),
            ModelConfig(path=DEFAULT_TEXT_ENCODER_PATH, **vram_config),
            ModelConfig(path=DEFAULT_VAE_PATH, **vram_config),
        ],
        tokenizer_config=ModelConfig(path=DEFAULT_TOKENIZER_PATH),
        args=argparse.Namespace(
            ref_cfg_scale=2.5,
            audio_cfg_scale=10.0,
            audio_feat_window_size=0,
            num_inference_steps=50,
            seed=42,
        ),
        whisper_ckpt_path=DEFAULT_WHISPER_PATH,
        wav2vec_ckpt_path=DEFAULT_WAV2VEC_PATH,
    )
    pipe.whisper_processor.to(dtype=torch.float32)
    pipe.wav2vec_processor.to(dtype=torch.float32)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    return pipe


def validate_preprocessed_sample(sample: PreprocessedLipSyncSample):
    assert os.path.exists(sample.audio_path), f"audio path not found: {sample.audio_path}"
    assert len(sample.ref_video) > 0, "ref_video cannot be empty."
    assert len(sample.raw_video) == len(sample.ref_video), "raw_video and ref_video must have the same length."
    assert len(sample.bboxes) == len(sample.ref_video), "bboxes and ref_video must have the same length."
    for bbox in sample.bboxes:
        assert len(bbox) == 4, f"bbox must have 4 ints, got: {bbox}"
    return sample


def infer_one_sample(pipe, sample: PreprocessedLipSyncSample, sample_name: str, output_dir: str,
                     ref_cfg_scale: float = 2.5, audio_cfg_scale: float = 10.0,
                     num_inference_steps: int = 30, seed: int = 42):
    sample = validate_preprocessed_sample(sample)
    raw_video = sample.raw_video
    ref_video = sample.ref_video
    bboxes = sample.bboxes
    audio_path = sample.audio_path

    original_num_frames = len(raw_video)

    print(f"[Sample {sample_name}] audio_path={audio_path}")
    print(f"[Sample {sample_name}] num_ref_frames={len(ref_video)}")

    # Check if ref_video is already 512x512 or needs resizing
    first_frame_size = ref_video[0].size
    needs_resize = first_frame_size != (512, 512)
    original_crop_size = first_frame_size[0]  # Assuming square crops

    if needs_resize:
        print(f"[Sample {sample_name}] Resizing crops from {first_frame_size} to 512x512 for inference")
        # Store original sizes for resizing back
        original_sizes = [(bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in bboxes]
        # Resize ref_video to 512x512 for inference
        ref_video_resized = [frame.resize((512, 512), Image.BILINEAR) for frame in ref_video]
    else:
        ref_video_resized = ref_video
        original_sizes = None

    total_length = get_total_length(audio_path, clip_num_frames=CLIP_NUM_FRAMES, motion_num_frames=MOTION_NUM_FRAMES)
    num_clips = 1 + max(0, total_length - CLIP_NUM_FRAMES) // (CLIP_NUM_FRAMES - MOTION_NUM_FRAMES)
    pingpong_indices = make_pingpong_indices(len(ref_video_resized), total_length)
    ref_video_resized = [ref_video_resized[index] for index in pingpong_indices]
    raw_video = [raw_video[index] for index in pingpong_indices]
    bboxes = [bboxes[index] for index in pingpong_indices]

    print(f"[Sample {sample_name}] total_length={total_length}, num_clips={num_clips}")

    motion_video = None
    whisper_feat = None
    wav2vec_feat = None
    latents_segments = []
    start_idx = 0

    for clip_idx in range(num_clips):
        print(f"[Sample {sample_name}] [{clip_idx + 1}/{num_clips}] start_idx={start_idx}")
        ref_video_clip = ref_video_resized[start_idx: start_idx + CLIP_NUM_FRAMES]
        output_video, outputs = pipe(
            ref_video=ref_video_clip,
            start_idx=start_idx,
            audio_npy_path=None,
            audio_wav_path=audio_path,
            whisper_feat=whisper_feat,
            wav2vec_feat=wav2vec_feat,
            prompt="",
            motion_video=motion_video,
            height=512,
            width=512,
            num_frames=CLIP_NUM_FRAMES,
            motion_latents_num_frames=2,
            ref_cfg_scale=ref_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            use_dynamic_cfg=True,
            replace_border_latents=True,
            replace_border_latents_width=1,
        )

        if whisper_feat is None:
            whisper_feat = outputs.get("whisper_feat", None)
        if wav2vec_feat is None:
            wav2vec_feat = outputs.get("wav2vec_feat", None)

        output_video = color_correction(output_video, ref_video_clip, global_align=False)
        motion_video = output_video[-MOTION_NUM_FRAMES:]
        output_latents = outputs["latents"]

        if clip_idx == 0:
            latents_segments.append(output_latents)
        else:
            latents_segments[-1], output_latents_to_append = smooth_transition_latent(latents_segments[-1], output_latents)
            latents_segments.append(output_latents_to_append)

        start_idx += CLIP_NUM_FRAMES - MOTION_NUM_FRAMES

    output_latents = torch.cat(latents_segments, dim=2)
    pipe.load_models_to_device(["vae"])
    final_video = pipe.vae.decode(output_latents, device=pipe.device, tiled=False, tile_size=(32, 32), tile_stride=(16, 16))
    final_output_video = pipe.vae_output_to_video(final_video)[:total_length]
    final_output_video = color_correction(
        final_output_video, ref_video_resized[:total_length], global_align=False
    )

    # Resize back to original crop sizes if needed
    if needs_resize and original_sizes is not None:
        print(f"[Sample {sample_name}] Resizing output back to original crop sizes")
        final_output_video_resized = []
        for frame, size in zip(final_output_video, original_sizes[:total_length]):
            if size[0] > 0 and size[1] > 0:
                final_output_video_resized.append(frame.resize(size, Image.BILINEAR))
            else:
                final_output_video_resized.append(frame)
        final_output_video = final_output_video_resized

    blended_crop_video = blend_crop_video_with_ref(
        final_output_video,
        ref_video[:total_length],
        replace_border_latents_width=1,
    )

    final_pasted_video = paste_video_back(blended_crop_video, raw_video, bboxes)
    final_pasted_video = final_pasted_video[:original_num_frames]

    output_path = save_video_with_audio(final_pasted_video, audio_path, f"{sample_name}_output", output_dir)
    return output_path


# Step 1: Upload video and detect faces on its first frame
def detect_faces_in_video(video_file):
    """Detect faces on the first frame of the uploaded video and return the annotated image.

    Returns updates for: face selection image, status, face selection group,
    face index, selected face preview, preview status.
    """
    global _selected_video_path, _detected_bboxes, _current_output_dir

    # Video cleared: drop the old selection and hide the selection area
    if video_file is None:
        _selected_video_path = None
        _detected_bboxes = None
        return None, "", gr.update(visible=False), 0, None, ""

    # Create a new output directory for each video upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    _current_output_dir = os.path.join("./results", f"{video_name}_{timestamp}")
    os.makedirs(_current_output_dir, exist_ok=True)

    # Copy video to output directory
    video_path = os.path.join(_current_output_dir, os.path.basename(video_file))
    shutil.copy(video_file, video_path)

    _selected_video_path = video_path
    _detected_bboxes = None  # Stays unset if detection fails, so faces from a previous video can't be used

    try:
        annotated_image, bboxes = prepare_face_selection_image(video_path)
        _detected_bboxes = bboxes

        # Save annotated image
        annotated_path = os.path.join(_current_output_dir, "face_selection.png")
        annotated_image.save(annotated_path)
    except Exception as e:
        import traceback
        error_msg = f"Error detecting faces: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        raise gr.Error(error_msg)

    if not bboxes:
        return annotated_path, "No Face Detected", gr.update(visible=False), 0, None, ""

    status = f"Detected {len(bboxes)} face(s). Please select a face by entering the index (0-{len(bboxes)-1})."
    return annotated_path, status, gr.update(visible=True), 0, None, ""


# Step 1.5: Preview selected face
def preview_selected_face(face_idx):
    """Preview the selected face on the image."""
    global _selected_video_path, _detected_bboxes, _current_output_dir

    if _selected_video_path is None or _detected_bboxes is None:
        return None, "Please upload a video first."

    try:
        face_idx = int(face_idx)
    except ValueError:
        return None, "Face index must be an integer."

    if face_idx < 0 or face_idx >= len(_detected_bboxes):
        return None, f"Invalid face index. Please enter a number between 0 and {len(_detected_bboxes) - 1}."

    # Read first frame
    import imageio.v2 as imageio
    import cv2
    reader = imageio.get_reader(_selected_video_path)
    first_frame = reader.get_data(0)
    reader.close()

    first_frame_pil = Image.fromarray(first_frame)

    # Draw only the selected face bbox
    annotated = draw_face_bboxes_on_frame(first_frame_pil, _detected_bboxes, selected_idx=face_idx)

    # Save preview (use the same output directory)
    preview_path = os.path.join(_current_output_dir, "face_preview.png")
    annotated.save(preview_path)

    return preview_path, f"Selected face {face_idx}. Click 'Generate Lip Sync' to proceed."


# Step 2: Process with selected face
def process_with_selected_face(face_idx, audio_file, use_video_audio=True):
    """Process video with selected face and audio.

    Audio resolution order:
      1. the reference video's own track (when ``use_video_audio`` is on and the
         video actually has one),
      2. the separately uploaded audio file,
      3. error.
    """
    global pipe, _selected_video_path, _detected_bboxes, _current_output_dir, _use_video_audio, _video_has_audio

    if pipe is None:
        raise gr.Error("Model not loaded. Please wait for the model to load.")

    if _selected_video_path is None:
        raise gr.Error("Please upload a video first.")

    if not _detected_bboxes:
        raise gr.Error("No Face Detected. Please upload a video with a visible face.")

    _use_video_audio = bool(use_video_audio)

    # Validate face index (only if multiple faces detected)
    if len(_detected_bboxes) > 1:
        try:
            face_idx = int(face_idx)
        except ValueError:
            raise gr.Error("Face index must be an integer.")

        if face_idx < 0 or face_idx >= len(_detected_bboxes):
            raise gr.Error(f"Invalid face index. Please enter a number between 0 and {len(_detected_bboxes) - 1}.")
    else:
        # Single face, use index 0
        face_idx = 0

    # Use the existing output directory
    output_dir = _current_output_dir

    # Resolve the audio source
    sample_name = os.path.splitext(os.path.basename(_selected_video_path))[0]
    audio_source = None
    if _use_video_audio and _video_has_audio:
        audio_source = "video"
        audio_path = os.path.join(output_dir, f"{sample_name}_from_video.wav")
        try:
            extract_audio_from_video(_selected_video_path, audio_path)
        except Exception as e:
            raise gr.Error(f"Failed to extract audio from the video: {e}\n\nUntick 'Use audio from reference video' and upload an audio file instead.")
        print(f"Using audio extracted from the reference video: {audio_path}")
    elif audio_file is not None:
        audio_source = "upload"
        audio_path = os.path.join(output_dir, os.path.basename(audio_file))
        shutil.copy(audio_file, audio_path)
        print(f"Using uploaded audio: {audio_path}")
    else:
        if _use_video_audio:
            raise gr.Error("The reference video has no audio track, and no audio file was uploaded. Please upload a target audio file.")
        raise gr.Error("Please upload an audio file, or enable 'Use audio from reference video'.")

    try:
        print(f"Processing with face index {face_idx}, bbox {_detected_bboxes[face_idx]}...")

        # Process video with face tracking
        raw_video, face_video, tracked_bboxes = process_video_with_face_tracking(
            _selected_video_path,
            selected_bbox=_detected_bboxes[face_idx],
            device="cuda",
        )

        print(f"Tracked {len(tracked_bboxes)} frames")

        # Create sample for inference
        sample = PreprocessedLipSyncSample(
            raw_video=raw_video,
            ref_video=face_video,
            bboxes=tracked_bboxes,
            audio_path=audio_path,
        )

        print("Running inference...")
        output_path = infer_one_sample(
            pipe, sample, sample_name, output_dir,
            ref_cfg_scale=2.5,
            audio_cfg_scale=10.0,
            num_inference_steps=30,
            seed=42,
        )

        # Collect the final output into the shared review directory, if configured.
        run_name = os.path.basename(os.path.normpath(output_dir))
        collected_main = collect_final_output(output_path, run_name)

        print("Done!")
        source_label = "the reference video's own audio track" if audio_source == "video" else "the uploaded audio file"
        status = f"Success! (audio from {source_label}) Output saved to: {output_path}"
        if _final_output_path is not None:
            status += f" | collected into: {_final_output_path} as {os.path.basename(collected_main)}"
        return output_path, status

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        raise gr.Error(error_msg)


def create_ui():
    """Create the Gradio UI with interactive face selection."""
    with gr.Blocks(title="X-Dub Lip Sync") as demo:
        gr.Markdown("# X-Dub Lip Sync with Face Tracking")
        gr.Markdown("Upload a video to detect faces, select a person, then upload audio to generate lip-synced video. If the video already contains an audio track it is used automatically, so the audio upload can be skipped.")

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Reference Video", sources=["upload"])
                video_info_text = gr.Textbox(label="Video Info", interactive=False, value="No video uploaded")
                face_selection_image = gr.Image(label="Face Selection", interactive=False)
                status_text = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                # Hidden initially, shown after a face is detected
                with gr.Group(visible=False) as face_selection_group:
                    face_idx_input = gr.Number(
                        label="Face Index",
                        value=0,
                        minimum=0,
                        precision=0,
                        info="Enter the index of the face you want to track (shown in the image above)"
                    )
                    preview_btn = gr.Button("Preview Selected Face", variant="secondary")
                    face_preview_image = gr.Image(label="Selected Face Preview", interactive=False)
                    preview_status = gr.Textbox(label="Preview Status", interactive=False)
                    audio_input = gr.Audio(label="Target Audio", sources=["upload"], type="filepath")
                    use_video_audio_checkbox = gr.Checkbox(
                        label="Use audio from reference video",
                        value=True,
                        info="When the uploaded video already has an audio track, use it directly and skip the "
                             "audio upload above. Untick to use the uploaded audio file instead. "
                             "The video's audio is extracted automatically (16 kHz mono)."
                    )
                    process_btn = gr.Button("Generate Lip Sync", variant="primary")

        with gr.Row():
            output_video = gr.Video(label="Output Video")
            output_status = gr.Textbox(label="Output Status", interactive=False)

        # Event handlers: uploading (or clearing) a video re-runs face detection on its first frame
        video_input.change(
            fn=get_video_info,
            inputs=[video_input],
            outputs=[video_info_text]
        ).then(
            fn=detect_faces_in_video,
            inputs=[video_input],
            outputs=[face_selection_image, status_text, face_selection_group, face_idx_input,
                     face_preview_image, preview_status]
        )

        preview_btn.click(
            fn=preview_selected_face,
            inputs=[face_idx_input],
            outputs=[face_preview_image, preview_status]
        )

        process_btn.click(
            fn=process_with_selected_face,
            inputs=[face_idx_input, audio_input, use_video_audio_checkbox],
            outputs=[output_video, output_status]
        )

    return demo


def main():
    import argparse

    global _final_output_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--server_name", type=str, default=None, help="Server name for share link")
    parser.add_argument(
        "--final_output_path",
        type=str,
        default=None,
        help="Directory that collects a copy of every final output video, one file per run.",
    )
    args = parser.parse_args()

    if args.final_output_path:
        _final_output_path = os.path.abspath(os.path.expanduser(args.final_output_path))
        os.makedirs(_final_output_path, exist_ok=True)
        print(f"Final outputs will also be collected into: {_final_output_path}")

    # Load model at startup
    load_model()

    # Create and launch UI
    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
