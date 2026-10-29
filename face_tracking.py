"""
Face tracking module using dwpose detection and arcface similarity.
Supports interactive face selection and tracking across video frames.
"""

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
import imageio.v2 as imageio

# Vendored third-party code (ArcFace, DWPose) lives next to this file; weights are on the shared storage.
THIRD_PARTY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")
ARCFACE_CKPT_PATH = "/afs-mt/shanhai-ai/zhouyang/zy_projects/omnisync/face_model_weights/arcface/backbone.pth"

# ArcFace: id_loss.py + insightface arcface_torch backbones
sys.path.append(os.path.join(THIRD_PARTY_DIR, "arcface"))

from id_loss import ArcFaceLoss

# DWPose (ONNX version), importable as annotator.dwpose
sys.path.insert(0, THIRD_PARTY_DIR)

_MAX_CROP_SIZE = 1024  # Maximum crop size to avoid memory issues
_ARCFACE_MODEL = None
_DWPOSE_DETECTOR = None


def get_arcface_model(device='cuda'):
    """Get or create arcface model for face similarity."""
    global _ARCFACE_MODEL
    if _ARCFACE_MODEL is None:
        ckpt_path = ARCFACE_CKPT_PATH
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ArcFace checkpoint not found: {ckpt_path}")
        _ARCFACE_MODEL = ArcFaceLoss(ckpt_path=ckpt_path, device=device, backbone='r100', input_is_bgr=True)
    return _ARCFACE_MODEL


def get_dwpose_detector():
    """Get or create dwpose detector (ONNX version)."""
    global _DWPOSE_DETECTOR
    if _DWPOSE_DETECTOR is None:
        from annotator.dwpose import DWposeDetector
        _DWPOSE_DETECTOR = DWposeDetector()
    return _DWPOSE_DETECTOR


def detect_faces(frame_bgr: np.ndarray) -> List[List[int]]:
    """
    Detect all faces in a frame using dwpose landmarks (ONNX version).
    Returns one square bbox per face, [x1, y1, x2, y2] in pixel coordinates,
    sized from the face landmarks (see ``_bbox_from_landmarks``).
    """
    height, width = frame_bgr.shape[:2]

    # candidate shape: [N, 68, 2] face landmarks for each detected person
    candidate, _ = get_dwpose_detector().get_lmks(frame_bgr)

    face_bboxes = []
    for face_landmarks in candidate:
        # Filter out invalid points (coordinates out of bounds or -1)
        x_coords = face_landmarks[:, 0]
        y_coords = face_landmarks[:, 1]
        valid_mask = (x_coords > 0) & (y_coords > 0) & (x_coords < width) & (y_coords < height)
        if valid_mask.sum() < 5:
            continue

        face_bboxes.append(_bbox_from_landmarks(x_coords[valid_mask], y_coords[valid_mask], width, height))

    return face_bboxes


def extract_face_embedding(frame_bgr: np.ndarray, bbox: List[int], device='cuda') -> torch.Tensor:
    """
    Extract arcface embedding for a face from given bbox.
    """
    x1, y1, x2, y2 = bbox

    # Crop face
    face = frame_bgr[y1:y2, x1:x2]

    if face.size == 0:
        # Return zero embedding if crop is empty
        return torch.zeros(512, device=device)

    # Resize to 112x112 (arcface input size)
    face_resized = cv2.resize(face, (112, 112))

    # Convert to tensor: BGR -> RGB -> [-1, 1]
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
    face_tensor = face_tensor.unsqueeze(0).to(device)  # [1, 3, 112, 112]

    # Get embedding
    arcface = get_arcface_model(device)
    with torch.no_grad():
        embedding = arcface.net(arcface.preprocess(face_tensor))
        embedding = embedding.squeeze(0)  # [512]

    return embedding


def compute_face_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """Compute cosine similarity between two face embeddings."""
    cos = torch.nn.functional.cosine_similarity(
        embedding1.unsqueeze(0), embedding2.unsqueeze(0), dim=-1
    )
    return cos.item()


def track_face_across_frames_with_scores(
    frames_bgr: List[np.ndarray],
    initial_bbox: List[int],
    initial_embedding: torch.Tensor,
    device='cuda',
    similarity_threshold: float = 0.3,
) -> Tuple[List[List[int]], List[float]]:
    """
    Detect on every frame and match each detection against the reference face with
    ArcFace, returning both the chosen bbox per frame and the best similarity
    reached on that frame.

    Frames where the tracker fell back to the previous bbox (nothing detected, or
    nothing scored above ``similarity_threshold``) are reported with a score of
    -1.0, so callers can exclude them from aggregate statistics.
    """
    tracked_bboxes = [initial_bbox]
    scores: List[float] = [1.0]  # the seed bbox is by definition the reference face

    for i in range(1, len(frames_bgr)):
        frame = frames_bgr[i]
        current_face_bboxes = detect_faces(frame)

        if not current_face_bboxes:
            print(f"Frame {i}: No faces detected, using previous bbox")
            tracked_bboxes.append(tracked_bboxes[-1])
            scores.append(-1.0)
            continue

        # Pick the detected face that is most similar to the reference identity
        best_bbox = tracked_bboxes[-1]
        best_similarity = -1.0

        for bbox in current_face_bboxes:
            try:
                embedding = extract_face_embedding(frame, bbox, device)
                similarity = compute_face_similarity(initial_embedding, embedding)
            except Exception:
                continue
            if similarity > best_similarity:
                best_similarity = similarity
                best_bbox = bbox

        if best_similarity < similarity_threshold:
            print(f"Frame {i}: Best similarity {best_similarity:.3f} below threshold {similarity_threshold}, using previous bbox")
            best_bbox = tracked_bboxes[-1]
            best_similarity = -1.0

        tracked_bboxes.append(best_bbox)
        scores.append(best_similarity)

    return tracked_bboxes, scores


def _bbox_from_landmarks(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    width: int,
    height: int,
) -> List[int]:
    """Build a square face bbox from valid landmark coordinates (shared helper)."""
    center_x = float(np.mean(x_coords))
    center_y = float(np.mean(y_coords))

    # Side from the landmark extent plus margin (similar to lip_sync_preprocess.py)
    face_width = float(np.max(x_coords)) - float(np.min(x_coords))
    face_height = float(np.max(y_coords)) - float(np.min(y_coords))
    max_side = min(max(face_width, face_height) * 1.45, _MAX_CROP_SIZE)

    x1 = int(center_x - max_side / 2)
    y1 = int(center_y - max_side / 2)
    x2 = int(center_x + max_side / 2)
    y2 = int(center_y + max_side / 2)

    return [
        max(0, x1),
        max(0, y1),
        min(width, x2),
        min(height, y2),
    ]


def compute_stable_crop_window(
    bboxes: List[List[int]],
    scores: List[float],
    frame_shape: Tuple[int, ...],
    similarity_threshold: float = 0.3,
    percentile: float = 95.0,
    fallback_bbox: Optional[List[int]] = None,
) -> List[int]:
    """
    Collapse per-frame face bboxes into ONE fixed crop window used for every frame.

    Face detection still runs on every frame (so ArcFace can keep identifying the
    right person in multi-person footage), but the window used for cropping and
    pasting is computed once. The face therefore keeps its natural size and
    position *within* the window as the person moves, instead of the crop
    chasing the face frame by frame.

    Only frames whose ArcFace score reached ``similarity_threshold`` contribute.
    The side length comes from the ``percentile``-th percentile of those frames'
    sizes (nearest-rank, so a drifting frame is excluded rather than blended in),
    widened if needed to also cover the spread of the face centers. A plain
    max() over every frame would let one bad detection blow the window up.
    Pass ``percentile=100`` to disable trimming: the window then covers every
    frame that detected the face, and frames with no detection are cropped with
    that same window.

    Returns [x1, y1, x2, y2] in pixel coordinates, clamped to the frame.
    """
    height, width = frame_shape[:2]

    qualified = [
        bbox for bbox, score in zip(bboxes, scores)
        if score >= similarity_threshold and bbox is not None and len(bbox) == 4
    ]
    # Frames where the face was never confidently found give us nothing to
    # aggregate; fall back to the seed bbox so the pipeline can still run.
    if not qualified:
        if fallback_bbox is not None:
            print("Stable window: no frames above threshold, using fallback bbox")
            return list(fallback_bbox)
        raise ValueError("No frames above similarity threshold; cannot compute a stable crop window.")

    valid = [b for b in qualified if b[2] > b[0] and b[3] > b[1]]
    if not valid:
        if fallback_bbox is not None:
            return list(fallback_bbox)
        raise ValueError("All qualified bboxes are degenerate; cannot compute a stable crop window.")

    centers_x = np.array([(b[0] + b[2]) / 2.0 for b in valid])
    centers_y = np.array([(b[1] + b[3]) / 2.0 for b in valid])
    sides = np.array([max(b[2] - b[0], b[3] - b[1]) for b in valid], dtype=np.float64)

    def _trimmed_bounds(values: np.ndarray) -> Tuple[float, float]:
        """
        Bounds of ``values`` after trimming the extreme ``1 - percentile`` from both
        tails (at least one value per tail when there is enough data).

        Plain ``np.percentile`` would not protect against drift on short clips: with
        linear interpolation a P95 over 10 frames blends the drifting frame into the
        result instead of excluding it.

        ``percentile >= 100`` disables trimming entirely and returns the plain
        min/max, i.e. every contributing frame's box is covered by the window.
        """
        ordered = np.sort(values)
        n = ordered.size
        if percentile >= 100.0 or n < 3:
            return float(ordered[0]), float(ordered[-1])
        trim = int(np.ceil((100.0 - percentile) / 100.0 * n))
        trim = max(1, min(trim, (n - 1) // 2))
        return float(ordered[trim]), float(ordered[n - 1 - trim])

    # Face size: the largest size that survived trimming (with percentile=100,
    # that is simply the largest box among the frames that detected a face).
    _, face_side = _trimmed_bounds(sides)

    # Face travel: how far the face center moved between frames. A face centered at
    # lo and later at hi, with box size s, covers lo-s/2 .. hi+s/2, so the window
    # has to span ``travel + s`` to keep it fully inside at both extremes.
    lo_x, hi_x = _trimmed_bounds(centers_x)
    lo_y, hi_y = _trimmed_bounds(centers_y)

    center_x = (lo_x + hi_x) / 2.0
    center_y = (lo_y + hi_y) / 2.0
    travel = max(hi_x - lo_x, hi_y - lo_y)
    side = face_side + travel

    # A square window cannot be larger than the frame's short side. When the face
    # both moves a long way and needs a large box, the two demands together can
    # exceed that; the window is then capped and the face may drift slightly past
    # its edge at the extremes.
    max_side = float(min(width, height))
    if side > max_side:
        print(
            f"Stable window: required side {side:.0f} exceeds frame short side "
            f"{max_side:.0f}; capping. The face may leave the crop near the "
            f"extremes of its motion."
        )
        side = max_side

    x1 = int(round(center_x - side / 2))
    y1 = int(round(center_y - side / 2))
    x2 = int(round(center_x + side / 2))
    y2 = int(round(center_y + side / 2))

    # Shift (rather than shrink) when the window runs off the frame edge, so the
    # window keeps its size and the face keeps its scale.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        x1 -= x2 - width
        x2 = width
    if y2 > height:
        y1 -= y2 - height
        y2 = height

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    size_tag = "max over qualifying frames" if percentile >= 100.0 else f"P{percentile:g}"
    print(
        f"Stable window: {len(valid)}/{len(bboxes)} frames qualified "
        f"(threshold={similarity_threshold}), side={side:.1f} ({size_tag}), "
        f"window=[{x1}, {y1}, {x2}, {y2}]"
    )
    return [x1, y1, x2, y2]


def crop_face_from_frames(frames: List[Image.Image], bboxes: List[List[int]]) -> List[Image.Image]:
    """Crop the face from each frame using the given bboxes, keeping the bbox size."""
    cropped_frames = []

    for frame, bbox in zip(frames, bboxes):
        x1, y1, x2, y2 = bbox

        # Ensure bbox is within bounds
        w, h = frame.size
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        cropped_frames.append(frame.crop((x1, y1, x2, y2)))

    return cropped_frames


def draw_face_bboxes_on_frame(
    frame: Image.Image,
    bboxes: List[List[int]],
    selected_idx: Optional[int] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> Image.Image:
    """
    Draw face bboxes on frame for visualization.
    selected_idx: index of selected face (will be highlighted)
    colors: list of RGB colors for each bbox
    """
    frame_copy = frame.copy()
    draw = ImageDraw.Draw(frame_copy)

    # Try to load a larger font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()

    if colors is None:
        colors = [(255, 0, 0) for _ in bboxes]  # Default red

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        color = colors[i]

        # Only draw bbox if this is the selected face (or if no face is selected yet)
        if selected_idx is None or i == selected_idx:
            # Draw bbox with thicker line
            draw.rectangle([x1, y1, x2, y2], outline=color, width=6)

            # Draw index number with larger font
            text = str(i)
            # Draw text background for better visibility
            text_bbox = draw.textbbox((x1 + 10, y1 + 10), text, font=font)
            draw.rectangle(text_bbox, fill=(0, 0, 0))
            draw.text((x1 + 10, y1 + 10), text, fill=(255, 255, 255), font=font)

    return frame_copy


def prepare_face_selection_image(video_path: str) -> Tuple[Image.Image, List[List[int]]]:
    """
    Detect faces on the first frame and draw their bboxes for user selection.
    Returns the annotated image and list of bboxes (empty if no face was found).
    """
    reader = imageio.get_reader(video_path)
    first_frame = reader.get_data(0)
    reader.close()

    # imageio returns RGB; detection runs on BGR (OpenCV convention)
    bboxes = detect_faces(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    print(f"DEBUG: Detected {len(bboxes)} faces on the first frame")

    annotated = draw_face_bboxes_on_frame(Image.fromarray(first_frame), bboxes)
    return annotated, bboxes


def process_video_with_face_tracking(
    video_path: str,
    selected_bbox: List[int],
    device='cuda',
) -> Tuple[List[Image.Image], List[Image.Image], List[List[int]]]:
    """
    Process video with face tracking.
    Returns:
        raw_video: original frames
        face_video: cropped face frames
        bboxes: bbox used for each frame (the same stable window on every frame)
    selected_bbox: the chosen face's bbox on the first frame, as returned by
        ``prepare_face_selection_image``.

    Per-frame detection still runs so ArcFace can pick the right person on every
    frame, but the crop/paste window is computed once from all frames (see
    ``compute_stable_crop_window``). The face then moves and scales naturally
    inside a fixed window instead of the window jumping to each frame's detection.
    """
    # Read video
    reader = imageio.get_reader(video_path)
    raw_video = []
    frames_bgr = []

    for frame in reader:
        frame_rgb = np.asarray(frame).astype(np.uint8)
        raw_video.append(Image.fromarray(frame_rgb).convert("RGB"))
        frames_bgr.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    reader.close()

    # Get initial embedding
    initial_embedding = extract_face_embedding(frames_bgr[0], selected_bbox, device)

    # Detection + ArcFace still run on every frame; only the window is collapsed.
    tracked_bboxes, scores = track_face_across_frames_with_scores(
        frames_bgr, selected_bbox, initial_embedding, device,
    )
    window = compute_stable_crop_window(
        tracked_bboxes,
        scores,
        frames_bgr[0].shape,
        fallback_bbox=selected_bbox,
    )
    tracked_bboxes = [list(window) for _ in frames_bgr]

    # Crop at the window's native size; resized to 512 later for inference
    face_video = crop_face_from_frames(raw_video, tracked_bboxes)

    return raw_video, face_video, tracked_bboxes


if __name__ == "__main__":
    # Test the module
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_video.mp4")

    # Prepare face selection image
    annotated, bboxes = prepare_face_selection_image(video_path)
    print(f"Detected {len(bboxes)} faces")
    annotated.save("face_selection.png")

    # Process with selected face (index 0)
    raw_video, face_video, tracked_bboxes = process_video_with_face_tracking(video_path, bboxes[0])

    # print(f"Processed {len(face_video)} frames")