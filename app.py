#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
CoWTracker Gradio Demo.

Interactive web demo for dense point tracking using CoWTracker.

Usage:
    python app.py
    python app.py --checkpoint /path/to/model.pth --port 8086
"""

import glob
import os
import uuid
from typing import Optional

import gradio as gr
import matplotlib
import mediapy
import numpy as np
import PIL.Image
import torch

from cowtracker import CoWTracker
from cowtracker.utils.padding import (
    apply_padding,
    compute_padding_params,
    remove_padding_and_scale_back,
)
from cowtracker.utils.visualization import (
    get_2d_colors,
    get_colors_from_cmap,
    paint_point_track,
)

# --- Constants ---
PREVIEW_WIDTH = 1024
PREVIEW_HEIGHT = 1024
FRAME_LIMIT = 512
# Default checkpoint: None means use the model's default HuggingFace URL
DEFAULT_CHECKPOINT = None

# --- Model Initialization ---


def initialize_model(checkpoint_path: Optional[str] = None):
    """Initialize and load the CoWTracker model once at startup.
    
    Args:
        checkpoint_path: Path to local checkpoint file.
                         If None, downloads from HuggingFace Hub.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    ckpt_path = checkpoint_path if checkpoint_path is not None else DEFAULT_CHECKPOINT
    if ckpt_path:
        print(f"Initializing CoWTracker model from {ckpt_path}...")
    else:
        print("Initializing CoWTracker model from HuggingFace Hub...")

    model = CoWTracker.from_checkpoint(
        ckpt_path,
        device=device,
        dtype=dtype,
    )

    print("Model initialized successfully!")
    return model


# Initialize model once at module level
GLOBAL_MODEL = None


def get_model():
    """Get the global model, initializing if needed."""
    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        GLOBAL_MODEL = initialize_model()
    return GLOBAL_MODEL


# --- Core Logic Functions ---


def preprocess_video_input(video_path):
    """Process uploaded video for tracking."""
    if not video_path:
        return None

    video_arr = mediapy.read_video(video_path)
    video_fps = video_arr.metadata.fps
    num_frames = video_arr.shape[0]

    if num_frames > FRAME_LIMIT:
        gr.Warning(
            f"Video is too long. Truncating to first {FRAME_LIMIT} frames.", duration=5
        )
        video_arr = video_arr[:FRAME_LIMIT]
        num_frames = FRAME_LIMIT

    height, width = video_arr.shape[1:3]
    if height > width:
        new_height, new_width = PREVIEW_HEIGHT, int(PREVIEW_WIDTH * width / height)
    else:
        new_height, new_width = int(PREVIEW_WIDTH * height / width), PREVIEW_WIDTH

    # Resize logic to keep manageable size
    if new_height * new_width > 768 * 1024:
        new_height = new_height * 3 // 4
        new_width = new_width * 3 // 4

    # Make divisible by 16 for ffmpeg compatibility
    new_height, new_width = new_height // 16 * 16, new_width // 16 * 16

    preview_video = mediapy.resize_video(video_arr, (new_height, new_width))
    input_video = preview_video  # using preview size for processing

    preview_video = np.array(preview_video)
    input_video = np.array(input_video)

    return (
        video_arr,
        preview_video,
        preview_video.copy(),
        input_video,
        video_fps,
        preview_video[0],
        gr.update(minimum=0, maximum=num_frames - 1, value=0, interactive=True),
        gr.update(interactive=True),
    )


def choose_frame(frame_num, video_preview_array):
    """Select frame for preview."""
    if video_preview_array is None:
        return None
    return video_preview_array[int(frame_num)]


def paint_video(
    video_preview,
    query_frame,
    video_fps,
    tracks,
    visibs,
    rate=1,
    show_bkg=True,
    cmap="gist_rainbow",
):
    """Paint tracks onto video and save to file."""
    T, H, W, _ = video_preview.shape

    # Get colors based on colormap choice
    if cmap == "bremm":
        xy0 = tracks[:, query_frame]
        colors = get_2d_colors(xy0, H, W)
    else:
        query_count = tracks.shape[0]
        colors = get_colors_from_cmap(query_count, cmap)

    painted_video = paint_point_track(
        video_preview, tracks, visibs, colors, rate=rate, show_bkg=show_bkg
    )

    # Save video
    video_file_name = uuid.uuid4().hex + ".mp4"
    video_path = os.path.join(os.path.dirname(__file__), "tmp")
    video_file_path = os.path.join(video_path, video_file_name)
    os.makedirs(video_path, exist_ok=True)

    # Cleanup old jpgs
    for f in glob.glob(os.path.join(video_path, "*.jpg")):
        os.remove(f)

    # Save frames and compile with ffmpeg
    for ti in range(T):
        temp_out_f = "%s/%03d.jpg" % (video_path, ti)
        im = PIL.Image.fromarray(painted_video[ti])
        im.save(temp_out_f)

    os.system(
        f'ffmpeg -y -hide_banner -loglevel error -f image2 -framerate {video_fps} '
        f'-pattern_type glob -i "{video_path}/*.jpg" -c:v libx264 -crf 20 '
        f'-pix_fmt yuv420p {video_file_path}'
    )

    # Cleanup used jpgs
    for ti in range(T):
        temp_out_f = "%s/%03d.jpg" % (video_path, ti)
        if os.path.exists(temp_out_f):
            os.remove(temp_out_f)

    return video_file_path


def update_vis(
    rate, show_bkg, cmap, video_preview, query_frame, video_fps, tracks, visibs
):
    """Update visualization with new settings."""
    if video_preview is None or len(tracks) == 0:
        return None
    T, H, W, _ = video_preview.shape
    tracks_ = tracks.reshape(H, W, T, 2)[::rate, ::rate].reshape(-1, T, 2)
    visibs_ = visibs.reshape(H, W, T)[::rate, ::rate].reshape(-1, T)
    return paint_video(
        video_preview,
        query_frame,
        video_fps,
        tracks_,
        visibs_,
        rate=rate,
        show_bkg=show_bkg,
        cmap=cmap,
    )


def track(video_preview, video_input, video_fps, query_frame, rate, show_bkg, cmap):
    """Run tracking on video with bidirectional propagation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    video_tensor = torch.tensor(video_input).unsqueeze(0).to(dtype)

    # Use the globally initialized model
    model = get_model()
    print("Using pre-loaded model for tracking...")

    video_tensor = video_tensor.permute(0, 1, 4, 2, 3)
    _, T, _, H, W = video_tensor.shape

    # Store original resolution
    orig_H, orig_W = H, W

    # Configure inference size and compute padding parameters
    inf_H, inf_W = 336, 560
    skip_upscaling = True

    print(f"Original video size: {orig_H}x{orig_W}")
    print(f"Inference size: {inf_H}x{inf_W}")

    # Compute padding parameters
    padding_info = compute_padding_params(
        orig_H, orig_W, inf_H, inf_W, skip_upscaling=skip_upscaling
    )
    print(f"Scale factor: {padding_info['scale']:.4f}")
    if padding_info["upscaling_skipped"]:
        print(
            f"Upscaling skipped (scale > 1.0) - using original size: {orig_H}x{orig_W}"
        )
    else:
        print(
            f"Scaled size (before padding): {padding_info['scaled_H']}x{padding_info['scaled_W']}"
        )
    print(
        f"Padding: top={padding_info['pad_top']}, bottom={padding_info['pad_bottom']}, "
        f"left={padding_info['pad_left']}, right={padding_info['pad_right']}"
    )

    torch.cuda.empty_cache()

    # Initialize output tensors for INFERENCE resolution
    traj_maps_e = torch.zeros(
        (1, T, inf_H, inf_W, 2), dtype=torch.float32, device="cpu"
    )
    visconf_maps_e = torch.zeros(
        (1, T, inf_H, inf_W), dtype=torch.float32, device="cpu"
    )

    with torch.no_grad():
        # Forward pass
        if query_frame < T - 1:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # Apply padding to forward video
                forward_video = video_tensor[0, query_frame:]
                forward_video_padded = apply_padding(forward_video, padding_info).to(device)

                predictions = model.forward(
                    video=forward_video_padded,
                    queries=None,
                )

                # Extract dense predictions (at INFERENCE resolution)
                tracks_dense = predictions["track"][0]  # (T_forward, inf_H, inf_W, 2)
                visibility_dense = predictions["vis"][0]  # (T_forward, inf_H, inf_W)
                confidence_dense = predictions["conf"][0]  # (T_forward, inf_H, inf_W)

                # Store forward predictions
                T_forward = tracks_dense.shape[0]
                traj_maps_e[0, query_frame : query_frame + T_forward] = (
                    tracks_dense.cpu()
                )
                visconf_maps_e[0, query_frame : query_frame + T_forward] = (
                    visibility_dense * confidence_dense
                ).cpu()

        # Backward pass
        if query_frame > 0:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # Flip video for backward tracking and apply padding
                backward_video = video_tensor[0, : query_frame + 1].flip([0])
                backward_video_padded = apply_padding(
                    backward_video, padding_info
                ).to(device)

                predictions = model.forward(
                    video=backward_video_padded,
                    queries=None,
                )

                # Extract dense predictions (at INFERENCE resolution)
                tracks_dense = predictions["track"][0]  # (T_backward, inf_H, inf_W, 2)
                visibility_dense = predictions["vis"][0]  # (T_backward, inf_H, inf_W)
                confidence_dense = predictions["conf"][0]  # (T_backward, inf_H, inf_W)

                # Flip back to original temporal order
                backward_tracks = tracks_dense.flip([0]).cpu()
                backward_visconf = (visibility_dense * confidence_dense).flip([0]).cpu()

                # Store backward predictions (excluding query frame if needed)
                end_idx = query_frame if query_frame < T - 1 else query_frame + 1
                traj_maps_e[0, :end_idx] = backward_tracks[:end_idx]
                visconf_maps_e[0, :end_idx] = backward_visconf[:end_idx]

    # Remove padding and scale back to original resolution
    print(f"Removing padding and scaling back to {orig_H}x{orig_W}")
    tracks_final, _, confidence_final = remove_padding_and_scale_back(
        traj_maps_e[0],  # (T, inf_H, inf_W, 2)
        torch.ones_like(visconf_maps_e[0]),  # dummy visibility (not used here)
        visconf_maps_e[0],  # (T, inf_H, inf_W)
        padding_info,
    )
    print(f"Tracks shape after unpadding: {tracks_final.shape}")
    print(f"Confidence shape after unpadding: {confidence_final.shape}")

    # Convert to numpy format
    tracks = tracks_final.permute(1, 2, 0, 3).reshape(-1, T, 2).numpy()
    confs = confidence_final.permute(1, 2, 0).reshape(-1, T).numpy()
    visibs = confs > 0.1

    return (
        update_vis(
            rate, show_bkg, cmap, video_preview, query_frame, video_fps, tracks, visibs
        ),
        tracks,
        visibs,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


# --- Gradio UI Layout ---

custom_css = """
h1 {text-align: center; margin-bottom: 0 !important;}
.contain {max-width: 95% !important;}
#examples-accordion {margin-top: 10px;}
"""


def create_demo():
    """Create and return the Gradio demo interface."""
    with gr.Blocks(title="CoWTracker Demo") as demo:
        # State Variables
        video_state = gr.State()
        video_queried_preview = gr.State()
        video_preview = gr.State()
        video_input = gr.State()
        video_fps = gr.State(24)
        tracks = gr.State([])
        visibs = gr.State([])

        # Header
        gr.Markdown(
            """
            <div style="text-align: center; max-width: 800px; margin: 0 auto;">
                <h1 style="font-weight: 900; margin-bottom: 7px;">🐮 CoWTracker</h1>
                <p style="margin-bottom: 10px; font-size: 94%">
                    Cost-Volume Free Warping-Based Dense Point Tracking.
                    <a href='https://cowtracker.github.io/' target='_blank'>Project Page</a> |
                    <a href='https://github.com/facebookresearch/cowtracker/' target='_blank'>GitHub</a> |
                    <a href='' target='_blank'>Paper</a>
                </p>
            </div>
            """
        )

        with gr.Row():
            # --- Left Column: Input & Query ---
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 1. Upload Video")
                    video_in = gr.Video(label="Input Video", format="mp4", height=300)
                    submit_btn = gr.Button("Step 1: Process Video", variant="primary")

                # Query Frame Preview
                with gr.Group():
                    gr.Markdown("### 2. Select Query Frame")
                    query_frame_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=1,
                        label="Frame Number",
                        interactive=False,
                    )
                    current_frame = gr.Image(
                        label="Query Frame Preview",
                        type="numpy",
                        interactive=False,
                        height=300,
                    )

            # --- Right Column: Visualization & Output ---
            with gr.Column(scale=2):
                gr.Markdown("### 3. Configure & Track")

                with gr.Group():
                    with gr.Row():
                        rate_radio = gr.Radio(
                            [1, 2, 4, 8, 16],
                            value=8,
                            label="Subsampling Rate",
                            interactive=False,
                        )
                        cmap_radio = gr.Radio(
                            ["gist_rainbow", "rainbow", "jet", "turbo", "bremm"],
                            value="gist_rainbow",
                            label="Colormap",
                            interactive=False,
                        )

                    with gr.Row():
                        bkg_check = gr.Checkbox(
                            value=True, label="Overlay on Video", interactive=False
                        )
                        track_button = gr.Button(
                            "Step 2: Start Tracking", variant="primary", interactive=False
                        )

                # Output takes entire width of this column
                output_video = gr.Video(
                    label="Tracking Result",
                    interactive=False,
                    autoplay=True,
                    loop=True,
                    height=550,
                )

        # --- Full Width Row: Examples ---
        with gr.Row():
            with gr.Column():
                video_folder = "videos"
                gr.Markdown("### 📚 Example Videos")
                video_dir = os.path.join(os.path.dirname(__file__), video_folder)
                video_files = []
                if os.path.exists(video_dir):
                    for filename in sorted(os.listdir(video_dir)):
                        if filename.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                            video_files.append(os.path.join(video_dir, filename))

                if video_files:
                    gr.Examples(
                        examples=video_files,
                        inputs=[video_in],
                        examples_per_page=16,
                    )

        # --- Interaction Logic ---

        # 1. Submit Video
        submit_btn.click(
            fn=preprocess_video_input,
            inputs=[video_in],
            outputs=[
                video_state,
                video_preview,
                video_queried_preview,
                video_input,
                video_fps,
                current_frame,
                query_frame_slider,
                track_button,
            ],
            queue=False,
        )

        # 2. Update Preview Frame on Slider Change
        query_frame_slider.change(
            fn=choose_frame,
            inputs=[query_frame_slider, video_queried_preview],
            outputs=[current_frame],
            queue=False,
        )

        # 3. Run Tracking
        track_button.click(
            fn=track,
            inputs=[
                video_preview,
                video_input,
                video_fps,
                query_frame_slider,
                rate_radio,
                bkg_check,
                cmap_radio,
            ],
            outputs=[
                output_video,
                tracks,
                visibs,
                rate_radio,
                bkg_check,
                cmap_radio,
            ],
            queue=True,
        )

        # 4. Instant Updates for Visualization Settings (after tracking is done)
        vis_args = [
            rate_radio,
            bkg_check,
            cmap_radio,
            video_preview,
            query_frame_slider,
            video_fps,
            tracks,
            visibs,
        ]
        rate_radio.change(
            fn=update_vis, inputs=vis_args, outputs=[output_video], queue=False
        )
        cmap_radio.change(
            fn=update_vis, inputs=vis_args, outputs=[output_video], queue=False
        )
        bkg_check.change(
            fn=update_vis, inputs=vis_args, outputs=[output_video], queue=False
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoWTracker Gradio Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8086,
        help="Server port",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )
    parser.add_argument(
        "--keyfile",
        type=str,
        default=None,
        help="Path to SSL key file",
    )
    parser.add_argument(
        "--crtfile",
        type=str,
        default=None,
        help="Path to SSL certificate file",
    )
    args = parser.parse_args()

    # Initialize model with custom checkpoint if provided
    if args.checkpoint:
        GLOBAL_MODEL = initialize_model(args.checkpoint)
    else:
        GLOBAL_MODEL = initialize_model()

    # Get server name
    server_name = os.uname()[1]

    # SSL certificate paths
    keyfile = args.keyfile
    crtfile = args.crtfile

    # Check if SSL certs exist
    use_ssl = keyfile is not None and crtfile is not None and os.path.exists(keyfile) and os.path.exists(crtfile)

    print("=" * 60)
    print("Starting CoWTracker Gradio Demo")
    print("=" * 60)
    print(f"Server: {server_name}")
    print(f"Port: {args.port}")
    print(f"SSL: {'Enabled' if use_ssl else 'Disabled'}")
    print(f"Share: {args.share}")
    print("=" * 60)

    demo = create_demo()

    # Launch with appropriate configuration
    launch_kwargs = {
        "share": args.share,
        "show_error": True,
        "server_name": server_name,
        "server_port": args.port,
        "theme": gr.themes.Ocean(),
        "css": custom_css,
    }

    if use_ssl:
        launch_kwargs.update({
            "ssl_keyfile": keyfile,
            "ssl_certfile": crtfile,
            "ssl_verify": False,
        })

    demo.launch(**launch_kwargs)

