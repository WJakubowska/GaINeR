import gradio as gr
import numpy as np
import hashlib
import os
from datetime import datetime
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

# Load SAM model and predictor
sam = sam_model_registry["vit_h"](checkpoint="ckpts/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
_last_image_hash = None  # cache key to avoid recomputing image features on each click


def _hash_image(image: Image.Image) -> str:
    arr = np.asarray(image)
    # md5 over raw bytes; fast enough and reliable for change detection
    return hashlib.md5(arr.tobytes()).hexdigest()


def segment(image, selection, evt: gr.SelectData):
    """Run SAM using a single click from the Image.select event.

    Args:
        image: PIL.Image provided by gr.Image(type="pil").
        evt: gr.SelectData containing click coordinates in evt.index (x, y).

    Returns:
        A mask image (numpy array or PIL) where the first predicted mask is shown.
    """
    if image is None or evt is None:
        return selection if selection is not None else image, selection

    # Compute image features only when the image actually changes
    global _last_image_hash
    img_hash = _hash_image(image)
    if _last_image_hash != img_hash:
        predictor.set_image(np.array(image))
        _last_image_hash = img_hash
        # Reset selection for a new image
        selection = None

    masks, _, _ = predictor.predict(
        point_coords=np.array([[evt.index[0], evt.index[1]]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    # Choose the largest-area mask (avoid tiny sub-segments)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    largest_idx = int(np.argmax(areas))
    mask = masks[largest_idx].astype(np.uint8) * 255

    # Initialize selection if needed, then union with new mask
    if selection is None or (
        isinstance(selection, np.ndarray) and selection.shape != mask.shape
    ):
        selection = np.zeros_like(mask, dtype=np.uint8)

    updated = np.maximum(selection, mask)
    # Return an overlay preview for display, keep raw mask in state
    overlay = _overlay_mask_on_image(image, updated)
    return overlay, updated


# Gradio v4: use Blocks and bind the Image.select event instead of tool="select"
def _blank_mask_like(image: Image.Image):
    if image is None:
        return None
    # PIL image size is (W, H)
    w, h = image.size
    return np.zeros((h, w), dtype=np.uint8)


def clear_selection(image):
    # Keep cached image features; just clear current selection
    blank = _blank_mask_like(image)
    # For preview, just show the original image with no overlay
    return image, blank


# --- Inpainting utilities ---
_pipe_cache = {}

def _get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_diffusers_inpaint_pipeline(model_id: str):
    """Load a Hugging Face inpainting pipeline (FLUX if possible; fallback to SD inpaint).

    Returns (pipeline, used_model_id) or (None, model_id) if loading failed.
    """
    if model_id in _pipe_cache:
        return _pipe_cache[model_id], model_id

    device = _get_device()
    pipe = None
    used_model_id = model_id
    try:
        from diffusers import AutoPipelineForInpainting
        import torch
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            dtype=dtype,
            variant="fp16" if device == "cuda" else None,
        )
        pipe = pipe.to(device)
    except Exception:
        # Try a widely available SD inpainting model
        try:
            from diffusers import AutoPipelineForInpainting
            import torch
            used_model_id = "runwayml/stable-diffusion-inpainting"
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = AutoPipelineForInpainting.from_pretrained(
                used_model_id,
                dtype=dtype,
            ).to(device)
        except Exception:
            pipe = None

    if pipe is not None:
        _pipe_cache[used_model_id] = pipe
    return pipe, used_model_id


def inpaint_with_hf(image: Image.Image, mask: np.ndarray, prompt: str, model_id: str):
    """Inpaint using a Hugging Face pipeline (prefers FLUX if supported by the model id)."""
    try:
        from diffusers.utils import make_image_grid  # noqa: F401 (sanity import)
    except Exception as e:
        raise RuntimeError(
            "Hugging Face diffusers not available. Install 'diffusers' and 'transformers'."
        ) from e

    pipe, used_model = _load_diffusers_inpaint_pipeline(model_id)
    if pipe is None:
        raise RuntimeError("Could not load a diffusers inpainting pipeline.")

    # Ensure binary mask and match original size
    mask_u8 = _to_binary_mask_u8(mask)
    mask_u8 = _resize_mask_to_image(mask_u8, image)
    mask_pil = Image.fromarray(mask_u8).convert("L")
    if not prompt or not prompt.strip():
        prompt = "Realistic"

    result = pipe(prompt=prompt, image=image, mask_image=mask_pil)
    out_img = result.images[0]

    # Guarantee output image matches original size and only replace masked regions
    if out_img.size != image.size:
        out_img = out_img.resize(image.size, Image.LANCZOS)
    final_img = Image.composite(out_img, image, mask_pil)
    return final_img, f"Inpainted with {used_model}"


def _to_binary_mask_u8(mask_like) -> np.ndarray:
    """Convert any incoming mask to single-channel uint8 {0,255}."""
    if mask_like is None:
        return None
    arr = np.array(mask_like)
    if arr.ndim == 3:
        # Any channel nonzero -> mask
        arr = (arr > 0).any(axis=2).astype(np.uint8) * 255
    else:
        arr = (arr > 0).astype(np.uint8) * 255
    return arr


def _resize_mask_to_image(mask_u8: np.ndarray, image: Image.Image) -> np.ndarray:
    if mask_u8 is None or image is None:
        return mask_u8
    h, w = mask_u8.shape[:2]
    if (w, h) == image.size:
        return mask_u8
    return np.array(Image.fromarray(mask_u8).resize(image.size, Image.NEAREST))


def _dilate_mask(mask_u8: np.ndarray, pixels: int = 3) -> np.ndarray:
    """Dilate a binary uint8 mask by the given number of pixels.

    Tries OpenCV for speed; falls back to PIL's MaxFilter. Returns uint8 {0,255}.
    """
    if mask_u8 is None:
        return None
    if pixels is None or pixels <= 0:
        return mask_u8
    try:
        import cv2
        k = int(max(1, 2 * int(pixels) + 1))  # odd kernel size
        kernel = np.ones((k, k), np.uint8)
        dil = cv2.dilate(mask_u8.astype(np.uint8), kernel, iterations=1)
        return (dil > 0).astype(np.uint8) * 255
    except Exception:
        # Fallback: PIL max filter behaves like dilation
        try:
            from PIL import ImageFilter
            k = int(max(1, 2 * int(pixels) + 1))  # odd size
            pil = Image.fromarray(mask_u8)
            dil = pil.filter(ImageFilter.MaxFilter(size=k))
            arr = np.array(dil)
            return (arr > 0).astype(np.uint8) * 255
        except Exception:
            return mask_u8


def _overlay_mask_on_image(image: Image.Image, mask_like, color=(30, 144, 255), alpha: float = 0.45) -> Image.Image:
    """Return an RGB image preview with a semi-transparent colored overlay on masked pixels.

    - color: RGB tuple for overlay (default DodgerBlue)
    - alpha: 0..1 transparency for the overlay on masked regions
    """
    if image is None:
        return None
    try:
        mask_u8 = _to_binary_mask_u8(mask_like)
        if mask_u8 is None:
            return image
        mask_u8 = _resize_mask_to_image(mask_u8, image)

        base_rgba = image.convert("RGBA")
        # Alpha mask scaled by desired alpha only where mask>0
        alpha_val = int(max(0, min(1, alpha)) * 255)
        alpha_mask = Image.fromarray((mask_u8 > 0).astype(np.uint8) * alpha_val, mode="L")

        color_img = Image.new("RGBA", image.size, (color[0], color[1], color[2], 0))
        color_img.putalpha(alpha_mask)

        blended = Image.alpha_composite(base_rgba, color_img)
        return blended.convert("RGB")
    except Exception:
        # If anything goes wrong, fall back to showing original image
        return image


def _pil_to_cv2(img: Image.Image):
    import cv2  # optional dependency

    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _cv2_to_pil(arr) -> Image.Image:
    import cv2

    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def inpaint_with_opencv(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Heuristic inpaint using OpenCV Telea as a fallback backend.

    mask: uint8 0-255, nonzero -> inpaint region
    """
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(
            "OpenCV not available. Install opencv-python or configure a FLUX backend."
        ) from e

    img_cv = _pil_to_cv2(image)
    mask_u8 = _to_binary_mask_u8(mask)
    mask_u8 = _resize_mask_to_image(mask_u8, image)
    radius = max(3, int(round(max(image.size) * 0.004)))
    out = cv2.inpaint(img_cv, mask_u8, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return _cv2_to_pil(out)


def inpaint_action(image: Image.Image, selection_mask: np.ndarray, prompt: str, model_id: str, dilation_px: int = 3):
    """Run inpainting using the current selection mask.

    Prefer Hugging Face (FLUX/SD-inpaint) backend; fallback to OpenCV if unavailable.
    """
    if image is None:
        return None, "No image."
    # Sanitize mask to 2D binary and match image size
    selection_mask = _to_binary_mask_u8(selection_mask)
    if selection_mask is None or np.max(selection_mask) == 0:
        return image, "Mask empty; returning original image."
    selection_mask = _resize_mask_to_image(selection_mask, image)
    # Dilate mask by a few pixels to cover seams
    try:
        selection_mask = _dilate_mask(selection_mask, int(dilation_px))
    except Exception:
        selection_mask = _dilate_mask(selection_mask, 3)

    # Try Hugging Face first
    try:
        out_img, msg = inpaint_with_hf(image, selection_mask, prompt, model_id)
        return out_img, msg
    except Exception as e:
        # Fallback to OpenCV
        try:
            out_img = inpaint_with_opencv(image, selection_mask)
            return out_img, f"HF inpaint failed ({e}); used OpenCV fallback."
        except Exception as ee:
            return image, f"Inpaint failed: HF error: {e}; OpenCV error: {ee}"


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_mask_action(selection_mask: np.ndarray, save_dir: str):
    if selection_mask is None:
        return "No mask to save."
    base = save_dir or "outputs/inpaint"
    mask_dir = os.path.join(base, "masks")
    _ensure_dir(mask_dir)
    fname = os.path.join(mask_dir, f"mask_{_timestamp_str()}.png")
    # Ensure black & white binary mask 0/255 saved
    mask_u8 = _to_binary_mask_u8(selection_mask)
    Image.fromarray(mask_u8).save(fname)
    return f"Saved mask to {fname}"


def save_inpaint_action(inpaint_img: Image.Image, original_image: Image.Image, selection_mask: np.ndarray, save_dir: str):
    if inpaint_img is None:
        return "No inpainted image to save."
    base = save_dir or "outputs/inpaint"
    out_dir = os.path.join(base, "images")
    _ensure_dir(out_dir)
    ts = _timestamp_str()
    # Save standard RGB inpainted image
    fname_rgb = os.path.join(out_dir, f"inpaint_{ts}.png")
    inpaint_img = Image.fromarray(np.array(inpaint_img))
    inpaint_img.save(fname_rgb)

    # Also save RGBA of the ORIGINAL image where alpha channel is the binary mask
    rgba_saved = None
    try:
        if original_image is not None:
            base_rgb = Image.fromarray(np.array(original_image)).convert("RGB")
            mask_u8 = _to_binary_mask_u8(selection_mask)
            if mask_u8 is not None:
                mask_u8 = _resize_mask_to_image(mask_u8, base_rgb)
                alpha = Image.fromarray(mask_u8, mode="L")
                r, g, b = base_rgb.split()
                rgba = Image.merge("RGBA", (r, g, b, alpha))
                fname_rgba = os.path.join(out_dir, f"fragment_{ts}_alpha.png")
                rgba.save(fname_rgba)
                rgba_saved = fname_rgba
    except Exception:
        rgba_saved = None

    if rgba_saved:
        return f"Saved inpainted image to {fname_rgb} and original RGBA fragment (alpha=mask) to {rgba_saved}"
    else:
        return f"Saved inpainted image to {fname_rgb}"


with gr.Blocks() as demo:
    gr.Markdown("# SAM click segmentation + Inpaint")
    with gr.Row():
        inp = gr.Image(type="pil", label="Image")
    with gr.Row():
        # Display an overlay preview, keep the raw mask in state for saving/inpaint
        out = gr.Image(type="pil", label="Mask Preview (blue overlay)")
    with gr.Row():
        inpaint_out = gr.Image(label="Inpainted Image")
    with gr.Row():
        prompt_tb = gr.Textbox(label="Prompt", placeholder="Describe what to inpaint / how to fill", value="")
    with gr.Row():
        model_tb = gr.Textbox(
            label="Hugging Face model id",
            value="runwayml/stable-diffusion-inpainting",
            placeholder="e.g., runwayml/stable-diffusion-inpainting or another inpainting repo",
        )
    with gr.Row():
        dilate_px = gr.Slider(0, 100, value=5, step=1, label="Mask dilation (px)")
    with gr.Row():
        clear_btn = gr.Button("Clear selection")
        inpaint_btn = gr.Button("Inpaint")
        save_dir = gr.Textbox(label="Save directory", value="outputs/inpaint")
        save_mask_btn = gr.Button("Save mask")
        save_inpaint_btn = gr.Button("Save inpaint")
    status = gr.Markdown(visible=True)
    sel_state = gr.State(value=None)

    # Clicking on the input image triggers prediction
    inp.select(fn=segment, inputs=[inp, sel_state], outputs=[out, sel_state])
    clear_btn.click(fn=clear_selection, inputs=inp, outputs=[out, sel_state])
    inpaint_btn.click(
        fn=inpaint_action,
        inputs=[inp, sel_state, prompt_tb, model_tb, dilate_px],
        outputs=[inpaint_out, status],
    )
    save_mask_btn.click(fn=save_mask_action, inputs=[sel_state, save_dir], outputs=[status])
    save_inpaint_btn.click(fn=save_inpaint_action, inputs=[inpaint_out, inp, sel_state, save_dir], outputs=[status])

demo.launch()