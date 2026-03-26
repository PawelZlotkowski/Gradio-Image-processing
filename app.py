import gradio as gr
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import cv2
from typing import Tuple
import io

# For object detection, we'll use torchvision's pre-trained model
try:
    import torch
    from torchvision import transforms
    from torchvision.models import efficientnet_b0
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, object detection will use basic color detection")

# ============================================================================
# CORE FEATURE 1: GRAYSCALE CONVERSION
# ============================================================================
def convert_to_grayscale(image: Image.Image) -> Tuple[Image.Image, str]:
    """Convert an image to grayscale."""
    if image is None:
        return None, "Error: No image provided"
    
    gray_image = ImageOps.grayscale(image)
    return gray_image, "✓ Successfully converted to grayscale"


# ============================================================================
# CORE FEATURE 2: IMAGE DETAIL EXTRACTION
# ============================================================================
def extract_image_details(image: Image.Image) -> str:
    """Extract and display image details."""
    if image is None:
        return "Error: No image provided"
    
    # Convert to numpy array for additional analysis
    img_array = np.array(image)
    
    # Get basic info
    width, height = image.size
    format_type = image.format or "Unknown"
    mode = image.mode
    
    # Calculate color statistics
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        avg_color = np.mean(img_array, axis=(0, 1)).astype(int)
        color_info = f"RGB({avg_color[0]}, {avg_color[1]}, {avg_color[2]})"
    elif len(img_array.shape) == 2:
        avg_color = np.mean(img_array)
        color_info = f"Grayscale ({int(avg_color)})"
    else:
        color_info = "Color mode: " + str(mode)
    
    # File size
    file_size = img_array.nbytes / 1024  # KB
    
    # Calculate histogram info
    if len(img_array.shape) == 2:
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
    else:
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
    
    details = f"""
📊 **IMAGE DETAILS**

• **Dimensions**: {width} x {height} pixels
• **Aspect Ratio**: {width/height:.2f}:1
• **Format**: {format_type}
• **Color Mode**: {mode}
• **Average Color**: {color_info}
• **Memory Size**: {file_size:.2f} KB
• **Brightness (Avg)**: {brightness:.0f}
• **Contrast (Std Dev)**: {contrast:.0f}
• **Total Pixels**: {width * height:,}
    """
    
    return details.strip()


# ============================================================================
# CORE FEATURE 3: BASIC OBJECT RECOGNITION
# ============================================================================
def detect_objects(image: Image.Image) -> str:
    """Perform basic object recognition using color-based detection."""
    if image is None:
        return "Error: No image provided"
    
    # Convert to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for common objects
    objects_detected = []
    
    # Red objects
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(img_hsv, lower_red1, upper_red1) + cv2.inRange(img_hsv, lower_red2, upper_red2)
    if cv2.countNonZero(mask_red) > 100:
        objects_detected.append(("Red objects", f"{cv2.countNonZero(mask_red)/img_hsv.size*100:.1f}%"))
    
    # Green objects
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    if cv2.countNonZero(mask_green) > 100:
        objects_detected.append(("Green objects", f"{cv2.countNonZero(mask_green)/img_hsv.size*100:.1f}%"))
    
    # Blue objects
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    if cv2.countNonZero(mask_blue) > 100:
        objects_detected.append(("Blue objects", f"{cv2.countNonZero(mask_blue)/img_hsv.size*100:.1f}%"))
    
    # Yellow objects
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    if cv2.countNonZero(mask_yellow) > 100:
        objects_detected.append(("Yellow objects", f"{cv2.countNonZero(mask_yellow)/img_hsv.size*100:.1f}%"))
    
    # Edge detection for shape recognition
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    result = "🎯 **OBJECT DETECTION RESULTS**\n\n"
    
    if objects_detected:
        result += "**Colors Detected:**\n"
        for obj_name, percentage in objects_detected:
            result += f"• {obj_name}: {percentage} of image\n"
    else:
        result += "• No significant colored regions detected\n"
    
    result += f"\n**Shape Features:**\n"
    result += f"• Contours found: {len(contours)}\n"
    result += f"• Image complexity: {'High' if len(contours) > 50 else 'Medium' if len(contours) > 10 else 'Low'}\n"
    
    return result.strip()


# ============================================================================
# BONUS FEATURE 1: EDGE DETECTION
# ============================================================================
def detect_edges(image: Image.Image, method: str = "Canny") -> Tuple[Image.Image, str]:
    """Detect edges in image using Canny or Sobel method."""
    if image is None:
        return None, "Error: No image provided"
    
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    if method == "Canny":
        edges = cv2.Canny(cv_image, 100, 200)
        msg = "✓ Canny edge detection applied"
    elif method == "Sobel":
        sobelx = cv2.Sobel(cv_image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(cv_image, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.hypot(sobelx, sobely)
        edges = np.uint8(255 * edges / np.max(edges)) if np.max(edges) > 0 else np.uint8(edges)
        msg = "✓ Sobel edge detection applied"
    elif method == "Laplacian":
        edges = cv2.Laplacian(cv_image, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
        msg = "✓ Laplacian edge detection applied"
    else:
        edges = cv2.Canny(cv_image, 100, 200)
        msg = "✓ Canny edge detection applied (default)"
    
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    result_image = Image.fromarray(edges_rgb)
    
    return result_image, msg


# ============================================================================
# BONUS FEATURE 2: FILTERS
# ============================================================================
def apply_filter(image: Image.Image, filter_type: str) -> Tuple[Image.Image, str]:
    """Apply various filters to the image."""
    if image is None:
        return None, "Error: No image provided"
    
    if filter_type == "Blur":
        filtered = image.filter(ImageFilter.GaussianBlur(radius=5))
        msg = "✓ Gaussian blur applied"
    elif filter_type == "Sharpen":
        filtered = image.filter(ImageFilter.SHARPEN)
        msg = "✓ Sharpen filter applied"
    elif filter_type == "Contour":
        filtered = image.filter(ImageFilter.CONTOUR)
        msg = "✓ Contour filter applied"
    elif filter_type == "Emboss":
        filtered = image.filter(ImageFilter.EMBOSS)
        msg = "✓ Emboss filter applied"
    elif filter_type == "Sepia":
        img_array = np.array(image).astype(np.float32)
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                  [0.349, 0.686, 0.168],
                                  [0.393, 0.769, 0.189]])
        filtered_array = cv2.transform(img_array, sepia_filter)
        filtered_array = np.clip(filtered_array, 0, 255).astype(np.uint8)
        filtered = Image.fromarray(filtered_array)
        msg = "✓ Sepia tone applied"
    elif filter_type == "Edge Enhance":
        filtered = image.filter(ImageFilter.EDGE_ENHANCE)
        msg = "✓ Edge enhance filter applied"
    else:
        filtered = image
        msg = "No filter applied"
    
    return filtered, msg


# ============================================================================
# BONUS FEATURE 3: IMAGE BRIGHTNESS/CONTRAST ADJUSTMENT
# ============================================================================
def adjust_brightness_contrast(image: Image.Image, brightness: float, contrast: float) -> Tuple[Image.Image, str]:
    """Adjust brightness and contrast of the image."""
    if image is None:
        return None, "Error: No image provided"
    
    # Brightness adjustment (1.0 = no change, <1 = darker, >1 = brighter)
    enhancer_brightness = ImageEnhance.Brightness(image)
    adjusted = enhancer_brightness.enhance(brightness)
    
    # Contrast adjustment (1.0 = no change, <1 = lower, >1 = higher)
    enhancer_contrast = ImageEnhance.Contrast(adjusted)
    adjusted = enhancer_contrast.enhance(contrast)
    
    msg = f"✓ Brightness: {brightness:.1f}x, Contrast: {contrast:.1f}x"
    return adjusted, msg


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
def create_gradio_app():
    """Create and launch the Gradio application."""
    
    with gr.Blocks(title="Image Processing Lab", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖼️ Image Processing Application")
        gr.Markdown("A professional Gradio-based image processing tool with multiple features for image analysis and enhancement.")
        
        with gr.Tabs():
            # ============================================================
            # TAB 1: GRAYSCALE CONVERSION
            # ============================================================
            with gr.TabItem("📐 Grayscale Conversion"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Convert Image to Grayscale")
                        image_input_gray = gr.Image(type="pil", label="Upload Image")
                        btn_gray = gr.Button("Convert to Grayscale", variant="primary")
                    
                    with gr.Column():
                        image_output_gray = gr.Image(label="Grayscale Output")
                        status_gray = gr.Textbox(label="Status", interactive=False)
                
                btn_gray.click(
                    fn=convert_to_grayscale,
                    inputs=[image_input_gray],
                    outputs=[image_output_gray, status_gray]
                )
            
            # ============================================================
            # TAB 2: IMAGE DETAILS
            # ============================================================
            with gr.TabItem("📊 Image Details"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Extract Image Information")
                        image_input_details = gr.Image(type="pil", label="Upload Image")
                        btn_details = gr.Button("Extract Details", variant="primary")
                    
                    with gr.Column():
                        details_output = gr.Textbox(label="Image Details", lines=12, interactive=False)
                
                btn_details.click(
                    fn=extract_image_details,
                    inputs=[image_input_details],
                    outputs=[details_output]
                )
            
            # ============================================================
            # TAB 3: OBJECT RECOGNITION
            # ============================================================
            with gr.TabItem("🎯 Object Detection"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Detect Objects and Colors")
                        image_input_detect = gr.Image(type="pil", label="Upload Image")
                        btn_detect = gr.Button("Detect Objects", variant="primary")
                    
                    with gr.Column():
                        detection_output = gr.Textbox(label="Detection Results", lines=12, interactive=False)
                
                btn_detect.click(
                    fn=detect_objects,
                    inputs=[image_input_detect],
                    outputs=[detection_output]
                )
            
            # ============================================================
            # TAB 4: EDGE DETECTION (BONUS)
            # ============================================================
            with gr.TabItem("🔍 Edge Detection"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Detect Edges in Image")
                        image_input_edges = gr.Image(type="pil", label="Upload Image")
                        edge_method = gr.Radio(
                            choices=["Canny", "Sobel", "Laplacian"],
                            value="Canny",
                            label="Detection Method"
                        )
                        btn_edges = gr.Button("Detect Edges", variant="primary")
                    
                    with gr.Column():
                        image_output_edges = gr.Image(label="Edge Detection Output")
                        status_edges = gr.Textbox(label="Status", interactive=False)
                
                btn_edges.click(
                    fn=detect_edges,
                    inputs=[image_input_edges, edge_method],
                    outputs=[image_output_edges, status_edges]
                )
            
            # ============================================================
            # TAB 5: FILTERS (BONUS)
            # ============================================================
            with gr.TabItem("✨ Image Filters"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Apply Filters to Image")
                        image_input_filter = gr.Image(type="pil", label="Upload Image")
                        filter_type = gr.Radio(
                            choices=["Blur", "Sharpen", "Contour", "Emboss", "Sepia", "Edge Enhance"],
                            value="Blur",
                            label="Filter Type"
                        )
                        btn_filter = gr.Button("Apply Filter", variant="primary")
                    
                    with gr.Column():
                        image_output_filter = gr.Image(label="Filtered Output")
                        status_filter = gr.Textbox(label="Status", interactive=False)
                
                btn_filter.click(
                    fn=apply_filter,
                    inputs=[image_input_filter, filter_type],
                    outputs=[image_output_filter, status_filter]
                )
            
            # ============================================================
            # TAB 6: BRIGHTNESS/CONTRAST (BONUS)
            # ============================================================
            with gr.TabItem("🌞 Brightness & Contrast"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Adjust Brightness and Contrast")
                        image_input_adjust = gr.Image(type="pil", label="Upload Image")
                        brightness_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Brightness (0.5 = darker, 2.0 = brighter)"
                        )
                        contrast_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Contrast (0.5 = lower, 2.0 = higher)"
                        )
                        btn_adjust = gr.Button("Apply Adjustments", variant="primary")
                    
                    with gr.Column():
                        image_output_adjust = gr.Image(label="Adjusted Output")
                        status_adjust = gr.Textbox(label="Status", interactive=False)
                
                btn_adjust.click(
                    fn=adjust_brightness_contrast,
                    inputs=[image_input_adjust, brightness_slider, contrast_slider],
                    outputs=[image_output_adjust, status_adjust]
                )
    
    return demo


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
