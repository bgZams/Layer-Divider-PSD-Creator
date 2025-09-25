import gradio as gr
import sys
sys.path.append("./scripts/")

from ldivider.ld_convertor import pil2cv, cv2pil, df2bgra
from ldivider.ld_processor import get_base, get_normal_layer, get_composite_layer, get_seg_base
# Import original functions
from ldivider.ld_utils import load_masks, divide_folder, load_seg_model

# Import our safe save function
import time
import zipfile
from PIL import Image

def save_layers_as_images(input_image, layer_lists, layer_names, output_dir, mode):
    """Simpan layers sebagai gambar terpisah jika PSD gagal"""
    timestamp = int(time.time())
    folder_name = f"layers_{mode}_{timestamp}"
    layers_dir = os.path.join(output_dir, folder_name)
    os.makedirs(layers_dir, exist_ok=True)
    
    # Simpan gambar asli
    original_path = os.path.join(layers_dir, "00_original.png")
    if isinstance(input_image, np.ndarray):
        cv2.imwrite(original_path, input_image)
    else:
        cv2.imwrite(original_path, cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR))
    
    # Simpan setiap layer group
    for group_idx, (layers, name) in enumerate(zip(layer_lists, layer_names)):
        group_dir = os.path.join(layers_dir, f"{group_idx+1:02d}_{name}")
        os.makedirs(group_dir, exist_ok=True)
        
        for layer_idx, layer in enumerate(layers):
            layer_filename = f"{layer_idx:03d}_{name}_layer.png"
            layer_path = os.path.join(group_dir, layer_filename)
            
            if isinstance(layer, np.ndarray):
                if len(layer.shape) == 3 and layer.shape[2] == 4:
                    layer_rgba = cv2.cvtColor(layer, cv2.COLOR_BGRA2RGBA)
                    Image.fromarray(layer_rgba).save(layer_path)
                else:
                    cv2.imwrite(layer_path, layer)
            else:
                layer.save(layer_path)
    
    # Buat file info.txt
    info_path = os.path.join(layers_dir, "layer_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Layer Division Result - Mode: {mode}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Groups: {len(layer_lists)}\n\n")
        for i, name in enumerate(layer_names):
            f.write(f"Group {i+1}: {name} ({len(layer_lists[i])} layers)\n")
        f.write(f"\nNote: PSD creation failed, layers saved as PNG files.\n")
        f.write(f"Import these into Photoshop manually.\n")
    
    return layers_dir

def safe_save_psd(input_image, layer_lists, layer_names, blend_modes, output_dir, mode):
    """Coba save PSD, jika gagal fallback ke save gambar"""
    try:
        from ldivider.ld_utils import save_psd
        print("🔄 Attempting to save PSD...")
        return save_psd(input_image, layer_lists, layer_names, blend_modes, output_dir, mode)
    except Exception as e:
        print(f"⚠️  PSD save failed: {str(e)}")
        print("🔄 Fallback: Saving as separate images...")
        
        layers_folder = save_layers_as_images(input_image, layer_lists, layer_names, output_dir, mode)
        
        # Buat file zip untuk download
        zip_filename = f"{output_dir}/layers_{mode}_{int(time.time())}.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(layers_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, layers_folder)
                    zipf.write(file_path, arcname)
        
        print(f"✅ Layers saved as ZIP: {zip_filename}")
        return zip_filename
from ldivider.ld_segment import get_mask_generator, get_masks, show_anns

import cv2
from pytoshop.enums import BlendMode
import os

import numpy as np

path = os.getcwd()
output_dir = f"{path}/output"
input_dir = f"{path}/input"
model_dir = f"{path}/segment_model"

# Pastikan directory output ada
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/tmp/seg_layer", exist_ok=True)
os.makedirs(input_dir, exist_ok=True)

load_seg_model(model_dir)

class webui:
    def __init__(self):
        self.demo = gr.Blocks()
        self.masks = None  # Store masks after segmentation
        
    def segment_image(self, input_image, pred_iou_thresh, stability_score_thresh, crop_n_layers, crop_n_points_downscale_factor, min_mask_region_area):
        if input_image is None:
            return None
            
        mask_generator = get_mask_generator(pred_iou_thresh, stability_score_thresh, min_mask_region_area, model_dir, "demo")
        self.masks = get_masks(pil2cv(input_image), mask_generator)  # Store masks
        input_image.putalpha(255)
        masked_image = show_anns(input_image, self.masks, output_dir)
        return masked_image

    def divide_layer(self, divide_mode, input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg, area_th):
        if input_image is None:
            return None, None, None, None, None
            
        if divide_mode == "segment_mode":
            return self.segment_divide(input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg, area_th)
        elif divide_mode == "color_base_mode":
            return self.color_base_divide(input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg)

    def segment_divide(self, input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg, area_th):
        image = pil2cv(input_image)
        self.input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        
        # Check if masks exist, if not try to load or show error
        masks = None
        try:
            masks = load_masks(output_dir)
        except FileNotFoundError:
            if self.masks is not None:
                masks = self.masks
            else:
                # Return error message instead of crashing
                error_msg = "No masks found! Please run segmentation first by clicking 'Segment' button."
                return None, None, None, None, error_msg

        df = get_seg_base(self.input_image, masks, area_th)

        base_image = cv2pil(df2bgra(df))
        image = cv2pil(image)
        
        if layer_mode == "composite":
            base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(self.input_image, df)
            filename = safe_save_psd(
                self.input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
                ["base", "screen", "multiply", "subtract", "addition"],
                [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
                output_dir,
                layer_mode
            )
            base_layer_list = [cv2pil(layer) for layer in base_layer_list]
            divide_folder(filename, input_dir, layer_mode)
            return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
            
        elif layer_mode == "normal":
            base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(self.input_image, df)
            filename = safe_save_psd(
                self.input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list],
                ["base", "bright", "shadow"],
                [BlendMode.normal, BlendMode.normal, BlendMode.normal],
                output_dir,
                layer_mode
            )
            divide_folder(filename, input_dir, layer_mode)
            return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
        else:
            return None, None, None, None, None
        
    def color_base_divide(self, input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg):
        image = pil2cv(input_image)
        self.input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

        df = get_base(self.input_image, loops, init_cluster, ciede_threshold, blur_size, h_split, v_split, n_cluster, alpha, th_rate, split_bg, False)        
        
        base_image = cv2pil(df2bgra(df))
        image = cv2pil(image)
        
        if layer_mode == "composite":
            base_layer_list, shadow_layer_list, bright_layer_list, addition_layer_list, subtract_layer_list = get_composite_layer(self.input_image, df)
            filename = safe_save_psd(
                self.input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list, subtract_layer_list, addition_layer_list],
                ["base", "screen", "multiply", "subtract", "addition"],
                [BlendMode.normal, BlendMode.screen, BlendMode.multiply, BlendMode.subtract, BlendMode.linear_dodge],
                output_dir,
                layer_mode,
            )
            base_layer_list = [cv2pil(layer) for layer in base_layer_list]
            return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
            
        elif layer_mode == "normal":
            base_layer_list, bright_layer_list, shadow_layer_list = get_normal_layer(self.input_image, df)
            filename = safe_save_psd(
                self.input_image,
                [base_layer_list, bright_layer_list, shadow_layer_list],
                ["base", "bright", "shadow"],
                [BlendMode.normal, BlendMode.normal, BlendMode.normal],
                output_dir,
                layer_mode,
            )
            return [image, base_image], base_layer_list, bright_layer_list, shadow_layer_list, filename
        else:
            return None, None, None, None, None

    def launch(self, share):
        with self.demo:
            gr.Markdown("# Layer Divider - PSD Creator")
            gr.Markdown("**Important:** For segment mode, you must run 'Segment' first before creating PSD!")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Upload Image")
                    divide_mode = gr.Dropdown(["segment_mode", "color_base_mode"], value="segment_mode", label="Division Mode", show_label=True)

                    with gr.Accordion("Segment Settings", open=True):
                        area_th = gr.Slider(1, 100000, value=20000, step=100, label="Area Threshold", show_label=True)
                        
                    with gr.Accordion("ColorBase Settings", open=False):
                        loops = gr.Slider(1, 20, value=1, step=1, label="Loops", show_label=True)
                        init_cluster = gr.Slider(1, 50, value=10, step=1, label="Initial Cluster", show_label=True)
                        ciede_threshold = gr.Slider(1, 50, value=5, step=1, label="CIEDE Threshold", show_label=True)
                        blur_size = gr.Slider(1, 20, value=5, label="Blur Size", show_label=True)
                        layer_mode = gr.Dropdown(["normal", "composite"], value="normal", label="Output Layer Mode", show_label=True)
                        
                    with gr.Accordion("Background Settings", open=False):
                        split_bg = gr.Checkbox(label="Split Background", show_label=True)
                        h_split = gr.Slider(1, 2048, value=256, step=4, label="Horizontal Split", show_label=True)
                        v_split = gr.Slider(1, 2048, value=256, step=4, label="Vertical Split", show_label=True)
                        n_cluster = gr.Slider(1, 1000, value=500, step=10, label="Cluster Number", show_label=True)
                        alpha = gr.Slider(1, 255, value=100, step=1, label="Alpha Threshold", show_label=True)
                        th_rate = gr.Slider(0, 1, value=0.1, step=0.01, label="Mask Content Ratio", show_label=True)

                    submit = gr.Button(value="Create PSD", variant="primary")

                with gr.Column():
                    with gr.Accordion("Segmentation Preview", open=True):
                        SAM_output = gr.Image(type="pil", label="Segmented Image")
                        with gr.Row():
                            pred_iou_thresh = gr.Slider(0, 1, value=0.8, step=0.01, label="Pred IoU Thresh", show_label=True)
                            stability_score_thresh = gr.Slider(0, 1, value=0.8, step=0.01, label="Stability Score Thresh", show_label=True)
                        with gr.Row():
                            crop_n_layers = gr.Slider(1, 10, value=1, step=1, label="Crop Layers", show_label=True)
                            crop_n_points_downscale_factor = gr.Slider(1, 10, value=2, step=1, label="Downscale Factor", show_label=True)
                        min_mask_region_area = gr.Slider(1, 1000, value=100, step=1, label="Min Mask Region Area", show_label=True)
                        segment = gr.Button(value="Segment Image", variant="secondary")

            with gr.Row():
                with gr.Tab("Output Images"):
                    output_0 = gr.Gallery(label="Original & Base")
                with gr.Tab("Base Layers"):
                    output_1 = gr.Gallery(label="Base Layers")
                with gr.Tab("Bright Layers"):
                    output_2 = gr.Gallery(label="Bright Layers")
                with gr.Tab("Shadow Layers"):
                    output_3 = gr.Gallery(label="Shadow Layers")

            output_file = gr.File(label="Download PSD File")
                    
            # Event handlers
            segment.click(
                self.segment_image,
                inputs=[input_image, pred_iou_thresh, stability_score_thresh, crop_n_layers, crop_n_points_downscale_factor, min_mask_region_area], 
                outputs=[SAM_output]
            )
            
            submit.click(
                self.divide_layer, 
                inputs=[divide_mode, input_image, loops, init_cluster, ciede_threshold, blur_size, layer_mode, h_split, v_split, n_cluster, alpha, th_rate, split_bg, area_th], 
                outputs=[output_0, output_1, output_2, output_3, output_file]
            )

        self.demo.queue()
        self.demo.launch(share=share)


if __name__ == "__main__":
    ui = webui()
    if len(sys.argv) > 1:
        if sys.argv[1] == "share":
            ui.launch(share=True)
        else:
            ui.launch(share=False)
    else:
        ui.launch(share=False)