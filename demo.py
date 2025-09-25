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
import os
import cv2
import numpy as np
from pytoshop.enums import BlendMode
from psd_tools import PSDImage
from psd_tools.api.layers import Layer, Group

def save_psd_with_pytoshop(input_image, layer_lists, layer_names, blend_modes, output_dir, mode):
    """Save PSD using pytoshop directly"""
    try:
        # Import yang benar untuk pytoshop
        import pytoshop
        from pytoshop.user import nested_layers
        from pytoshop.enums import BlendMode as PytoshopBlendMode
        
        # Konversi input image ke format yang tepat
        if isinstance(input_image, np.ndarray):
            if len(input_image.shape) == 3 and input_image.shape[2] == 4:
                # Konversi BGRA ke RGBA
                input_image_rgba = cv2.cvtColor(input_image, cv2.COLOR_BGRA2RGBA)
                input_image_pil = Image.fromarray(input_image_rgba)
            else:
                input_image_pil = Image.fromarray(input_image)
        else:
            input_image_pil = input_image
            
        # Konversi ke RGB jika RGBA
        if input_image_pil.mode == 'RGBA':
            input_image_pil = input_image_pil.convert('RGB')
        
        # Buat struktur layers untuk pytoshop
        layers = []
        
        # Tambahkan background layer
        layers.append(('Background', input_image_pil))
        
        # Tambahkan setiap group layer
        for layers_group, name, blend_mode in zip(layer_lists, layer_names, blend_modes):
            group_layers = []
            
            # Tambahkan setiap layer ke dalam group
            for i, layer in enumerate(layers_group):
                if isinstance(layer, np.ndarray):
                    if len(layer.shape) == 3 and layer.shape[2] == 4:
                        # Konversi BGRA ke RGBA lalu ke RGB
                        layer_rgba = cv2.cvtColor(layer, cv2.COLOR_BGRA2RGBA)
                        layer_pil = Image.fromarray(layer_rgba)
                    else:
                        layer_pil = Image.fromarray(layer)
                else:
                    layer_pil = layer
                
                # Konversi ke RGB jika RGBA
                if layer_pil.mode == 'RGBA':
                    layer_pil = layer_pil.convert('RGB')
                
                group_layers.append((f"{name}_{i}", layer_pil))
            
            # Tambahkan group sebagai nested layer
            if group_layers:
                layers.append((name, group_layers))
        
        # Simpan PSD
        timestamp = int(time.time())
        filename = f"{output_dir}/layers_{mode}_{timestamp}.psd"
        
        # Gunakan nested_layers untuk membuat PSD
        nested_layers(layers, filename)
        
        print(f"✅ PSD saved successfully with pytoshop: {filename}")
        return filename
    except Exception as e:
        print(f"⚠️  Pytoshop PSD save failed: {str(e)}")
        raise e

def save_psd_with_psdtools_simple(input_image, layer_lists, layer_names, blend_modes, output_dir, mode):
    """Save PSD using psd-tools with simple approach"""
    try:
        from psd_tools import PSDImage
        from psd_tools.api.layers import PixelLayer, Group
        
        # Buat PSD baru dengan ukuran yang sesuai
        if isinstance(input_image, np.ndarray):
            height, width = input_image.shape[:2]
        else:
            width, height = input_image.size
            
        # Konversi input image ke PIL Image RGB
        if isinstance(input_image, np.ndarray):
            if len(input_image.shape) == 3 and input_image.shape[2] == 4:
                # Konversi BGRA ke RGB
                input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGRA2RGB)
                input_pil = Image.fromarray(input_rgb)
            else:
                input_pil = Image.fromarray(input_image)
        else:
            if input_image.mode == 'RGBA':
                input_pil = input_image.convert('RGB')
            else:
                input_pil = input_image
        
        # Buat PSD baru dari gambar
        psd = PSDImage.frompil(input_pil)
        
        # Tambahkan setiap group layer
        for layers, name, blend_mode in zip(layer_lists, layer_names, blend_modes):
            # Buat group untuk setiap jenis layer
            group_data = []
            
            # Tambahkan setiap layer ke dalam group
            for i, layer in enumerate(layers):
                if isinstance(layer, np.ndarray):
                    if len(layer.shape) == 3 and layer.shape[2] == 4:
                        # Konversi BGRA ke RGB
                        layer_rgb = cv2.cvtColor(layer, cv2.COLOR_BGRA2RGB)
                        layer_pil = Image.fromarray(layer_rgb)
                    else:
                        layer_pil = Image.fromarray(layer)
                else:
                    if layer.mode == 'RGBA':
                        layer_pil = layer.convert('RGB')
                    else:
                        layer_pil = layer
                
                # Tambahkan layer ke group data
                group_data.append(layer_pil)
            
            # Jika ada layer dalam group, buat composite dari semua layer
            if group_data:
                # Gabungkan semua layer dalam group menjadi satu
                composite = group_data[0].copy()
                for layer_img in group_data[1:]:
                    # Blend layers (simple paste untuk sekarang)
                    composite.paste(layer_img, (0, 0), layer_img if layer_img.mode == 'RGBA' else None)
                
                # Buat layer dari composite
                new_psd = PSDImage.frompil(composite)
                if new_psd.layers:
                    # Copy layer pertama dari composite PSD
                    layer_to_add = new_psd[0]
                    psd.append(layer_to_add)
        
        # Simpan PSD
        timestamp = int(time.time())
        filename = f"{output_dir}/layers_{mode}_{timestamp}.psd"
        psd.save(filename)
        
        print(f"✅ PSD saved successfully with psd-tools: {filename}")
        return filename
    except Exception as e:
        print(f"⚠️  PSD-tools simple save failed: {str(e)}")
        raise e

def save_psd_alternative(input_image, layer_lists, layer_names, blend_modes, output_dir, mode):
    """Alternative PSD save method using basic approach"""
    try:
        from psd_tools import PSDImage
        
        # Konversi input image ke PIL Image
        if isinstance(input_image, np.ndarray):
            if len(input_image.shape) == 3 and input_image.shape[2] == 4:
                input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGRA2RGB)
                input_pil = Image.fromarray(input_rgb)
            else:
                input_pil = Image.fromarray(input_image)
        else:
            input_pil = input_image.convert('RGB') if input_image.mode == 'RGBA' else input_image
        
        # Buat PSD sederhana dari gambar utama
        psd = PSDImage.frompil(input_pil)
        
        # Simpan PSD
        timestamp = int(time.time())
        filename = f"{output_dir}/layers_{mode}_{timestamp}.psd"
        psd.save(filename)
        
        print(f"✅ Basic PSD saved successfully: {filename}")
        return filename
    except Exception as e:
        print(f"⚠️  Alternative PSD save failed: {str(e)}")
        raise e

def safe_save_psd(input_image, layer_lists, layer_names, blend_modes, output_dir, mode):
    """Coba save PSD dengan beberapa metode, jika gagal fallback ke save gambar"""
    
    # Metode 1: Pytoshop dengan import yang benar
    try:
        print("🔄 Attempting to save PSD with pytoshop...")
        return save_psd_with_pytoshop(input_image, layer_lists, layer_names, blend_modes, output_dir, mode)
    except Exception as e:
        print(f"⚠️  Pytoshop PSD save failed: {str(e)}")
        
        # Metode 2: PSD-tools simple dengan perbaikan
        try:
            print("🔄 Attempting to save PSD with psd-tools simple...")
            return save_psd_with_psdtools_simple(input_image, layer_lists, layer_names, blend_modes, output_dir, mode)
        except Exception as e2:
            print(f"⚠️  PSD-tools simple save failed: {str(e2)}")
            
            # Metode 3: Alternative basic PSD
            try:
                print("🔄 Attempting to save PSD with alternative method...")
                return save_psd_alternative(input_image, layer_lists, layer_names, blend_modes, output_dir, mode)
            except Exception as e3:
                print(f"⚠️  Alternative PSD save failed: {str(e3)}")
                
                # Metode 4: Fungsi asli (jika ada)
                try:
                    from ldivider.ld_utils import save_psd
                    print("🔄 Attempting to save PSD with original method...")
                    
                    # Periksa format input_image
                    if isinstance(input_image, Image.Image):
                        input_image = pil2cv(input_image)
                    
                    # Pastikan semua layer dalam format numpy array
                    formatted_layer_lists = []
                    for layer_list in layer_lists:
                        formatted_layers = []
                        for layer in layer_list:
                            if isinstance(layer, Image.Image):
                                layer = pil2cv(layer)
                            formatted_layers.append(layer)
                        formatted_layer_lists.append(formatted_layers)
                    
                    # Coba simpan PSD dengan data yang sudah diformat
                    return save_psd(input_image, formatted_layer_lists, layer_names, blend_modes, output_dir, mode)
                except Exception as e4:
                    print(f"⚠️  Original PSD save method failed: {str(e4)}")
                    
                    # Fallback: Simpan sebagai gambar terpisah
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

                    # Hapus folder sementara
                    import shutil
                    shutil.rmtree(layers_folder)
                    print(f"✅ Layers saved as ZIP: {zip_filename}. Temporary folder deleted.")
                    return zip_filename

from ldivider.ld_segment import get_mask_generator, get_masks, show_anns

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
            # Hanya panggil divide_folder jika file yang dibuat adalah PSD (bukan ZIP)
            if filename.endswith('.psd'):
                try:
                    divide_folder(filename, input_dir, layer_mode)
                except Exception as e:
                    print(f"⚠️  Error in divide_folder: {str(e)}")
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
            # Hanya panggil divide_folder jika file yang dibuat adalah PSD (bukan ZIP)
            if filename.endswith('.psd'):
                try:
                    divide_folder(filename, input_dir, layer_mode)
                except Exception as e:
                    print(f"⚠️  Error in divide_folder: {str(e)}")
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
            # Hanya panggil divide_folder jika file yang dibuat adalah PSD (bukan ZIP)
            if filename.endswith('.psd'):
                try:
                    divide_folder(filename, input_dir, layer_mode)
                except Exception as e:
                    print(f"⚠️  Error in divide_folder: {str(e)}")
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
            # Hanya panggil divide_folder jika file yang dibuat adalah PSD (bukan ZIP)
            if filename.endswith('.psd'):
                try:
                    divide_folder(filename, input_dir, layer_mode)
                except Exception as e:
                    print(f"⚠️  Error in divide_folder: {str(e)}")
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