# File: ld_utils_patch.py
# Simpan file ini di folder ldivider/ dan import sebagai pengganti ld_utils

import cv2
import numpy as np
from PIL import Image
import os
import time
import zipfile
from pytoshop.enums import BlendMode
from pytoshop import PsdFile, Group, Layer
import pytoshop.user

# Import fungsi lain dari ld_utils asli
try:
    from .ld_utils import *
except:
    # Fallback jika import relatif gagal
    pass

def save_layers_as_images(input_image, layer_lists, layer_names, output_dir, mode):
    """
    Simpan layers sebagai gambar terpisah jika PSD gagal
    """
    # Buat folder output
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
            
            # Konversi layer ke format yang bisa disimpan
            if isinstance(layer, np.ndarray):
                if len(layer.shape) == 3 and layer.shape[2] == 4:
                    # BGRA to RGBA
                    layer_rgba = cv2.cvtColor(layer, cv2.COLOR_BGRA2RGBA)
                    Image.fromarray(layer_rgba).save(layer_path)
                else:
                    cv2.imwrite(layer_path, layer)
            else:
                layer.save(layer_path)
    
    # Buat file info.txt dengan informasi layers
    info_path = os.path.join(layers_dir, "layer_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Layer Division Result - Mode: {mode}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Groups: {len(layer_lists)}\n\n")
        
        for i, name in enumerate(layer_names):
            f.write(f"Group {i+1}: {name} ({len(layer_lists[i])} layers)\n")
        
        f.write(f"\nNote: PSD creation failed due to pytoshop packbits error.\n")
        f.write(f"Layers saved as individual PNG files with transparency.\n")
        f.write(f"You can manually import these into Photoshop:\n")
        f.write(f"1. Create new document\n")
        f.write(f"2. Drag & drop PNG files as layers\n")
        f.write(f"3. Set blend modes manually if needed\n")
    
    return layers_dir

def safe_save_psd(input_image, layer_lists, layer_names, blend_modes, output_dir, mode):
    """
    Coba save PSD dengan berbagai metode, fallback ke images jika gagal
    """
    # Method 1: Coba tanpa kompresi
    try:
        print("🔄 Trying PSD save without compression...")
        # Modifikasi temporary untuk disable compression
        return save_psd_no_compression(input_image, layer_lists, layer_names, blend_modes, output_dir, mode)
    except Exception as e1:
        print(f"⚠️  No compression failed: {e1}")
    
    # Method 2: Coba dengan psd-tools sebagai alternatif
    try:
        print("🔄 Trying alternative PSD method...")
        return save_psd_alternative(input_image, layer_lists, layer_names, blend_modes, output_dir, mode)
    except Exception as e2:
        print(f"⚠️  Alternative method failed: {e2}")
    
    # Method 3: Fallback ke images
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

def save_psd_no_compression(input_image, layer_lists, layer_names, blend_modes, output_dir, mode):
    """
    Save PSD tanpa kompresi untuk menghindari packbits error
    """
    # Implementasi sederhana tanpa kompresi RLE
    # Ini akan membuat file lebih besar tapi menghindari packbits error
    
    timestamp = int(time.time())
    filename = f"{output_dir}/result_{mode}_{timestamp}_uncompressed.psd"
    
    # Buat PSD dengan compression mode raw
    height, width = input_image.shape[:2]
    psd = PsdFile(width=width, height=height)
    
    # Set compression ke raw (no compression)
    for layer_list, name, blend_mode in zip(layer_lists, layer_names, blend_modes):
        group = Group(name=name, blend_mode=blend_mode)
        
        for i, layer_data in enumerate(layer_list):
            layer_name = f"{name}_{i:03d}"
            
            # Konversi layer data ke format yang tepat
            if isinstance(layer_data, np.ndarray):
                if len(layer_data.shape) == 3 and layer_data.shape[2] == 4:
                    # BGRA format
                    layer_array = layer_data
                else:
                    # Add alpha channel if missing
                    layer_array = np.dstack([layer_data, np.ones(layer_data.shape[:2], dtype=np.uint8) * 255])
            else:
                layer_array = np.array(layer_data)
            
            # Buat layer dengan compression raw
            layer = pytoshop.user.Layer.from_array(layer_array, compression='raw')
            layer.name = layer_name
            layer.blend_mode = BlendMode.normal
            
            group.layers.append(layer)
        
        psd.layers.append(group)
    
    # Save dengan error handling
    with open(filename, 'wb') as fd:
        psd.write(fd)
    
    return filename

def save_psd_alternative(input_image, layer_lists, layer_names, blend_modes, output_dir, mode):
    """
    Metode alternatif menggunakan psd-tools untuk write
    """
    from psd_tools import PSDImage
    from psd_tools.constants import BlendMode as PSDBlendMode
    
    # Buat PSD document baru
    timestamp = int(time.time())
    filename = f"{output_dir}/result_{mode}_{timestamp}_alt.psd"
    
    # Implementasi dengan psd-tools (lebih sederhana)
    # Note: psd-tools biasanya untuk read, tapi bisa dicoba untuk write sederhana
    
    # Fallback: simpan sebagai layered TIFF jika PSD tidak bisa
    tiff_filename = f"{output_dir}/result_{mode}_{timestamp}.tiff"
    
    # Buat multi-layer TIFF
    images = []
    for layer_list in layer_lists:
        for layer_data in layer_list:
            if isinstance(layer_data, np.ndarray):
                if len(layer_data.shape) == 3 and layer_data.shape[2] == 4:
                    # BGRA to RGBA
                    layer_rgba = cv2.cvtColor(layer_data, cv2.COLOR_BGRA2RGBA)
                    images.append(Image.fromarray(layer_rgba))
                else:
                    images.append(Image.fromarray(layer_data))
            else:
                images.append(layer_data)
    
    # Save as multi-page TIFF
    if images:
        images[0].save(tiff_filename, format='TIFF', save_all=True, append_images=images[1:])
        print(f"✅ Saved as multi-layer TIFF: {tiff_filename}")
        return tiff_filename
    
    raise Exception("No images to save")

# Override fungsi save_psd asli
def save_psd(input_image, layer_lists, layer_names, blend_modes, output_dir, mode):
    """
    Fungsi save_psd yang aman dengan fallback
    """
    return safe_save_psd(input_image, layer_lists, layer_names, blend_modes, output_dir, mode)