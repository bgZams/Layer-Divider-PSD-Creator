# File: patch_pytoshop.py
# Jalankan script ini untuk mempatch pytoshop
import os
import site

# Temukan lokasi pytoshop
def find_pytoshop_path():
    for path in site.getsitepackages():
        pytoshop_path = os.path.join(path, 'pytoshop', 'codecs.py')
        if os.path.exists(pytoshop_path):
            return pytoshop_path
    
    # Coba di virtual environment
    import sys
    venv_path = os.path.join(sys.prefix, 'lib', 'site-packages', 'pytoshop', 'codecs.py')
    if os.path.exists(venv_path):
        return venv_path
    
    return None

def patch_codecs():
    codecs_path = find_pytoshop_path()
    if not codecs_path:
        print("❌ File codecs.py tidak ditemukan!")
        return False
    
    print(f"📁 Ditemukan: {codecs_path}")
    
    # Backup file asli
    backup_path = codecs_path + '.backup'
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(codecs_path, backup_path)
        print(f"💾 Backup dibuat: {backup_path}")
    
    # Baca file
    with open(codecs_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check apakah sudah dipatch
    if 'from . import packbits' in content:
        print("✅ File sudah dipatch!")
        return True
    
    # Lakukan patch
    # Cari import statements dan tambahkan import packbits
    if 'from . import' in content:
        content = content.replace(
            'from . import',
            'from . import packbits\nfrom . import'
        )
    else:
        # Jika tidak ada import relatif, tambahkan di awal
        lines = content.split('\n')
        # Cari baris pertama yang bukan comment/docstring
        insert_line = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
                insert_line = i
                break
        
        lines.insert(insert_line, 'from . import packbits')
        content = '\n'.join(lines)
    
    # Simpan file yang sudah dipatch
    with open(codecs_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Patch berhasil diterapkan!")
    return True

if __name__ == "__main__":
    print("🔧 Mempatch pytoshop untuk mengatasi packbits error...")
    if patch_codecs():
        print("\n🎉 Patch selesai! Coba jalankan aplikasi lagi.")
    else:
        print("\n❌ Patch gagal!")