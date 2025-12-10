"""
AGGRESSIVE Disk Space Management for Colab
Prevents 235GB limit crashes with multiple cleanup strategies
"""

import subprocess
import shutil
import gc
import torch
from pathlib import Path


def ultra_aggressive_cleanup():
    """
    ULTRA AGGRESSIVE disk cleanup - keeps ONLY what's absolutely necessary
    Run this if disk usage goes above 100GB
    """
    print("\n🚨 ULTRA AGGRESSIVE CLEANUP ACTIVATED")
    print("="*60)

    freed_space = 0

    # 1. Clear ALL pip cache
    print("1. Clearing pip cache...")
    try:
        result = subprocess.run(['pip', 'cache', 'purge'], capture_output=True, text=True)
        print(f"  ✓ Pip cache cleared")
    except:
        pass

    # 2. Clear ALL CUDA cache
    print("2. Clearing CUDA cache...")
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print(f"  ✓ CUDA cache cleared")
    except:
        pass

    # 3. Remove ALL HuggingFace cache EXCEPT actively used models
    print("3. Cleaning HuggingFace cache (keeping only IgT5 & ESM-2)...")
    try:
        cache_dir = Path.home() / '.cache' / 'huggingface'

        if cache_dir.exists():
            # Models we're using
            keep_models = [
                'Exscientia--IgT5',
                'facebook--esm2_t33_650M_UR50D'
            ]

            # Clear hub cache
            hub_dir = cache_dir / 'hub'
            if hub_dir.exists():
                for item in hub_dir.iterdir():
                    if item.is_dir():
                        should_keep = any(model in item.name for model in keep_models)
                        if not should_keep:
                            try:
                                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                                shutil.rmtree(item)
                                freed_space += size
                                print(f"  ✓ Removed: {item.name[:40]}... ({size/1e9:.2f} GB)")
                            except Exception as e:
                                print(f"  ⚠ Could not remove {item.name}: {e}")

            # Clear downloads
            downloads_dir = cache_dir / 'downloads'
            if downloads_dir.exists():
                try:
                    size = sum(f.stat().st_size for f in downloads_dir.rglob('*') if f.is_file())
                    shutil.rmtree(downloads_dir)
                    downloads_dir.mkdir()
                    freed_space += size
                    print(f"  ✓ Cleared downloads ({size/1e9:.2f} GB)")
                except:
                    pass

    except Exception as e:
        print(f"  ⚠ HuggingFace cleanup warning: {e}")

    # 4. Clear torch hub cache
    print("4. Clearing torch hub cache...")
    try:
        torch_cache = Path.home() / '.cache' / 'torch'
        if torch_cache.exists():
            for item in torch_cache.iterdir():
                try:
                    if item.is_dir():
                        size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        shutil.rmtree(item)
                        freed_space += size
                    elif item.is_file():
                        size = item.stat().st_size
                        item.unlink()
                        freed_space += size
                except:
                    pass
            print(f"  ✓ Torch cache cleared")
    except:
        pass

    # 5. Clear temporary files
    print("5. Clearing /tmp...")
    try:
        tmp_dir = Path('/tmp')
        for item in tmp_dir.glob('tmp*'):
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                elif item.is_file():
                    item.unlink()
            except:
                pass
        print(f"  ✓ /tmp cleared")
    except:
        pass

    # 6. Clear Python cache
    print("6. Clearing Python __pycache__...")
    try:
        for pycache in Path('/content').rglob('__pycache__'):
            try:
                shutil.rmtree(pycache)
            except:
                pass
        print(f"  ✓ Python cache cleared")
    except:
        pass

    # 7. Force garbage collection
    print("7. Running garbage collection...")
    for _ in range(3):
        gc.collect()
    print(f"  ✓ Garbage collected")

    print(f"\n📊 Total space freed: ~{freed_space/1e9:.2f} GB")
    print("="*60)


def monitor_and_auto_cleanup(threshold_gb=150):
    """
    Monitor disk usage and auto-cleanup if needed
    Returns: (used_gb, total_gb, percentage)
    """
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if '/' in line and 'Filesystem' not in line:
                parts = line.split()
                if len(parts) >= 5:
                    # Parse disk usage
                    used_str = parts[2]  # e.g., "150G"
                    total_str = parts[1]  # e.g., "236G"
                    percent_str = parts[4]  # e.g., "64%"

                    # Convert to GB
                    used_gb = float(used_str.replace('G', '').replace('M', '0.'))
                    total_gb = float(total_str.replace('G', '').replace('M', '0.'))
                    percent = int(percent_str.replace('%', ''))

                    print(f"📊 Disk: {used_gb:.1f}GB / {total_gb:.1f}GB ({percent}% used)")

                    # Auto-cleanup if over threshold
                    if used_gb > threshold_gb:
                        print(f"\n⚠️  WARNING: Disk usage {used_gb:.1f}GB > {threshold_gb}GB threshold!")
                        print("Running ultra aggressive cleanup...")
                        ultra_aggressive_cleanup()

                    return used_gb, total_gb, percent

    except Exception as e:
        print(f"⚠️  Could not monitor disk: {e}")

    return None, None, None


def setup_continuous_monitoring(check_every_n_batches=250, threshold_gb=150):
    """
    Setup a function to call every N batches during training
    Usage in training loop:

    if (batch_idx + 1) % 250 == 0:
        monitor_and_auto_cleanup(threshold_gb=150)
    """
    pass  # This is just documentation


# Quick cleanup function for regular use
def standard_cleanup():
    """Standard cleanup - run at start of each epoch"""
    print("\n🧹 Standard cleanup...")

    # Clear pip cache
    try:
        subprocess.run(['pip', 'cache', 'purge'], capture_output=True)
        print("  ✓ Cleared pip cache")
    except:
        pass

    # Clear CUDA cache
    try:
        torch.cuda.empty_cache()
        gc.collect()
        print("  ✓ Cleared CUDA cache")
    except:
        pass

    # Monitor disk
    monitor_and_auto_cleanup(threshold_gb=180)  # Higher threshold for standard cleanup


if __name__ == "__main__":
    # Test the cleanup
    print("Testing disk cleanup...")
    ultra_aggressive_cleanup()
    print("\nDisk status after cleanup:")
    monitor_and_auto_cleanup()
