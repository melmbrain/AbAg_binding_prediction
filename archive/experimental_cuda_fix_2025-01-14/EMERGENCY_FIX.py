# ============================================================================
# EMERGENCY FIX - Paste this into Colab if error persists
# ============================================================================
# This script patches the training file to FORCE disable torch.compile
# Run this BEFORE running the training script
# ============================================================================

import os

print("="*70)
print("EMERGENCY FIX: Patching train_ultra_speed_v26.py")
print("="*70)

script_path = 'train_ultra_speed_v26.py'

if not os.path.exists(script_path):
    print(f"❌ ERROR: {script_path} not found!")
    print("Make sure you're in the correct directory:")
    print("  cd /content/drive/MyDrive/AbAg_Training")
else:
    # Read the script
    with open(script_path, 'r') as f:
        content = f.read()

    # Check if it has the old broken argument parser
    if "type=bool, default=True" in content and "'--use_compile'" in content:
        print("⚠️  Found OLD broken code - applying emergency fix...")

        # Fix 1: Change argument parser
        content = content.replace(
            "parser.add_argument('--use_compile', type=bool, default=True)",
            "parser.add_argument('--use_compile', type=lambda x: x.lower() == 'true', default=False)  # EMERGENCY FIX"
        )

        # Fix 2: Add global compile disable at the top (after imports)
        if "torch.compiler.disable()" not in content:
            import_section_end = content.find("# ============================================================================\n# OPTIMIZATIONS:")
            if import_section_end > 0:
                patch = '''
# ============================================================================
# EMERGENCY FIX: Force disable torch.compile globally
# ============================================================================
import torch.compiler
torch.compiler.disable()
print("🚨 EMERGENCY FIX: torch.compile FORCEFULLY DISABLED")

'''
                content = content[:import_section_end] + patch + content[import_section_end:]

        # Save the fixed file
        with open(script_path, 'w') as f:
            f.write(content)

        print("✅ Emergency fix applied!")
        print("\nChanges made:")
        print("  1. Fixed argparse boolean conversion")
        print("  2. Added global torch.compiler.disable()")
        print("  3. Set default use_compile=False")

    elif "type=lambda x: x.lower() == 'true'" in content:
        print("✅ Script already has the fix!")

        # Still add global disable as extra safety
        if "torch.compiler.disable()" not in content:
            print("Adding extra safety: global torch.compiler.disable()...")

            with open(script_path, 'r') as f:
                lines = f.readlines()

            # Find line after imports (look for first "# ====" comment)
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.startswith("# ============================================================================"):
                    if "OPTIMIZATIONS:" in lines[i+1] if i+1 < len(lines) else False:
                        insert_pos = i
                        break

            if insert_pos > 0:
                patch_lines = [
                    "# ============================================================================\n",
                    "# EMERGENCY FIX: Force disable torch.compile globally\n",
                    "# ============================================================================\n",
                    "import torch.compiler\n",
                    "torch.compiler.disable()\n",
                    "print(\"🚨 EMERGENCY FIX: torch.compile FORCEFULLY DISABLED\")\n",
                    "\n"
                ]
                lines = lines[:insert_pos] + patch_lines + lines[insert_pos:]

                with open(script_path, 'w') as f:
                    f.writelines(lines)

                print("✅ Added global disable as extra safety!")
            else:
                print("⚠️  Could not find insertion point, skipping global disable")

    else:
        print("⚠️  Unknown script version - manual check needed")

    # Verify the fix
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    with open(script_path, 'r') as f:
        content = f.read()

    checks = [
        ("Fixed argparse", "type=lambda x: x.lower() == 'true'" in content),
        ("Default False", "default=False" in content and "'--use_compile'" in content),
        ("Global disable", "torch.compiler.disable()" in content),
    ]

    all_good = True
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"{status} {check_name}: {check_result}")
        if not check_result:
            all_good = False

    print("\n" + "="*70)
    if all_good:
        print("✅✅✅ ALL FIXES APPLIED - Ready to train!")
        print("\nRun this now:")
        print("  !python train_ultra_speed_v26.py")
    else:
        print("⚠️⚠️⚠️ Some fixes missing - manual intervention needed")
        print("\nTry these steps:")
        print("  1. Delete train_ultra_speed_v26.py from Google Drive")
        print("  2. Re-upload the fixed version from your local machine")
        print("  3. Restart Colab runtime")
    print("="*70)
