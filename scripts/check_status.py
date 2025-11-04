"""
Check status of database downloads and integration
Quick diagnostic tool to see what's been downloaded and what's ready
"""

import sys
from pathlib import Path

def check_downloads():
    """Check which databases have been downloaded"""
    print("\n" + "="*80)
    print("DATABASE DOWNLOAD STATUS")
    print("="*80)

    external_dir = Path('external_data')

    if not external_dir.exists():
        print("\n⚠ external_data directory not found")
        print("Creating directory...")
        external_dir.mkdir(parents=True, exist_ok=True)
        return False

    status = {
        'abbibench': False,
        'saaint': False,
        'pdbbind': False,
        'merged': False
    }

    # Check AbBiBench
    abbibench_files = list(external_dir.glob('abbibench*.csv'))
    if abbibench_files:
        file_size = abbibench_files[0].stat().st_size / 1024 / 1024
        print(f"\n✅ AbBiBench: {abbibench_files[0].name} ({file_size:.1f} MB)")
        status['abbibench'] = True
    else:
        print("\n❌ AbBiBench: Not downloaded")
        print("   Run: python scripts/download_abbibench.py")

    # Check SAAINT
    saaint_files = list(external_dir.glob('saaint*.csv'))
    if saaint_files:
        file_size = saaint_files[0].stat().st_size / 1024 / 1024
        print(f"\n✅ SAAINT-DB: {saaint_files[0].name} ({file_size:.1f} MB)")
        status['saaint'] = True
    else:
        print("\n❌ SAAINT-DB: Not downloaded")
        print("   Run: python scripts/download_saaint.py")

    # Check PDBbind
    pdbbind_files = list(external_dir.glob('pdbbind*.csv'))
    if pdbbind_files:
        file_size = pdbbind_files[0].stat().st_size / 1024 / 1024
        print(f"\n✅ PDBbind: {pdbbind_files[0].name} ({file_size:.1f} MB)")
        status['pdbbind'] = True
    else:
        print("\n⚠ PDBbind: Not downloaded (optional)")
        print("   Requires manual download from: http://www.pdbbind.org.cn/download.php")
        print("   Then run: python scripts/download_pdbbind.py")

    # Check merged dataset
    merged_files = list(external_dir.glob('merged*.csv'))
    if merged_files:
        file_size = merged_files[0].stat().st_size / 1024 / 1024
        print(f"\n✅ Merged Dataset: {merged_files[0].name} ({file_size:.1f} MB)")
        status['merged'] = True
    else:
        print("\n❌ Merged Dataset: Not created")
        print("   Run: python scripts/integrate_all_databases.py --existing YOUR_DATA.csv")

    return status

def check_integration():
    """Check integration report"""
    print("\n" + "="*80)
    print("INTEGRATION STATUS")
    print("="*80)

    report_file = Path('external_data/integration_report.txt')

    if report_file.exists():
        print("\n✅ Integration report found")
        print(f"   File: {report_file}")
        print(f"   Size: {report_file.stat().st_size / 1024:.1f} KB")
        print("\nReport contents:")
        print("-" * 80)
        print(report_file.read_text())
        print("-" * 80)
    else:
        print("\n❌ Integration report not found")
        print("   Integration has not been run yet")

def check_requirements():
    """Check Python packages"""
    print("\n" + "="*80)
    print("PYTHON REQUIREMENTS")
    print("="*80)

    required = {
        'pandas': 'Data processing',
        'numpy': 'Numerical operations',
        'tqdm': 'Progress bars',
        'datasets': 'Hugging Face datasets (for AbBiBench)'
    }

    all_installed = True

    for package, description in required.items():
        try:
            if package == 'datasets':
                import datasets
                version = datasets.__version__
            elif package == 'pandas':
                import pandas
                version = pandas.__version__
            elif package == 'numpy':
                import numpy
                version = numpy.__version__
            elif package == 'tqdm':
                import tqdm
                version = tqdm.__version__

            print(f"\n✅ {package:15s} {version:10s} - {description}")
        except ImportError:
            print(f"\n❌ {package:15s} {'NOT FOUND':10s} - {description}")
            all_installed = False

    if not all_installed:
        print("\n⚠ Missing packages detected")
        print("Install with: pip install pandas numpy tqdm datasets")

def show_next_steps(status):
    """Show recommended next steps"""
    print("\n" + "="*80)
    print("RECOMMENDED NEXT STEPS")
    print("="*80)

    if not status['abbibench'] and not status['saaint']:
        print("\n📥 Step 1: Download databases")
        print("   Option A (Windows): RUN_DOWNLOAD_AND_INTEGRATE.bat")
        print("   Option B (Manual):  python scripts/download_all.sh")
        print("   Option C (Individual):")
        print("     - python scripts/download_abbibench.py")
        print("     - python scripts/download_saaint.py")

    elif (status['abbibench'] or status['saaint']) and not status['merged']:
        print("\n🔗 Step 2: Integrate with your data")
        print("   python scripts/integrate_all_databases.py \\")
        print('     --existing "C:/Users/401-24/Desktop/Docking prediction/data/processed/phase6/final_205k_dataset.csv" \\')
        print("     --output external_data/merged_all_databases.csv")

    elif status['merged']:
        print("\n✅ Download and integration complete!")
        print("\n🎯 Next steps:")
        print("   1. Review: external_data/integration_report.txt")
        print("   2. Generate ESM2 embeddings for new sequences")
        print("   3. Train model:")
        print("      python train_balanced.py \\")
        print("        --data external_data/merged_all_databases.csv \\")
        print("        --loss weighted_mse \\")
        print("        --sampling stratified")

def main():
    """Main status check"""
    print("="*80)
    print("DATABASE INTEGRATION STATUS CHECKER")
    print("="*80)

    # Check requirements
    check_requirements()

    # Check downloads
    status = check_downloads()

    # Check integration
    check_integration()

    # Show next steps
    show_next_steps(status)

    print("\n" + "="*80)
    print("STATUS CHECK COMPLETE")
    print("="*80)

    print("\nFor detailed help, see:")
    print("  - QUICK_START_GUIDE.md")
    print("  - EXTERNAL_DATA_README.md")
    print("  - DOWNLOAD_INSTRUCTIONS.md")

if __name__ == "__main__":
    main()
