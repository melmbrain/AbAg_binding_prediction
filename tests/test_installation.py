"""
Installation and Functionality Test Suite
Tests that the package is properly installed and functional
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_1_imports():
    """Test that all required packages can be imported"""
    print("="*70)
    print("Test 1: Package Imports")
    print("="*70)

    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm'
    }

    failed = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name:<20} imported successfully")
        except ImportError:
            print(f"  ❌ {name:<20} FAILED to import")
            failed.append(package)

    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print(f"   Install with: pip install {' '.join(failed)}")
        return False

    print(f"\n✅ All required packages available")
    return True


def test_2_package_import():
    """Test that abag_affinity package can be imported"""
    print("\n" + "="*70)
    print("Test 2: AbAg Package Import")
    print("="*70)

    try:
        from abag_affinity import AffinityPredictor
        print("  ✅ abag_affinity package imported")

        from abag_affinity import __version__
        print(f"  ✅ Package version: {__version__}")

        return True

    except ImportError as e:
        print(f"  ❌ Failed to import abag_affinity: {e}")
        return False


def test_3_model_files():
    """Test that model files exist"""
    print("\n" + "="*70)
    print("Test 3: Model Files")
    print("="*70)

    required_files = {
        'models/agab_phase2_model.pth': 'Phase 2 Model',
        'models/agab_phase2_results.json': 'Model Metadata'
    }

    failed = []
    for filepath, name in required_files.items():
        path = Path(filepath)
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ {name:<20} found ({size:.1f} MB)")
        else:
            print(f"  ❌ {name:<20} NOT FOUND at {filepath}")
            failed.append(filepath)

    if failed:
        print(f"\n❌ Missing files: {', '.join(failed)}")
        return False

    print(f"\n✅ All model files present")
    return True


def test_4_predictor_init():
    """Test predictor initialization"""
    print("\n" + "="*70)
    print("Test 4: Predictor Initialization")
    print("="*70)

    try:
        from abag_affinity import AffinityPredictor

        print("  Initializing predictor...")
        predictor = AffinityPredictor(verbose=False)

        print("  ✅ Predictor initialized successfully")

        # Check device
        print(f"  ✅ Device: {predictor.device}")

        return True

    except Exception as e:
        print(f"  ❌ Predictor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_single_prediction():
    """Test single prediction"""
    print("\n" + "="*70)
    print("Test 5: Single Prediction")
    print("="*70)

    try:
        from abag_affinity import AffinityPredictor

        predictor = AffinityPredictor(verbose=False)

        # Test sequences
        antibody_heavy = "EVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCARDGPGNFDESSGYVWGQGTLVTVSS"
        antigen = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"

        print("  Making prediction...")
        result = predictor.predict(
            antibody_heavy=antibody_heavy,
            antigen=antigen
        )

        print("  ✅ Prediction successful")

        # Check result format
        required_keys = ['pKd', 'Kd_nM', 'Kd_uM', 'category', 'interpretation']
        for key in required_keys:
            if key in result:
                print(f"  ✅ Result contains '{key}'")
            else:
                print(f"  ❌ Result missing '{key}'")
                return False

        print(f"\n  Result: pKd = {result['pKd']:.2f}, Kd = {result['Kd_nM']:.1f} nM")
        print(f"  Category: {result['category']}")

        return True

    except Exception as e:
        print(f"  ❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*70)
    print("Test 6: Batch Prediction")
    print("="*70)

    try:
        from abag_affinity import AffinityPredictor

        predictor = AffinityPredictor(verbose=False)

        # Test pairs
        antigen = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"
        pairs = [
            {
                'id': 'test_1',
                'antibody_heavy': 'EVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCARDGPGNFDESSGYVWGQGTLVTVSS',
                'antigen': antigen
            },
            {
                'id': 'test_2',
                'antibody_heavy': 'QVQLQESGGGSVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAINSRGGSTYYADSVKGRFTISRDNAKNTLYLQMSSLKPEDTAVYYCAAGDVWGQGTQVTVSS',
                'antigen': antigen
            }
        ]

        print(f"  Processing {len(pairs)} pairs...")
        results = predictor.predict_batch(pairs, show_progress=False)

        print(f"  ✅ Batch prediction successful")
        print(f"  ✅ Processed {len(results)} pairs")

        # Check all succeeded
        success_count = sum(1 for r in results if r.get('status') == 'success')
        print(f"  ✅ {success_count}/{len(results)} predictions succeeded")

        return success_count == len(pairs)

    except Exception as e:
        print(f"  ❌ Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_input_validation():
    """Test input validation"""
    print("\n" + "="*70)
    print("Test 7: Input Validation")
    print("="*70)

    try:
        from abag_affinity import AffinityPredictor

        predictor = AffinityPredictor(verbose=False)

        # Test empty sequence
        print("  Testing empty sequence rejection...")
        try:
            predictor.predict(antibody_heavy="", antigen="KVFGRC")
            print("  ❌ Failed to reject empty sequence")
            return False
        except ValueError:
            print("  ✅ Correctly rejected empty sequence")

        # Test invalid amino acids
        print("  Testing invalid amino acid rejection...")
        try:
            predictor.predict(antibody_heavy="EVQLXYZ", antigen="KVFGRC")
            print("  ❌ Failed to reject invalid amino acids")
            return False
        except ValueError:
            print("  ✅ Correctly rejected invalid amino acids")

        # Test short sequence
        print("  Testing short sequence rejection...")
        try:
            predictor.predict(antibody_heavy="EVQ", antigen="KVFGRC")
            print("  ❌ Failed to reject short sequence")
            return False
        except ValueError:
            print("  ✅ Correctly rejected short sequence")

        print("\n✅ Input validation working correctly")
        return True

    except Exception as e:
        print(f"  ❌ Validation test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*70)
    print("AbAg Binding Affinity Prediction - Installation Tests")
    print("="*70 + "\n")

    tests = [
        ("Package Imports", test_1_imports),
        ("AbAg Package", test_2_package_import),
        ("Model Files", test_3_model_files),
        ("Predictor Init", test_4_predictor_init),
        ("Single Prediction", test_5_single_prediction),
        ("Batch Prediction", test_6_batch_prediction),
        ("Input Validation", test_7_input_validation)
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:<12} {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nYour installation is working correctly!")
        print("\nNext steps:")
        print("  1. Read README.md for usage instructions")
        print("  2. Try examples: python examples/basic_usage.py")
        print("  3. Start making predictions!")
        print("="*70 + "\n")
        return 0
    else:
        failed_count = sum(1 for _, p in results if not p)
        print(f"❌ {failed_count} TEST(S) FAILED")
        print("="*70)
        print("\nPlease fix the failed tests before proceeding.")
        print("Check the error messages above for details.")
        print("="*70 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
