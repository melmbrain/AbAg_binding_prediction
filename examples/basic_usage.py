"""
Basic Usage Example - AbAg Binding Affinity Prediction

This script demonstrates how to use the AffinityPredictor for single and batch predictions.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from abag_affinity import AffinityPredictor


def example_1_single_prediction():
    """Example 1: Single antibody-antigen prediction"""
    print("="*70)
    print("Example 1: Single Prediction")
    print("="*70)

    # Initialize predictor
    print("\nInitializing predictor...")
    predictor = AffinityPredictor(verbose=True)

    # Example sequences (truncated for display)
    antibody_heavy = "EVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCARDGPGNFDESSGYVWGQGTLVTVSS"
    antibody_light = "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCQRYNRAPYTFGQGTKVEIK"
    antigen = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"

    # Predict binding affinity
    print("\nPredicting binding affinity...")
    result = predictor.predict(
        antibody_heavy=antibody_heavy,
        antibody_light=antibody_light,
        antigen=antigen
    )

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"pKd:            {result['pKd']:.2f}")
    print(f"Kd (nM):        {result['Kd_nM']:.2f}")
    print(f"Kd (μM):        {result['Kd_uM']:.4f}")
    print(f"Category:       {result['category']}")
    print(f"Interpretation: {result['interpretation']}")
    print("="*70)


def example_2_heavy_only():
    """Example 2: Heavy chain only (VHH/nanobody)"""
    print("\n\n" + "="*70)
    print("Example 2: Heavy Chain Only (VHH/Nanobody)")
    print("="*70)

    predictor = AffinityPredictor(verbose=False)

    # VHH sequence (single domain antibody)
    vhh_sequence = "QVQLQESGGGSVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAINSRGGSTYYADSVKGRFTISRDNAKNTLYLQMSSLKPEDTAVYYCAAGDVWGQGTQVTVSS"
    antigen = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"

    print("\nPredicting VHH binding...")
    result = predictor.predict(
        antibody_heavy=vhh_sequence,
        antigen=antigen
    )

    print("\nResults:")
    print(f"  pKd: {result['pKd']:.2f}")
    print(f"  Kd: {result['Kd_nM']:.1f} nM")
    print(f"  {result['interpretation']}")


def example_3_batch_processing():
    """Example 3: Batch processing multiple pairs"""
    print("\n\n" + "="*70)
    print("Example 3: Batch Processing")
    print("="*70)

    predictor = AffinityPredictor(verbose=False)

    # Target antigen
    antigen = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"

    # Library of antibody candidates
    pairs = [
        {
            'id': 'Ab001',
            'antibody_heavy': 'EVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCARDGPGNFDESSGYVWGQGTLVTVSS',
            'antigen': antigen
        },
        {
            'id': 'Ab002',
            'antibody_heavy': 'QVQLQESGGGSVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAINSRGGSTYYADSVKGRFTISRDNAKNTLYLQMSSLKPEDTAVYYCAAGDVWGQGTQVTVSS',
            'antigen': antigen
        },
        {
            'id': 'Ab003',
            'antibody_heavy': 'EVQLQQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARGLRFLEWFAYWGQGTLVTVSS',
            'antigen': antigen
        },
        {
            'id': 'Ab004',
            'antibody_heavy': 'QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARRHWPGGFDYWGQGTLVTVSS',
            'antigen': antigen
        },
        {
            'id': 'Ab005',
            'antibody_heavy': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDGYCSGGSCYSWFAYWGQGTLVTVSS',
            'antigen': antigen
        }
    ]

    print(f"\nProcessing {len(pairs)} antibody candidates...")
    results = predictor.predict_batch(pairs, show_progress=True)

    # Sort by affinity
    results_sorted = sorted(results, key=lambda x: x.get('pKd', 0), reverse=True)

    print("\n" + "="*70)
    print("RANKED RESULTS")
    print("="*70)
    print(f"{'Rank':<6} {'ID':<10} {'pKd':<8} {'Kd (nM)':<12} {'Category':<12}")
    print("-"*70)

    for i, result in enumerate(results_sorted, 1):
        if result['status'] == 'success':
            print(f"{i:<6} {result['id']:<10} {result['pKd']:<8.2f} {result['Kd_nM']:<12.1f} {result['category']:<12}")
        else:
            print(f"{i:<6} {result['id']:<10} ERROR: {result['error']}")

    print("="*70)


def example_4_compare_variants():
    """Example 4: Compare antibody variants"""
    print("\n\n" + "="*70)
    print("Example 4: Compare Antibody Variants")
    print("="*70)

    predictor = AffinityPredictor(verbose=False)

    antigen = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"

    # Original and mutant variants
    variants = {
        'Original': 'EVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCARDGPGNFDESSGYVWGQGTLVTVSS',
        'Mutant_1': 'EVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAFNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCARDGPGNFDESSGYVWGQGTLVTVSS',
        'Mutant_2': 'EVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCARDGPGNFDESSAYVWGQGTLVTVSS'
    }

    print("\nComparing variants...")
    results = {}
    for name, heavy_chain in variants.items():
        result = predictor.predict(
            antibody_heavy=heavy_chain,
            antigen=antigen
        )
        results[name] = result

    print("\n" + "="*70)
    print("VARIANT COMPARISON")
    print("="*70)
    print(f"{'Variant':<15} {'pKd':<8} {'Kd (nM)':<12} {'ΔpKd':<10}")
    print("-"*70)

    original_pKd = results['Original']['pKd']
    for name, result in results.items():
        delta = result['pKd'] - original_pKd
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        print(f"{name:<15} {result['pKd']:<8.2f} {result['Kd_nM']:<12.1f} {delta_str:<10}")

    print("="*70)


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("AbAg Binding Affinity Prediction - Usage Examples")
    print("="*70 + "\n")

    try:
        # Run examples
        example_1_single_prediction()
        example_2_heavy_only()
        example_3_batch_processing()
        example_4_compare_variants()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
