"""
Test evaluation functions for biology benchmarks using real examples from bio_data.
"""

from bio_benchmark.evaluation import evaluate_response


def test_b1_examples():
    """Test REL-B1 (homoplasy detection) evaluation with 3 examples."""
    examples = [
        {
            "question": "Homoplasy refers to structured convergence:\npairs or groups of distantly related taxa that repeatedly share the same nucleotide motifs\nacross many independent alignment columns more often than expected, while other taxa\nwith similar overall sequences do not share those nucleotide motifs as consistently.\n\nYour job is to examine the entire alignment and provided tree and decide whether such structured\nhomoplasy is likely to be present and which taxa are involved.\n\n[Alignment and tree data would be here...]",
            "answer": {"label": "yes", "taxa": [15, 49, 18, 28, 20, 39, 29, 42, 16, 31]},
            "responses": [
                "Yes. The taxa involved are taxon_15, taxon_49, taxon_18, taxon_28, taxon_20, taxon_39, taxon_29, taxon_42, taxon_16, taxon_31.",  # Correct
                "Yes. I identified taxon_15, taxon_49, taxon_18, taxon_28, taxon_20.",  # Partial match
                "No. There is no evidence of structured homoplasy.",  # Incorrect (label mismatch)
            ]
        },
        {
            "question": "Homoplasy refers to structured convergence:\npairs or groups of distantly related taxa that repeatedly share the same nucleotide motifs\nacross many independent alignment columns more often than expected, while other taxa\nwith similar overall sequences do not share those nucleotide motifs as consistently.\n\nYour job is to examine the entire alignment and provided tree and decide whether such structured\nhomoplasy is likely to be present and which taxa are involved.\n\n[Alignment and tree data would be here...]",
            "answer": {"label": "no", "taxa": []},
            "responses": [
                "No. There is no evidence of structured homoplasy in this alignment.",  # Correct
                "No.",  # Correct
                "Yes. The taxa involved are taxon_5, taxon_12, taxon_23.",  # Incorrect (label mismatch)
            ]
        },
        {
            "question": "Homoplasy refers to structured convergence:\npairs or groups of distantly related taxa that repeatedly share the same nucleotide motifs\nacross many independent alignment columns more often than expected, while other taxa\nwith similar overall sequences do not share those nucleotide motifs as consistently.\n\nYour job is to examine the entire alignment and provided tree and decide whether such structured\nhomoplasy is likely to be present and which taxa are involved.\n\n[Alignment and tree data would be here...]",
            "answer": {"label": "yes", "taxa": [5, 11, 4, 15, 21, 23, 10, 16, 46, 48, 37, 50]},
            "responses": [
                "Yes. The convergent taxa are: 5, 11, 4, 15, 21, 23, 10, 16, 46, 48, 37, 50.",  # Correct (exact match)
                "Yes. Taxa 5, 11, 4, 15, 21, 23 are involved.",  # Partial match
                "Yes. I found homoplasy involving taxa 1, 2, 3, 4, 5.",  # Partial match (some overlap)
            ]
        },
    ]
    
    print("=" * 80)
    print("Testing REL-B1 (Homoplasy Detection) Evaluation")
    print("=" * 80)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Question: {example['question'][:100]}...")
        print(f"Gold Answer: label={example['answer']['label']}, taxa={example['answer']['taxa']}")
        
        for j, response in enumerate(example['responses'], 1):
            result = evaluate_response(example['question'], example['answer'], response, task="REL-B1")
            print(f"\n  Response {j}: {response[:80]}...")
            print(f"    Predicted label: {result.get('pred_label')}")
            print(f"    Correct: {result.get('correct')}")
            print(f"    Predicted taxa: {result.get('pred_taxa', [])}")
            print(f"    Precision: {result.get('precision', -1):.3f}, Recall: {result.get('recall', -1):.3f}, F1: {result.get('f1', -1):.3f}")


if __name__ == "__main__":
    test_b1_examples()
    print("\n" + "=" * 80)
    print("All biology evaluation tests completed!")
    print("=" * 80)
