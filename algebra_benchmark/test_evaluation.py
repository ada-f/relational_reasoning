"""
Test evaluation functions for algebra benchmarks using real examples from REL-A1.
"""

from algebra_benchmark.evaluation import evaluate_response


def test_algebra_examples():
    """Test REL-A1 (Raven's Progressive Matrix) evaluation with 3 examples."""
    examples = [
        {
            "question": "Complete the Raven's progressive matrix. Only return the missing panel index (1-8)!\nPanel 0:\n[639.4267984578837, 25.010755222666937, 275.02931836911927]\n[223.21073814882274, 736.4712141640124, 676.6994874229113]\n[892.1795677048455, 86.93883262941615, 421.9218196852704]\n\nPanel 1:\n[639.4267984578837, 25.010755222666937, 275.02931836911927]\n[223.21073814882274, 736.4712141640124, 676.6994874229113]\n[892.1795677048455, 86.93883262941615, 421.9218196852704]\n\n[... all panels are the same ...]\n\nAnswer set:\nAnswer 1: [639.4267984578837, 25.010755222666937, 275.02931836911927]\n[223.21073814882274, 736.4712141640124, 676.6994874229113]\n[892.1795677048455, 86.93883262941615, 421.9218196852704]\nAnswer 2: [93.69523986159246, ...]\n...",
            "answer": {"target": 0},
            "responses": [
                "Answer 1",  # Correct (1-based, maps to 0-based index 0)
                "1",  # Correct
                "Answer 2",  # Incorrect
            ]
        },
        {
            "question": "Complete the Raven's progressive matrix. Only return the missing panel index (1-8)!\nPanel 0:\n[717.9142707189759, 212.6265440541406, 499.23125949227733]\n[884.6831538867025, 642.8518702004242, 142.87159044206265]\n[139.6303195255063, 744.9889820916065, 538.9772896672904]\n\n[... all panels are the same ...]\n\nAnswer set:\nAnswer 1: [897.822883602477, ...]\n...\nAnswer 7: [717.9142707189759, 212.6265440541406, 499.23125949227733]\n[884.6831538867025, 642.8518702004242, 142.87159044206265]\n[139.6303195255063, 744.9889820916065, 538.9772896672904]\nAnswer 8: [760.6021652572316, ...]",
            "answer": {"target": 6},
            "responses": [
                "Answer 7",  # Correct (1-based, maps to 0-based index 6)
                "7",  # Correct
                "Answer 1",  # Incorrect
            ]
        },
        {
            "question": "Complete the Raven's progressive matrix. Only return the missing panel index (1-8)!\nPanel 0:\n[539.2960887794584, 729.9310690899762, 201.1510633896959]\n[311.71629130089497, 995.1493566608947, 649.8780576394535]\n[438.10008391450407, 517.5758410355907, 121.00419586826573]\n\n[... all panels are the same ...]\n\nAnswer set:\nAnswer 1: [64.02589632634492, ...]\n...\nAnswer 4: [539.2960887794584, 729.9310690899762, 201.1510633896959]\n[311.71629130089497, 995.1493566608947, 649.8780576394535]\n[438.10008391450407, 517.5758410355907, 121.00419586826573]\n...",
            "answer": {"target": 3},
            "responses": [
                "Answer 4",  # Correct (1-based, maps to 0-based index 3)
                "4",  # Correct
                "The answer is 4",  # Correct (extracted as 4)
            ]
        },
    ]
    
    print("=" * 80)
    print("Testing REL-A1 (Raven's Progressive Matrix) Evaluation")
    print("=" * 80)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Question: {example['question'][:100]}...")
        print(f"Gold Answer: target index {example['answer']['target']} (Answer {example['answer']['target'] + 1})")
        
        for j, response in enumerate(example['responses'], 1):
            result = evaluate_response(example['question'], example['answer'], response, task="REL-A1")
            print(f"\n  Response {j}: {response}")
            print(f"    Predicted index: {result.get('pred')} (Answer {result.get('pred') + 1 if result.get('pred') is not None else None})")
            print(f"    Correct: {result.get('correct')}")
            assert result.get('gold') == example['answer']['target'], f"Gold mismatch: {result.get('gold')} != {example['answer']['target']}"


if __name__ == "__main__":
    test_algebra_examples()
    print("\n" + "=" * 80)
    print("All algebra evaluation tests completed!")
    print("=" * 80)
