"""SLM Integration Test"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from slm.engine import SLMEngine

def main():
    e = SLMEngine()

    print("=" * 60)
    print("  SLM INTEGRATION TEST")
    print("=" * 60)

    # Test 1: Train
    print("\n[TEST 1] Training on seed corpus...")
    t = e.train_file("corpus/seed.txt")
    print(f"  Trained on {t} tokens")
    print(f"  Vocab size: {e.vocab_size()}")
    assert t > 0, "Training should process tokens"
    assert e.vocab_size() > 0, "Should have vocabulary"
    print("  PASS")

    # Test 2: Generate
    print("\n[TEST 2] Generating text...")
    prompts = ["the world is", "language is the", "science has"]
    for prompt in prompts:
        result = e.generate(prompt, 25, 0.7)
        print(f"  Prompt: '{prompt}'")
        print(f"  Output: {result}")
        assert len(result) > len(prompt), "Generated text should be longer than prompt"
    print("  PASS")

    # Test 3: Predict
    print("\n[TEST 3] Predicting next words...")
    preds = e.predict_next("the", 5)
    print(f"  Predictions for 'the': {preds}")
    assert len(preds) > 0, "Should have predictions"
    print("  PASS")

    # Test 4: Save
    print("\n[TEST 4] Saving model...")
    os.makedirs("models", exist_ok=True)
    success = e.save_model("models/test.slm")
    assert success, "Save should succeed"
    print("  Saved to models/test.slm")
    print("  PASS")

    # Test 5: Load
    print("\n[TEST 5] Loading model...")
    e2 = SLMEngine()
    assert e2.vocab_size() == 0, "New engine should be empty"
    success = e2.load_model("models/test.slm")
    assert success, "Load should succeed"
    print(f"  Loaded vocab size: {e2.vocab_size()}")
    assert e2.vocab_size() == e.vocab_size(), "Loaded model should match original"
    print("  PASS")

    # Test 6: Generate from loaded model
    print("\n[TEST 6] Generating from loaded model...")
    result = e2.generate("the world", 20, 0.7)
    print(f"  Output: {result}")
    assert len(result) > len("the world"), "Should generate text"
    print("  PASS")

    # Test 7: Train on inline text
    print("\n[TEST 7] Training on inline text...")
    before_vocab = e.vocab_size()
    e.train_text("quantum computing is the future of technology and innovation in our digital age")
    after_vocab = e.vocab_size()
    print(f"  Vocab before: {before_vocab}, after: {after_vocab}")
    assert after_vocab >= before_vocab, "Vocab should grow or stay same"
    print("  PASS")

    # Test 8: Reset
    print("\n[TEST 8] Resetting model...")
    e.reset()
    assert e.vocab_size() == 0, "Vocab should be 0 after reset"
    assert not e.is_trained, "Should not be trained after reset"
    print("  PASS")

    # Test 9: Top vocab
    print("\n[TEST 9] Top vocabulary...")
    e.train_file("corpus/seed.txt")
    top = e.top_vocab(10)
    print(f"  Top 10 words: {[(w, c) for w, c in top]}")
    assert len(top) == 10, "Should return 10 words"
    print("  PASS")

    print("\n" + "=" * 60)
    print("  ALL 9 TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
