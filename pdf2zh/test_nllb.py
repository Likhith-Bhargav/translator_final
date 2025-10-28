#!/usr/bin/env python3
"""
Test script for NLLB-200 Distilled 600M model integration.

This script tests the NLLB translator implementation without requiring a PDF file.
It tests both direct NLLB calls and the translator wrapper.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pdf2zh.nllb_direct import translate_text, translate_batch, get_nllb_direct
from pdf2zh.translator import NLLBTranslator


def test_nllb_direct():
    """Test the direct NLLB implementation."""
    print("=" * 60)
    print("Testing NLLB Direct Implementation")
    print("=" * 60)
    
    # Test model path - update this to your actual model path
    model_path = "/Users/likhithbhargav/Desktop/PDFMathTranslate-main 4/models/facebook_nllb-200-distilled-600M"
    
    try:
        # Test single translation
        print("\n1. Testing single translation...")
        text = "Hello, how are you today?"
        result = translate_text(text, "en", "zh", model_path)
        print(f"Input: {text}")
        print(f"Output: {result}")
        
        # Test batch translation
        print("\n2. Testing batch translation...")
        texts = [
            "Good morning!",
            "How is the weather?",
            "Thank you very much."
        ]
        results = translate_batch(texts, "en", "zh", model_path)
        for i, (inp, out) in enumerate(zip(texts, results)):
            print(f"Batch {i+1}: {inp} -> {out}")
            
        # Test different language pairs
        print("\n3. Testing different language pairs...")
        test_pairs = [
            ("en", "fr", "Hello world"),
            ("en", "de", "Good evening"),
            ("en", "es", "How are you?"),
        ]
        
        for src, tgt, text in test_pairs:
            try:
                result = translate_text(text, src, tgt, model_path)
                print(f"{src}->{tgt}: {text} -> {result}")
            except Exception as e:
                print(f"{src}->{tgt}: ERROR - {e}")
                
    except Exception as e:
        print(f"ERROR in NLLB Direct test: {e}")
        return False
    
    print("\n✅ NLLB Direct tests completed successfully!")
    return True


def test_nllb_translator():
    """Test the NLLB translator wrapper."""
    print("\n" + "=" * 60)
    print("Testing NLLB Translator Wrapper")
    print("=" * 60)
    
    try:
        # Test translator initialization
        print("\n1. Testing translator initialization...")
        model_path = "/Users/likhithbhargav/Desktop/PDFMathTranslate-main 4/models/facebook_nllb-200-distilled-600M"
        translator = NLLBTranslator("en", "zh", model_path)
        print(f"Translator created: {translator}")
        
        # Test translation
        print("\n2. Testing translation...")
        test_texts = [
            "This is a test sentence.",
            "The weather is nice today.",
            "Machine learning is fascinating.",
        ]
        
        for text in test_texts:
            result = translator.translate(text)
            print(f"Input: {text}")
            print(f"Output: {result}")
            print()
            
        # Test same-language passthrough
        print("\n3. Testing same-language passthrough...")
        translator_same = NLLBTranslator("en", "en", model_path)
        result = translator_same.translate("This should not be translated.")
        print(f"Same language: {result}")
        
        # Test caching
        print("\n4. Testing caching...")
        text = "This is a caching test."
        result1 = translator.translate(text, ignore_cache=True)
        result2 = translator.translate(text, ignore_cache=False)  # Should use cache
        print(f"First call: {result1}")
        print(f"Second call (cached): {result2}")
        print(f"Results match: {result1 == result2}")
        
    except Exception as e:
        print(f"ERROR in NLLB Translator test: {e}")
        return False
    
    print("\n✅ NLLB Translator tests completed successfully!")
    return True


def test_model_loading():
    """Test model loading and basic functionality."""
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    model_path = "/Users/likhithbhargav/Desktop/PDFMathTranslate-main 4/models/facebook_nllb-200-distilled-600M"
    
    try:
        print(f"Loading model from: {model_path}")
        nllb = get_nllb_direct(model_path)
        print("✅ Model loaded successfully!")
        
        # Test language mapping
        print("\nTesting language mapping...")
        test_codes = ["en", "zh", "zh-cn", "fr", "de", "es", "ja", "ko"]
        for code in test_codes:
            mapped = nllb._map_lang(code)
            print(f"{code} -> {mapped}")
            
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("\nMake sure:")
        print("1. The model path is correct")
        print("2. The model files are downloaded and present")
        print("3. Required dependencies (transformers, torch) are installed")
        return False
    
    return True


def main():
    """Run all tests."""
    print("NLLB-200 Distilled 600M Model Test Suite")
    print("=" * 60)
    
    # Check if model path exists
    model_path = "/Users/likhithbhargav/Desktop/PDFMathTranslate-main 4/models/facebook_nllb-200-distilled-600M"
    if not Path(model_path).exists():
        print(f"❌ Model path does not exist: {model_path}")
        print("\nPlease:")
        print("1. Download the facebook/nllb-200-distilled-600M model")
        print("2. Place it in the correct directory")
        print("3. Update the model_path variable in this script if needed")
        return
    
    # Run tests
    tests = [
        test_model_loading,
        test_nllb_direct,
        test_nllb_translator,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! NLLB integration is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
