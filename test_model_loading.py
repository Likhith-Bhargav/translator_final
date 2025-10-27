#!/usr/bin/env python3
"""
Test script to verify Facebook mBART model is working correctly
"""

import sys
import os
from pathlib import Path

def test_model_loading():
    """Test if the mBART model loads successfully"""
    print("🔍 Testing mBART model loading...")

    try:
        # Add the project root to Python path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))

        # Import the M2M100Direct class
        from pdf2zh.m2m100_direct import M2M100Direct

        # Try to load the model (using default path or user can modify)
        model_path = "/Users/likhithbhargav/Desktop/PDFMathTranslate-main 4/models/mbart-large-50-many-to-many-mmt"

        print(f"📁 Loading model from: {model_path}")

        # Check if model directory exists
        if not Path(model_path).exists():
            print(f"❌ Model directory not found: {model_path}")
            print("💡 Make sure the model files are in the correct directory")
            return False

        # Load the model
        translator = M2M100Direct(model_path)
        print("✅ Model loaded successfully!")

        # Test supported languages
        languages = translator.get_supported_languages()
        print(f"🌍 Model supports {len(languages)} languages:")
        print(f"   Available: {', '.join(list(languages.keys())[:10])}...")  # Show first 10

        # Test a simple translation
        print("\n🧪 Testing translation...")
        test_text = "Hello, how are you today?"
        try:
            result = translator.translate_text(test_text, "en", "zh")
            print(f"✅ Translation test successful!")
            print(f"   English: {test_text}")
            print(f"   Chinese: {result}")
        except Exception as e:
            print(f"❌ Translation test failed: {e}")
            return False

        print("\n🎉 All tests passed! Your mBART model is working correctly.")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure transformers and torch are installed:")
        print("   pip install transformers torch sentencepiece")
        return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check if all model files are present in the model directory")
        print("2. Verify the model path is correct")
        print("3. Ensure transformers, torch, and sentencepiece are installed")
        print("4. Check if there are any permission issues with model files")
        return False

def test_batch_translation():
    """Test batch translation functionality"""
    print("\n🔄 Testing batch translation...")

    try:
        from pdf2zh.m2m100_direct import translate_batch

        test_texts = [
            "Hello world",
            "How are you?",
            "This is a test"
        ]

        results = translate_batch(test_texts, "en", "zh")
        print("✅ Batch translation successful!")
        for i, (original, translated) in enumerate(zip(test_texts, results)):
            print(f"   {i+1}. {original} → {translated}")

        return True
    except Exception as e:
        print(f"❌ Batch translation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 mBART Model Test Script")
    print("=" * 50)

    success = test_model_loading()

    if success:
        test_batch_translation()

    print("\n" + "=" * 50)
    if success:
        print("🎉 Model test completed successfully!")
        sys.exit(0)
    else:
        print("❌ Model test failed. Check the error messages above.")
        sys.exit(1)
