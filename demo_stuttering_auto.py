#!/usr/bin/env python3
"""
Auto-running terminal demo showing stuttering detection
Runs all test cases automatically without user input
"""

import time
import sys
from speech_processor import SpeechProcessor

def print_section(title, color_code=36):
    """Print a colored section header"""
    print(f"\n\033[{color_code}m{'='*70}\033[0m")
    print(f"\033[{color_code}m{title:^70}\033[0m")
    print(f"\033[{color_code}m{'='*70}\033[0m\n")

def print_result(label, text, color_code=32):
    """Print a result with color"""
    print(f"\033[{color_code}m{label}:\033[0m {text}")

def demo_stuttering_auto():
    """Auto-running demo with various stuttering patterns"""
    
    print_section("🎤 Real-Time Stuttering Detection Demo", 36)
    print("This demo shows how the system processes stuttering patterns.\n")
    
    # Initialize processor
    print("Loading Whisper model (tiny for faster loading)...")
    processor = SpeechProcessor(model_size="tiny")
    print("✓ Model loaded!\n")
    
    # Test cases with different stuttering patterns
    test_cases = [
        {
            "name": "Repetition Stuttering",
            "input": "the the the the main idea is that we need to understand this concept",
            "description": "Multiple word repetitions"
        },
        {
            "name": "Partial Word Repetition",
            "input": "th-th-the concept is very important for our understanding",
            "description": "Partial word stuttering (th-th-the)"
        },
        {
            "name": "Mixed Stuttering",
            "input": "i i i think that we we need to to to focus on this topic",
            "description": "Multiple words with repetitions"
        },
        {
            "name": "Filler Words",
            "input": "um uh the the main point is er that we should ah consider this",
            "description": "Filler words and repetitions"
        },
        {
            "name": "Complex Stuttering",
            "input": "this this is a very very important concept that that we need to understand understand",
            "description": "Multiple stuttering patterns"
        },
        {
            "name": "Natural Speech with Stutters",
            "input": "so so the the the question is um how do we approach this problem problem",
            "description": "Natural conversation with stutters"
        },
        {
            "name": "Real-world Example",
            "input": "i i want to to explain that the the the machine learning algorithm works by by processing data",
            "description": "Technical speech with stutters"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print_section(f"Test Case {i}: {test['name']}", 33)
        print(f"Description: {test['description']}\n")
        
        # Show input
        print("\033[36m[Input Text]\033[0m")
        print(f"  {test['input']}\n")
        
        time.sleep(0.3)
        
        # Process the input
        print("\033[33m[Processing...]\033[0m")
        time.sleep(0.2)
        
        # Detect and clean stuttering
        cleaned_text, stutters = processor.detect_stuttering(test['input'])
        
        # Show results
        print_result("📝 Raw Text", test['input'], 31)
        print_result("✨ Cleaned Text", cleaned_text, 32)
        
        if stutters:
            print(f"\n\033[33m⚠️  Detected Stutters:\033[0m")
            for stutter in stutters:
                print(f"   • {stutter}")
        else:
            print("\n\033[32m✅ No stutters detected\033[0m")
        
        # Predict next sentences
        predictions = processor.predict_next_sentences(cleaned_text)
        print(f"\n\033[36m🔮 Predicted Next Sentences:\033[0m")
        for j, pred in enumerate(predictions, 1):
            print(f"   {j}. {pred}")
        
        time.sleep(1)
        
        if i < len(test_cases):
            print("\n\033[90m" + "-"*70 + "\033[0m")
            time.sleep(0.5)
    
    print_section("✅ Demo Complete!", 32)
    print("\nSummary:")
    print("  ✓ Successfully detects word repetitions")
    print("  ✓ Removes partial word stutters (th-th-the → the)")
    print("  ✓ Filters out filler words (um, uh, er, etc.)")
    print("  ✓ Provides cleaned, readable text")
    print("  ✓ Generates context-aware next sentence predictions")
    print("\n\033[36mTry the interactive mode: python3 test_stuttering_demo.py --interactive\033[0m\n")

if __name__ == "__main__":
    try:
        demo_stuttering_auto()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\033[31mError: {e}\033[0m")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        import traceback
        traceback.print_exc()
        sys.exit(1)

