#!/usr/bin/env python3
"""
Terminal demo showing stuttering detection and cleaning in real-time
Simulates speech input with various stuttering patterns
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

def simulate_typing(text, delay=0.05):
    """Simulate typing effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def demo_stuttering():
    """Demo various stuttering patterns"""
    
    print_section("🎤 Real-Time Stuttering Detection Demo", 36)
    print("This demo simulates speech with stuttering patterns")
    print("and shows how the system processes them in real-time.\n")
    
    # Initialize processor (we'll simulate transcriptions, so we don't need Whisper)
    # But we'll use the stuttering detection part
    processor = SpeechProcessor(model_size="tiny")  # Use tiny to load faster
    
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
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print_section(f"Test Case {i}: {test['name']}", 33)
        print(f"Description: {test['description']}\n")
        
        # Simulate real-time transcription
        print("\033[36m[Simulating Speech Input...]\033[0m")
        print("\033[90mRaw transcription: ", end='')
        simulate_typing(test['input'], delay=0.03)
        print("\033[0m")
        
        time.sleep(0.5)
        
        # Process the input
        print("\033[33m[Processing...]\033[0m")
        time.sleep(0.3)
        
        # Detect and clean stuttering
        cleaned_text, stutters = processor.detect_stuttering(test['input'])
        
        # Show results
        print_result("📝 Raw Text", test['input'], 31)
        print_result("✨ Cleaned Text", cleaned_text, 32)
        
        if stutters:
            print(f"\033[33m⚠️  Detected Stutters:\033[0m")
            for stutter in stutters:
                print(f"   • {stutter}")
        else:
            print("\033[32m✅ No stutters detected\033[0m")
        
        # Predict next sentences
        predictions = processor.predict_next_sentences(cleaned_text)
        print(f"\n\033[36m🔮 Predicted Next Sentences:\033[0m")
        for j, pred in enumerate(predictions, 1):
            print(f"   {j}. {pred}")
        
        time.sleep(1.5)
        
        if i < len(test_cases):
            print("\n\033[90m" + "-"*70 + "\033[0m")
            input("\nPress Enter to continue to next test case...")
    
    print_section("✅ Demo Complete!", 32)
    print("Summary:")
    print("• The system successfully detects word repetitions")
    print("• Removes partial word stutters (th-th-the → the)")
    print("• Filters out filler words (um, uh, er, etc.)")
    print("• Provides cleaned, readable text")
    print("• Generates context-aware next sentence predictions")

def interactive_demo():
    """Interactive demo where user can type stuttering text"""
    print_section("🎤 Interactive Stuttering Demo", 36)
    print("Type text with stuttering patterns (e.g., 'the the the main idea')")
    print("Type 'quit' or 'exit' to stop\n")
    
    processor = SpeechProcessor(model_size="tiny")
    
    while True:
        try:
            user_input = input("\033[36mEnter text with stuttering: \033[0m").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            print("\n\033[33m[Processing...]\033[0m")
            time.sleep(0.3)
            
            cleaned_text, stutters = processor.detect_stuttering(user_input)
            predictions = processor.predict_next_sentences(cleaned_text)
            
            print_result("📝 Raw Text", user_input, 31)
            print_result("✨ Cleaned Text", cleaned_text, 32)
            
            if stutters:
                print(f"\n\033[33m⚠️  Detected Stutters:\033[0m")
                for stutter in stutters:
                    print(f"   • {stutter}")
            else:
                print("\n\033[32m✅ No stutters detected\033[0m")
            
            print(f"\n\033[36m🔮 Predicted Next Sentences:\033[0m")
            for j, pred in enumerate(predictions, 1):
                print(f"   {j}. {pred}")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\033[31mError: {e}\033[0m")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stuttering Detection Demo')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    args = parser.parse_args()
    
    try:
        if args.interactive:
            interactive_demo()
        else:
            demo_stuttering()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\033[31mError: {e}\033[0m")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)





