#!/usr/bin/env python3
"""
Simple test script to verify the Flask server can start
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing imports...")
    from flask import Flask
    print("✓ Flask imported")
    
    from flask_socketio import SocketIO
    print("✓ Flask-SocketIO imported")
    
    import numpy as np
    print("✓ NumPy imported")
    
    print("\nTesting app creation...")
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*")
    print("✓ Flask app created")
    
    @app.route('/test')
    def test():
        return "Server is working!"
    
    print("\n✓ All basic imports and setup successful!")
    print("\nTo start the server, run:")
    print("  python app.py")
    print("\nThen open: http://localhost:5000")
    
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    print("\nPlease install missing dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    sys.exit(1)





