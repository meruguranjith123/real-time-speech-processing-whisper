#!/usr/bin/env python3
"""
Generate system architecture figure as PNG image
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
color_client = '#4A90E2'  # Blue
color_server = '#50C878'  # Green
color_process = '#FFA500'  # Orange
color_output = '#FF6B6B'  # Red

# Define box style
box_style = dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black', linewidth=2)

# Client Side
client_y = 8
ax.text(2, client_y, 'Client Side\n(Web Browser)', ha='center', va='center', 
        fontsize=14, weight='bold', bbox=dict(boxstyle="round,pad=0.8", facecolor=color_client, edgecolor='black', linewidth=2))

# Client modules
client_modules = [
    ('Audio Capture\nWeb Audio API', 2, 6.5),
    ('PCM Processing\nChunk Buffering', 2, 5),
    ('Base64 Encoding', 2, 3.5),
    ('WebSocket\nTransmission', 2, 2)
]

for text, x, y in client_modules:
    ax.text(x, y, text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', edgecolor='black', linewidth=1.5))

# Server Side
server_y = 8
ax.text(8, server_y, 'Server Side\n(Flask + Whisper)', ha='center', va='center',
        fontsize=14, weight='bold', bbox=dict(boxstyle="round,pad=0.8", facecolor=color_server, edgecolor='black', linewidth=2))

# Server modules
server_modules = [
    ('Audio Reception\nBase64 Decode', 8, 6.5),
    ('Format Conversion\n(WebM→WAV)', 8, 5),
    ('Whisper ASR\nTranscription', 8, 3.5),
    ('Stutter Detection\nPattern Matching', 8, 2)
]

for text, x, y in server_modules:
    ax.text(x, y, text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', edgecolor='black', linewidth=1.5))

# Processing modules
processing_modules = [
    ('Stutter Cleaning\nRule-based', 12, 5),
    ('Next Sentence\nPrediction', 12, 3.5)
]

for text, x, y in processing_modules:
    ax.text(x, y, text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color_process, edgecolor='black', linewidth=1.5))

# Output
ax.text(8, 0.5, 'Output Display\n(Web Interface)', ha='center', va='center',
        fontsize=14, weight='bold', bbox=dict(boxstyle="round,pad=0.8", facecolor=color_output, edgecolor='black', linewidth=2))

# Output modules
output_modules = [
    ('Raw Transcription', 5, -0.5),
    ('Cleaned Text', 8, -0.5),
    ('Stutter Patterns', 11, -0.5),
    ('Predictions', 14, -0.5)
]

for text, x, y in output_modules:
    ax.text(x, y, text, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcoral', edgecolor='black', linewidth=1.5))

# Arrows - Client flow
for i in range(len(client_modules)-1):
    ax.annotate('', xy=(client_modules[i+1][1], client_modules[i+1][2]+0.3),
                xytext=(client_modules[i][1], client_modules[i][2]-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

# Arrow from client to server
ax.annotate('', xy=(7, 2), xytext=(3, 2),
            arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
ax.text(5, 2.3, 'Audio Data', ha='center', fontsize=10, weight='bold', color='purple')

# Arrows - Server flow
for i in range(len(server_modules)-1):
    ax.annotate('', xy=(server_modules[i+1][1], server_modules[i+1][2]+0.3),
                xytext=(server_modules[i][1], server_modules[i][2]-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))

# Arrow from server to processing
ax.annotate('', xy=(11.5, 3.5), xytext=(8.5, 3.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
ax.annotate('', xy=(11.5, 5), xytext=(8.5, 5),
            arrowprops=dict(arrowstyle='->', lw=2, color='orange'))

# Arrows to output
output_y = 0.5
for x in [5, 8, 11, 14]:
    ax.annotate('', xy=(x, output_y-0.3), xytext=(x, 1.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# Title
ax.text(8, 9.5, 'Real-Time Assistive Speech Captioning System Architecture', 
        ha='center', va='center', fontsize=16, weight='bold')

# Stage labels
stages = [
    ('Stage 1:\nCapture', 2, 7.5),
    ('Stage 2:\nTranscription', 8, 7.5),
    ('Stage 3:\nCleaning', 12, 4.5),
    ('Stage 4:\nDisplay', 8, 1.5)
]

for text, x, y in stages:
    ax.text(x, y, text, ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', edgecolor='black', linewidth=1, alpha=0.7))

plt.tight_layout()
plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('system_architecture.pdf', bbox_inches='tight', facecolor='white')
print("System architecture figure saved as 'system_architecture.png' and 'system_architecture.pdf'")
