# How to Add System Architecture Image to Overleaf

## Step 1: Upload the Image File

1. Go to [Overleaf](https://www.overleaf.com)
2. Open your project (or create a new one)
3. Click the **"Upload"** button in the left sidebar (or use the menu)
4. Upload `system_architecture.png` to your project
5. Make sure it's in the same folder as your `main.tex` file

## Step 2: Verify the Image is Included

The `main.tex` file already includes the image with this code:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{system_architecture.png}
\caption{Complete system architecture diagram showing all components and data flow from audio capture to output display. Generated visualization of the real-time assistive speech captioning pipeline.}
\label{fig:system_diagram}
\end{figure}
```

## Step 3: Compile

1. Click the **"Recompile"** button in Overleaf
2. The image should appear in your PDF

## Troubleshooting

### Image not showing?
- Make sure `system_architecture.png` is uploaded to the project
- Check that the filename matches exactly: `system_architecture.png`
- Verify the file is in the same directory as `main.tex`

### Image too large/small?
You can adjust the size by changing `width=\textwidth` to:
- `width=0.8\textwidth` (80% of page width)
- `width=0.6\textwidth` (60% of page width)
- `width=15cm` (fixed 15cm width)

### Need different format?
The script also generated `system_architecture.pdf` which you can use instead:
```latex
\includegraphics[width=\textwidth]{system_architecture.pdf}
```

## Alternative: Use the TikZ Diagrams

If you prefer vector graphics, the LaTeX file also includes TikZ diagrams that will render directly in Overleaf without needing to upload images. These are already in the document as Figure 1 and Figure 2.
