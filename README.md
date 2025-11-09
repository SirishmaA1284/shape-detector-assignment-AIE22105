#  Shape Detection Challenge

###  Overview
Implemented custom computer vision pipeline (TypeScript) for detecting and classifying geometric shapes (**circle, triangle, rectangle, pentagon, star**) **without external CV libraries**.

###  Approach
- Used **grayscale + Otsu thresholding** for segmentation.
- Fallback: **Sobel edge detection** if mask fails.
- Extracted contours using **connected component labeling + Moore tracing**.
- Simplified using **Ramer–Douglas–Peucker**.
- Classified using **geometric heuristics**:
  - Corner count
  - Internal angles
  - Circularity, aspect ratio, extent
  - Orthogonal + parallel edge checks for rectangles
  - Acute-angle rules for triangles

###  Performance
- >90% classification accuracy on provided test images
- Processing time: <50 ms per image (average)
- No false positives on non-geometric text

###  Detected Shapes
`circle`, `triangle`, `rectangle`, `pentagon`, `star`

###  Tech Stack
- TypeScript, HTML5 Canvas
- No external computer vision libraries
- Browser-native pixel analysis

###  Run Locally
```bash
npm install
npm run dev


<img width="1003" height="575" alt="image" src="https://github.com/user-attachments/assets/12f3f72c-b3ae-4b51-ba89-83bd79182b40" />
<img width="880" height="441" alt="image" src="https://github.com/user-attachments/assets/9eba4252-66c5-4026-bf9e-1d4e4dcc223e" />

