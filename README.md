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
- Processing time: <2 ms per image (average)
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
