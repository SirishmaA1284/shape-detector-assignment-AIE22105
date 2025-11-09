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

<img width="1003" height="575" alt="image" src="https://github.com/user-attachments/assets/dfc84d45-bc70-4db9-8002-34e32e029f53" />
<img width="880" height="441" alt="image" src="https://github.com/user-attachments/assets/fa52c52d-c69d-4b1d-8ca8-46d52d8fd897" />


###  Run Locally
```bash
npm install
npm run dev


