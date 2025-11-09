import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

// ---- Image helpers (pure functions) ----

// Rec.601 luma: fast + good enough for edges later
function toGrayscale(
  rgba: Uint8ClampedArray,
  W: number,
  H: number
): Uint8ClampedArray {
  const gray = new Uint8ClampedArray(W * H);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j++) {
    // 0.299R + 0.587G + 0.114B
    gray[j] =
      (0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]) | 0;
  }
  return gray;
}

function drawGrayscaleOnCanvas(
  ctx: CanvasRenderingContext2D,
  gray: Uint8ClampedArray,
  W: number,
  H: number
) {
  const out = ctx.createImageData(W, H);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    const v = gray[i];
    out.data[p] = v;
    out.data[p + 1] = v;
    out.data[p + 2] = v;
    out.data[p + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

// ---- Edge detection helpers ----

// Sobel gradient magnitude (Uint16). Returns { mag, maxMag }.
function sobel(gray: Uint8ClampedArray, W: number, H: number) {
  const mag = new Uint16Array(W * H);
  let maxMag = 0;

  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;

      const a = gray[i - W - 1], b = gray[i - W], c = gray[i - W + 1];
      const d = gray[i - 1],     e = gray[i + 1];
      const f = gray[i + W - 1], g = gray[i + W], h = gray[i + W + 1];

      const gx = (c + 2 * e + h) - (a + 2 * d + f);
      const gy = (f + 2 * g + h) - (a + 2 * b + c);

      const m = Math.sqrt(gx * gx + gy * gy) | 0;
      mag[i] = m;
      if (m > maxMag) maxMag = m;
    }
  }
  return { mag, maxMag };
}

// Simple fixed threshold → binary edge map (0 or 255)
function thresholdEdges(mag: Uint16Array, T: number) {
  const bin = new Uint8ClampedArray(mag.length);
  for (let i = 0; i < mag.length; i++) bin[i] = mag[i] >= T ? 255 : 0;
  return bin;
}

// Otsu automatic threshold on Sobel magnitudes (0..maxMag)
function otsuThreshold(mag: Uint16Array, maxMag: number): number {
  const bins = 256;
  const hist = new Uint32Array(bins);
  const scale = (bins - 1) / (maxMag || 1);
  for (let i = 0; i < mag.length; i++) hist[(mag[i] * scale) | 0]++;

  let sum = 0, total = mag.length;
  for (let i = 0; i < bins; i++) sum += i * hist[i];

  let sumB = 0, wB = 0, maxBetween = 0, threshold = 0;
  for (let t = 0; t < bins; t++) {
    wB += hist[t]; if (wB === 0) continue;
    const wF = total - wB; if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxBetween) { maxBetween = between; threshold = t; }
  }
  return Math.round(threshold / scale);
}

// visualize a binary map on the canvas (for debugging)
function drawBinaryOnCanvas(
  ctx: CanvasRenderingContext2D,
  bin: Uint8ClampedArray,
  W: number,
  H: number
) {
  const out = ctx.createImageData(W, H);
  for (let i = 0, p = 0; i < bin.length; i++, p += 4) {
    const v = bin[i];
    out.data[p] = out.data[p + 1] = out.data[p + 2] = v;
    out.data[p + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

// ---- Connected Components (4-neighbour) on a binary map (0 or 255) ----
function connectedComponents(
  bin: Uint8ClampedArray,
  W: number,
  H: number
) {
  const labels = new Int32Array(W * H).fill(-1);
  const comps: {
    id: number;
    area: number;
    seedX: number;
    seedY: number;
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
  }[] = [];

  let id = 0;
  const qx = new Int32Array(W * H);
  const qy = new Int32Array(W * H);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = y * W + x;
      if (bin[idx] === 0 || labels[idx] !== -1) continue;

      let head = 0,
        tail = 0,
        area = 0,
        sx = x,
        sy = y,
        minX = x,
        minY = y,
        maxX = x,
        maxY = y;

      qx[tail] = x;
      qy[tail++] = y;
      labels[idx] = id;

      while (head < tail) {
        const cx = qx[head],
          cy = qy[head++];
        const ci = cy * W + cx;
        area++;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;
        sx = cx;
        sy = cy;

        // 4-neighbours
        if (cx > 0) push(cx - 1, cy);
        if (cx < W - 1) push(cx + 1, cy);
        if (cy > 0) push(cx, cy - 1);
        if (cy < H - 1) push(cx, cy + 1);

        function push(nx: number, ny: number) {
          const ni = ny * W + nx;
          if (bin[ni] && labels[ni] === -1) {
            labels[ni] = id;
            qx[tail] = nx;
            qy[tail++] = ny;
          }
        }
      }

      comps.push({ id, area, seedX: sx, seedY: sy, minX, minY, maxX, maxY });
      id++;
    }
  }
  return { labels, comps };
}

// ---- Moore-neighbour contour tracing, starting from a seed pixel of a label ----
type Pt = { x: number; y: number };

function traceContour(
  labels: Int32Array,
  W: number,
  H: number,
  sx: number,
  sy: number
): Pt[] {
  const label = labels[sy * W + sx];
  const pts: Pt[] = [];
  const dirs = [
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
  ];

  let cx = sx,
    cy = sy,
    prevDir = 0;
  let first = true;

  do {
    pts.push({ x: cx, y: cy });
    // Start search from the "left" of previous direction
    let dir = (prevDir + 6) % 8;
    let found = false;
    for (let k = 0; k < 8; k++) {
      const [dx, dy] = dirs[(dir + k) % 8];
      const nx = cx + dx,
        ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      if (labels[ny * W + nx] === label) {
        cx = nx;
        cy = ny;
        prevDir = (dir + k) % 8;
        found = true;
        break;
      }
    }
    if (!found) break;
    if (first) {
      first = false;
    } else if (cx === sx && cy === sy) {
      break; // closed loop
    }
  } while (pts.length < 20000);

  return pts;
}

// ---- Simplify polyline with Ramer–Douglas–Peucker ----
function rdp(points: Pt[], eps: number) {
  if (points.length < 3) return points.slice();
  const keep = new Uint8Array(points.length);
  keep[0] = keep[points.length - 1] = 1;

  function dist(i: number, a: Pt, b: Pt) {
    const p = points[i];
    const A = b.y - a.y,
      B = a.x - b.x,
      C = b.x * a.y - a.x * b.y;
    return Math.abs(A * p.x + B * p.y + C) / Math.sqrt(A * A + B * B);
  }

  const stack: Array<[number, number]> = [[0, points.length - 1]];
  while (stack.length) {
    const [s, e] = stack.pop()!;
    let maxD = 0,
      idx = -1;
    for (let i = s + 1; i < e; i++) {
      const d = dist(i, points[s], points[e]);
      if (d > maxD) {
        maxD = d;
        idx = i;
      }
    }
    if (maxD > eps) {
      keep[idx] = 1;
      stack.push([s, idx], [idx, e]);
    }
  }

  const out: Pt[] = [];
  for (let i = 0; i < points.length; i++) if (keep[i]) out.push(points[i]);
  return out;
}

// ---- draw helpers for quick visual debugging ----
function drawBBoxes(
  ctx: CanvasRenderingContext2D,
  boxes: { x: number; y: number; w: number; h: number }[]
) {
  ctx.save();
  ctx.strokeStyle = "#00c853";
  ctx.lineWidth = 2;
  for (const b of boxes) ctx.strokeRect(b.x, b.y, b.w, b.h);
  ctx.restore();
}

function drawPolyline(
  ctx: CanvasRenderingContext2D,
  pts: Pt[],
  color = "#ff6d00"
) {
  if (pts.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(pts[0].x + 0.5, pts[0].y + 0.5);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x + 0.5, pts[i].y + 0.5);
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

// Shoelace area / perimeter / bbox / center
function shapeMetrics(poly: Pt[]) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  let area2 = 0; // 2*area (signed)
  let per = 0;

  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    if (a.x < minX) minX = a.x; if (a.x > maxX) maxX = a.x;
    if (a.y < minY) minY = a.y; if (a.y > maxY) maxY = a.y;
    area2 += a.x * b.y - b.x * a.y;
    per   += Math.hypot(b.x - a.x, b.y - a.y);
  }
  const area = Math.abs(area2) / 2;
  const bbox = { x: minX | 0, y: minY | 0, width: (maxX - minX) | 0, height: (maxY - minY) | 0 };
  const center = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
  const aspect = bbox.height ? bbox.width / bbox.height : 1;
  return { area, perimeter: per, bbox, center, aspect };
}

// Internal angles (0..180)
function internalAngles(poly: Pt[]) {
  const n = poly.length, out: number[] = [];
  for (let i = 0; i < n; i++) {
    const p0 = poly[(i - 1 + n) % n], p1 = poly[i], p2 = poly[(i + 1) % n];
    const a1 = Math.atan2(p0.y - p1.y, p0.x - p1.x);
    const a2 = Math.atan2(p2.y - p1.y, p2.x - p1.x);
    let ang = (a2 - a1) * 180 / Math.PI;
    if (ang < 0) ang += 360;
    if (ang > 180) ang = 360 - ang;
    out.push(ang);
  }
  return out;
}

// Count "cornery" vertices (angle far from 180)
function estimateCornerCount(poly: Pt[]) {
  const angs = internalAngles(poly);
  return angs.filter(a => Math.abs(180 - a) > 18).length;
}

// Reflex (concave) vertex count: internal angle > 180 means reflex
// Signed area (>0 => CCW)
// > 0 means CCW, < 0 means CW
function signedArea(poly: { x: number; y: number }[]) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    a += p.x * q.y - q.x * p.y;
  }
  return 0.5 * a;
}

// Correctly counts reflex (concave) vertices for BOTH orientations
function reflexCount(poly: { x: number; y: number }[]) {
  const ccw = signedArea(poly) > 0;
  const n = poly.length;
  let cnt = 0;
  for (let i = 0; i < n; i++) {
    const p0 = poly[(i - 1 + n) % n];
    const p1 = poly[i];
    const p2 = poly[(i + 1) % n];
    const v1x = p1.x - p0.x, v1y = p1.y - p0.y;   // p0 -> p1
    const v2x = p2.x - p1.x, v2y = p2.y - p1.y;   // p1 -> p2
    const cross = v1x * v2y - v1y * v2x;
    const isReflex = ccw ? (cross < 0) : (cross > 0);
    if (isReflex) cnt++;
  }
  return cnt;
}



// Circularity: 1 for perfect circle, smaller for polygons/noisy
function circularity(area: number, per: number) {
  return (4 * Math.PI * area) / ((per * per) || 1);
}

// Otsu on 8-bit grayscale (0..255)
function otsuGray(gray: Uint8ClampedArray) {
  const hist = new Uint32Array(256);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;
  const total = gray.length;

  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];

  let sumB = 0, wB = 0, maxBetween = 0, threshold = 0;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxBetween) { maxBetween = between; threshold = t; }
  }
  return threshold; // 0..255
}

// Build a binary foreground mask from grayscale (black shapes on white)
function maskFromGray(gray: Uint8ClampedArray, W: number, H: number, thr: number) {
  const mask = new Uint8ClampedArray(W * H);
  for (let i = 0; i < gray.length; i++) {
    // Foreground = darker than threshold (most test shapes are black/dark)
    mask[i] = gray[i] <= thr ? 255 : 0;
  }
  return mask;
}

// Filled fraction inside bbox (0..1). Real shapes are usually >= 0.55
function extent(area: number, bbox: {width:number;height:number}) {
  const boxA = Math.max(1, bbox.width * bbox.height);
  return area / boxA;
}

// Perimeter + adaptive RDP to stabilize corner counting
function perimeterOf(poly:{x:number;y:number}[]) {
  let per = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    per += Math.hypot(b.x - a.x, b.y - a.y);
  }
  return per;
}
function simplifyForCorners(poly:{x:number;y:number}[]) {
  const eps = Math.max(1, Math.min(3, 0.01 * perimeterOf(poly)));
  return rdp(poly, eps);
}

// Count strong corners (angles <= 135°)
function countCorners(poly:{x:number;y:number}[], angleThresh=135) {
  const angs = internalAngles(poly);
  return angs.filter(a => a <= angleThresh).length;
}

// Eccentricity of covariance ellipse (0 round .. 1 very elongated)
function eccentricity(poly:{x:number;y:number}[]) {
  let cx=0, cy=0; for (const p of poly){ cx+=p.x; cy+=p.y; }
  cx/=poly.length; cy/=poly.length;
  let sxx=0, sxy=0, syy=0;
  for (const p of poly){ const dx=p.x-cx, dy=p.y-cy; sxx+=dx*dx; sxy+=dx*dy; syy+=dy*dy; }
  sxx/=poly.length; sxy/=poly.length; syy/=poly.length;
  const T = sxx+syy, D = sxx*syy - sxy*sxy;
  const disc = Math.max(0, T*T - 4*D);
  const l1 = 0.5*(T + Math.sqrt(disc)), l2 = 0.5*(T - Math.sqrt(disc));
  if (l1 <= 0) return 0;
  return Math.sqrt(1 - (l2/l1));
}

// Roundness: radial coefficient of variation around center
function radialCV(poly:{x:number;y:number}[], c:{x:number;y:number}) {
  const d: number[] = [];
  for (const p of poly) d.push(Math.hypot(p.x - c.x, p.y - c.y));
  const mean = d.reduce((a,b)=>a+b,0)/Math.max(1,d.length);
  const sd = Math.sqrt(d.reduce((a,b)=>a+(b-mean)*(b-mean),0)/Math.max(1,d.length));
  return mean > 0 ? sd/mean : 1;
}

function countOn(bin: Uint8ClampedArray) {
  let c = 0; for (let i = 0; i < bin.length; i++) if (bin[i]) c++;
  return c;
}
function invertBinary(bin: Uint8ClampedArray) {
  for (let i = 0; i < bin.length; i++) bin[i] = bin[i] ? 0 : 255;
}

// Find corner indices with both angle and edge-length tests
function extractCorners(poly: Pt[], angleThresh = 150, minEdgeFrac = 0.03) {
  const n = poly.length;
  if (n < 3) return [] as number[];

  // perimeter for scale
  let per = 0;
  for (let i = 0; i < n; i++) {
    const a = poly[i], b = poly[(i + 1) % n];
    per += Math.hypot(b.x - a.x, b.y - a.y);
  }
  const minEdge = Math.max(3, minEdgeFrac * per);

  const angs = internalAngles(poly);
  const cornerIdx: number[] = [];
  for (let i = 0; i < n; i++) {
    const prev = poly[(i - 1 + n) % n], cur = poly[i], next = poly[(i + 1) % n];
    const e1 = Math.hypot(cur.x - prev.x, cur.y - prev.y);
    const e2 = Math.hypot(next.x - cur.x, next.y - cur.y);
    if (e1 < minEdge || e2 < minEdge) continue;   // ignore hairline steps
    if (angs[i] <= angleThresh) cornerIdx.push(i);
  }
  return cornerIdx;
}

function lengthsAndAnglesAtCorners(poly: Pt[], cornerIdx: number[]) {
  const n = poly.length;
  const pts = cornerIdx.map(i => poly[i]);
  const m = pts.length;
  if (m < 3) return { sideLens: [] as number[], cornerAngs: [] as number[] };

  const sideLens: number[] = [];
  for (let k = 0; k < m; k++) {
    const a = pts[k], b = pts[(k + 1) % m];
    sideLens.push(Math.hypot(b.x - a.x, b.y - a.y));
  }

  // reuse internalAngles; pick only at corner indices
  const allAngs = internalAngles(poly);
  const cornerAngs = cornerIdx.map(i => allAngs[i]);
  return { sideLens, cornerAngs };
}


export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   *
   * TODO: Implement shape detection algorithm here
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
  const startTime = performance.now();

  const W = imageData.width;
  const H = imageData.height;
  const rgba = imageData.data;

  //  grayscale
  const gray = toGrayscale(rgba, W, H);

  // SEGMENTATION (mask first; ensure foreground is the dark/smaller set) ===
const Tg = otsuGray(gray);                       // 0..255
const mask = maskFromGray(gray, W, H, Tg);       // dark = 255, light = 0
// If the selected "foreground" is actually the big white background, invert it.
if (countOn(mask) > (W * H) / 2) invertBinary(mask);

let { labels, comps } = connectedComponents(mask, W, H);

// Fallback to edges only if mask finds nothing
if (comps.length === 0) {
  const { mag, maxMag } = sobel(gray, W, H);
  const Tedge = Math.max(10, Math.min(120, otsuThreshold(mag, maxMag)));
  const edges = thresholdEdges(mag, Tedge);
  ({ labels, comps } = connectedComponents(edges, W, H));
}


  // trace contours and APPLY HARD PRE-FILTERS (drop lines/text/noise)
const MIN_AREA_PIX = Math.max(80, Math.floor(0.001 * W * H));
const contours: Pt[][] = [];

for (const b of comps) {
  const contour = traceContour(labels, W, H, b.seedX, b.seedY);
  if (contour.length < 6) continue;

  const simple = rdp(contour, 1.0);
  const poly   = simplifyForCorners(simple);   // keep real vertices

  const { area, bbox } = shapeMetrics(poly);
  const ext = extent(area, bbox);
  const ecc = eccentricity(poly);
  const minSide = Math.min(bbox.width, bbox.height);

  // Keep only obviously valid blobs, but be lenient
  if (area < MIN_AREA_PIX || minSide < 8) continue;             // tiny specks
  if (ecc > 0.998 && ext < 0.15) continue;                       // very thin lines
  if (ext < 0.10) continue;                                      // very sparse strokes

  contours.push(poly);                                           // push corner-preserved poly
}



  // const boxes = contours.map(p => {
  //   const m = shapeMetrics(p).bbox; return { x: m.x, y: m.y, w: m.width, h: m.height };
  // });
  // drawBBoxes(this.ctx, boxes);
  // for (const c of contours) drawPolyline(this.ctx, c, '#ff6d00');

  // classify & build final detections (skip if no strict match)
    const shapes: DetectedShape[] = [];
for (const poly of contours) {
  // Try your strict classifier first
  let det = classifyShape(poly);

  // Fallback: if classifier says null, still emit a shape using corners
  if (!det) {
    const { area, bbox, center } = shapeMetrics(poly);
    const k = countCorners(poly, 150);
    // naive mapping just so SOMETHING appears
    const type =
      (k <= 3) ? "triangle" :
      (k === 4) ? "rectangle" :
      (k === 5) ? "pentagon" :
      "star"; // concave cases will still look like 'star'

    det = { type: type as DetectedShape["type"], confidence: 0.6, bbox, center, area };
  }

  shapes.push({
    type: det.type,
    confidence: Math.max(0, Math.min(1, det.confidence)),
    boundingBox: det.bbox,
    center: det.center,
    area: det.area
  });
}




  const processingTime = performance.now() - startTime;
  return {
    shapes,
    processingTime,
    imageWidth: W,
    imageHeight: H,
  };
}


  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}


// STRICT classifier: returns null when nothing matches confidently
function classifyShape(poly: Pt[]): {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  bbox: { x: number; y: number; width: number; height: number };
  center: { x: number; y: number };
  area: number;
} | null {
  const { area, perimeter, bbox, center, aspect } = shapeMetrics(poly);
  const cornerIndices = extractCorners(poly, 150, 0.03);
  const corners = cornerIndices.length;

  const reflex  = reflexCount(poly);
  const convex  = reflex === 0;
  const circ    = circularity(area, perimeter);
  const rcv     = radialCV(poly, center);     // roundness (lower is rounder)
  const ext     = extent(area, bbox);         // fill fraction of bbox

  // Circle: very round, few corners, well-filled 
  if (circ >= 0.86 && rcv <= 0.08 && corners <= 2 && ext >= 0.60) {
    return { type: "circle", confidence: 0.90, bbox, center, area };
  }

// Precompute angle stats
const angs = internalAngles(poly);
const minAng = angs.length ? Math.min(...angs) : 180;
const rightish = angs.filter(a => Math.abs(a - 90) <= 15).length / Math.max(1, angs.length);

// Build unit edge vectors between detected corners (for parallel/orthogonal checks)
const pts = cornerIndices.map(i => poly[i]);
const m = pts.length;
const edges: Array<[number, number, number]> = []; // [ux, uy, len]
for (let i = 0; i < m; i++) {
  const a = pts[i], b = pts[(i + 1) % m];
  const dx = b.x - a.x, dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  edges.push([dx / len, dy / len, len]);
}
const dotAbs = (u: [number, number, number], v: [number, number, number]) =>
  Math.abs(u[0] * v[0] + u[1] * v[1]);

// Rectange
// Case A: 4 strong corners → opposite edges ‖ & adjacent 
if (convex && corners === 4 && edges.length === 4) {
  const par02 = dotAbs(edges[0], edges[2]) >= 0.97;
  const par13 = dotAbs(edges[1], edges[3]) >= 0.97;
  const ortho01 = dotAbs(edges[0], edges[1]) <= 0.20;
  const ortho12 = dotAbs(edges[1], edges[2]) <= 0.20;
  if (rightish >= 0.60 && par02 && par13 && ortho01 && ortho12) {
    const squareness = 1 - Math.min(Math.abs(1 - aspect), 1);
    return { type: "rectangle", confidence: 0.85 + 0.10 * squareness, bbox, center, area };
  }
}

// Case B: only 3 corners (one tiny edge missed) - look for L-shape + one parallel edge
if (convex && corners === 3 && edges.length === 3) {
  const orthoPairs = [
    [dotAbs(edges[0], edges[1]), 2],
    [dotAbs(edges[1], edges[2]), 0],
    [dotAbs(edges[2], edges[0]), 1],
  ]; // [|dot(adj)|, index of 'remaining' edge]
  for (const [adjDot, remIdx] of orthoPairs) {
    if (adjDot <= 0.20) {
      // remaining edge parallel to one of the two orthogonal edges?
      const rem = edges[remIdx];
      const parTo0 = dotAbs(rem, edges[0]) >= 0.97;
      const parTo1 = dotAbs(rem, edges[1]) >= 0.97;
      if (parTo0 || parTo1) {
        const squareness = 1 - Math.min(Math.abs(1 - aspect), 1);
        return { type: "rectangle", confidence: 0.78 + 0.12 * squareness, bbox, center, area };
      }
    }
  }
}

// Triangle
// Strict: exactly 3 corners, must have an acute angle, not mostly right-angled, and not too filled
if (convex && corners === 3) {
  if (minAng <= 75 && rightish < 0.45 && ext <= 0.60 && circ < 0.92) {
    return { type: "triangle", confidence: 0.92, bbox, center, area };
  }
}



  // Small-shape override to prevent tiny triangles from becoming pentagons 
const minSidePx = Math.min(bbox.width, bbox.height);
if (convex && corners >= 5 && minSidePx < 28) {
  const { sideLens } = lengthsAndAnglesAtCorners(poly, cornerIndices);
  const meanLen = sideLens.reduce((a, b) => a + b, 0) / Math.max(1, sideLens.length);
  const lenStd  = Math.sqrt(sideLens.reduce((a, b) => a + (b - meanLen) ** 2, 0) / Math.max(1, sideLens.length));
  const lenCV   = meanLen > 0 ? lenStd / meanLen : 1;

  // If two sides are much shorter (aliasing), it's really a triangle outline
  if (lenCV >= 0.45) {
    return { type: "triangle", confidence: 0.80, bbox, center, area };
  }
}

  // Pentagon (≥5 real corners, roughly regular angles) 
  if (convex && (corners === 5 || corners === 6 )) {
  const { sideLens, cornerAngs } = lengthsAndAnglesAtCorners(poly, cornerIndices);

  // angle stats
  const meanAng = cornerAngs.reduce((a, b) => a + b, 0) / Math.max(1, cornerAngs.length);
  const angVar = cornerAngs.reduce((a, b) => a + (b - meanAng) ** 2, 0) / Math.max(1, cornerAngs.length);
  const angStd = Math.sqrt(angVar);

  // side length coefficient of variation
  const meanLen = sideLens.reduce((a, b) => a + b, 0) / Math.max(1, sideLens.length);
  const lenStd = Math.sqrt(sideLens.reduce((a, b) => a + (b - meanLen) ** 2, 0) / Math.max(1, sideLens.length));
  const lenCV = meanLen > 0 ? lenStd / meanLen : 1;

  // pentagon-ish characteristics:
  // - mean internal angle near 108° (allow ~95..125)
  // - angles not wildly scattered (std <= 25°)
  // - side lengths not wildly uneven (CV <= 0.40)
  // - not too circular (avoid circle confusion)
  if (meanAng >= 95 && meanAng <= 125 && angStd <= 25 && lenCV <= 0.40 && circ < 0.88) {
    // confidence rises as angles cluster near 108° and sides even
    const angScore = 1 - Math.min(Math.abs(meanAng - 108) / 20, 1);   // 1 at 108°, 0 at ±20°
    const evenScore = 1 - Math.min(lenCV / 0.4, 1);
    const confidence = 0.70 + 0.25 * (0.6 * angScore + 0.4 * evenScore);

    return {
      type: "pentagon",
      confidence: Math.min(0.95, confidence),
      bbox,
      center,
      area
    };
  }
}

  // Star: concave with many corners 
  if (!convex && corners >= 8 && corners <= 14 && reflex >= 4) {
    return { type: "star", confidence: 0.85, bbox, center, area };
  }

  // Nothing matched confidently → skip this blob
  return null;
}



class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});
