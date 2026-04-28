# Trait Extraction Methods: How the Algorithms Detect Cap Shape, Colour, and Texture

**Date:** 2026-04-28
**File:** `models/visual_trait_extractor.py`

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Colour Analysis](#2-colour-analysis)
3. [Shape Analysis](#3-shape-analysis)
4. [Texture Analysis](#4-texture-analysis)
5. [Brightness Analysis](#5-brightness-analysis)
6. [Species Scoring](#6-species-scoring)
7. [The Role of Masks](#7-the-role-of-masks)
8. [Summary: From Pixels to Traits](#8-summary-from-pixels-to-traits)

---

## 1. The Big Picture

Your trait extractor is a collection of classical computer vision algorithms — no neural networks, no training required. It takes a photo and extracts four types of visual information:

| Trait | What It Tells You | Example Use |
|-------|-------------------|-------------|
| **Colour** | Dominant and secondary colours | "Red cap with white spots = Fly Agaric" |
| **Shape** | Cap geometry | "Funnel-shaped = Chanterelle" |
| **Texture** | Surface smoothness / ridges | "Ridged underside = Chanterelle" |
| **Brightness** | Light level | "Dark photo = Black Trumpet" |

These traits are then fed into a rule-based scoring system that guesses which species the mushroom might be.

**Important:** There are two versions of every analysis:
- **Full-image** (`analyse_colours`, `analyse_shape`, etc.) — uses the entire photo
- **Masked** (`analyse_colours_masked`, `analyse_shape_masked`, etc.) — uses only the mushroom region from the segmentation mask

The masked versions are what your fine-tuned YOLO model enables.

---

## 2. Colour Analysis

### How Humans See Colour vs. How Computers See Colour

Humans describe colour with words like "red," "orange," or "brown." Computers see colour as three numbers per pixel:
- **RGB:** Red (0–255), Green (0–255), Blue (0–255)
- **HSV:** Hue (0–179), Saturation (0–255), Value/Brightness (0–255)

The trait extractor uses **HSV** because it separates "what colour is this?" (hue) from "how intense is it?" (saturation) and "how bright is it?" (value). This makes it easier to say "this is red" regardless of lighting.

### Step 1: Reduce the Image

The photo is resized to a tiny 128 x 128 pixels. This is fast and removes noise.

### Step 2: Find Dominant Colours with KMeans Clustering

**What is KMeans?** Imagine you have 16,384 pixels (128 x 128), each with its own colour. KMeans groups them into clusters — like sorting coloured marbles into a few buckets by similarity.

The algorithm:
1. Picks 5 random "centre" colours
2. Assigns every pixel to the nearest centre
3. Moves each centre to the average of its assigned pixels
4. Repeats steps 2–3 until stable

**Result:** 5 representative colours, sorted by how many pixels belong to each. The largest cluster = dominant colour, second largest = secondary colour.

### Step 3: Map HSV Numbers to Colour Names

Each cluster centre (H, S, V) is converted to a human-readable name using a set of rules:

| Colour | Hue Range | Saturation | Value |
|--------|-----------|------------|-------|
| Red | 0–10 or 160–179 | > 80 | > 60 |
| Orange | 10–25 | > 80 | > 80 |
| Yellow | 25–35 | > 70 | > 80 |
| Green | 70–85 | > 60 | > 50 |
| Brown | 10–25 | > 40 | > 30 (but not too bright) |
| White | any | < 25 | > 190 |
| Black | any | any | < 30 |

**Special case — Fly Agaric:** The code has a hardcoded override. If > 9% of pixels are red AND > 5% are white, it forces the dominant colour to "red" and secondary to "white." This catches the iconic red-and-white Fly Agaric even when forest background confuses KMeans.

### Step 4: Compute Colour Ratios

The algorithm also computes the fraction of pixels that match specific colour recipes:
- **red_ratio:** R > 150, R > 1.25x G, R > 1.25x B
- **orange_red_ratio:** R > 140, G between 50–130, B < 80 (captures Fly Agaric caps in natural light)
- **orange_yellow_ratio:** R > 140, G > 90, B < 140 (Chanterelle colour)
- **brown_ratio:** R > 80, G > 45, B < 90, R > G > B
- **white_ratio:** All channels > 185
- **dark_ratio:** All channels < 60

These ratios are used later for species scoring.

---

## 3. Shape Analysis

### The Goal

Classify the mushroom cap into one of these shapes:
- **convex** — dome-shaped, like a button
- **flat** — level top
- **funnel-shaped** — dips in the middle like a cone
- **bell-shaped** — tall and narrow
- **wavy** — irregular, wavy edges
- **irregular** — none of the above

### Step 1: Convert to Black and White

The image is converted to grayscale and blurred slightly. Then a technique called **Otsu's thresholding** automatically finds the best brightness cutoff to separate the mushroom from the background.

**Otsu's method:** Imagine a histogram of all pixel brightnesses. Otsu finds the single brightness value that splits the histogram into two groups (foreground/background) with the least overlap.

### Step 2: Find the Outline (Contour)

The black-and-white image is traced to find the outer edge of the white blob. This edge is called a **contour** — a list of (x, y) points around the object.

### Step 3: Compute Two Key Numbers

**Circularity:** How circle-like is the shape?
```
circularity = 4 * pi * area / (perimeter ^ 2)
```
- 1.0 = perfect circle
- 0.0 = very spiky or thin

**Aspect ratio:** How wide vs. tall?
```
aspect_ratio = width / height
```
- 1.0 = perfectly square
- 2.0 = twice as wide as tall
- 0.5 = twice as tall as wide

### Step 4: Classify by Rules

| Shape | Circularity | Aspect Ratio |
|-------|-------------|--------------|
| convex | > 0.80 | 0.8 – 1.3 |
| flat | < 0.60 | > 1.6 |
| bell-shaped | — | < 0.7 |
| wavy | < 0.45 | — |
| funnel-shaped | < 0.65 | 0.5 – 1.0 |
| convex (default) | everything else | everything else |

---

## 4. Texture Analysis

### The Goal

Determine if the mushroom surface is:
- **smooth** — few edges
- **fibrous** — moderate edges
- **scaly** — lots of edges

And detect **ridges** (parallel lines) which indicate chanterelle-like gills.

### Step 1: Edge Detection (Canny Algorithm)

**What is Canny edge detection?** Imagine tracing all the sharp boundaries in a photo — where colour changes abruptly. Canny finds these boundaries and draws them as white lines on a black background.

The algorithm looks at every pixel and asks: "Is this pixel on a boundary between light and dark?" It uses two thresholds:
- Pixels above the high threshold = definitely an edge
- Pixels between thresholds = edge only if connected to a definite edge
- Pixels below the low threshold = not an edge

### Step 2: Measure Edge Density

```
edge_density = (number of edge pixels) / (total pixels)
```

| Edge Density | Texture |
|--------------|---------|
| < 0.05 | smooth |
| 0.05 – 0.15 | fibrous |
| > 0.15 | scaly |

### Step 3: Detect Ridges (Hough Line Transform)

**What is Hough Transform?** Imagine every edge pixel "votes" for all possible lines that could pass through it. If many pixels vote for the same line, that line is real.

The algorithm:
1. Takes the edge image from Canny
2. Every edge point votes in "line space" (angle vs. distance from origin)
3. Peaks in the vote histogram = detected lines

**Result:** If more than 15 lines are found, `has_ridges = True`.

**Why this matters:** Chanterelles have distinctive forked gills (ridges) on the underside. This is a strong identification cue.

---

## 5. Brightness Analysis

The simplest analysis of all:

1. Convert image to HSV
2. Take the average of the V (value/brightness) channel
3. Classify:

| Mean V | Brightness |
|--------|------------|
| < 70 | dark |
| 70 – 160 | medium |
| > 160 | bright |

This helps distinguish mushrooms photographed in deep forest shade (Black Trumpet) from those in bright sunlight.

---

## 6. Species Scoring

### How It Works

After extracting colour, shape, and texture, the system assigns a score to each of the 12 mushroom species. It is like a detective building a case:

- "The cap is red and white" → points for Fly Agaric
- "The cap is funnel-shaped and ridged" → points for Chanterelle
- "The cap is black and the photo is dark" → points for Black Trumpet

### Scoring Rules (Simplified)

**Colour signals:**
- High red + white → Fly Agaric (+5.0 for red, +1.4 for white)
- High orange-yellow → Chanterelle (+4.0)
- High brown → Porcini (+4.2) or Other Boletus (+3.3)
- High dark → Black Trumpet (+3.5)
- Dominant white → Amanita virosa (+2.5)

**Shape signals:**
- Funnel-shaped → Chanterelle (+0.5), False Chanterelle (+0.3), Black Trumpet (+0.2)
- Convex/flat → Porcini (+0.25), Other Boletus (+0.2)
- Bell-shaped → Fly Agaric (+0.35), Amanita virosa (+0.25)
- Wavy → Chanterelle (+0.4)

**Texture signals:**
- Has ridges → Chanterelle (+0.5), False Chanterelle (+0.25)
- Scaly → Fly Agaric (+0.3)
- Smooth → Chanterelle (+0.1), Porcini (+0.1)

### Normalisation

All scores are converted to probabilities that sum to 1.0. The species with the highest score is the top guess.

**Example:** A bright orange, funnel-shaped mushroom with ridges would score heavily for Chanterelle and lightly for everything else.

---

## 7. The Role of Masks

### Without Masks (Full-Image Analysis)

Before your fine-tuned YOLO model, the trait extractor analysed the **entire photo**. This caused problems:
- A red mushroom on green grass might be classified as "green" because grass covers 70% of the photo
- A brown mushroom held by a hand might be classified by the hand's skin tone
- Background leaves and sky pollute every measurement

### With Masks (Mushroom-Only Analysis)

Your fine-tuned YOLO model generates a mask that says "these pixels are the mushroom, everything else is background." The masked analysis functions work exactly like the full-image versions, but they **ignore all non-mushroom pixels**.

**Masked colour analysis:**
- Only counts pixels where the mask is white
- KMeans runs only on mushroom pixels
- Colour ratios compute only inside the mask

**Masked shape analysis:**
- Uses the mask contour directly (the outline drawn by YOLO)
- No thresholding needed — the mask IS the shape
- More accurate because it does not confuse background clutter with mushroom edges

**Masked texture analysis:**
- Canny edges outside the mask are set to zero
- Only edges inside the mushroom are counted
- Ridge detection only looks at the mushroom underside

**Masked brightness analysis:**
- Only averages brightness inside the mask
- A bright sky behind a dark mushroom no longer skews the result

### Quality Gates

The system only uses masked traits if the mask passes quality checks:
- Confidence >= 0.50
- Area between 2% and 85% of the image
- Not too fragmented (<= 3 pieces)
- Not too many holes (<= 10%)
- Not too irregular boundary (<= 0.35)

If the mask fails these checks, the system falls back to full-image analysis.

---

## 8. Summary: From Pixels to Traits

Here is the complete pipeline for a single photo:

```
USER PHOTO
    |
    v
[YOLO SEGMENTATION]  <-- your fine-tuned model
    |
    v
MASK (white = mushroom, black = background)
    |
    v
+-------------------------------------------+
|  COLOUR: KMeans on masked HSV pixels      |
|  → dominant: red, secondary: white        |
+-------------------------------------------+
|  SHAPE: Contour of mask                   |
|  → convex, circularity: 0.85              |
+-------------------------------------------+
|  TEXTURE: Canny edges inside mask         |
|  → scaly, edge_density: 0.12              |
+-------------------------------------------+
|  BRIGHTNESS: Mean V inside mask           |
|  → medium                                 |
+-------------------------------------------+
    |
    v
[SPECIES SCORING]
Red + white + convex + scaly = Fly Agaric (85%)
    |
    v
TRAIT DICTIONARY → sent to LLM classifier
```

### Why This Approach Works

- **No training required** for trait extraction — it is all classical computer vision
- **Interpretable** — you can see exactly why a trait was assigned
- **Fast** — all operations are simple pixel math, no neural network inference
- **Mask-aware** — your YOLO fine-tuning directly improves accuracy by isolating the mushroom

### Limitations

- **Colour naming is approximate** — "brown" vs. "tan" can be ambiguous
- **Shape depends on photo angle** — a convex cap photographed from above looks flat
- **Texture ignores scale** — a tiny scaly texture and a large one produce similar edge density
- **Species scoring is rule-based** — it cannot learn new patterns, only what was explicitly coded

These limitations are why the system also includes a CNN classifier (`cnn_classifier.py`) as a parallel prediction path. The trait extractor provides interpretable features; the CNN provides learned pattern recognition.
