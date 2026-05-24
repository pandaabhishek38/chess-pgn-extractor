# Chess Video to PGN Pipeline

A computer vision pipeline that processes overhead chess game videos and outputs PGN (Portable Game Notation) files. The system detects move transitions using frame differencing, validates detected moves against legal chess positions, and exports results in standard PGN format.

---

## How to Run

### Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
opencv-python
numpy
chess
```

### Running the Pipeline

```bash
python main.py
```

You will be prompted to:

1. Select an existing video from the `videos/` folder
2. Or upload a new video by providing its full path

### Calibration

Before processing, each video requires a one-time manual calibration. If no calibration exists for a video, the calibration tool launches automatically.

**Click the four board corners in this order:**

1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

Calibration data is saved to `config.json` and reused on subsequent runs. Pre-calibrated configurations for all 5 sample videos are included in the repository.

### Output

- PGN files are saved to `outputs/`
- Debug board images are saved to `debug/`
- Sample board screenshots are in `screenshots/`

---

## Pipeline Architecture

```
Video File
    │
    ▼
Frame Stabilization       ← detect movement start/stop
    │
    ▼
Board Detection           ← perspective warp to 800x800
    │
    ▼
Square Mapping            ← split into 64 named squares
    │
    ▼
Occupancy Detection       ← pixel diff between frames
    │
    ▼
Move Inference            ← validate against legal moves
    │
    ▼
PGN Export
```

### 1. Frame Stabilization (`src/frame_sequence.py`)

Scans the video frame by frame computing pixel-level movement scores using `cv2.absdiff`. When movement exceeds a threshold (250,000 pixels), a move is considered in progress. When movement drops back below the threshold, the next stable frame is captured as a post-move snapshot.

A cooldown of 30 frames (~1 second at 30fps) prevents duplicate captures after each detected move.

**Key parameters:**

- `movement_threshold = 250000`
- `cooldown_frames = 30`

### 2. Board Detection (`src/board_detector.py`)

Uses perspective transform (`cv2.getPerspectiveTransform`) to warp the detected board region into a normalized 800×800 image. Corner points are provided via manual calibration and stored in `config.json`.

After warping, the board is rotated 90° clockwise to correct for sideways camera orientation (white pieces appear on the left in the source videos).

### 3. Square Mapping (`src/square_mapper.py`)

The 800×800 warped board is divided into a uniform 8×8 grid. Each cell is named using standard chess notation (a1–h8), with `a1` at bottom-left and `h8` at top-right.

Each square is 100×100 pixels. Square names are assigned as:

```python
square_name = f"{FILES[col]}{8 - row}"
```

### 4. Occupancy Detection (`src/occupancy_detector.py`)

For each stable frame, all 64 squares are compared against the previous accepted board state using:

1. Grayscale conversion
2. 15% center crop per square (reduces edge bleed between adjacent squares)
3. `cv2.absdiff` between before/after
4. Gaussian blur (5×5) to reduce noise
5. Binary threshold at pixel value 25
6. Sum of non-zero pixels as the diff score

Squares exceeding a diff score of 50,000 are flagged as changed.

### 5. Move Inference (`src/inference.py`)

Changed squares are matched against all legal moves in the current board position using `python-chess`.

- If exactly 2 squares changed, those are used directly as move candidates
- Otherwise, the top-6 highest-diff squares are used as a fallback for noisy frames

A move is confirmed only when both `from_square` and `to_square` of a legal move appear in the candidate set. Confirmed moves are pushed to the board state via `board.push(move)`.

The reference frame (baseline for diff comparison) is updated **only on confirmed moves**. This prevents baseline drift from camera shake or partial transitions.

---

## Design Decisions & Tradeoffs

### Manual Calibration over Automatic Detection

Automatic corner detection (Hough line transforms, contour detection) was prototyped but proved unstable on handheld footage with varying lighting and camera angles. Manual calibration trades setup time for reliability. This matches real-world chess camera products (DGT, Chessnut) which also require board registration as a setup step.

### Update Baseline Only on Confirmed Moves

Early versions updated the reference frame on every stable frame. This caused baseline drift. After a few frames of camera shake, the diff was comparing against a shifted baseline, generating false positives that cascaded through the game state.

Updating only on confirmed moves trades recall (some real moves get missed) for precision (confirmed moves are more likely correct). This was a deliberate engineering decision to maintain board state integrity.

### Top-6 Fallback for Noisy Frames

When more than 2 squares change (camera shake, lighting variation), taking the top-6 highest-diff squares as candidates gives the inference engine enough signal to find the real move while tolerating noise. A stricter filter (top-2 only) rejected too many real moves on shakier videos.

### Legal Move Filtering as Validation Layer

Every detected move is validated against `python-chess` legal moves for the current position. This prevents physically impossible moves from corrupting board state and provides a meaningful confidence signal. If no legal move matches the detected squares, the frame is rejected rather than forcing a guess.

### Gap Filter (Abandoned)

A ratio-based gap filter was implemented to distinguish real moves (clear top-2 diff signal) from noise (flat diff distribution). While theoretically sound, threshold tuning was highly video-dependent. A threshold that worked for game_1 degraded game_2, and vice versa. Reverted to top-6 fallback which generalised better across all 5 videos.

### Additional experiments

Cooldown tuning, rotation direction, occupancy margin, and per-video recalibration were all tested iteratively. In each case the original parameters proved most stable across all 5 videos, confirming that global tuning outperforms per-video optimization when ground truth PGNs are unavailable for validation.

---

## Results

| Video      | Duration | Stable Frames | Confirmed Moves | Legal Failures |
| ---------- | -------- | ------------- | --------------- | -------------- |
| game_1.mp4 | 4:44     | 70            | 14              | 55             |
| game_2.mp4 | 10:12    | 183           | 12              | 170            |
| game_3.mp4 | 2:35     | 84            | 10              | 73             |
| game_4.mp4 | 4:02     | 153           | 6               | 146            |
| game_5.mp4 | 3:08     | 101           | 4               | 96             |

**Game 1** performed best due to minimal camera shake and slower move tempo. **Game 2**, despite being the longest video with the most stable frames captured, suffered from frequent rapid exchanges that caused baseline drift. **Games 4 and 5** were the most challenging due to heavy camera shake and fast back-and-forth play within short windows.

The high legal failure rate across all videos reflects the frame stabilization capturing noise frames (camera shake triggering false move detections) rather than actual move transitions.

**A note on PGN accuracy:**
The confirmed move counts above reflect move transitions detected, not PGN accuracy. Due to the systematic square naming offset from manual calibration, detected moves are valid legal chess moves but mapped to incorrect squares. This means the PGN files are structurally valid but semantically inaccurate. The pipeline correctly identifies that a move happened, but not which squares were involved. Fixing this likely requires more robust board localization and orientation handling. (see Future Improvements). PGN accuracy is currently the primary gap between this proof-of-concept and a production-ready system.

---

## Known Limitations

### Square Naming Offset

Manual calibration introduces a systematic one column offset in square naming. Detected moves are valid legal chess moves but mapped to incorrect squares. For example, detecting `f3` when the actual move was `e4`. This is the primary driver of PGN inaccuracy.

### Camera Shake

Handheld footage generates false stable frames. The pipeline captures a frame when movement stops, but movement sometimes stops mid-shake rather than post-move. This inflates stable frame counts and legal failure rates.

### Rapid Move Sequences

The 30-frame cooldown (~1 second) misses moves that occur in rapid back-and-forth exchanges. Games 4 and 5 contained sequences of 4-5 moves within 7-8 seconds, exceeding the capture rate of the stabilization system.

### Edge Square Instability

Pieces near board edges are occasionally clipped by calibration imprecision, causing persistent high diff scores on edge squares (particularly the a-file and h-file) that generate false move candidates.

---

## Future Improvements

The current pixel diff approach has a clear accuracy ceiling. The following improvements would achieve production-ready detection rates:

### 1. CNN-Based Piece Detector (Highest Impact)

Replace pixel diff with a piece-aware classifier. A lightweight CNN trained on chess piece images (readily available datasets exist) would distinguish between "piece moved" and "lighting changed", which is the core confusion in the current system. Expected improvement: detection rate from ~15% to 60%+.

### 2. Automatic Board Corner Detection

Replace manual calibration with automatic corner detection using Hough line transforms to detect the board grid, eliminating the systematic square naming offset entirely. This would fix PGN accuracy independently of detection rate.

### 3. Per-Video Orientation Configuration

Videos in this dataset were filmed from different camera angles with different board orientations. A per-video orientation flag (or automatic orientation detection) would handle this robustly.

### 4. Temporal Voting

Instead of confirming a move on a single frame, vote across 3-5 consecutive frames. A move confirmed by majority vote is significantly more reliable than single-frame detection.

### 5. Adaptive Thresholds

Movement threshold and cooldown could be calibrated per-video based on frame rate and detected game tempo, rather than fixed global values.

---

## Project Structure

```
chess-pgn-extractor/
├── main.py                  # entry point
├── config.json              # calibration corner data
├── requirements.txt
├── src/
│   ├── frame_sequence.py    # move transition detection
│   ├── board_detector.py    # perspective warp
│   ├── square_mapper.py     # 8x8 grid splitting
│   ├── occupancy_detector.py # pixel diff scoring
│   ├── inference.py         # legal move matching
│   └── calibrate.py         # interactive calibration tool
├── videos/                  # input videos
├── outputs/                 # generated PGN files
├── screenshots/             # sample board images
└── debug/                   # debug board images (gitignored)
```
