# Test Data

`samples/test_manifest.csv` contains every frame from the final held-out test split:

- Buoy: 2 videos, 196 frames
- SMD: 2 videos, 598 frames
- TMD: 4 videos, 1402 frames
- Total: 8 videos, 2196 frames

The video files are not copied into this repository because the complete test videos are large and are better stored outside Git.

Expected layout after preparing data:

```text
samples/
  Buoy/<test video>.avi
  SMD/<test video>.mp4
  TMD/<test video>.avi
  annotations/<dataset>/<annotation sidecar>
  test_manifest.csv
  test_videos.csv
```

To copy or symlink videos from an external dataset folder:

```bash
python3 tools/prepare_test_data.py --source-root /path/to/HL --symlink
```

If checksums for local source videos are required, compute them after preparation with:

```bash
sha256sum samples/Buoy/* samples/SMD/* samples/TMD/*
```
