# photo_cluster_router.py
"""
Unsupervised face clustering & routing with constraints:
- Cluster folder (person_xxx) is created if the cluster has at least one photo
  with faces_in_image <= group_thr (default: 3 → allows singles, pairs, triples).
- If a cluster has faces only from group photos (faces_in_image > group_thr),
  then all such source images are routed once to sorted/__unused_group_only__/.
- Noise/outliers from DBSCAN go to sorted/__unknown__.

Also writes clustering_report.csv with assignments and reasons.

Usage:
    pip install insightface onnxruntime scikit-learn opencv-python pillow numpy tqdm
    python photo_cluster_router.py \
        --input-dir ./input_photos \
        --out-dir ./sorted \
        --eps-sim 0.55 \
        --min-samples 2 \
        --min-face 110 \
        --group-thr 3
"""

import argparse
import shutil
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import insightface
from insightface.app import FaceAnalysis

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_bgr(path: Path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im)[:, :, ::-1]
    return arr

def laplacian_variance(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def center_crop_square(bgr, size_min=256):
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s)//2
    x0 = (w - s)//2
    crop = bgr[y0:y0+s, x0:x0+s]
    if s < size_min:
        crop = cv2.resize(crop, (size_min, size_min), interpolation=cv2.INTER_CUBIC)
    return crop

@dataclass
class FaceRec:
    img_path: Path
    face_index: int
    faces_in_image: int
    bbox: np.ndarray
    det_score: float
    embedding: np.ndarray

class Embedder:
    def __init__(self, det_size=640, ctx_id=0):
        self.app = FaceAnalysis(allowed_modules=['detection','recognition'])
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def detect(self, bgr):
        return self.app.get(bgr)

    def embed_face(self, face):
        return np.array(face.embedding, dtype=np.float32)

def collect_faces(input_dir: Path, min_face=110, blur_thr=45.0, det_size=640, gpu_id=0):
    emb = Embedder(det_size=det_size, ctx_id=gpu_id)
    records = []
    files = [p for p in input_dir.rglob("*") if is_image(p)]
    for img_path in tqdm(files, desc="Detecting/embedding"):
        bgr = load_bgr(img_path)
        if min(bgr.shape[:2]) < min_face:
            bgr = center_crop_square(bgr, size_min=min_face)
        if laplacian_variance(bgr) < blur_thr:
            continue
        faces = emb.detect(bgr)
        n_faces = len(faces)
        for idx, f in enumerate(faces):
            rec = FaceRec(
                img_path=img_path,
                face_index=idx,
                faces_in_image=n_faces,
                bbox=np.array(f.bbox, dtype=np.float32),
                det_score=float(f.det_score),
                embedding=emb.embed_face(f)
            )
            records.append(rec)
    return records

def cluster_faces(records, eps_sim=0.55, min_samples=2):
    if not records:
        return np.array([])
    X = np.stack([r.embedding for r in records]).astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    eps = max(1e-6, 1.0 - float(eps_sim))
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
    labels = db.fit_predict(X)
    return labels







def route_by_clusters(records, labels, out_dir: Path, group_thr: int):
    """
    Routing policy (corrected):
    - Eligible cluster: exists a record with faces_in_image <= group_thr (solo/joint) -> create person_xxx.
    - Ineligible (group-only): all records have faces_in_image > group_thr -> no person_xxx.
    - For each image:
        * Copy to every eligible person's folder that appears on this image (solo/joint/group), once per (person,image).
        * If at least one ineligible cluster appears on the image, also copy the image once to __group_only__.
        * If the image has no clustered faces at all (only noise), copy it once to __unknown__.
    """
    ensure_dir(out_dir)
    unknown_dir = out_dir / "__unknown__"
    group_only_dir = out_dir / "__group_only__"
    ensure_dir(unknown_dir)
    ensure_dir(group_only_dir)

    # Build cluster -> indices and image -> indices
    cluster_indices = {}
    image_indices = {}
    for idx, (rec, lab) in enumerate(zip(records, labels)):
        cluster_indices.setdefault(lab, []).append(idx)
        image_indices.setdefault(rec.img_path, []).append(idx)

    # Determine eligibility (solo or joint present)
    eligible_clusters = set()
    for lab, idxs in cluster_indices.items():
        if lab == -1:
            continue
        if any(records[i].faces_in_image <= group_thr for i in idxs):
            eligible_clusters.add(lab)

    # Prepare person folders for eligible clusters
    eligible_sorted = sorted(eligible_clusters)
    cluster_to_name = {cl: f"person_{i:03d}" for i, cl in enumerate(eligible_sorted)}
    for name in cluster_to_name.values():
        ensure_dir(out_dir / name)

    # Dedup helpers
    copied_person_image = set()   # (person_name, src_path)
    staged_group_only = set()     # src_path
    staged_unknown = set()        # src_path

    log = []

    # Route per image
    for src, idxs in image_indices.items():
        labs_on_image = {labels[i] for i in idxs}
        eligible_on_image = {lab for lab in labs_on_image if lab in eligible_clusters}
        ineligible_on_image = {lab for lab in labs_on_image if (lab not in eligible_clusters and lab != -1)}
        has_any_cluster = len(labs_on_image - {-1}) > 0

        # Copy to eligible persons that actually appear on this image
        for lab in eligible_on_image:
            person_name = cluster_to_name[lab]
            key = (person_name, src)
            if key not in copied_person_image:
                dst_name = f"{src.stem}__{person_name}{src.suffix}"
                dst = (out_dir / person_name) / dst_name
                if not (dst.exists() and dst.is_file()):
                    shutil.copy2(src, dst)
                copied_person_image.add(key)
            # log per face occurrence for this person on the image
            for i in idxs:
                if labels[i] == lab:
                    rec = records[i]
                    log.append([src.name, person_name, rec.face_index, rec.faces_in_image, rec.det_score, "eligible_person_image"])

        # Stage group-only copy if any ineligible cluster appears
        if ineligible_on_image:
            staged_group_only.add(src)
            for i in idxs:
                if labels[i] in ineligible_on_image:
                    rec = records[i]
                    log.append([src.name, "__group_only__", rec.face_index, rec.faces_in_image, rec.det_score, "group_only_person_on_image"])

        # If no clustered faces at all, stage for unknown
        if not has_any_cluster:
            staged_unknown.add(src)
            for i in idxs:
                if labels[i] == -1:
                    rec = records[i]
                    log.append([src.name, "__unknown__", rec.face_index, rec.faces_in_image, rec.det_score, "noise_only_image"])

    # Write staged group_only and unknown once per image
    for src in staged_group_only:
        dst = group_only_dir / src.name
        if not (dst.exists() and dst.is_file()):
            shutil.copy2(src, dst)
    for src in staged_unknown:
        dst = unknown_dir / src.name
        if not (dst.exists() and dst.is_file()):
            shutil.copy2(src, dst)

    # Save report
    import csv
    report_path = out_dir / "clustering_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "assigned_folder", "face_index", "faces_in_image", "det_score", "reason"])
        w.writerows(log)

    return cluster_to_name, report_path, eligible_clusters


def main():
    ap = argparse.ArgumentParser(description="Unsupervised face clustering & routing with cluster eligibility <= group_thr.")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eps-sim", type=float, default=0.55)
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--min-face", type=int, default=110)
    ap.add_argument("--blur-thr", type=float, default=45.0)
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--group-thr", type=int, default=3, help="threshold for group photo (faces_in_image > group_thr).")
    args = ap.parse_args()

    records = collect_faces(args.input_dir, min_face=args.min_face, blur_thr=args.blur_thr,
                            det_size=args.det_size, gpu_id=args.gpu_id)
    if not records:
        print("No usable faces found.")
        return

    labels = cluster_faces(records, eps_sim=args.eps_sim, min_samples=args.min_samples)
    cluster_to_name, report_path, eligible_clusters = route_by_clusters(records, labels, args.out_dir, group_thr=args.group_thr)

    print("[OK] Eligible clusters (with at least one photo <= group_thr people):")
    for cl in sorted(eligible_clusters):
        print(f"  {cluster_to_name[cl]} (cluster {cl})")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()
