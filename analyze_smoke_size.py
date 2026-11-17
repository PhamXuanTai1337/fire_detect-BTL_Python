"""
Analyze Smoke Size Distribution & Generate Milestone Representative Samples
Dataset: SmokeSeg (Pascal VOC format: JPEGImages & SegmentationClass)
"""

import os
import sys
import glob
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Set Matplotlib temporary config dir to avoid permission warnings on Windows
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image, ImageDraw


def find_dataset_subdirs(base_path):
    """
    Auto-detect JPEGImages and SegmentationClass directories.
    Handles nested structures like:
    - base_path/SmokeSeg/SegmentationClass
    - base_path/SegmentationClass
    """
    candidates = [
        (os.path.join(base_path, 'SmokeSeg', 'JPEGImages'),
         os.path.join(base_path, 'SmokeSeg', 'SegmentationClass')),
        (os.path.join(base_path, 'JPEGImages'),
         os.path.join(base_path, 'SegmentationClass'))
    ]

    for img_dir, mask_dir in candidates:
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            return img_dir, mask_dir

    raise FileNotFoundError(
        f"Không tìm thấy thư mục JPEGImages và SegmentationClass trong: {base_path}\n"
        f"Vui lòng kiểm tra lại đường dẫn dataset."
    )


def process_single_mask(mask_dir, mask_filename):
    """Read a single mask and calculate smoke pixel count."""
    mask_path = os.path.join(mask_dir, mask_filename)
    try:
        with Image.open(mask_path) as img:
            arr = np.array(img)
            # Smoke pixels are non-zero
            smoke_pixels = int(np.count_nonzero(arr > 0))
            h, w = arr.shape[:2]
            total_pixels = h * w
            ratio = (smoke_pixels / total_pixels) * 100.0
            return mask_filename, smoke_pixels, total_pixels, ratio, (h, w)
    except Exception as e:
        print(f"Lỗi khi đọc file {mask_filename}: {e}", file=sys.stderr)
        return mask_filename, -1, 0, 0.0, (0, 0)


def scan_dataset(mask_dir, num_workers=16):
    """Scan all mask files concurrently and extract smoke statistics."""
    mask_files = [
        f for f in os.listdir(mask_dir)
        if f.lower().endswith('.png') and not f.startswith('.')
    ]
    total_files = len(mask_files)
    print(f"[*] Tìm thấy {total_files} file mask trong {mask_dir}")
    print(f"[*] Đang quét và đếm pixel khói bằng {num_workers} threads...")

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for item in executor.map(lambda f: process_single_mask(mask_dir, f), mask_files):
            if item[1] >= 0:
                results.append(item)

    # Sort results by smoke pixel count ascending
    results.sort(key=lambda x: x[1])
    return results


def find_milestone_samples(results, img_dir, percentiles=[10, 25, 50, 75, 95]):
    """
    Find 5 representative samples closest to the 5 milestone percentiles.
    """
    counts = np.array([r[1] for r in results])
    target_values = np.percentile(counts, percentiles)

    milestones = []
    labels = [
        "1. Khói siêu nhỏ (P10)",
        "2. Khói nhỏ (P25 - Q1)",
        "3. Khói trung bình (P50 - Median)",
        "4. Khói lớn (P75 - Q3)",
        "5. Khói rất lớn (P95)"
    ]

    for p, target, label in zip(percentiles, target_values, labels):
        idx = int(np.abs(counts - target).argmin())
        fname, px_count, total_px, ratio, shape = results[idx]
        stem = os.path.splitext(fname)[0]

        # Find matching image file (.jpg, .jpeg, .png)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = os.path.join(img_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        milestones.append({
            'percentile': p,
            'label': label,
            'target_px': int(target),
            'actual_px': px_count,
            'total_px': total_px,
            'ratio': ratio,
            'mask_name': fname,
            'img_path': img_path,
            'index': idx
        })

    return milestones


def plot_distribution_curves(results, milestones, output_path):
    """
    Plot 2 comprehensive line charts:
    1. Cumulative Distribution Function (CDF) showing percentage and count of images <= X pixels.
    2. Frequency / Density curve showing image distribution across pixel bins (log-scaled).
    """
    counts = np.array([r[1] for r in results])
    n = len(counts)

    # Sort counts for CDF
    sorted_counts = np.sort(counts)
    cumulative_pct = (np.arange(1, n + 1) / n) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=150)
    fig.suptitle(
        f"THỐNG KÊ PHÂN BỐ KÍCH THƯỚC KHÓI THEO SỐ LƯỢNG PIXEL (TỔNG {n:,} ẢNH)",
        fontsize=15, fontweight='bold', y=0.98
    )

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

    # --- Subplot 1: Cumulative Distribution Curve ---
    ax1.plot(sorted_counts, cumulative_pct, color='#0d6efd', linewidth=2.5, label='Đường tích lũy (%)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Số lượng pixel khói trong ảnh (Thang đo Logarithmic)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Tỷ lệ phần trăm tích lũy (%)', fontsize=11, fontweight='bold', color='#0d6efd')
    ax1.set_title('Đường phân bố tích lũy (Cumulative Curve)\nGiúp chọn ngưỡng pixel cho tập test khói nhỏ', fontsize=12, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.set_ylim(-2, 102)

    # Secondary Y-axis for raw image count
    ax1_count = ax1.twinx()
    ax1_count.set_ylabel('Số lượng ảnh tích lũy (Ảnh)', fontsize=11, fontweight='bold', color='#6c757d')
    ax1_count.set_ylim(-2 * n / 100, 102 * n / 100)
    ax1_count.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))

    # Mark standard COCO small object threshold (1024 px = 32x32)
    coco_thresh = 1024
    coco_pct = (np.count_nonzero(counts <= coco_thresh) / n) * 100.0
    ax1.axvline(x=coco_thresh, color='#fd7e14', linestyle=':', linewidth=2, label=f'Chuẩn COCO Small (1024 px: {coco_pct:.1f}%)')

    # Draw vertical dashed lines for the 5 milestones
    for m, col in zip(milestones, colors):
        px = m['actual_px']
        pct = (np.count_nonzero(counts <= px) / n) * 100.0
        ax1.axvline(x=px, color=col, linestyle='--', linewidth=1.8, alpha=0.85)
        ax1.scatter([px], [pct], color=col, s=60, zorder=5)
        ax1.annotate(
            f"P{m['percentile']} ({px:,} px\n~{pct:.0f}%)",
            xy=(px, pct),
            xytext=(px * 1.15, max(5, pct - 8)),
            fontsize=9,
            fontweight='bold',
            color=col,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=col, alpha=0.85)
        )

    ax1.legend(loc='lower right', framealpha=0.9)

    # --- Subplot 2: Frequency Distribution (Histogram & Density Line) ---
    log_bins = np.logspace(np.log10(max(1, counts.min())), np.log10(counts.max()), 50)
    bin_counts, bin_edges = np.histogram(counts, bins=log_bins)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    ax2.bar(bin_edges[:-1], bin_counts, width=np.diff(bin_edges), align='edge',
            color='#adb5bd', alpha=0.5, edgecolor='#6c757d', label='Số ảnh trong khoảng')
    ax2.plot(bin_centers, bin_counts, color='#dc3545', linewidth=2.5, marker='o', markersize=4, label='Đường mật độ số lượng ảnh')
    ax2.set_xscale('log')
    ax2.set_xlabel('Số lượng pixel khói trong ảnh (Thang đo Logarithmic)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Số lượng ảnh', fontsize=11, fontweight='bold')
    ax2.set_title('Biểu đồ đường mật độ số lượng ảnh theo dải pixel\nĐánh dấu 5 mốc đại diện', fontsize=12, fontweight='bold')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)

    for m, col in zip(milestones, colors):
        px = m['actual_px']
        ax2.axvline(x=px, color=col, linestyle='--', linewidth=1.8, alpha=0.85,
                    label=f"P{m['percentile']}: {px:,} px ({m['ratio']:.2f}%)")

    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Đã lưu biểu đồ đường phân bố tại: {output_path}")


def crop_smoke_region(img_np, mask_np, pad=30, crop_size=160):
    """Extract a cropped square region around the smoke center for detailed inspection."""
    coords = np.argwhere(mask_np > 0)
    if len(coords) == 0:
        # Fallback to image center if empty
        cy, cx = img_np.shape[0] // 2, img_np.shape[1] // 2
    else:
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        cy = (min_y + max_y) // 2
        cx = (min_x + max_x) // 2

    h, w = img_np.shape[:2]
    half = crop_size // 2

    # Clamp bbox coordinates
    y1 = max(0, cy - half)
    y2 = min(h, y1 + crop_size)
    if y2 == h:
        y1 = max(0, h - crop_size)

    x1 = max(0, cx - half)
    x2 = min(w, x1 + crop_size)
    if x2 == w:
        x1 = max(0, w - crop_size)

    crop_img = img_np[y1:y2, x1:x2].copy()
    crop_mask = mask_np[y1:y2, x1:x2].copy()
    return crop_img, crop_mask, (x1, y1, x2, y2)


def plot_milestone_images(milestones, mask_dir, output_path):
    """
    Visualize the 5 representative images corresponding to the 5 milestones.
    Row 1: Original Image with Bounding Box around smoke.
    Row 2: Overlay with Semi-Transparent Red Mask + Red Contour.
    Row 3: Zoomed-in Close-up (especially crucial for small smoke puffs).
    """
    n_cols = len(milestones)
    fig, axes = plt.subplots(3, n_cols, figsize=(4.2 * n_cols, 12), dpi=150)
    fig.suptitle(
        "5 ẢNH ĐẠI DIỆN CHO 5 MỐC SỐ LƯỢNG PIXEL KHÓI TRONG DATASET",
        fontsize=16, fontweight='bold', y=0.99
    )

    row_titles = [
        "1. Ảnh gốc + Bounding Box vị trí khói",
        "2. Lớp phủ phân đoạn khói (Red Overlay)",
        "3. Phóng to cận cảnh vùng khói (Zoom-in)"
    ]

    for col_idx, m in enumerate(milestones):
        # Read image and mask
        mask_path = os.path.join(mask_dir, m['mask_name'])
        with Image.open(mask_path) as mask_img:
            mask_arr = np.array(mask_img)

        if m['img_path'] and os.path.exists(m['img_path']):
            with Image.open(m['img_path']) as img:
                img_rgb = np.array(img.convert('RGB'))
        else:
            # Fallback black canvas
            img_rgb = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 3), dtype=np.uint8)

        # 1. Row 1: Bounding Box
        img_bbox = img_rgb.copy()
        coords = np.argwhere(mask_arr > 0)
        if len(coords) > 0:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)

            # Draw rectangle using PIL for clean thick lines
            pil_bbox = Image.fromarray(img_bbox)
            draw = ImageDraw.Draw(pil_bbox)
            # Add small padding to bbox
            pad = 5
            bx1, by1 = max(0, min_x - pad), max(0, min_y - pad)
            bx2, by2 = min(img_rgb.shape[1] - 1, max_x + pad), min(img_rgb.shape[0] - 1, max_y + pad)
            draw.rectangle([bx1, by1, bx2, by2], outline='#00ff00', width=3)
            img_bbox = np.array(pil_bbox)

        axes[0, col_idx].imshow(img_bbox)
        col_header = (
            f"{m['label']}\n"
            f"Số pixel: {m['actual_px']:,} px ({m['ratio']:.2f}% ảnh)\n"
            f"File: {m['mask_name'][:22]}..."
        )
        axes[0, col_idx].set_title(col_header, fontsize=10, fontweight='bold', pad=8)
        axes[0, col_idx].axis('off')

        # 2. Row 2: Overlay Mask
        img_overlay = img_rgb.copy().astype(float)
        smoke_mask = (mask_arr > 0)
        # Red overlay with alpha=0.55
        red_tint = np.array([255, 30, 30], dtype=float)
        img_overlay[smoke_mask] = (
            img_overlay[smoke_mask] * 0.45 + red_tint * 0.55
        )
        img_overlay = np.clip(img_overlay, 0, 255).astype(np.uint8)

        axes[1, col_idx].imshow(img_overlay)
        axes[1, col_idx].set_title(f"Mặt nạ khói: {m['actual_px']:,} pixels", fontsize=10)
        axes[1, col_idx].axis('off')

        # 3. Row 3: Zoom-in Close-up
        crop_img, crop_mask, (cx1, cy1, cx2, cy2) = crop_smoke_region(
            img_rgb, mask_arr, pad=20, crop_size=150
        )
        crop_overlay = crop_img.copy().astype(float)
        crop_smoke = (crop_mask > 0)
        crop_overlay[crop_smoke] = (
            crop_overlay[crop_smoke] * 0.45 + red_tint * 0.55
        )
        crop_overlay = np.clip(crop_overlay, 0, 255).astype(np.uint8)

        axes[2, col_idx].imshow(crop_overlay)
        axes[2, col_idx].set_title(f"Cận cảnh chi tiết (Zoom)", fontsize=10)
        axes[2, col_idx].axis('off')

    # Add row labels on the left
    for row_idx, rtitle in enumerate(row_titles):
        axes[row_idx, 0].text(
            -0.08, 0.5, rtitle,
            transform=axes[row_idx, 0].transAxes,
            rotation=90, va='center', ha='right',
            fontsize=11, fontweight='bold', color='#1a1a1a'
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Đã lưu ảnh 5 mốc đại diện tại: {output_path}")


def export_test_list_csv(results, output_csv_path, small_threshold=1000):
    """Export sorted pixel count metadata and flag 'small smoke'."""
    small_count = 0
    with open(output_csv_path, 'w', encoding='utf-8') as f:
        f.write("filename,smoke_pixels,total_pixels,area_percentage,is_small_smoke\n")
        for r in results:
            fname, px, total, ratio, _ = r
            is_small = 1 if px <= small_threshold else 0
            if is_small:
                small_count += 1
            f.write(f"{fname},{px},{total},{ratio:.4f},{is_small}\n")

    print(f"[+] Đã xuất metadata toàn bộ {len(results):,} ảnh ra: {output_csv_path}")
    print(f"[*] Số lượng ảnh 'khói nhỏ' (<= {small_threshold} px): {small_count:,} ảnh ({small_count/len(results)*100:.1f}%)")


def print_summary_and_guidance(results, milestones):
    """Print statistical breakdown and guidance on what counts as 'small smoke'."""
    counts = np.array([r[1] for r in results])
    n = len(counts)

    print("\n" + "=" * 70)
    print("               BÁO CÁO PHÂN TÍCH KÍCH THƯỚC KHÓI")
    print("=" * 70)
    print(f"- Tổng số ảnh phân tích: {n:,} ảnh")
    print(f"- Số pixel khói nhỏ nhất (Min): {counts.min():,} px ({counts.min()/262144*100:.4f}% ảnh)")
    print(f"- Số pixel khói lớn nhất (Max): {counts.max():,} px ({counts.max()/262144*100:.2f}% ảnh)")
    print(f"- Trung bình (Mean): {counts.mean():.1f} px ({counts.mean()/262144*100:.2f}% ảnh)")
    print(f"- Trung vị (Median / P50): {np.median(counts):.1f} px ({np.median(counts)/262144*100:.2f}% ảnh)")

    print("\n[+] BẢNG PHÂN VỊ CHI TIẾT:")
    pct_list = [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]
    pct_vals = np.percentile(counts, pct_list)
    for p, v in zip(pct_list, pct_vals):
        n_below = np.count_nonzero(counts <= v)
        print(f"  * P{p:02d} ({p:2d}%): <= {v:7.0f} px  (~{v/262144*100:5.2f}% diện tích) -> {n_below:5,d} ảnh")

    print("\n[+] 5 MỐC ĐẠI DIỆN ĐÃ ĐƯỢC CHỌN:")
    for m in milestones:
        print(f"  * {m['label']}: {m['actual_px']:,} px ({m['ratio']:.2f}% diện tích) -> File: {m['mask_name']}")

    print("\n" + "=" * 70)
    print("      GỢI Ý KHOA HỌC: BAO NHIÊU PIXEL KHÓI THÌ ĐƯỢC COI LÀ 'NHỎ'?")
    print("=" * 70)
    coco_small = np.count_nonzero(counts <= 1024)
    p25_val = int(np.percentile(counts, 25))
    p40_val = int(np.percentile(counts, 40))

    print(f"""
Dựa trên đặc trưng phân bố của tập SmokeSeg (ảnh gốc 512x512 = 262,144 px):

1. MỨC 1 - KHÓI SIÊU NHỎ / CHỚM BẮT ĐẦU (Tiny / Distant Smoke):
   - Ngưỡng đề xuất: <= 250 pixel (tương đương phân vị P25 = {p25_val} px, diện tích < 0.10%).
   - Số lượng ảnh: ~{np.count_nonzero(counts <= 250):,} ảnh ({np.count_nonzero(counts <= 250)/n*100:.1f}% dataset).
   - Đặc điểm: Đốm khói rất xa, mờ nhạt ở đường chân trời, chỉ vài chục đến hơn trăm pixel.
   - Ứng dụng: Test khả năng phát hiện cháy rừng từ cực sớm (Early-Warning / Hard cases).

2. MỨC 2 - KHÓI NHỎ TIÊU CHUẨN (Standard Small Smoke - Chuẩn COCO):
   - Ngưỡng đề xuất: <= 1,024 pixel (chuẩn COCO Small: diện tích < 32x32 = 1,024 px, chiếm < 0.39% ảnh).
   - Số lượng ảnh: ~{coco_small:,} ảnh ({coco_small/n*100:.1f}% dataset).
   - Đặc điểm: Cột khói vừa bốc lên rõ ràng, chiếm dưới 0.4% khung hình.
   - Ứng dụng: Chuẩn đánh giá học thuật phổ biến nhất để đo mIoU/F1 trên vật thể nhỏ.

3. MỨC 3 - KHÓI NHỎ GIAI ĐOẠN ĐẦU (Early Stage Medium-Small):
   - Ngưỡng đề xuất: <= 500 pixel (tương đương phân vị P40 = {p40_val} px, diện tích < 0.20%).
   - Số lượng ảnh: ~{np.count_nonzero(counts <= 500):,} ảnh ({np.count_nonzero(counts <= 500)/n*100:.1f}% dataset).
   - Ứng dụng: Cân bằng lý tưởng giữa số lượng ảnh đa dạng và độ khó cao cho mô hình.
""")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Smoke Size Distribution & Milestone Samples Visualizer")
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default=r"C:\Users\User\OneDrive\Documents\bai_tap\Python\BTL-Python\dataset\SmokeSeg",
        help="Đường dẫn đến thư mục chứa SmokeSeg dataset"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./smoke_analysis_output",
        help="Thư mục lưu trữ biểu đồ và ảnh kết quả"
    )
    parser.add_argument(
        '--small_thresh',
        type=int,
        default=1024,
        help="Ngưỡng pixel khói nhỏ để xuất danh sách test (mặc định 1024 px theo chuẩn COCO)"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help="Số luồng CPU để quét dataset nhanh"
    )

    args = parser.parse_args()

    # 1. Check directories
    img_dir, mask_dir = find_dataset_subdirs(args.dataset_dir)
    print(f"[+] Tìm thấy thư mục ảnh: {img_dir}")
    print(f"[+] Tìm thấy thư mục mặt nạ: {mask_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Scan dataset
    results = scan_dataset(mask_dir, num_workers=args.num_workers)

    # 3. Find 5 milestone samples
    milestones = find_milestone_samples(results, img_dir, percentiles=[10, 25, 50, 75, 95])

    # 4. Print statistical summary and guidance
    print_summary_and_guidance(results, milestones)

    # 5. Plot distribution curves
    curve_output = os.path.join(args.output_dir, "smoke_pixel_distribution_curve.png")
    plot_distribution_curves(results, milestones, curve_output)

    # 6. Plot 5 milestone images
    samples_output = os.path.join(args.output_dir, "smoke_5_milestone_samples.png")
    plot_milestone_images(milestones, mask_dir, samples_output)

    # 7. Export metadata CSV & Small smoke test list
    csv_output = os.path.join(args.output_dir, "smoke_dataset_stats.csv")
    export_test_list_csv(results, csv_output, small_threshold=args.small_thresh)

    small_txt_output = os.path.join(args.output_dir, f"test_small_smoke_under_{args.small_thresh}px.txt")
    with open(small_txt_output, 'w', encoding='utf-8') as f:
        for r in results:
            if r[1] <= args.small_thresh:
                f.write(f"{r[0]}\n")
    print(f"[+] Đã xuất danh sách tên file khói nhỏ ra: {small_txt_output}")

    print("\n[V] HOÀN TẤT THÀNH CÔNG! Tất cả file kết quả được lưu tại:")
    print(f"    -> {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
