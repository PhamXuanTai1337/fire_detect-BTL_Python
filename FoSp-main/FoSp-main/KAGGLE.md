# FoSp + full MAT trên Kaggle

Đây là bản FoSp tích hợp MAT pretrained **đủ hai stage**, MAT được đóng băng.
Backbone, Focus/BCG, projection và Fusion vẫn được train. Chưa có kết quả IoU
của một lượt train đầy đủ; đừng coi kiểm tra chạy được là bằng chứng về độ chính xác.

## File đưa lên GitHub

Dùng **nội dung bên trong `FoSp-main/FoSp-main/` làm gốc repository**:

```text
config/
mmseg/
mat_inpainter/       # gồm cả LICENSE và các file .cpp/.cu/.h
tools/
requirements/
tests/
imgs/               # hình minh họa trong README
.gitignore
setup.py
pyproject.toml
requirements.txt
README.md
FULL_MAT.md
KAGGLE.md
LICENSE
kaggle_train.ipynb
```

Không đưa `pretrained/`, `data/`, `work_dirs/`, `.venv-mat/`, cache hay file
`.pth/.pkl/.pt` lên Git. Không cần thư mục `MAT-main/` gốc hoặc
`inpainter_setting/` cũ: runtime MAT cần thiết đã nằm trong `mat_inpainter/`.
`.gitignore` đã loại dữ liệu và trọng số. Có thể tạo ZIP đúng danh sách bằng:

```bash
python tools/package_github.py --output /kaggle/working/fosp_mat_github.zip
```

Giữ nguyên hai file LICENSE. Trọng số MAT khoảng 661 MB; lưu dưới dạng Kaggle
Dataset hoặc tải bằng script bên dưới. Checkpoint FoSp khi train chứa cả MAT
và optimizer nên lớn hơn nhiều; `max_keep_ckpts=2` giới hạn checkpoint định kỳ.

## 1. Tạo notebook và cài môi trường

Bật GPU và Internet trong notebook. Hướng dẫn này dùng **một GPU**, batch 1,
FP32. Cấu hình `mat_impl='ref'` vẫn chạy GPU và đủ hai stage MAT; chỉ dùng
toán tử PyTorch thay cho extension CUDA cần biên dịch. Không cài MMCV 2.x
hoặc MMSegmentation từ pip đè lên mã trong repository.

Thay URL repo của bạn rồi chạy từng cell:

```python
!git clone https://github.com/YOUR_NAME/YOUR_REPO.git /kaggle/working/FoSp
%cd /kaggle/working/FoSp
!python -m venv --system-site-packages /kaggle/working/fosp-venv
!/kaggle/working/fosp-venv/bin/python tools/setup_kaggle.py
```

Script giữ nguyên cặp torch/torchvision có sẵn, cài MMCV 1.6.2 bản Python,
sửa phần setup cũ cho Python 3.13 và pin YAPF tương thích.
Các lệnh sau luôn gọi Python trong venv; không cần đổi kernel notebook.

## 2. Gắn dữ liệu và pretrained

Thêm dataset qua **Add Input**, sửa đường dẫn theo dataset thực tế:

```python
import os
os.environ['SMOKE_DATA_ROOT'] = '/kaggle/input/YOUR_DATASET/SmokeSeg'
# Nếu split nằm ở SmokeSeg_ImageSets/Segmentation thì sửa biến này:
os.environ['SMOKE_SPLIT_DIR'] = 'ImageSets/Segmentation'
```

Cấu trúc bắt buộc:

```text
SmokeSeg/
  JPEGImages/abc.jpg
  SegmentationClass/abc.png
  ImageSets/Segmentation/train.txt
  ImageSets/Segmentation/val.txt
  ImageSets/Segmentation/test.txt
```

Mỗi dòng split chỉ ghi ID `abc`, không kèm đuôi file. Mask là PNG một kênh
hoặc palette index, **0 = nền, 1 = khói, 255 = bỏ qua**. Mask đen/trắng 0/255
phải được chuyển đúng nhãn trước; loader không hiểu 255 là khói. Không dùng tập
test để chọn checkpoint. Bộ ảnh ví dụ trong repo gốc không thay thế dataset đầy đủ.

Cách A: tải MAT khi notebook có Internet:

```python
!/kaggle/working/fosp-venv/bin/python tools/download_mat.py
```

Cách B: gắn trọng số vào Kaggle Dataset, rồi đặt:

```python
os.environ['MAT_CHECKPOINT'] = '/kaggle/input/YOUR_WEIGHTS/Places_512_FullData.pkl'
```

MiT-B3 pretrained tự tải từ OpenMMLab ở lần train đầu. Nếu đã có file, đặt:

```python
os.environ['MIT_CHECKPOINT'] = '/kaggle/input/YOUR_WEIGHTS/mit_b3_20220624-13b1141c.pth'
```

Chỉ load trọng số từ nguồn tin cậy. MAT `.pkl` dùng pickle của tác giả.
Đường dẫn trong `/kaggle/input` chỉ để đọc; kết quả được ghi vào `/kaggle/working`.

## 3. Kiểm tra rồi train

```python
!/kaggle/working/fosp-venv/bin/python tools/check_kaggle_data.py config/fosp_kaggle.py
!/kaggle/working/fosp-venv/bin/python -m pytest tests -q
!/kaggle/working/fosp-venv/bin/python tools/check_training.py config/fosp_kaggle.py --work-dir /kaggle/working/fosp_check
```

Lệnh cuối train 1 iteration, validation 1 ảnh, lưu checkpoint, khởi động lại
và resume tới iteration 2. Nó kiểm tra cả optimizer đã tiếp tục từ bước 1.
Đây chỉ là kiểm tra luồng chạy. Xem `fosp_check/report.json` để xác nhận PASS.
Thư mục kiểm tra có checkpoint lớn; có thể xóa sau khi xác nhận PASS để dành chỗ
cho train thật. Khi chạy lại bài kiểm tra, chọn một `--work-dir` mới.

Train đầy đủ với 40.000 iterations:

```python
!/kaggle/working/fosp-venv/bin/python tools/train.py config/fosp_kaggle.py --work-dir /kaggle/working/fosp_runs
```

Validation mỗi 1.000 iterations. `IoU.smoke` là IoU lớp khói tính trên toàn bộ
pixel validation; `mIoU` là trung bình IoU của nền và khói. Checkpoint tốt nhất
được chọn theo `IoU.smoke`, không phải một chỉ số `mf_beta` chưa được triển khai.
Các giá trị lưu trong JSON ở thang 0–1; bảng console hiển thị phần trăm.

## 4. Tiếp tục sau một phiên Kaggle

Lưu output notebook hoặc tải checkpoint trước khi kết thúc phiên. Ở phiên mới,
gắn output cũ làm Input, thiết lập lại môi trường và đường dẫn pretrained, rồi:

```python
!/kaggle/working/fosp-venv/bin/python tools/train.py config/fosp_kaggle.py --resume-from /kaggle/input/YOUR_PREVIOUS_OUTPUT/fosp_runs/iter_1000.pth --work-dir /kaggle/working/fosp_runs
```

Dùng checkpoint `iter_*.pth` mới nhất để tiếp tục cả optimizer và lịch learning
rate; `--load-from` chỉ load trọng số. Nếu cùng phiên và thư mục đã có checkpoint,
có thể dùng `--auto-resume`.

Đánh giá test sau khi train, thay đúng đường dẫn checkpoint tốt nhất:

```python
!/kaggle/working/fosp-venv/bin/python tools/check_kaggle_data.py config/fosp_kaggle.py --splits train val test
!/kaggle/working/fosp-venv/bin/python tools/test.py --config config/fosp_kaggle.py --checkpoint /kaggle/working/fosp_runs/best_IoU.smoke_iter_1000.pth --eval mIoU mFscore --work-dir /kaggle/working/fosp_test
```

## Phạm vi rà soát

Các sửa chữa cho Kaggle gồm: dependency/setup, import Python cũ, GPU scatter của
MMCV với PyTorch mới, DataLoader workers=0, nhận kết quả FoSp khi validation,
tính metric theo từng class, streaming pre-eval và cấu hình chọn best checkpoint.
Resume cũng được sửa để không lặp lại iteration đã hoàn tất, giữ best score,
và tự chọn đúng checkpoint định kỳ mới nhất.
Kiến trúc MAT đủ hai stage, Focus/BCG, Fusion và công thức loss được giữ nguyên
trong lần rà soát này. Không tự động bật AMP hoặc huấn luyện MAT.

Chưa thực thi trực tiếp trên tài khoản Kaggle của bạn; môi trường, dataset và
thời gian phiên thực tế cần được xác nhận bằng lệnh kiểm tra ở bước 3.
