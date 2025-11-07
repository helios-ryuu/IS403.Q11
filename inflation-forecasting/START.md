# 🚀 START - Hướng dẫn khởi chạy

Hướng dẫn thiết lập và chạy dự án Inflation Forecasting.

---

## ⚡ Quick Start (3 bước)

```bash
# Bước 1: Setup Virtual Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Bước 2: Cài đặt thư viện
python -m pip install --upgrade pip
pip install -r requirements.txt

# Bước 3: Chạy Jupyter
jupyter notebook
```

Sau đó chạy notebooks theo thứ tự: **01 → 02 → ... → 12**

---

## 📋 Yêu cầu

- ✅ Python 3.8+ (khuyến nghị 3.10+)
- ✅ pip
- ✅ ~500MB dung lượng

---

## 🔧 Chi tiết từng bước

### 1. Tạo Virtual Environment

```bash
cd inflation-forecasting
python -m venv .venv
```

### 2. Kích hoạt Virtual Environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
.\.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Kiểm tra:** Thấy `(.venv)` ở đầu dòng = thành công

### 3. Cài đặt Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, statsmodels; print('✅ OK!')"
```

---

## 📓 Workflow

### Notebooks cần implement (12 files)

1. **01_preprocessing.ipynb** - Load data từ `data/raw/`, xử lý, lưu vào `data/processed/`
2. **02_random_walk.ipynb** - Random Walk model
3. **03_var.ipynb** - VAR model
4. **04_arima.ipynb** - ARIMA model
5. **05_lasso.ipynb** - LASSO + GridSearchCV
6. **06_ridge.ipynb** - Ridge + GridSearchCV
7. **07_elastic_net.ipynb** - Elastic Net + GridSearchCV
8. **08_lars.ipynb** - LARS model
9. **09_random_forest.ipynb** - Random Forest + GridSearchCV
10. **10_svr.ipynb** - SVR + GridSearchCV
11. **11_xgboost.ipynb** - XGBoost + GridSearchCV
12. **12_evaluation.ipynb** - So sánh tất cả, export LaTeX

### Tham khảo

Xem các file `example_*.ipynb` để biết cách implement từng bước.

---

## 💡 Tips

**Luôn kích hoạt venv:**
```bash
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac
```

**Chạy lại từ đầu:**
```bash
rm -r data/processed/ results/
# Sau đó chạy lại từ notebook 01
```

---

## 🔍 Troubleshooting

### Lỗi Execution Policy (Windows)

```powershell
# Chạy PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Lỗi "No module named..."

```bash
# Kiểm tra venv đã active (thấy (.venv))
pip install -r requirements.txt
```

### Jupyter không tìm kernel

```bash
pip install ipykernel
python -m ipykernel install --user
```

---

## 📊 Dataset Peru

**Có sẵn trong `data/raw/`:**
- `df_raw.csv`, `df_raw_core.csv`
- `df_lags.csv`, `df_lags_core.csv`

✅ Không cần crawl - bắt đầu từ notebook 01

---

## ✅ Checklist

- [ ] Tạo venv
- [ ] Kích hoạt venv (thấy `.venv`)
- [ ] Cài đặt requirements
- [ ] Verify installation
- [ ] Implement 12 notebooks
- [ ] Kiểm tra `results/` có đầy đủ files
- [ ] Export LaTeX table
- [ ] Viết báo cáo

---

**Project:** IS403.Q11 - Inflation Forecasting Peru  
**Status:** ✅ Ready to Run


---

## 📋 Yêu cầu hệ thống

- ✅ Python 3.8+ (khuyến nghị 3.10+)
- ✅ pip (Python package manager)
- ✅ ~500MB dung lượng trống
- ✅ Windows PowerShell / Linux Terminal / macOS Terminal

---

## 🔧 Chi tiết từng bước

### Bước 1: Tạo Virtual Environment

Virtual environment giúp cách ly dependencies, tránh conflict với các dự án khác.

```bash
# Di chuyển vào thư mục dự án
cd inflation-forecasting

# Tạo virtual environment tên ".venv"
python -m venv .venv
```

### Bước 2: Kích hoạt Virtual Environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
.\.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Kiểm tra thành công:** Sẽ thấy `(.venv)` ở đầu dòng lệnh.

### Bước 3: Cài đặt Dependencies

```bash
# Upgrade pip (khuyến nghị)
python -m pip install --upgrade pip 

# Cài đặt tất cả thư viện từ requirements.txt
pip install -r requirements.txt
```

**Thời gian:** ~2-5 phút (tùy tốc độ mạng)

### Bước 4: Verify Installation

```bash
# Kiểm tra các package chính
python -c "import pandas, numpy, sklearn, xgboost, statsmodels; print('✅ All packages installed!')"
```

Nếu không có lỗi → Cài đặt thành công!

### Bước 5: Khởi chạy Jupyter Notebook

```bash
# Đảm bảo venv đã được kích hoạt (thấy (.venv) ở đầu dòng)
jupyter notebook
```

Browser sẽ tự động mở với Jupyter interface.

---

## 📓 Chạy Notebooks

### Quy trình bắt buộc (theo thứ tự):

#### 1️⃣ Notebook 01: Preprocessing
📂 `notebooks/01_preprocessing.ipynb`

**Mục đích:**
- Load dataset Peru từ `data/raw/`
- Kiểm tra dữ liệu, xử lý missing values
- Tạo train/test split (80/20)
- Lưu processed data

**Output:**
```
data/processed/
├── df_processed.csv
├── df_train.csv
└── df_test.csv
```

**Thời gian chạy:** ~2-5 phút

---

#### 2️⃣ Notebook 02: Econometric Models
📂 `notebooks/02_econometric_models.ipynb`

**Mô hình chạy:**
- Random Walk (RW)
- ARIMA
- VAR (nếu có nhiều biến)

**Output:**
```
results/tables/
├── econometric_results.csv
└── econometric_predictions.csv

results/figures/
└── econometric_forecasts.png
```

**Thời gian chạy:** ~5-10 phút

---

#### 3️⃣ Notebook 03: Linear ML Models
📂 `notebooks/03_linear_ml_models.ipynb`

**Mô hình chạy:**
- LASSO
- Ridge
- Elastic Net
- LARS

**Đặc điểm:**
- Tự động hyperparameter tuning với GridSearchCV
- Feature importance analysis

**Output:**
```
results/tables/
├── linear_ml_results.csv
└── linear_ml_predictions.csv

results/figures/
└── linear_ml_forecasts.png
```

**Thời gian chạy:** ~10-20 phút (do GridSearchCV)

---

#### 4️⃣ Notebook 04: Nonlinear ML Models
📂 `notebooks/04_nonlinear_ml_models.ipynb`

**Mô hình chạy:**
- Random Forest
- Support Vector Regression (SVR)
- XGBoost

**Đặc điểm:**
- GridSearchCV cho hyperparameters
- Feature importance từ tree-based models

**Output:**
```
results/tables/
├── nonlinear_ml_results.csv
└── nonlinear_ml_predictions.csv

results/figures/
├── nonlinear_ml_forecasts.png
├── rf_feature_importance.png
└── xgb_feature_importance.png
```

**Thời gian chạy:** ~15-30 phút (do GridSearchCV + complex models)

---

#### 5️⃣ Notebook 05: Evaluation & Comparison
📂 `notebooks/05_evaluation.ipynb`

**Mục đích:**
- So sánh tất cả 10 mô hình
- Tạo bảng tổng hợp
- Export LaTeX table cho báo cáo
- Phân tích insights

**Output:**
```
results/tables/
├── all_models_results.csv
├── model_comparison_latex.tex  ⭐ Import vào LaTeX
└── category_summary.csv

results/figures/
├── all_models_comparison.png
└── category_comparison_boxplot.png

results/
└── key_insights.txt  ⭐ Đọc để hiểu kết quả
```

**Thời gian chạy:** ~2-5 phút

---

## 📊 Sau khi hoàn thành

### 1. Kiểm tra kết quả

```bash
# Xem files đã tạo
ls results/tables/
ls results/figures/

# Đọc insights
cat results/key_insights.txt
```

### 2. Mô hình tốt nhất

```bash
# Xem ranking
cat results/tables/all_models_results.csv
```

### 3. Sử dụng cho báo cáo

**LaTeX Table:**
```bash
cat results/tables/model_comparison_latex.tex
# Copy & paste vào file .tex
```

**Figures:**
```
results/figures/all_models_comparison.png
# Include vào báo cáo
```

---

## 💡 Tips & Best Practices

### Virtual Environment

**Luôn kích hoạt trước khi làm việc:**
```bash
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac
```

**Tắt virtual environment:**
```bash
deactivate
```

### Chạy lại notebooks

Nếu muốn chạy lại từ đầu:
```bash
# Xóa dữ liệu processed và results
rm -r data/processed/
rm -r results/

# Chạy lại từ notebook 01
```

### Điều chỉnh hyperparameters

Trong notebooks 03 và 04, bạn có thể chỉnh:

```python
# Ví dụ: LASSO
lasso_params = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]  # Thêm/bớt giá trị
}

# Ví dụ: Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],  # Tăng để chính xác hơn
    'max_depth': [5, 10, 15, None]
}
```

### Thay đổi số lag features

```python
# Trong notebooks 03, 04
n_lags = 12  # Thử 6, 12, 24
```

### Thay đổi train/test split

```python
# Trong notebook 01
train_size = int(len(df_processed) * 0.8)  # 80/20 split
# Thử 0.7 (70/30) hoặc 0.85 (85/15)
```

---

## 🔍 Troubleshooting

### Lỗi: "python: command not found"

**Giải pháp:**
```bash
# Thử python3 thay vì python
python3 -m venv .venv
```

### Lỗi: "Cannot activate virtual environment"

**Windows PowerShell - Lỗi Execution Policy:**
```powershell
# Chạy PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Sau đó chạy lại
.\.venv\Scripts\Activate.ps1
```

### Lỗi: "No module named 'xxx'"

**Giải pháp:**
```bash
# Kiểm tra venv đã active chưa
# Phải thấy (.venv) ở đầu dòng

# Cài lại requirements
pip install -r requirements.txt

# Hoặc cài từng package bị thiếu
pip install pandas numpy scikit-learn
```

### Lỗi: "FileNotFoundError: data/raw/..."

**Giải pháp:**
```bash
# Kiểm tra đang ở đúng thư mục
pwd
# Phải hiển thị: .../inflation-forecasting

# Kiểm tra dataset có trong data/raw/
ls data/raw/
# Phải thấy: df_raw.csv, df_raw_core.csv, ...
```

### Lỗi: "Kernel died" trong Jupyter

**Giải pháp:**
```bash
# Cài lại ipykernel
pip install ipykernel
python -m ipykernel install --user --name=inflation-venv

# Trong Jupyter: Kernel → Change Kernel → inflation-venv
```

### Notebook chạy quá lâu (GridSearchCV)

**Giải pháp:**
```python
# Giảm số lượng params để test nhanh hơn
lasso_params = {'alpha': [0.01, 0.1, 1.0]}  # Từ 5 xuống 3
rf_params = {
    'n_estimators': [100],  # Chỉ test 1 giá trị
    'max_depth': [10]
}
```

### Out of Memory

**Giải pháp:**
```python
# Giảm n_lags
n_lags = 6  # Thay vì 12

# Hoặc giảm cross-validation folds
GridSearchCV(..., cv=3)  # Thay vì cv=5
```

---

## 📊 Metrics Reference

### RMSFE (Root Mean Squared Forecast Error)

```python
from utils.metrics import rmsfe

score = rmsfe(y_true, y_pred)
# Càng nhỏ càng tốt
```

**Formula:**
```
RMSFE = sqrt(mean((y_true - y_pred)^2))
```

### MAPE (Mean Absolute Percentage Error)

```python
from utils.metrics import mape

score = mape(y_true, y_pred)
# Càng nhỏ càng tốt (đơn vị: %)
```

**Formula:**
```
MAPE = mean(|y_true - y_pred| / |y_true|) * 100
```

### Evaluate Model

```python
from utils.metrics import evaluate_model

results = evaluate_model(y_true, y_pred, model_name="My Model")
# In ra cả RMSFE và MAPE
# Trả về dict với kết quả
```

---

## 🎓 Dataset Peru

**Có sẵn trong `data/raw/`:**
- `df_raw.csv` - Headline inflation (dữ liệu gốc)
- `df_raw_core.csv` - Core inflation (dữ liệu gốc)
- `df_lags.csv` - Headline với lag features
- `df_lags_core.csv` - Core với lag features

**Không cần:**
- ❌ Crawl dữ liệu
- ❌ Download từ internet
- ❌ Data collection

**Chỉ cần:**
- ✅ Chạy notebook 01 để preprocess
- ✅ Bắt đầu phân tích

---

## 📝 Checklist hoàn thành

- [ ] Tạo virtual environment
- [ ] Kích hoạt venv (thấy `.venv` ở đầu dòng)
- [ ] Cài đặt requirements
- [ ] Verify installation (import packages OK)
- [ ] Chạy notebook 01 ✅
- [ ] Chạy notebook 02 ✅
- [ ] Chạy notebook 03 ✅
- [ ] Chạy notebook 04 ✅
- [ ] Chạy notebook 05 ✅
- [ ] Kiểm tra `results/` có đầy đủ files
- [ ] Đọc `key_insights.txt`
- [ ] Export LaTeX table
- [ ] Viết báo cáo

---

## 🎯 Next Steps

1. **Chạy xong notebooks** → Có đầy đủ kết quả
2. **Phân tích kết quả** → Đọc `key_insights.txt`
3. **So sánh models** → Xem `all_models_results.csv`
4. **Viết báo cáo** → Dùng template `../latex-inflation-report.tex`
5. **Include results** → LaTeX table + Figures

---

## 📧 Cần trợ giúp?

1. Đọc **README.md** cho tổng quan
2. Xem **comments trong notebooks** cho chi tiết kỹ thuật
3. Check **docstrings** trong `utils/metrics.py`
4. Xem **troubleshooting** ở trên

---

**Ready to start? Chạy lệnh:**

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

**Good luck! 🚀**

---

**Project:** IS403.Q11 - Inflation Forecasting Peru  
**Last Updated:** November 7, 2025  
**Status:** ✅ Ready to Run