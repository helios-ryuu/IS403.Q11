# Inflation Forecasting - Peru Dataset

Dự án so sánh 10 mô hình dự báo lạm phát trên dataset Peru: Econometric (RW, VAR, ARIMA), Linear ML (LASSO, Ridge, Elastic Net, LARS), và Nonlinear ML (Random Forest, SVR, XGBoost).

> **🚀 Muốn bắt đầu ngay?** Xem **[START.md](START.md)** - Hướng dẫn chi tiết từng bước thiết lập và khởi chạy dự án.

---

## � Mục lục

- [📋 Tổng quan](#-tổng-quan)
  - [Mô hình (10 models)](#mô-hình-10-models)
  - [Metrics đánh giá](#metrics-đánh-giá)
- [📁 Cấu trúc dự án](#-cấu-trúc-dự-án)
- [🚀 Quick Start](#-quick-start)
- [📊 Workflow](#-workflow)
- [📦 Dependencies](#-dependencies-tối-thiểu)
- [📈 Kết quả](#-kết-quả)
- [🔧 Custom Metrics](#-custom-metrics)
- [💡 Tips](#-tips)
- [📝 Viết báo cáo](#-viết-báo-cáo)
- [📖 Cấu trúc báo cáo đề xuất](#-cấu-trúc-báo-cáo-đề-xuất)
- [🎓 Dataset Peru](#-dataset-peru)
- [⚙️ Troubleshooting](#️-troubleshooting)
- [📧 Support](#-support)

---

## �📋 Tổng quan

### Mô hình (10 models)

**Kinh tế lượng:**
- Random Walk (RW)
- Vector Autoregression (VAR)
- ARIMA

**Machine Learning - Tuyến tính:**
- LASSO Regression
- Ridge Regression
- Elastic Net
- LARS (Least Angle Regression)

**Machine Learning - Phi tuyến:**
- Random Forest
- Support Vector Regression (SVR)
- XGBoost

### Metrics đánh giá

- **RMSFE** (Root Mean Squared Forecast Error): $\sqrt{\frac{1}{T} \sum_{t=1}^T (y_t - \hat{y}_t)^2}$
- **MAPE** (Mean Absolute Percentage Error): $\frac{1}{T} \sum_{t=1}^T |\frac{y_t - \hat{y}_t}{y_t}| \times 100\%$

---

## 📁 Cấu trúc dự án

```
inflation-forecasting/
├── data/
│   ├── raw/              # Dataset Peru (có sẵn)
│   └── processed/        # Tự động tạo
├── notebooks/
│   ├── example_*.ipynb   # 5 notebooks mẫu (tham khảo)
│   ├── 01-12.ipynb       # 12 notebooks trống (team tự làm)
│   └── ...
├── utils/
│   └── metrics.py        # RMSFE, MAPE
└── results/
    ├── figures/          # PNG (300 DPI)
    └── tables/           # CSV + LaTeX
```

---

## 🚀 Quick Start

```bash
# 1. Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Chạy Jupyter
jupyter notebook
```

**Chi tiết:** [START.md](START.md)

---

## 📊 Workflow

**12 Notebooks cần implement:**

1. **01_preprocessing.ipynb** - Load & xử lý data từ `data/raw/`
2. **02-11** - 10 mô hình (RW, VAR, ARIMA, LASSO, Ridge, Elastic Net, LARS, RF, SVR, XGBoost)
3. **12_evaluation.ipynb** - So sánh kết quả, export LaTeX

**Tham khảo:** Xem `example_*.ipynb` để biết cách implement

---

## 📦 Dependencies (Tối thiểu)

```
numpy>=1.24.0
pandas>=2.0.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## � Kết quả

Sau khi chạy xong, bạn có:

**Tables:**
- `all_models_results.csv` - Tổng hợp tất cả
- `model_comparison_latex.tex` - Import vào LaTeX
- `category_summary.csv` - Tóm tắt theo nhóm

**Figures:**
- `all_models_comparison.png` - So sánh toàn bộ
- `econometric_forecasts.png` - Dự báo econometric
- `linear_ml_forecasts.png` - Dự báo linear ML
- `nonlinear_ml_forecasts.png` - Dự báo nonlinear ML

**Summary:**
- `key_insights.txt` - Phân tích chính

---

## 🔧 Custom Metrics

```python
from utils.metrics import rmsfe, mape, evaluate_model

# Tính RMSFE
rmsfe_value = rmsfe(y_true, y_pred)

# Tính MAPE
mape_value = mape(y_true, y_pred)

# Đánh giá model
results = evaluate_model(y_true, y_pred, model_name="My Model")
```

---

## 💡 Tips

1. **Virtual Environment**: Luôn kích hoạt `.venv` trước khi làm việc
2. **Chạy theo thứ tự**: 01 → 02 → 03 → 04 → 05 (quan trọng!)
3. **Dataset có sẵn**: Không cần crawl, bắt đầu từ preprocessing
4. **Hyperparameters**: Điều chỉnh trong GridSearchCV nếu cần
5. **Lag features**: Thay đổi `n_lags` để thử nghiệm

---

## 📝 Viết báo cáo

1. Chạy tất cả notebooks
2. Import LaTeX table từ `results/tables/model_comparison_latex.tex`
3. Include figures từ `results/figures/`
4. Tham khảo insights từ `results/key_insights.txt`

Template LaTeX: `../latex-inflation-report.tex`

---

## 📖 Cấu trúc báo cáo đề xuất

1. **Giới thiệu** - Tầm quan trọng dự báo lạm phát
2. **Nghiên cứu liên quan** - Review literature
3. **Phương pháp** - Chi tiết 10 models + metrics
4. **Thực nghiệm** - Dataset Peru, kết quả, phân tích
5. **Kết luận** - Tổng kết, ý nghĩa, hạn chế

---

## 🎓 Dataset Peru

- **Nguồn**: Từ dự án `inflation-prediction`
- **Đã có sẵn**: `data/raw/*.csv`
- **Không cần crawl**: Bắt đầu trực tiếp từ preprocessing
- **Biến**: Headline inflation, Core inflation, lag features

---

## ⚙️ Troubleshooting

**Lỗi import module:**
```bash
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Jupyter không tìm kernel:**
```bash
pip install ipykernel
python -m ipykernel install --user
```

---

**Project:** IS403.Q11 - Inflation Forecasting Peru  
**Status:** ✅ Ready to use
