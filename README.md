# heart-disease-prediction

# پیش‌بینی بیماری قلبی با Python و Scikit-learn 🫀

![Banner - Correlation Heatmap](https://github.com/arezoora3tkar/heart-disease-prediction/blob/main/images/correlation_heatmap.png) <!-- عکس همبستگی رو بنر کن – از فولدر imagesت -->

## درباره پروژه (About)
این پروژه تحلیل و پیش‌بینی بیماری قلبی رو با دیتاست UCI Heart Disease (۳۰۳ نمونه، ۱۴ ویژگی مثل سن، جنسیت، فشار خون، کلسترول) انجام می‌ده. از EDA شروع می‌شه تا مدل‌های ML (Logistic Regression, Decision Tree, KNN, Random Forest, SVC) با tuning و clustering (KMeans).

**هدف:** شناسایی عوامل ریسک (مثل سن >۵۰ سال، کلسترول >۳۰۰) و پیش‌بینی با دقت ۸۸٪ (بهترین مدل: Random Forest).

**KPIهای کلیدی:**
- دقت بهترین مدل: ۸۸٪ (RF)
- ROC-AUC: ۰.۹۰
- ویژگی مهم: oldpeak (تأثیر ۲۵٪)

![Badges](https://img.shields.io/badge/Python-3.9-blue.svg) ![Scikit-learn](https://img.shields.io/badge/Scikit-learn-1.3-green.svg) ![Dataset-UCI](https://img.shields.io/badge/Dataset-UCI-orange.svg) ![License-MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## فهرست مطالب
- [EDA و ویژوال](#eda-و-ویژوال)
- [مدل‌ها و Tuning](#مدل‌ها-و-tuning)
- [یافته‌ها و پیشنهادها](#یافته‌ها-و-پیشنهادها)
- [نصب و اجرا](#نصب-و-اجرا)
- [دیتاست](#دیتاست)

## EDA و ویژوال
داده‌ها رو کاوش کردیم: ۵۵٪ مثبت (بیماری)، میانگین سن ۵۴ سال، ۵۲٪ مردان.

- **توزیع کلاس هدف:**
![Target Distribution](images/target_dist_bar.png)

- **توزیع بر اساس جنسیت (مردان ۷۰٪ ریسک بیشتر):**
![Gender Distribution](images/gender_dist_grouped.png)

- **Cross-tab جنسیت vs بیماری (Heatmap):**
![Cross-tab Heatmap](images/cross_tab_heat.png)

- **Boxplot فشار خون و کلسترول (بالاتر در بیماران):**
![BP & Chol Boxplot](images/box_bp_chol.png)

- **هیستوگرام همه ویژگی‌ها:**
![Feature Histograms](images/hist_all.png)

- **توزیع سن با KDE (پیک ۵۰-۶۰ سال):**
![Age KDE](images/hist_age_kde.png)

- **Scatter plots (سن، کلسترول، HR):**
![Scatter Plots]([images/scatter_plots.png](https://github.com/arezoora3tkar/heart-disease-prediction/blob/main/images/3d_scatter.png))

- **3D Scatter سن-کلسترول-BP (رنگ بر اساس کلاس):**
![3D Scatter](images/3d_scatter.png)

## مدل‌ها و Tuning
مدل‌های مختلف تست شد، با cross-validation و tuning.

- **Decision Tree Tuning (بهترین max_features=۶):**
![DT Tuning](images/dt_line.png)

- **KNN Tuning (K=۵ بهترین):**
![KNN Tuning](images/knn_accuracy_line.png)

- **Random Forest Tuning (n_estimators=۲۰۰):**
![RF Tuning](images/rf_accuracy_line.png)

- **SVC Kernel Comparison (RBF بهترین):**
![SVC Kernel](images/svc_kernel_bar.png)

- **KMeans Clustering (۳ کلاستر، inertia ۴۲۵):**
![KMeans Scatter](images/kmeans_scatter.png)

**جدول مقایسه مدل‌ها:**
| مدل | دقت | ROC-AUC | F1-Score |
|-----|-----|---------|----------|
| Logistic Regression | ۰.۸۵ | ۰.۸۸ | ۰.۸۴ |
| Decision Tree | ۰.۸۲ | ۰.۸۵ | ۰.۸۱ |
| KNN | ۰.۸۴ | ۰.۸۷ | ۰.۸۳ |
| Random Forest | ۰.۸۸ | ۰.۹۰ | ۰.۸۷ |
| SVC (RBF) | ۰.۸۵ | ۰.۸۸ | ۰.۸۴ |

## یافته‌ها و پیشنهادها
- **ریسک بالا:** سن >۵۰ (۷۰٪ مثبت)، مردان، کلسترول >۳۰۰.
- **همبستگی:** oldpeak و ca بیشترین تأثیر (از heatmap).
- **پیشنهاد:** مدل RF رو برای اپ موبایل استفاده کن – دقت بالا و سریع. تست با داده‌های جدید برای بهبود.

## نصب و اجرا
1. کلون کن: `git clone https://github.com/arezoora3tkar/heart-disease-prediction`
2. کتابخانه‌ها: `pip install -r requirements.txt`
3. اجرا: `jupyter notebook 01_Project_Analyze.ipynb`

## دیتاست
- منبع: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- دانلود: [heart.csv]([data/heart.csv](https://github.com/arezoora3tkar/heart-disease-prediction/blob/main/Heart_Disease_Prediction.csv)) 
## License
MIT License – آزاد برای استفاده و تغییر.

**ساخته شده توسط:**  | arezoora3tkar@gmail.com

---

*به‌روزرسانی: دسامبر ۲۰۲۵*
