# 🏡 House Price Prediction – Hongshan District, Wuhan  
# 🏡 武汉洪山区二手房房价预测项目

---

## 📑 Table of Contents | 目录

- [1. Project Overview | 项目概述](#1-project-overview--项目概述)  
- [2. Project Structure | 项目结构](#2-project-structure--项目结构)  
- [3. Dataset Description | 数据集说明](#3-dataset-description--数据集说明)  
- [4. Data Cleaning & Feature Engineering | 数据清洗与特征工程](#4-data-cleaning--feature-engineering--数据清洗与特征工程)  
- [5. Exploratory Data Analysis (EDA) | 探索性数据分析](#5-exploratory-data-analysis-eda--探索性数据分析)  
- [6. Modeling | 建模](#6-modeling--建模)  
- [7. Feature Importance | 特征重要性](#7-feature-importance--特征重要性)  
- [8. SHAP Interpretation | SHAP 可解释性分析](#8-shap-interpretation--shap-可解释性分析)  
- [9. Key Insights | 关键洞察](#9-key-insights--关键洞察)  
- [10. Future Work | 未来优化方向](#10-future-work--未来优化方向)  
- [11. Requirements | 环境依赖](#11-requirements--环境依赖)

---

## 1. Project Overview | 项目概述

**English**  
This project builds a machine learning model to predict second-hand housing prices in Hongshan District, Wuhan. The workflow includes web scraping, data cleaning, feature engineering, exploratory data analysis (EDA), multiple regression models, and SHAP-based interpretability.  
The goal is to identify the key drivers of housing prices and provide transparent, data-driven insights.

**中文**  
本项目通过机器学习预测武汉洪山区的二手房房价，流程包含数据爬取、清洗、特征工程、EDA、多模型训练及 SHAP 可解释性分析。  
目标是识别房价的关键影响因素，为房产估值与决策提供清晰、可靠的洞察。

---

## 2. Project Structure | 项目结构

```plaintext
📦 House-Price-Prediction-Hongshan-District-Wuhan
│
├── notebooks/  # Jupyter notebooks
│     ├── 01_data_cleaning.ipynb
│     ├── 02_eda.ipynb
│     ├── 03_modeling.ipynb
│     └── 04_shap_interpretation.ipynb
│
├── src/        # Python scripts for modular code
│     ├── data_cleaning.py
│     ├── feature_engineering.py
│     ├── train_model.py
│     └── utils.py
│
├── data/
│     ├── raw/        # scraped raw data
│     └── processed/  # cleaned dataset
│
├── README.md
└── requirements.txt
```

## 3. Dataset Description | 数据集说明

**English**  
Dataset fields include:  
- Area (㎡)  
- Rooms / Halls  
- Orientation  
- Decoration level  
- Floor level (High / Middle / Low)  
- Total floors  
- Building type  
- Subdistrict (location)  
- Price (target variable)

**中文**  
数据字段包含：  
- 面积（㎡）  
- 室/厅数量  
- 朝向  
- 装修情况（精装/简装/毛坯）  
- 楼层类型（高/中/低）  
- 总楼层  
- 建筑类型（板楼/塔楼）  
- 片区（subdistrict）  
- 房价（预测目标）

---

## 4. Data Cleaning & Feature Engineering | 数据清洗与特征工程

**English**

- Parsed raw text fields into structured variables  
- Normalized price into numeric value  
- Extracted rooms, halls, area, floor info  
- One-hot encoded all categorical features  
- Drop missing value
- Combined preprocessing and model training via Pipeline

**中文**

- 将文本字段解析为结构化变量  
- 将房价格式标准化为可计算的数值  
- 提取室/厅/面积/楼层等信息  
- 对分类变量进行 OneHot 编码  
- 删除缺失值 
- 使用 Pipeline 统一处理预处理和建模

---

## 5. Exploratory Data Analysis (EDA) | 探索性数据分析

**English Findings**

- Price distribution is right-skewed  
- Area strongly correlates with price  
- Clear price differences across subdistricts  
- High floors generally more expensive  
- Decoration/layout moderate influence

**中文结论**

- 房价呈右偏分布  
- 面积与房价高度正相关  
- 片区房价差异明显  
- 高楼层更受青睐  
- 装修/户型影响中等

---

## 6. Modeling | 建模

### Models | 模型

- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  

### Performance | 模型表现

| Model | RMSE | R² |
|-------|------|--------|
| Linear Regression | 833k | 0.787 |
| XGBoost | 774k | 0.816 |
| **Random Forest** | ⭐ **662k** | ⭐ **0.866** |

👉 **Random Forest performs best**  
👉 **随机森林表现最佳**

---

## 7. Feature Importance | 特征重要性

### Aggregated Importance | 聚合后特征重要性

| Feature | Importance |
|---------|------------|
| Area | 0.73 |
| Subdistrict | 0.12 |
| Total Floor | 0.066 |
| Floor Level | 0.027 |
| Rooms/Halls | ~0.015 |
| Decoration | ~0.014 |

**English**  
Area dominates; location is second; floor characteristics matter; decoration/layout minor.

**中文**  
面积最重要；区域第二；楼层有影响；装修与户型影响较小。

---

## 8. SHAP Interpretation | SHAP 可解释性分析

**English**

- Larger area increases predicted price  
- High-value subdistricts raise prices  
- High floors push price up; low floors down  
- Decoration/layout show small effects  

**中文**

- 面积越大，预测价格越高  
- 高价值片区拉升房价  
- 高楼层正向影响，低楼层负向影响  
- 装修/户型影响有限

---

## 9. Key Insights | 关键洞察

### 中文 

- **面积决定房屋的基础价值**，是最核心、最底层的定价逻辑。  
- **片区体现区域质量差异**，是驱动房价分层的主要外部因素。  
- **楼层特征反映居住体验**（采光、噪音），进而影响价格偏好。  
- **装修与户型属于辅助价值**，影响成交意愿但不改变价值本质。  
- 房价形成逻辑总结为：  
  **空间为底，区域为核，楼层为体验，装修为偏好。**

### English 

- **Area forms the foundation of housing value**, driving most price variation.  
- **Subdistrict differentiates value**, reflecting neighborhood quality and accessibility.  
- **Floor characteristics influence livability**, refining perceived value.  
- **Decoration and layout enhance appeal**, but do not fundamentally determine price.  
- Price logic summarized as:  
  **“Area sets the foundation, location drives differentiation, livability refines value, decoration enhances perception.”**

---

## 10. Future Work | 未来优化方向

- Add POI data (schools, subway distance, business density)  
- Include year-built or renovation year  
- Use LightGBM/CatBoost  
- Deploy as Streamlit / FastAPI app  
- Add temporal analysis for trends  

---

## 11. Requirements | 环境依赖

```plaintext
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
shap


