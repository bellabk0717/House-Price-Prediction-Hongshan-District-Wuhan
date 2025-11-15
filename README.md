# 🏠 House Price Prediction and Market Insights in Hongshan District, Wuhan  
# 🏠 武汉洪山区二手房房价预测与市场洞察  

## 📌 Project Overview | 项目概述  
This project leverages second-hand housing data from **Lianjia (链家网)** to build an end-to-end machine learning pipeline.  
The workflow includes:  
- Data cleaning & feature engineering  
- Exploratory data analysis (EDA)  
- Model benchmarking (Linear Regression, Ridge, Random Forest, XGBoost)  
- SHAP-based interpretability  

本项目基于 **链家网二手房数据**，构建了一个端到端的机器学习流程，主要包括： 
- 数据爬取
- 数据清洗与特征工程  
- 探索性数据分析（EDA）  
- 多模型对比（线性回归、岭回归、随机森林、XGBoost）  
- 基于 SHAP 的模型可解释性分析  

---

## 📂 Project Structure | 项目结构  
- `data/` : Raw and processed datasets (原始与处理后的数据, GitHub未上传)  
- `notebooks/` : Jupyter notebooks for analysis (分析过程笔记本)  
- `src/` : Python scripts (数据清洗、特征工程、建模与可视化脚本)  
- `reports/` : Figures and final report (图表与最终报告)  

---

## 🛠️ Tech Stack | 技术栈  
- **Languages / 语言**: Python  
- **Libraries / 库**: pandas, numpy, scikit-learn, XGBoost, shap, matplotlib, seaborn  
- **Environment / 环境**: Jupyter Notebook, Git  

---

## 🎯 Objectives | 项目目标  
- Identify key drivers of housing prices (area, layout, decoration, location, floor level)  
- Deliver a reproducible framework for house price prediction and market insights  
- Provide data-driven support for buyers, developers, and policymakers  

- 识别影响房价的关键因素（面积、户型、装修、地段、楼层等）  
- 构建可复现的房价预测与市场洞察框架  
- 为购房者、开发商和政策制定者提供数据驱动的决策参考  

---

## 📈 Example Insights | 示例洞察  
- Larger area and better decoration significantly increase house prices  
- Properties near key locations (subway, schools) show clear premium  
- SHAP analysis highlights **area, location, and decoration** as the most important features  

- 面积更大、装修更好显著提升房价  
- 临近地铁、学校等核心地段的房源存在明显溢价  
- SHAP 分析结果显示：面积、地段与装修是最重要的价格驱动因素  
