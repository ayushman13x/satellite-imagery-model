# Satellite–Tabular Fusion for Property Price Prediction

This project investigates whether satellite imagery can provide complementary neighborhood-level information to improve residential property price prediction when combined with traditional tabular housing data. The focus is not to replace structured features, but to critically evaluate the contribution of visual context in a multimodal machine learning pipeline.

## Project Overview
Traditional property valuation models rely on structured attributes such as living area, construction quality, and geographic coordinates. While these features capture internal property characteristics, they often fail to represent the surrounding environmental context. 

In this project, satellite images corresponding to property locations are used to capture visual information such as **green cover, built-up density, and urban layout**. These visual features are combined with a high-performance tabular model to assess whether multimodal learning provides a statistically significant improvement in price prediction.

---

## Modeling Approach

The project follows a rigorous, staged approach to ensure an "Honest Hybrid" evaluation:

### 1. Tabular-Only Baseline
A strong baseline is established using 18 raw features and 14 engineered features (e.g., `sqft_grade`, `luxury_index`). We utilize an **XGBoost regressor** trained on log-transformed prices to handle the non-linearities and heteroskedasticity common in real estate data.

### 2. Satellite Image Feature Extraction
Satellite images are processed using a **pretrained MobileNetV2 model** used strictly as a frozen feature extractor. By removing the final classification layer and applying **Global Average Pooling**, we generate 1,280-dimensional embeddings for every property location.

### 3. The "Golden Sweep" (Feature Selection)
To prevent the "Curse of Dimensionality" and ensure the model does not overfit on visual noise, we implemented a **feature selection sweep**. Using `SelectKBest` with an `f_regression` scoring function, we identified that the **top 20 visual features** provided the optimal balance between complexity and predictive power.

### 4. Multimodal Fusion Model
A **feature-level fusion strategy** is applied by concatenating the 32 tabular features with the 20 most predictive visual embeddings. The final model is an XGBoost regressor ($n=2000, \eta=0.02$) trained on this 52-feature combined representation.



---

## Repository Structure

```text
satellite-tabular-property-valuation/
├── notebooks/           # End-to-end experimental pipeline
│   ├── 1_image_download.ipynb   # Fetches satellite images for the dataset
│   ├── 2_preprocessing_eda.ipynb # Data cleaning, feature engineering, and EDA
│   └── 3_final_model.ipynb      # CNN extraction, Golden Sweep, and Hybrid Model
├── src/                 # Reusable production-grade code
│   └── data_fetcher.py          # Modular Mapbox image downloader (Utility script)
├── models/              # Serialized model artifacts
│   ├── honest_hybrid_k20.pkl    
│   └── final_feature_list.pkl   
├── requirements.txt     # Dependency list
└── README.md            # Project documentation

```

###How to Set Up the Project
1. Clone the repository
Bash

git clone [https://github.com/yourusername/satellite-tabular-property-valuation.git](https://github.com/yourusername/satellite-tabular-property-valuation.git)
cd satellite-tabular-property-valuation
2. Create a virtual environment
Mac / Linux

Bash

python -m venv venv
source venv/bin/activate
Windows

Bash

python -m venv venv
venv\Scripts\activate
3. Install dependencies
Bash

pip install -r requirements.txt
Running the Project
The notebooks are designed to be executed in the following order:

1. 1_image_download.ipynb
This notebook initializes the data ingestion pipeline. It uses the Mapbox Static Maps API and the data_fetcher.py utility to download satellite imagery for all property coordinates.

Command: Run all cells to populate the house_images/ directory.

2. 2_preprocessing_eda.ipynb
This stage focuses on data integrity and feature engineering. It handles missing values, applies log transformations to the price target, and calculates complex interactions like sqft_grade.

Command: Run all cells to generate the processed tabular features.

3. 3_final_model.ipynb
The core execution engine. It performs CNN feature extraction (MobileNetV2), executes the "Golden Sweep" feature selection (k=20), and trains the final Multimodal XGBoost model
