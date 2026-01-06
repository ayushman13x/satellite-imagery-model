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

---

## Running the Project
The project is structured as a sequential pipeline. To reproduce the results, the notebooks must be executed in the following order:

### 1. 1_image_download.ipynb
This notebook serves as the **Data Ingestion** layer. It bridges the gap between tabular geographic coordinates and visual environmental context.
* **Core Logic:** Iterates through the property dataset, passing `lat` and `long` coordinates to the Mapbox Static Maps API via the `data_fetcher.py` utility.
* **Key Parameters:** Captures satellite tiles at **Zoom Level 16** with a **400x400** pixel resolution to balance detail with computational efficiency.
* **Output:** Populates the `house_images/` directory with high-resolution satellite imagery used for subsequent feature extraction.

### 2. 2_preprocessing_eda.ipynb
This notebook handles **Data Integrity and Feature Engineering**. It transforms raw variables into a high-signal feature set optimized for Gradient Boosting.
* **Data Cleaning:** Implements log-transformations on the `price` target variable to normalize distribution and mitigate the impact of outliers.
* **Feature Engineering:** Constructs critical interaction variables, most notably `sqft_grade` (Living Area × Construction Quality), which serves as the primary tabular predictor.
* **Exploratory Analysis:** Includes spatial heatmaps and correlation matrices to visualize the relationship between geographic location and market value.
* **Output:** A curated tabular dataset ready for multimodal integration.

### 3. 3_final_model.ipynb
The **Modeling and Fusion** engine of the project. This notebook implements the multimodal learning strategy.
* **CNN Feature Extraction:** Passes the satellite imagery through a frozen **MobileNetV2** backbone (pretrained on ImageNet) to extract 1,280-dimensional visual embeddings.
* **The "Golden Sweep":** Applies `SelectKBest` with an `f_regression` scoring function to isolate the **top 20 visual features**, effectively reducing noise and preventing the "Curse of Dimensionality."
* **Multimodal Fusion:** Concatenates the 32 tabular features with the 20 visual features. This 52-feature matrix is then fed into an **XGBoost Regressor**.
* **Output:** Generates the final performance metrics ($R^2$, MAE) and the submission-ready `23117040_final.csv`.

  ---

## Key Findings

The experimental results provided several critical insights into the integration of computer vision with traditional real estate economics:

* **Tabular Dominance:** Structured housing attributes (living area, construction quality, and location) remain the primary drivers of property value.
* **Marginal Utility of Visuals:** Satellite imagery alone is insufficient for precise valuation but acts as a significant "refiner" of price.
* **The Hybrid Lift:** Integrating the top 20 visual features through the "Golden Sweep" method resulted in a measurable improvement in R² score, increasing from a tabular baseline of **0.894** to a hybrid peak of **0.898**.
* **Environmental Signals:** The model successfully identified visual premiums for properties with high vegetation density and price penalties for high-density industrial "gray" cover.

---

## Key Takeaways

1. **Feature Engineering is Vital:** Interaction features like `sqft_grade` capture more variance than raw features alone.
2. **Dimensionality Control:** Naive fusion of 1,280 CNN features introduces noise; targeted feature selection (k=20) is essential for maintaining model stability.
3. **Neighborhood Context:** Satellite imagery provides a unique proxy for neighborhood "desirability" that standard tabular data often misses.

---

## Explainability

To ensure the model was not making decisions based on artifacts or noise, explainability methods were applied:

* **Feature Importance (XGBoost):** Gain and Weight metrics confirmed that while structural features lead, specific visual embeddings rank high, indicating the model is actively utilizing the satellite data.
* **Gred-Cam:** Analysis shows that the selected visual features correlate with physical attributes such as green cover, proximity to arterial roads, and built-up density.

---

## Outputs

* **Final Predictions:** Located in `23117040_final.csv` (includes `id` and `predicted_price`).
* **Serialized Model:** The trained hybrid regressor is stored in `models/honest_hybrid_k20.pkl` for future inference tasks.
