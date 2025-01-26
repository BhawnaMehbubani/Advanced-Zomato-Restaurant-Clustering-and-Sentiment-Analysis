# Zomato Restaurant Clustering and Sentiment Analysis

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Dataset Description](#dataset-description)
4. [Pipeline Architecture Workflow](#pipeline-architecture-workflow)
5. [Step-by-Step Process with Algorithms](#step-by-step-process-with-algorithms)
6. [Enhanced Data Insights and Observations](#enhanced-data-insights-and-observations)
7. [How to Run the Project](#how-to-run-the-project)
8. [Future Scope](#future-scope)



## Introduction

Zomato, a global leader in restaurant aggregation and food delivery, collects rich data on restaurants, customer reviews, and ratings. This project leverages this data to perform clustering and sentiment analysis, providing actionable insights for customers and business growth.



## Problem Statement

The project aims to:
- Cluster restaurants into meaningful segments based on their features.
- Perform sentiment analysis on customer reviews to gauge satisfaction.
- Derive insights into cuisine popularity, cost analysis, and customer preferences.



## Dataset Description

### Zomato Restaurant Metadata
- Attributes like name, cost per person, cuisines, categories, and operating hours.
- Used for clustering analysis.

### Zomato Restaurant Reviews
- Includes review text, ratings, reviewer metadata, and timestamps.
- Used for sentiment analysis.



## Pipeline Architecture Workflow

```plaintext
Data Collection
    |
    v
+-----------------------+
| Zomato Restaurant Data|
| - Metadata            |
| - Reviews             |
+-----------------------+
    |
    v
Data Cleaning and Preprocessing
    |
    +--------------------------------------------------------------+
    |                                                              |
    |                                                              |
    v                                                              v
Restaurant Metadata Cleaning                              Review Text Cleaning
- Handle missing values                                   - Remove punctuation
- Drop duplicates                                         - Convert to lowercase
- Parse cost and timings                                  - Tokenization
- Standardize cuisine names                               - Stopword removal
- Merge datasets on restaurant name                       - Lemmatization
    |                                                              |
    v                                                              v
+--------------------------------------------------------------------------+
|                Feature Engineering                                       |
| - Create derived features (e.g., avg rating, review count)               |
| - Normalize numeric data (cost, ratings)                                |
| - Convert categorical features into one-hot encoding (cuisine types)     |
+--------------------------------------------------------------------------+
    |
    v
Exploratory Data Analysis (EDA)
    |
    +--------------------------------------------------------------+
    |                                                              |
    |                                                              |
    v                                                              v
Visualizations on Restaurant Metadata                          Insights from Reviews
- Distribution of ratings and costs                             - Sentiment distribution
- Popular cuisines and categories                               - Influential reviewers
- Geographical trends                                           - Frequency of positive/negative keywords
    |
    v
+--------------------------------------------------------------------------+
|                       Clustering                                         |
| - Use K-Means Clustering for restaurant segmentation                     |
| - Reduce dimensions with PCA for clustering                              |
| - Determine optimal clusters using Elbow and Silhouette scores           |
+--------------------------------------------------------------------------+
    |
    v
Cluster Exploration
    |
    +--------------------------------------------------------------+
    |                                                              |
    |                                                              |
    v                                                              v
Insights for Zomato                                              Targeted Recommendations
- Identify high-cost, high-rated clusters                        - Suggest best restaurants to customers
- Segment mid-tier budget-friendly options                       - Recommend improvement areas to businesses
    |
    v
+--------------------------------------------------------------------------+
|                  Sentiment Analysis                                       |
| - Text Preprocessing (cleaning reviews)                                  |
| - Feature Extraction using TF-IDF                                        |
| - Train Naive Bayes classifier for sentiment classification              |
| - Validate model performance using accuracy and F1-score                 |
+--------------------------------------------------------------------------+
    |
    v
Results & Visualization
    |
    +--------------------------------------------------------------+
    |                                                              |
    |                                                              |
    v                                                              v
Restaurant Clustering Insights                                    Sentiment Analysis Insights
- Restaurant segmentation                                          - Percentage of positive and negative reviews
- Cost-benefit insights                                            - Common themes in customer feedback
    |
    v
Final Reporting & Dashboard
- Combine clustering and sentiment analysis results into actionable insights

```



## Step-by-Step Process with Algorithms

### 1. **Data Collection**
   - **Objective**: Gather Zomato metadata and review datasets.
   - **Tools Used**: Python (`pandas`, `numpy`).
   - **Algorithm**: None.

### 2. **Data Cleaning and Preprocessing**
   - **Objective**: Handle missing values, remove duplicates, and preprocess text.
   - **Tools Used**: Python (`pandas`, `nltk`).
   - **Algorithms**:
     - Missing value imputation: Replace null values with median/mean (for numerical data) or "unknown" (for categorical data).
     - Text cleaning: Tokenization, lowercasing, stopword removal, and stemming.

### 3. **Feature Engineering**
   - **Objective**: Create new features for clustering and analysis.
   - **Tools Used**: Python (`pandas`, `sklearn`).
   - **Algorithms**:
     - PCA (Principal Component Analysis): Reduce dimensionality of features.
     - Feature scaling: StandardScaler for normalization of cost and ratings.

### 4. **Exploratory Data Analysis (EDA)**
   - **Objective**: Visualize trends and uncover patterns.
   - **Tools Used**: Python (`matplotlib`, `seaborn`).
   - **Algorithms**:
     - Histogram: Analyze cost and rating distributions.
     - Bar plots: Identify popular cuisines and cities with the highest-rated restaurants.

### 5. **Clustering**
   - **Objective**: Group restaurants into meaningful clusters.
   - **Tools Used**: Python (`sklearn`).
   - **Algorithms**:
     - K-Means:
       - Clusters restaurants based on cost and ratings.
       - Optimal number of clusters determined using the Elbow Method and Silhouette Score.
     - Multi-dimensional clustering:
       - Incorporates additional features like cuisines and categories.

### 6. **Sentiment Analysis**
   - **Objective**: Classify reviews as positive or negative.
   - **Tools Used**: Python (`nltk`, `sklearn`).
   - **Algorithms**:
     - Naive Bayes: Simple probabilistic model for text classification.
     - TF-IDF: Transform textual data into numerical vectors for model input.



## Enhanced Data Insights and Observations

### Key Observations
1. **Top-Rated Cities**:
   - Delhi, Mumbai, and Bangalore feature the highest density of 4+ star-rated restaurants.
   - Smaller cities like Pune and Hyderabad show emerging trends in high-quality dining.

2. **Popular Cuisines**:
   - North Indian and Chinese cuisines dominate across most cities.
   - Regional specialties like Hyderabadi Biryani and Kolkata Mishti Doi are highly rated.

3. **Cost Insights**:
   - Average cost per person is highest in Delhi and Mumbai.
   - Cost distribution reveals mid-range restaurants form the majority.

4. **Review Patterns**:
   - Positive reviews often highlight good service and ambiance.
   - Negative reviews commonly mention delays, pricing, and food quality issues.

5. **Cluster Characteristics**:
   - Cluster 1: High-end restaurants with premium pricing and excellent ratings.
   - Cluster 2: Budget-friendly restaurants with moderate ratings.
   - Cluster 3: Popular restaurants offering diverse cuisines at affordable costs.

### Sentiment Analysis Insights
- Positive reviews account for 75% of the dataset.
- Reviewers with a high follower count tend to leave more detailed reviews, influencing sentiment significantly.



## How to Run the Project

### Prerequisites
- Python 3.7+
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `nltk`, `sklearn`, `wordcloud`.

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/BhawnaMehbubani/Advanced-Zomato-Restaurant-Clustering-and-Sentiment-Analysis.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Zomato_Restaurant_Clustering_and_Sentiment_Analysis.ipynb
   ```
4. Run all cells to execute the pipeline and visualize results.




## Future Scope

1. Incorporate geospatial data for location-specific insights.
2. Use advanced NLP models like BERT or GPT for more nuanced sentiment analysis.
3. Build an interactive dashboard for real-time analysis and recommendations.
4. Add restaurant images and user demographic data for enhanced clustering.


