 SpaceX Launch Success Prediction

A comprehensive data science project analyzing SpaceX Falcon 9 rocket launches to predict landing outcomes and identify key factors influencing mission success. This project demonstrates the complete data science pipeline from data collection to machine learning model deployment.

## 📋 Table of Contents
- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)

## 🎯 Overview

SpaceX revolutionized the space industry by offering Falcon 9 rocket launches at $62 million compared to competitors' $165 million+. The key to cost reduction is the reusability of the first stage boosters. This project builds a machine learning pipeline to predict the probability of successful first stage landings, enabling better pricing strategies for competing against SpaceX.

## 💼 Business Problem

**Objective**: Create a machine learning model to predict the landing outcome of SpaceX Falcon 9 first stage boosters.

**Key Questions**:
- What factors influence successful rocket landings?
- How do variables like payload mass, orbit type, and launch site affect outcomes?
- What are the optimal conditions for successful landings?
- How has SpaceX's success rate evolved over time?

## 📊 Dataset

**Data Sources**:
- **SpaceX REST API**: Historical launch data
- **Wikipedia Web Scraping**: Additional Falcon 9 launch records
- **Launch Sites**: Geographic and proximity analysis

**Key Features**:
- Flight Number, Launch Site, Payload Mass
- Orbit Type, Mission Outcome, Booster Version
- Landing Outcome, Launch Date
- Geographic coordinates and site proximities

**Dataset Size**: 101 launches (2010-2020)
**Target Variable**: Landing Success (Binary: 0=Failure, 1=Success)

## 🔬 Methodology

### 1. Data Collection
- **REST API Integration**: Automated data retrieval from SpaceX API
- **Web Scraping**: BeautifulSoup for Wikipedia data extraction
- **Data Validation**: Comprehensive cleaning and preprocessing

### 2. Data Wrangling & EDA
- Missing value imputation and data type conversions
- Feature engineering for categorical variables
- Statistical analysis and correlation studies
- **SQL Analysis**: Complex queries for data insights

### 3. Data Visualization
- **Matplotlib/Seaborn**: Statistical plots and trend analysis
- **Folium Maps**: Interactive geospatial analysis of launch sites
- **Plotly Dash**: Interactive dashboards for data exploration

### 4. Machine Learning Pipeline
- **Algorithms Tested**:
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
  - Logistic Regression
- **Model Selection**: GridSearchCV for hyperparameter optimization
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

## 🏆 Key Findings

### Launch Success Insights
- **Overall Success Rate**: Improved from 33% (2013) to 83% (2020)
- **Best Performing Site**: KSC LC-39A (76.9% success rate)
- **Optimal Orbit Types**: ES-L1, GEO, HEO, SSO (100% success rate)
- **Payload Impact**: Lighter payloads (<4000kg) show higher success rates

### Site Analysis
- **Geographic Distribution**: All launch sites located within the United States
- **Proximity Factors**: 
  - Launch sites maintain distance from cities for safety
  - Close proximity to coastlines for trajectory optimization
  - No direct railway access to launch sites

### Temporal Trends
- **Learning Curve**: Clear improvement in success rates over time
- **First Successful Ground Landing**: December 22, 2015
- **Technology Evolution**: Consistent advancement in landing capabilities

## 🛠 Technologies Used

**Programming Languages**:
- Python 3.8+

**Data Science Libraries**:
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn
- **Web Scraping**: BeautifulSoup, Requests
- **Geospatial Analysis**: Folium
- **Database**: SQLite

**Development Environment**:
- Jupyter Notebooks
- IBM Watson Studio (Development Platform)

## 📁 Project Structure

```
spacex-launch-prediction/
│
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Cleaned datasets
│   └── external/               # External data sources
│
├── notebooks/
│   ├── 01-data-collection-api.ipynb
│   ├── 02-data-collection-webscraping.ipynb
│   ├── 03-data-wrangling.ipynb
│   ├── 04-eda-dataviz.ipynb
│   ├── 05-eda-sql.ipynb
│   ├── 06-interactive-visual-analytics-folium.ipynb
│   ├── 07-plotly-dash-dashboard.ipynb
│   └── 08-machine-learning-prediction.ipynb
│
├── src/
│   ├── data/                   # Data processing modules
│   ├── features/               # Feature engineering
│   ├── models/                 # ML model implementations
│   └── visualization/          # Plotting utilities
│
├── reports/
│   ├── figures/                # Generated plots and charts
│   └── presentation.pdf        # Final presentation
│
├── requirements.txt
├── README.md
└── LICENSE
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/spacex-launch-prediction.git
cd spacex-launch-prediction
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**:
```bash
jupyter notebook
```

## 🚀 Usage

### Running the Complete Pipeline

1. **Data Collection**: Start with `01-data-collection-api.ipynb`
2. **Data Processing**: Continue through notebooks sequentially
3. **Model Training**: Execute `08-machine-learning-prediction.ipynb`
4. **Interactive Dashboard**: Run `07-plotly-dash-dashboard.ipynb`

### Quick Start - Prediction Model

```python
from src.models.predict_model import SpaceXPredictor

# Initialize predictor
predictor = SpaceXPredictor()

# Load trained model
predictor.load_model('models/best_model.pkl')

# Make prediction
prediction = predictor.predict({
    'payload_mass': 5000,
    'orbit_type': 'LEO',
    'launch_site': 'KSC LC-39A',
    'booster_version': 'F9 v1.1'
})

print(f"Landing Success Probability: {prediction:.2%}")
```

## 📈 Results

### Model Performance
- **Best Algorithm**: Decision Tree Classifier
- **Accuracy**: 86.25%
- **Key Features**: Payload mass, orbit type, launch site, booster version

### Business Impact
- **Cost Optimization**: 62% cost reduction compared to traditional launches
- **Risk Assessment**: Improved bidding strategies against SpaceX
- **Success Prediction**: 86% accuracy in landing outcome prediction

### Visualization Highlights
- Interactive launch site maps with success/failure indicators
- Temporal trend analysis showing improvement over time
- Payload vs. success rate correlation analysis
- Comprehensive dashboard for stakeholder presentations

## 🔮 Future Improvements

1. **Enhanced Features**:
   - Weather data integration
   - Real-time telemetry analysis
   - Advanced trajectory modeling

2. **Model Enhancements**:
   - Deep learning implementation
   - Ensemble methods
   - Real-time prediction capabilities

3. **Deployment**:
   - Web application development
   - API endpoint creation
   - Cloud platform integration

## 🙏 Acknowledgments

- **IBM Developer Skills Network** for the project framework
- **SpaceX** for providing public API access
- **NASA** for additional technical resources
- **Open Source Community** for the excellent libraries and tools
