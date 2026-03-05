# Grid Resilience & Carbon Analytics

**Characterizing Electricity Demand and Marginal CO₂ Emissions During Hazardous Grid Events**

[![Project Website](https://img.shields.io/badge/Website-Live-brightgreen)](https://your-website-url.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Project Overview

This project investigates how hazardous events (Public Safety Power Shutoffs, extreme weather, equipment failures) affect electricity demand patterns and marginal CO₂ emissions using high-resolution time-series data. We apply data mining and clustering techniques to identify distinct operational signatures that can inform grid management and decarbonization strategies.

**Research Question:** Can hazardous events be characterized by distinct electricity demand and marginal CO₂ emission profiles using time-resolved grid data?

## 👥 Team Members

- **Moulik Kumar** - Dataset Acquisition + Integration
- **Atharva Zodpe** - Website Development + Content Structuring
- **Pratik Patil** - Exploratory Analysis + Feature Engineering

## 🗂️ Repository Structure

```
grid-analytics/
│
├── data/
│   ├── raw/                    # Raw data from APIs
│   ├── processed/              # Cleaned and preprocessed data
│   └── events/                 # Hazardous event catalog
│
├── src/
│   ├── data_collection.py      # API data collection scripts
│   ├── data_cleaning.py        # Cleaning and preprocessing
│   ├── feature_engineering.py  # Feature extraction
│   ├── visualization.py        # Exploratory visualizations
│   └── utils/                  # Helper functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_event_clustering.ipynb
│
├── reports/
│   ├── milestone_1.pdf
│   ├── milestone_2.pdf
│   └── figures/                # Generated visualizations
│
├── website/                    # Website source files
│   └── index.html
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - WattTime API (register at https://www.watttime.org/)
  - EIA Open Data API (register at https://www.eia.gov/opendata/)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/grid-analytics.git
cd grid-analytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API credentials:
```bash
cp config.example.py config.py
# Edit config.py with your API keys
```

## 📚 Data Sources

### 1. WattTime API - Marginal Emissions
- **Type:** Dynamic API
- **Resolution:** 5-15 minutes
- **Coverage:** Multiple US grid regions
- **Variables:** Timestamp, region, marginal CO₂ intensity (lbs/MWh)

### 2. EIA-930 - Electricity Demand
- **Type:** Government API
- **Resolution:** Hourly
- **Coverage:** All US balancing authorities (2019-2026)
- **Variables:** Timestamp, balancing authority, demand (MW)

### 3. Hazardous Event Catalog
- **Type:** Curated static dataset
- **Sources:** Utility reports, CPUC filings, NOAA alerts
- **Variables:** Event type, start/end times, affected region, severity

## 🚀 Usage

### Data Collection

```bash
python src/data_collection.py --region CAISO --start-date 2024-01-01 --end-date 2024-12-31
```

### Data Cleaning

```bash
python src/data_cleaning.py --input data/raw/ --output data/processed/
```

### Generate Visualizations

```bash
python src/visualization.py --data data/processed/emissions_clean.csv
```

## 📈 Key Findings (Milestone 2)

- **Data Quality:** 96.8% completeness after cleaning 2.1M records
- **Event Coverage:** 52 validated hazardous events (2019-2026)
- **Demand-Emissions Correlation:** Strong baseline correlation (r=0.92) that weakens during events
- **Regional Differences:** CAISO shows bimodal emissions distribution; PSCO higher baseline; ERCOT highest volatility

## 📊 Visualizations

Our exploratory analysis includes 10 unique visualizations:

1. Time series analysis of demand and emissions during PSPS events
2. Regional distribution analysis of emissions intensity
3. Correlation heatmaps of feature relationships
4. Seasonal decomposition of demand patterns
5. Peak demand vs. peak emissions scatter analysis
6. Event duration distribution by type
7. Q-Q plots for normality assessment
8. Hourly demand patterns by day of week
9. Cumulative emissions impact ranking
10. Multivariate feature relationship analysis

## 🛠️ Preprocessing Pipeline

Our data cleaning pipeline addresses:

- **Missing Values:** 3.2% in demand data, 1.8% in emissions
  - Forward-fill for gaps <30 min
  - Linear interpolation for 30min-2hr gaps
  
- **Duplicates:** Removed 0.4% duplicate records

- **Outliers:** IQR-based detection and capping
  - Demand must be >0 MW
  - Emissions must be <1500 lbs/MWh
  
- **Temporal Standardization:** All timestamps converted to UTC with uniform 15-minute intervals

## 📝 Milestones

- ✅ **Milestone 1:** Project framing, research questions, website launch
- ✅ **Milestone 2:** Data collection, cleaning, exploratory analysis
- ⏳ **Milestone 3:** Feature engineering, clustering analysis
- ⏳ **Milestone 4:** Final analysis, comprehensive reporting

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the team members directly.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- WattTime for providing marginal emissions data
- U.S. Energy Information Administration for electricity demand data
- California Public Utilities Commission for PSPS event documentation
- Course instructors and TAs for guidance

## 📞 Contact

**Project Website:** [https://your-website-url.github.io](https://your-website-url.github.io)

**Team Email:** group4-grid-analytics@university.edu

---

*Last Updated: March 2026*
