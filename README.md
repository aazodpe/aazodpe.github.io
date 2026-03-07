# Grid Resilience & Carbon Analytics

**Characterizing Electricity Demand and Marginal CO₂ Emissions During Hazardous Grid Events**

[![Project Website](https://img.shields.io/badge/Website-Live-brightgreen)](https://aazodpe.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project investigates how hazardous events (Public Safety Power Shutoffs, extreme weather, equipment failures) affect electricity demand patterns and marginal CO₂ emissions using high-resolution time-series data. We apply data mining and clustering techniques to identify distinct operational signatures that can inform grid management and decarbonization strategies.

**Research Question:** Can hazardous events be characterized by distinct electricity demand and marginal CO₂ emission profiles using time-resolved grid data?

## Team Members

- **Moulik Kumar** - Dataset Acquisition + Integration
- **Atharva Zodpe** - Website Development + Content Structuring
- **Pratik Patil** - Exploratory Analysis + Feature Engineering

## Repository Structure

```
Data Mining Project/
│
├── data/
│   ├── raw/
│   │   ├── emissions_raw.csv       # Raw MOER readings from WattTime API
│   │   ├── demand_raw.csv          # Raw hourly demand from EIA-930 API
│   │   └── events_catalog.csv      # Curated hazardous event records
│   └── processed/
│       ├── emissions_clean.csv     # Cleaned emissions data
│       ├── demand_clean.csv        # Cleaned demand data
│       └── data_quality_report.json
│
├── figures/                        # Generated visualization PNGs (viz01–viz10)
│
├── data_collection.py              # API data collection script
├── data_cleaning.py                # Cleaning and preprocessing pipeline
├── visualization.py                # Exploratory visualizations (10 charts)
├── index.html                      # Project website
├── requirements.txt                # Python dependencies
└── README.md
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - WattTime API (register at https://www.watttime.org/)
  - EIA Open Data API (register at https://www.eia.gov/opendata/)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/aazodpe/aazodpe.github.io.git
cd aazodpe.github.io
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

4. Add API credentials to a `.env` file:
```
WATTTIME_USER=your_username
WATTTIME_PASSWORD=your_password
EIA_API_KEY=your_key
```

## Data Sources

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

## Usage

### Data Collection

```bash
python data_collection.py
```

### Data Cleaning

```bash
python data_cleaning.py
```

### Generate Visualizations

```bash
python visualization.py
```

## Key Findings (Milestone 2)

- **Data Quality:** 96.8% completeness after cleaning 2.1M records
- **Event Coverage:** 52 validated hazardous events (2019-2026)
- **Demand-Emissions Correlation:** Strong baseline correlation (r=0.92) that weakens during events
- **Regional Differences:** CAISO shows bimodal emissions distribution; PSCO higher baseline; ERCOT highest volatility

## Visualizations

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

## Preprocessing Pipeline

Our data cleaning pipeline addresses:

- **Missing Values:** 3.2% in demand data, 1.8% in emissions
  - Forward-fill for gaps <30 min
  - Linear interpolation for 30min-2hr gaps
  
- **Duplicates:** Removed 0.4% duplicate records

- **Outliers:** IQR-based detection and capping
  - Demand must be >0 MW
  - Emissions must be <1500 lbs/MWh
  
- **Temporal Standardization:** All timestamps converted to UTC with uniform 15-minute intervals

## Milestones

- ✅ **Milestone 1:** Project framing, research questions, website launch
- ✅ **Milestone 2:** Data collection, cleaning, exploratory analysis
- ⏳ **Milestone 3:** Feature engineering, clustering analysis
- ⏳ **Milestone 4:** Final analysis, comprehensive reporting

## Contributing

This is an academic project. For questions or suggestions, please contact the team members directly.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- WattTime for providing marginal emissions data
- U.S. Energy Information Administration for electricity demand data
- California Public Utilities Commission for PSPS event documentation
- Course instructors and TAs for guidance

## Contact

**Project Website:** [https://your-website-url.github.io](https://your-website-url.github.io)

**Team Email:** group4-grid-analytics@university.edu

---

*Last Updated: March 2026*
