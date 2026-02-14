# The Hype Paradox: Film & TV Predictive Analytics

[![GitHub Pages](https://img.shields.io/badge/Live_Report-GitHub_Pages-blue?style=for-the-badge&logo=github)](https://sarim-bit.github.io/hype-paradox/)

##  [View the Live Interactive Analysis](https://sarim-bit.github.io/hype-paradox/trailer-to-ratings)
> The live report now includes an **Interactive Success Matrix**. Hover over data points to explore show-specific metrics and isolate performance by network.

### **Executive Summary**
In the era of viral marketing, is a trailer's view count a reliable predictor of a show's quality? **The Hype Paradox** is a data-driven investigation into the decoupling of social media "buzz" from critical success. By merging **TMDB** metadata with **YouTube Data API** engagement metrics for 100 high-profile shows (2024–2025), this project identifies the mathematical point where marketing volume stops scaling with critical acclaim.

---

## [View the Full Analysis Report](https://sarim-bit.github.io/hype-paradox/)
*For a deep dive into Network Efficiency, Genre Volatility, and the Public Sentiment Leaderboard, visit the live site.*

---

## Key Findings
* **The Popularity Gap:** A weak correlation (**0.22**) between YouTube views and TMDB ratings proves that marketing spend does not guarantee quality.
* **The Buzz Penalty:** High comment-to-like ratios (**-0.25** correlation) often signal audience polarization rather than universal acclaim.
* **Passive Majority:** Viral hits show a sharp drop in **Review Density (-0.61)**, indicating that massive reach often attracts a passive audience that is less likely to engage with the product critically.

## Technical Stack
* **APIs:** YouTube Data API v3, TheMovieDatabase (TMDB) API.
* **NLP:** VADER Sentiment Analysis (Compound scoring for 5,000+ organic comments).
* **Data Science:** Pandas, NumPy, Scipy (Pearson Correlation, Log Transformation).
* **Static Visualization:** Seaborn, Matplotlib (Lollipop charts, Heatmaps, Quadrant Analysis).
* **Interactive Visualization:** Plotly (Interactive Success Matrix with quadrant mapping).

## Repository Structure
```text
├── code/
│   └── trailer_to_ratings_predictions.ipynb   # Full EDA & Analysis Pipeline
├── data/
│   ├── integrated_tv_data.csv                 # Final processed dataset
│   └── youtube_comments_repository.json       # Harvested audience feedback
├── images/                                    # Generated analytical plots
└── index.md                                   # Source for GitHub Pages report
```

## Future Roadmap
* **Predictive Modeling**: Training an XGBoost regressor to predict TMDB scores based on the first 24 hours of trailer engagement.

* **Multichannel Signals**: Integrating Reddit and X (Twitter) API streams for a broader sentiment "Social Signal".

* **Semantic Search**: Implementing a Vector Database (Pinecone) to categorize why audiences are dissatisfied (e.g., "Script Quality" vs. "Casting").

**Author**: Sarim Rizvi

**Source Code**: [Github Repository](https://github.com/sarim-bit/hype-paradox)
