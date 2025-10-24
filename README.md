# Predicting Player Match Placement in Fortnite Using Linear Regression

## Overview
This research project explores the feasibility of predicting a player’s match placement in *Fortnite* using key in-game performance metrics. The study serves as a proof of concept for the application of data analytics and predictive modeling in competitive gaming environments.

The model utilizes a Linear Regression approach to examine the relationship between match performance and player outcomes, demonstrating how game telemetry can be leveraged for future real-time analytics and strategy optimization.

---

## Objectives
- Analyze the influence of player performance statistics on match placement
- Develop and evaluate a predictive model using linear regression
- Establish a foundation for future predictive modeling in esports performance analytics

---

## Dataset and Features
The model is trained using a dataset consisting of player match statistics.  
Three features are currently used for prediction:

| Feature | Description |
|--------|-------------|
| **Eliminations** | Number of eliminations earned in a match |
| **Damage to Players** | Total damage dealt to enemy players |
| **Accuracy** | Hit percentage of all shots fired |

**Target Variable:** Final match placement

---

## Methodology
- Data preprocessing and normalization applied before training
- Train-test split implemented for model evaluation
- Performance assessed using:
  - **Coefficient of Determination (R²)**
  - **Mean Absolute Error (MAE)**

---

## Installation & Usage

### Requirements
Ensure Python is installed along with the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

### Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/jaywilliams9999-sudo/fortnitedataproject.git
