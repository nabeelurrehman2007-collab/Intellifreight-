# 🚢 IntelliFreight
### AI-Powered Maritime Freight Cost Forecasting & Vessel Optimization

**IntelliFreight** is a machine learning-powered maritime analytics application designed to forecast freight costs and support vessel optimization for international shipping routes. By combining predictive analytics with vessel and port operational parameters, the system helps users make more informed shipping and logistics decisions.

## 🌟 Key Features

- **Freight Cost Prediction:** Predicts estimated maritime freight costs using a trained machine learning model.
- **Vessel Optimization Analysis:** Evaluates vessel characteristics and operational constraints to support shipping decisions.
- **Port-to-Port Route Analysis:** Supports route-specific freight forecasting, including international shipping routes.
- **Operational Parameter Analysis:** Considers shipment volume, vessel dimensions, draft, length overall (LOA), and port cargo-handling capacity.
- **Prediction Range:** Displays an estimated prediction range to communicate uncertainty around the forecast.
- **Market Status Indicator:** Provides a monitoring status to help users interpret forecast results.
- **Interactive Dashboard:** Built with Streamlit for an accessible, user-friendly experience.

## 🎯 Problem Statement

Maritime freight costs are influenced by multiple factors, including vessel specifications, cargo volume, port infrastructure, and shipping route characteristics. Traditional estimation methods may require considerable manual effort and may not provide an integrated view of these operational factors.

IntelliFreight aims to simplify this process by combining machine learning-based cost forecasting with vessel and port parameter analysis in a single application.

## 💡 Proposed Solution

IntelliFreight uses a trained machine learning model to estimate freight costs based on the available dataset and input parameters. It combines the resulting forecast with operational information to help users evaluate shipping scenarios and understand potential cost implications.

The application is intended as a decision-support prototype rather than a replacement for real-time freight quotations or professional maritime planning systems.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Web Application:** Streamlit
- **Model Storage:** Pickle (`.pkl`)
- **Development Environment:** Visual Studio Code

## 🧠 Model Performance

The trained model was evaluated using a held-out test dataset.

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 1.78 USD/MT |
| Root Mean Squared Error (RMSE) | 2.21 |
| R² Score | 0.6623 |
| Dataset Size | 1,000 records |
| Training Set | 800 records |
| Test Set | 200 records |

**Note:** These metrics reflect performance on the project's evaluation dataset. Actual forecasting accuracy may vary across routes, market conditions, and input scenarios.

## ⚙️ How It Works

1. The user enters shipping and vessel-related parameters.
2. The application processes the inputs for prediction.
3. The trained machine learning model generates an estimated freight cost.
4. The application displays the forecast and its prediction range.
5. Vessel and port parameters provide additional operational context for evaluating the shipping scenario.

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or a compatible Python environment
- pip
- Git (optional, for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/IntelliFreight.git
cd IntelliFreight
```

Replace `YOUR-USERNAME` with your GitHub username and adjust the repository name if necessary.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn
```

If the project includes a `requirements.txt` file, use:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## 📂 Project Structure

```text
IntelliFreight/
├── app.py
├── models/
│   └── freight_forecasting_model.pkl
├── requirements.txt
└── README.md
```

*The structure above is illustrative. Update it to match the files included in your repository.*

## 🌍 Example Use Case

Consider a shipment from Newcastle, Australia, to Visakhapatnam, India, involving 70,000 metric tonnes of cargo.

The user provides vessel and port parameters, including:

- Vessel type: Panamax
- Draft: 16.0 metres
- Length overall (LOA): 300 metres
- Port handling capacity: 40,000 MT/day
- Cargo volume: 70,000 MT

IntelliFreight generates an estimated freight cost forecast and a market monitoring indicator based on the application's model and configured logic.

*Example inputs are illustrative project scenarios, not verified live shipping quotations.*

## 🔮 Future Enhancements

- Integration with real-time maritime freight and port data.
- Automatic vessel recommendations based on route and cargo requirements.
- More comprehensive voyage-cost analysis, including fuel and port expenses.
- Interactive route visualizations and vessel comparison tools.
- Improved model performance using larger and more representative datasets.
- Deployment through a cloud platform for wider accessibility.

## ⚠️ Limitations

- Predictions depend on the quality, coverage, and representativeness of the training dataset.
- Forecasts should not be treated as guaranteed freight quotations.
- Live market prices, weather, congestion, and other external conditions may not be reflected unless explicitly integrated.
- Vessel feasibility and port compatibility should be independently verified before operational use.



## 📄 License

This project is intended for educational, research, and demonstration purposes.
---

⭐ **If you find IntelliFreight interesting, consider starring the repository!**
