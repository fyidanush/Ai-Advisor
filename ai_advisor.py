from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Fetch market data
def fetch_market_data():
    tickers = ["^GSPC", "BND", "GLD", "BTC-USD"]  # S&P 500, Bonds, Gold, Bitcoin
    data = yf.download(tickers, start="2020-01-01", end="2024-03-01")["Close"]
    returns = data.pct_change().dropna()
    return returns

# Generate synthetic user profiles
def generate_user_profiles(n=1000):
    np.random.seed(42)

    age = np.random.randint(18, 65, n)
    income = np.random.randint(20000, 150000, n)
    risk_tolerance = np.random.choice(["Low", "Medium", "High"], n, p=[0.3, 0.5, 0.2])

    # Market Trends (average returns over past 7 days)
    stock_trend = np.random.uniform(-0.02, 0.02, n)
    bond_trend = np.random.uniform(-0.005, 0.005, n)
    gold_trend = np.random.uniform(-0.01, 0.01, n)
    crypto_trend = np.random.uniform(-0.05, 0.05, n)

    # Portfolio allocation based on risk tolerance with added noise
    portfolio_allocation = []

    for risk in risk_tolerance:
        if risk == "Low":
            base_allocation = [10, 70, 15, 5]  # More bonds, less crypto
        elif risk == "Medium":
            base_allocation = [40, 40, 15, 5]  # Balanced
        else:  # High risk
            base_allocation = [70, 10, 10, 10]  # More stocks & crypto

        # Add noise to the allocation
        noise = np.random.normal(0, 5, 4)  # Adding noise with mean 0 and std dev 5
        allocation = np.clip(base_allocation + noise, 0, 100)  # Ensure allocations are between 0% and 100%
        allocation = allocation / np.sum(allocation) * 100  # Normalize to sum to 100%
        portfolio_allocation.append(allocation)

    portfolio_allocation = np.array(portfolio_allocation)  # Convert to NumPy array

    df = pd.DataFrame({
        "Age": age,
        "Income": income,
        "Risk_Tolerance": risk_tolerance,
        "Stock_Trend": stock_trend,
        "Bond_Trend": bond_trend,
        "Gold_Trend": gold_trend,
        "Crypto_Trend": crypto_trend,
        "Stocks_%": portfolio_allocation[:, 0],
        "Bonds_%": portfolio_allocation[:, 1],
        "Gold_%": portfolio_allocation[:, 2],
        "Crypto_%": portfolio_allocation[:, 3],
    })

    return df

# Load or train the model
def load_or_train_model():
    try:
        # Try to load the pre-trained model
        model = joblib.load("random_forest_model.pkl")
        print("Model loaded from file.")
    except FileNotFoundError:
        # If the model file doesn't exist, train a new model
        print("Training new model...")
        user_data = generate_user_profiles()
        user_data["Risk_Tolerance"] = user_data["Risk_Tolerance"].map({"Low": 0, "Medium": 1, "High": 2})

        X = user_data[["Age", "Income", "Risk_Tolerance", "Stock_Trend", "Bond_Trend", "Gold_Trend", "Crypto_Trend"]]
        Y = user_data[["Stocks_%", "Bonds_%", "Gold_%", "Crypto_%"]]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, Y_train)

        # Save the trained model
        joblib.dump(model, "random_forest_model.pkl")
        print("Model trained and saved.")

    return model

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = load_or_train_model()

# Home route to display the form
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from the form
        age = int(request.form["age"])
        income = int(request.form["income"])
        risk_tolerance = request.form["risk_tolerance"]
        stock_trend = float(request.form["stock_trend"])
        bond_trend = float(request.form["bond_trend"])
        gold_trend = float(request.form["gold_trend"])
        crypto_trend = float(request.form["crypto_trend"])

        # Map risk tolerance to numerical value
        risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
        risk_tolerance_num = risk_mapping[risk_tolerance]

        # Prepare input for the model
        user_input = np.array([[age, income, risk_tolerance_num, stock_trend, bond_trend, gold_trend, crypto_trend]])

        # Make prediction
        allocation = model.predict(user_input)[0]

        # Prepare the result to display
        result = {
            "Stocks": f"{allocation[0]:.2f}%",
            "Bonds": f"{allocation[1]:.2f}%",
            "Gold": f"{allocation[2]:.2f}%",
            "Crypto": f"{allocation[3]:.2f}%"
        }

        # Render the result in the template
        return render_template("index.html", result=result)

    # Render the form for GET requests
    return render_template("index.html", result=None)

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)