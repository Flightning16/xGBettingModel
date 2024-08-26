import pandas as pd
import streamlit as st
from joblib import load
from urllib.error import HTTPError

# Load your pre-trained model
model = load("/Users/thomasfalzon/Desktop/csv_files/combined_model.joblib")

# Define leagues with file paths and division codes
league_info = {
    "Premier League": {
        "div_code": "E0",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Premier_League_stats.csv"
    },
    "EFL Championship": {
        "div_code": "E1",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Championship_stats.csv"
    },
    "La Liga": {
        "div_code": "SP1",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/La_Liga.csv"
    },
    "Spanish Segunda Division": {
        "div_code": "SP2",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Segunda_Division.csv"
    },
    "Ligue 1": {
        "div_code": "F1",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Ligue_1.csv"
    },
    "Ligue 2": {
        "div_code": "F2",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Ligue_2.csv"
    },
    "Serie A": {
        "div_code": "I1",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Serie_A.csv"
    },
    "Serie B": {
        "div_code": "I2",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Serie_B.csv"
    },
    "Bundesliga": {
        "div_code": "D1",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Bundesliga.csv"
    },
    "Bundesliga 2": {
        "div_code": "D2",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Bundesliga_2.csv"
    },
    "Eredivisie": {
        "div_code": "N1",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Eredivisie.csv"
    },
    "Belgian Pro League": {
        "div_code": "B1",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Pro_League.csv"
    },
    "Primeira Liga": {
        "div_code": "P1",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Primeira_Liga.csv"
    },
    "Brazilian Serie A": {
        "div_code": "BSA",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Brazilian_Serie_A.csv"
    },
    "Argentine Primera División": {
        "div_code": "ARG",
        "file": "/Users/thomasfalzon/Desktop/csv_files/League_Data/Primera_Division.csv"
    },
}

# Streamlit app layout
st.title("Football Match Predictions")
st.write("Select a league to view upcoming match predictions:")

# Dropdown menu for league selection
selected_league = st.selectbox("Choose a league:", list(league_info.keys()))

# Get the corresponding league information
selected_league_info = league_info[selected_league]
selected_file_path = selected_league_info["file"]
selected_div_code = selected_league_info["div_code"]

try:
    # Load and process the aggregated season data for the selected league
    df_league_current = pd.read_csv(selected_file_path)

    # Rename columns to match the feature names used during training
    df_league_clean = df_league_current.rename(columns={
        "GF": "gf",
        "GA": "ga",
        "Sh": "sh",
        "SoT": "sot",
        "xG": "xg",
        "xA": "xa",
        "npxG": "npxg"
    })

    # Extract the expected feature names and order from the model
    expected_feature_names = model.feature_names_in_

    # Reorder the features DataFrame to match the training order
    features = df_league_clean[expected_feature_names]

    # Calculate expected points using the model
    df_league_clean["exp_points"] = model.predict(features)

    # Load the upcoming fixtures from the website
    df_next_fix = pd.read_csv("https://football-data.co.uk/fixtures.csv")

    # Filter fixtures to include only the selected league based on division code
    df_next_fix_league = df_next_fix[df_next_fix["Div"] == selected_div_code]

    # Generate predictions for the next set of matches
    def make_predictions(df_league_clean, df_matches, model):
        predictions = []

        for _, row in df_matches.iterrows():
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]

            # Get expected points for home and away teams
            home_exp_points = df_league_clean[df_league_clean['Team'] == home_team]["exp_points"].values[0] if home_team in df_league_clean['Team'].values else 0
            away_exp_points = df_league_clean[df_league_clean['Team'] == away_team]["exp_points"].values[0] if away_team in df_league_clean['Team'].values else 0

            exp_points_diff_teams = home_exp_points - away_exp_points

            if exp_points_diff_teams > 4:
                result = "H"
                odds = row.get("PSH", 0)
            elif abs(exp_points_diff_teams) < 4:
                result = "D"
                odds = row.get("PSD", 0)
            else:
                result = "A"
                odds = row.get("PSA", 0)

            predictions.append({
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "Prediction": result,
                "Odds": odds
            })

        return pd.DataFrame(predictions)

    # Generate predictions
    df_predictions = make_predictions(df_league_clean, df_next_fix_league, model)

    # Display the predictions
    st.subheader(f"Predicted Outcomes for {selected_league}")
    st.dataframe(df_predictions)

except HTTPError as e:
    st.error(f"Error loading data for {selected_league}: {e}")
except FileNotFoundError as e:
    st.error(f"Error loading local data for {selected_league}: {e}")
except ValueError as e:
    st.error(f"ValueError: {e}")
