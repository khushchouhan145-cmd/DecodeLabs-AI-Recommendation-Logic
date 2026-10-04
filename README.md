# AI Recommendation Logic

## DecodeLabs - Artificial Intelligence Project 3

### Project Overview

This project is a simple AI-based recommendation system that recommends
relevant learning resources based on the user's interests and preferences.

The system uses text similarity to compare the user's preferences with
the descriptions of available items and ranks the items according to
their similarity score.

---

## Objective

The objective of this project is to build a simple recommendation
system based on user preferences using matching and similarity logic.

---

## Key Features

- Accepts user interests and preferences
- Matches user preferences with item descriptions
- Uses TF-IDF for text representation
- Calculates similarity using Cosine Similarity
- Ranks recommendations based on similarity score
- Displays the top relevant recommendations
- Handles empty input
- Handles irrelevant preferences

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

---

## How the System Works

The recommendation process follows these steps:

1. User enters their interests or preferences.
2. The system stores available items and their descriptions.
3. TF-IDF converts the text into numerical vectors.
4. Cosine Similarity compares the user's preferences with item descriptions.
5. Similarity scores are calculated.
6. Items are sorted according to their similarity scores.
7. The most relevant items are displayed as recommendations.

---

## Project Flow

User Preferences
        |
        v
Text Processing
        |
        v
TF-IDF Vectorization
        |
        v
Cosine Similarity
        |
        v
Similarity Scores
        |
        v
Ranking
        |
        v
Recommendations

---

## Installation

Make sure Python is installed on your system.

Install the required libraries using:

```bash
python -m pip install -r requirements.txt