import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Sample item dataset
data = {
    "Item": [
        "Python for Beginners",
        "Advanced Python Programming",
        "Machine Learning Basics",
        "Deep Learning with Python",
        "Artificial Intelligence Basics",
        "Data Science with Python",
        "Python Data Analysis",
        "Web Development Fundamentals",
        "Frontend Development",
        "Backend Development with Python",
        "SQL and Database Fundamentals",
        "Computer Vision with OpenCV",
        "Natural Language Processing",
        "Data Visualization",
        "AI Project Development"
    ],

    "Description": [
        "python programming coding beginner variables functions",
        "python programming advanced functions object oriented programming",
        "machine learning artificial intelligence algorithms models classification regression",
        "deep learning neural networks python artificial intelligence tensorflow",
        "artificial intelligence machine learning algorithms intelligent systems",
        "python data science pandas numpy statistics analysis",
        "python pandas numpy data analysis cleaning processing",
        "html css javascript web development frontend websites",
        "html css javascript frontend user interface web design",
        "python backend api server database development",
        "sql database mysql queries data management relational database",
        "computer vision image processing python opencv images",
        "natural language processing text artificial intelligence language",
        "data visualization charts graphs python pandas analysis",
        "artificial intelligence machine learning python project development"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)


def recommend_items(user_preferences, number_of_recommendations=5):

    # Add user preference to the dataset
    all_text = df["Description"].tolist() + [user_preferences]

    # Convert text into numerical TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(all_text)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    ).flatten()

    
    result = df.copy()
    result["Similarity Score"] = similarity_scores

    
    result = result.sort_values(
        by="Similarity Score",
        ascending=False
    )

    
    return result.head(number_of_recommendations)


print("=" * 55)
print("       AI RECOMMENDATION SYSTEM")
print("             DecodeLabs Project 3")
print("=" * 55)

print("\nAvailable interests:")
print("Python")
print("Machine Learning")
print("Artificial Intelligence")
print("Web Development")
print("Data Science")
print("Deep Learning")
print("SQL")
print("Computer Vision")
print("NLP")

print("\nEnter your interests/preferences.")
print("You can enter multiple interests separated by spaces.")

user_input = input("\nYour preferences: ").strip()


# -----------------------------------------
# Input Validation
# -----------------------------------------

if not user_input:
    print("\nError: Please enter at least one preference.")

else:

    recommendations = recommend_items(user_input)

    # Check whether any meaningful match was found
    highest_score = recommendations["Similarity Score"].max()

    if highest_score == 0:

        print("\nSorry, no relevant recommendation was found.")
        print("Please try interests such as:")
        print("Python, Machine Learning, AI, Web Development,")
        print("Data Science, Deep Learning, SQL, Computer Vision or NLP.")

    else:

        print("\n" + "=" * 55)
        print("        YOUR RECOMMENDATIONS")
        print("=" * 55)

        for index, row in recommendations.iterrows():

            score = row["Similarity Score"] * 100

            # Only show relevant recommendations
            if score > 0:

                print(f"\n{row['Item']}")
                print(f"Similarity Score: {score:.2f}%")
                print(f"Why recommended: {row['Description']}")

        print("\n" + "=" * 55)
        print("Recommendation process completed successfully.")
        print("=" * 55)