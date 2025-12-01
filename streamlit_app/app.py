import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1) Load Data (with cache)
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_books.csv")
    return df

df = load_data()

# =========================
# 2) Basic Page Config
# =========================
st.set_page_config(
    page_title="📚 Amazon Books – EDA & Price Insights",
    layout="wide"
)

st.title("📚 Amazon Books – EDA & Price Insights")
st.write(
    "This Streamlit app is built from the **cleaned Amazon books dataset**. "
    "You can explore prices, ratings, reviews, authors, and genres interactively."
)

# =========================
# 3) Sidebar Filters
# =========================
st.sidebar.header("🔍 Filters")

# Type filter
types = df["Type"].dropna().unique()
selected_types = st.sidebar.multiselect(
    "Book Type",
    options=sorted(types),
    default=list(sorted(types))
)

# Main Genre filter
genres = df["Main Genre"].dropna().unique()
selected_genres = st.sidebar.multiselect(
    "Main Genre",
    options=sorted(genres),
    default=list(sorted(genres))
)

# Rating filter
min_rating, max_rating = float(df["Rating"].min()), float(df["Rating"].max())
rating_range = st.sidebar.slider(
    "Rating range",
    min_value=min_rating,
    max_value=max_rating,
    value=(min_rating, max_rating),
    step=0.1
)

# Outlier filter (if column exists)
if "Price_outlier" in df.columns:
    exclude_outliers = st.sidebar.checkbox("Exclude price outliers", value=False)
else:
    exclude_outliers = False

# Apply filters
filtered_df = df.copy()

filtered_df = filtered_df[
    (filtered_df["Type"].isin(selected_types)) &
    (filtered_df["Main Genre"].isin(selected_genres)) &
    (filtered_df["Rating"] >= rating_range[0]) &
    (filtered_df["Rating"] <= rating_range[1])
]

if exclude_outliers and "Price_outlier" in df.columns:
    filtered_df = filtered_df[~filtered_df["Price_outlier"]]

st.sidebar.write(f"📊 Filtered rows: **{len(filtered_df)}**")

# =========================
# 4) Tabs for Sections
# =========================
tab_overview, tab_price, tab_rating, tab_authors, tab_corr = st.tabs(
    ["📘 Overview", "💰 Price Analysis", "⭐ Ratings & Reviews", "👤 Authors & Genres", "📈 Correlations"]
)

# =========================
# 5) Overview Tab
# =========================
with tab_overview:
    st.subheader("Dataset Overview")

    # Show basic info
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Shape (rows, columns):**", filtered_df.shape)
        st.write("**Columns:**", list(filtered_df.columns))
    with col2:
        st.write("**Sample rows:**")
        st.dataframe(filtered_df.head(10))

    st.markdown("---")
    st.subheader("Price – Summary Statistics")
    st.write(filtered_df["Price"].describe())

# =========================
# 6) Price Analysis Tab
# =========================
with tab_price:
    st.subheader("💰 Price Distribution ")

    # Histogram of Price_SAR
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered_df["Price"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Price ")
    ax.set_title("Distribution of Book Prices ")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Average Price by Type ")

    if not filtered_df.empty:
        avg_price_type = (
            filtered_df.groupby("Type", as_index=False)["Price"]
            .mean()
            .sort_values("Price", ascending=False)
        )
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=avg_price_type, x="Type", y="Price", ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.set_ylabel("Average Price")
        ax2.set_title("Average Book Price by Type")
        st.pyplot(fig2)

        st.write("**Average price by Type (table):**")
        st.dataframe(avg_price_type)
    else:
        st.info("No data available with the current filters.")

# =========================
# 7) Ratings & Reviews Tab
# =========================
with tab_rating:
    st.subheader("⭐ Rating Distribution")

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered_df["Rating"], bins=20, kde=True, ax=ax3)
    ax3.set_xlabel("Rating")
    ax3.set_title("Distribution of Ratings")
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Price vs Rating ")

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=filtered_df,
        x="Rating",
        y="Price",
        alpha=0.5,
        ax=ax4
    )
    ax4.set_xlabel("Rating")
    ax4.set_ylabel("Price")
    ax4.set_title("Price vs Rating")
    st.pyplot(fig4)

    st.markdown("---")
    st.subheader("Price vs Number of People Rated")

    fig5, ax5 = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=filtered_df,
        x="No. of People rated",
        y="Price",
        alpha=0.5,
        ax=ax5
    )
    ax5.set_xscale("log")
    ax5.set_xlabel("Number of People Rated (log scale)")
    ax5.set_ylabel("Price")
    ax5.set_title("Price vs Number of People Rated")
    st.pyplot(fig5)

# =========================
# 8) Authors & Genres Tab
# =========================
with tab_authors:
    st.subheader("👤 Top Authors (by number of books in filtered data)")

    if not filtered_df.empty:
        author_counts = (
            filtered_df["Author"]
            .value_counts()
            .head(10)
        )

        fig6, ax6 = plt.subplots(figsize=(8, 4))
        sns.barplot(
            x=author_counts.values,
            y=author_counts.index,
            ax=ax6
        )
        ax6.set_xlabel("Number of Books")
        ax6.set_ylabel("Author")
        ax6.set_title("Top 10 Authors by Book Count")
        st.pyplot(fig6)

        st.write("**Top 10 authors (table):**")
        st.dataframe(author_counts.rename("Book_Count"))
    else:
        st.info("No data available with the current filters.")

    st.markdown("---")
    st.subheader("Average Rating by Main Genre")

    if not filtered_df.empty:
        avg_rating_genre = (
            filtered_df.groupby("Main Genre", as_index=False)["Rating"]
            .mean()
            .sort_values("Rating", ascending=False)
        )

        fig7, ax7 = plt.subplots(figsize=(10, 4))
        sns.barplot(
            data=avg_rating_genre,
            x="Main Genre",
            y="Rating",
            ax=ax7
        )
        ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, ha="right")
        ax7.set_ylabel("Average Rating")
        ax7.set_title("Average Rating by Main Genre")
        st.pyplot(fig7)

        st.write("**Average Rating by Main Genre (table):**")
        st.dataframe(avg_rating_genre)
    else:
        st.info("No data available with the current filters.")

# =========================
# 9) Correlation Tab
# =========================
with tab_corr:
    st.subheader("📈 Correlation Heatmap (Numeric Features)")

    numeric_cols = ["Price",  "Rating", "No. of People rated"]
    existing_numeric = [c for c in numeric_cols if c in filtered_df.columns]

    if len(existing_numeric) >= 2:
        corr = filtered_df[existing_numeric].corr()

        fig8, ax8 = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax8)
        ax8.set_title("Correlation Heatmap")
        st.pyplot(fig8)

        st.write("**Correlation Matrix (table):**")
        st.dataframe(corr)
    else:
        st.info("Not enough numeric columns to compute correlations.")
