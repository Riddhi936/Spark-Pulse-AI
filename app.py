import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import re


# ============================================================
# SPARKPULSE AI
# Distributed Social Media Trend & Sentiment Analyzer
# ============================================================


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SparkPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 🎨 FINAL SPARKPULSE AI COLOR SYSTEM
# ============================================================

# ---------- Global Colors ----------
BG = "#F8F7FC"
WHITE = "#FFFFFF"
PRIMARY = "#A7A0C8"
SECONDARY = "#DCD8ED"
HEADING = "#403B59"
TEXT = "#6D687A"
HIGHLIGHT = "#EEEAF7"
BORDER = "#DCD8ED"

# ---------- Dataset Overview ----------
DATA_PRIMARY = "#9FAFD0"
DATA_SECONDARY = "#DDE4F2"
DATA_HIGHLIGHT = "#EEF2F8"
DATA_HEADING = "#3F4963"
DATA_TEXT = "#6D7180"

# ---------- Platform & Country ----------
INSTAGRAM = "#A7A0C8"
TWITTER = "#91A8C7"
FACEBOOK = "#B8B1D0"
COUNTRY = "#9FAFD0"

# ---------- Sentiment ----------
POSITIVE = "#A7A0C8"
NEUTRAL = "#DCD8ED"
NEGATIVE = "#817A9E"

# ---------- Hashtags ----------
HASHTAG_MAIN = "#A7A0C8"
HASHTAG_TOP = "#817A9E"
PROGRESS_BG = "#DCD8ED"
PROGRESS_FILL = "#A7A0C8"

# ---------- Activity & Engagement ----------
ACTIVITY = "#9FAFD0"
ENGAGEMENT = "#A7A0C8"

# ---------- ML Model ----------
ML_PRIMARY = "#817A9E"
ML_SECONDARY = "#A7A0C8"
MATRIX_BG = "#EEEAF7"
MATRIX_HIGHLIGHT = "#817A9E"

# ---------- Business Insights ----------
BUSINESS_PRIMARY = "#A7A0C8"
BUSINESS_HIGHLIGHT = "#EEEAF7"
BUSINESS_CARD = "#FFFFFF"
BUSINESS_BORDER = "#DCD8ED"


# ============================================================
# 🌸 SOFT LAVENDER + BLUE WEBSITE THEME
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       MAIN APPLICATION
       ======================================================== */

    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}

    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1, h2, h3 {{
        color: {HEADING} !important;
        font-weight: 700;
    }}

    h1 {{
        letter-spacing: -0.5px;
    }}

    h2 {{
        margin-top: 1rem;
    }}

    h3 {{
        margin-top: 0.8rem;
    }}


    /* ========================================================
       NORMAL TEXT
       ======================================================== */

    p {{
        color: {TEXT};
    }}

    label {{
        color: {TEXT} !important;
    }}

    .stCaption {{
        color: {TEXT} !important;
    }}


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {{
        background-color: {HIGHLIGHT};
        border-right: 1px solid {BORDER};
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {HEADING} !important;
    }}

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {{
        color: {TEXT} !important;
    }}


    /* ========================================================
       SIDEBAR NAVIGATION
       ======================================================== */

    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        background-color: transparent;
        border-radius: 10px;
        padding: 7px 10px;
        margin-bottom: 3px;
        transition: 0.2s;
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background-color: {SECONDARY};
    }}

    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
        background-color: {PRIMARY};
    }}


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(64, 59, 89, 0.06);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {TEXT} !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {HEADING} !important;
        font-weight: 700;
    }}


    /* ========================================================
       ALERT / INFO BOXES
       ======================================================== */

    div[data-testid="stAlert"] {{
        border-radius: 12px;
        border: 1px solid {BORDER};
    }}


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
        background-color: {WHITE};
    }}


    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    div[data-testid="stProgress"] {{
        background-color: transparent;
    }}

    div[data-testid="stProgress"] > div {{
        background-color: {PROGRESS_BG};
        border-radius: 10px;
    }}

    div[data-testid="stProgress"] > div > div {{
        background-color: {PROGRESS_FILL};
        border-radius: 10px;
    }}


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {{
        border-color: {BORDER} !important;
    }}


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {{
        background-color: {PRIMARY};
        color: {WHITE};
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }}

    .stButton > button:hover {{
        background-color: {HEADING};
        color: {WHITE};
    }}


    /* ========================================================
       SELECT BOX
       ======================================================== */

    div[data-baseweb="select"] > div {{
        border-color: {BORDER};
        border-radius: 10px;
        background-color: {WHITE};
    }}


    /* ========================================================
       EXPANDER
       ======================================================== */

    div[data-testid="stExpander"] {{
        background-color: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

DATA_PATH = os.path.join(
    "data",
    "social_media.csv"
)

CHART_PATH = os.path.join(
    "outputs",
    "charts"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    original_rows = len(df)

    # Remove unnecessary ID columns
    df = df.drop(
        columns=[
            "_c0",
            "Unnamed: 0"
        ],
        errors="ignore"
    )

    # Clean text columns
    text_columns = [
        "Text",
        "Sentiment",
        "Platform",
        "Country",
        "Hashtags"
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    # Clean Text
    if "Text" in df.columns:

        df["Text"] = (
            df["Text"]
            .str.lower()
            .str.replace(
                r"[^a-zA-Z0-9\s]",
                "",
                regex=True
            )
            .str.strip()
        )

    # Numeric columns
    numeric_columns = [
        "Likes",
        "Retweets",
        "Year",
        "Month",
        "Day",
        "Hour"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Timestamp
    if "Timestamp" in df.columns:

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

    return df, original_rows


df, original_rows = load_data()


# ============================================================
# VERIFIED PROJECT RESULTS
# ============================================================

processed_rows = 712
duplicates_removed = 20

positive_count = 349
neutral_count = 193
negative_count = 170

positive_percentage = round(
    positive_count / processed_rows * 100,
    1
)

neutral_percentage = round(
    neutral_count / processed_rows * 100,
    1
)

negative_percentage = round(
    negative_count / processed_rows * 100,
    1
)

emotion_categories = 191

ml_accuracy = "64.04%"


# ============================================================
# SENTIMENT WORD LISTS
# ============================================================

positive_words = {
    "happy",
    "joy",
    "love",
    "excited",
    "excitement",
    "amazing",
    "great",
    "good",
    "wonderful",
    "positive",
    "grateful",
    "gratitude",
    "hope",
    "peace",
    "serenity",
    "contentment",
    "euphoria",
    "determination",
    "acceptance",
    "awesome",
    "success",
    "celebrate",
    "celebration",
    "fun",
    "beautiful",
    "enjoy"
}


negative_words = {
    "sad",
    "sadness",
    "angry",
    "anger",
    "hate",
    "despair",
    "grief",
    "lonely",
    "loneliness",
    "confusion",
    "confused",
    "embarrassed",
    "negative",
    "fear",
    "afraid",
    "stress",
    "stressed",
    "pain",
    "disappointed",
    "disappointment",
    "frustrated",
    "frustration"
}


# ============================================================
# SENTIMENT CLASSIFICATION
# ============================================================

def classify_sentiment(text):

    words = set(
        re.findall(
            r"[a-zA-Z]+",
            str(text).lower()
        )
    )

    positive_count_local = len(
        words.intersection(
            positive_words
        )
    )

    negative_count_local = len(
        words.intersection(
            negative_words
        )
    )

    if positive_count_local > negative_count_local:

        return "Positive"

    elif negative_count_local > positive_count_local:

        return "Negative"

    else:

        return "Neutral"


df["Broad Sentiment"] = (
    df["Text"]
    .apply(classify_sentiment)
)


# ============================================================
# HASHTAG ANALYSIS
# ============================================================

hashtag_list = []

for hashtags in df["Hashtags"].dropna():

    found = re.findall(
        r"#\w+",
        str(hashtags)
    )

    hashtag_list.extend(found)


if len(hashtag_list) > 0:

    hashtag_counts = (
        pd.Series(hashtag_list)
        .value_counts()
    )

    top_hashtag = hashtag_counts.index[0]

    top_hashtag_count = int(
        hashtag_counts.iloc[0]
    )

else:

    hashtag_counts = pd.Series(
        dtype=int
    )

    top_hashtag = "N/A"

    top_hashtag_count = 0


# ============================================================
# MAIN INSIGHTS
# ============================================================

most_active_platform = (
    df["Platform"]
    .value_counts()
    .idxmax()
)

most_active_country = (
    df["Country"]
    .value_counts()
    .idxmax()
)

most_active_hour = int(
    df["Hour"]
    .value_counts()
    .idxmax()
)


# ============================================================
# CHART FUNCTION
# ============================================================

def show_chart(filename):

    chart_file = os.path.join(
        CHART_PATH,
        filename
    )

    if os.path.exists(chart_file):

        st.image(
            chart_file,
            use_container_width=True
        )

    else:

        st.warning(
            "Chart not found: " + filename
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ SparkPulse AI")

st.sidebar.caption(
    "Distributed Social Media Trend & Sentiment Analyzer"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Dataset Overview",
        "Platform & Country Insights",
        "Sentiment & Emotion Analysis",
        "Trending Hashtags",
        "Activity & Engagement",
        "ML Model Performance",
        "Business Insights"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Technology Stack")

st.sidebar.write("• Apache Spark")
st.sidebar.write("• Spark SQL")
st.sidebar.write("• Spark ML")
st.sidebar.write("• Python")
st.sidebar.write("• Streamlit")


# ============================================================
# 1. EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.title("⚡ SparkPulse AI")

    st.caption(
        "Distributed Social Media Trend & Sentiment Analyzer"
    )

    st.divider()

    st.subheader("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Posts",
            "712"
        )

    with col2:
        st.metric(
            "Positive",
            "49.0%"
        )

    with col3:
        st.metric(
            "Negative",
            "23.9%"
        )

    with col4:
        st.metric(
            "Emotion Categories",
            "191"
        )

    st.divider()

    st.subheader("📊 Sentiment Distribution")

    sentiment_data = pd.DataFrame(
        {
            "Sentiment": [
                "Positive",
                "Neutral",
                "Negative"
            ],
            "Posts": [
                positive_count,
                neutral_count,
                negative_count
            ]
        }
    )

    fig, ax = plt.subplots(
        figsize=(10, 3.5)
    )

    ax.bar(
        sentiment_data["Sentiment"],
        sentiment_data["Posts"],
        color=[
            POSITIVE,
            NEUTRAL,
            NEGATIVE
        ]
    )

    ax.set_ylabel(
        "Number of Posts",
        color=TEXT
    )

    ax.set_title(
        "Broad Sentiment Distribution",
        color=HEADING
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        colors=TEXT
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.divider()

    st.subheader("🔥 Trending Topics & Hashtags")

    if len(hashtag_counts) > 0:

        top_topics = hashtag_counts.head(8)

        maximum = int(
            top_topics.max()
        )

        for index, (topic, count) in enumerate(
            top_topics.items()
        ):

            percentage = int(
                count / maximum * 100
            )

            col1, col2 = st.columns(
                [2, 8]
            )

            with col1:
                st.write(
                    f"**{topic}**"
                )

            with col2:

                st.progress(
                    percentage / 100
                )

                st.caption(
                    f"{count} posts"
                )

    else:

        st.info(
            "No hashtags available."
        )

    st.divider()

    st.subheader("💡 Quick Insights")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            f"📱 **Top Platform**\n\n"
            f"{most_active_platform} has the highest number of posts."
        )

    with col2:

        st.info(
            f"⏰ **Peak Activity**\n\n"
            f"Most posts are created around "
            f"{most_active_hour:02d}:00."
        )

    with col3:

        st.info(
            f"🌍 **Top Country**\n\n"
            f"{most_active_country} has the highest number of posts."
        )

    st.success(
        f"🤖 **Machine Learning:** "
        f"Spark ML Logistic Regression achieved "
        f"{ml_accuracy} test accuracy."
    )


# ============================================================
# 2. DATASET OVERVIEW
# ============================================================

elif page == "Dataset Overview":

    st.title("Dataset Overview")

    st.caption(
        "Overview of the social media dataset used in SparkPulse AI"
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Original Rows", "732")

    with col2:
        st.metric("Processed Rows", "712")

    with col3:
        st.metric("Duplicates Removed", "20")

    with col4:
        st.metric("Columns", "13")

    st.divider()

    st.subheader("Dataset Sample")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Dataset Structure")

    structure = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(df[column].dtype)
                for column in df.columns
            ]
        }
    )

    st.dataframe(
        structure,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 3. PLATFORM & COUNTRY INSIGHTS
# ============================================================

elif page == "Platform & Country Insights":

    st.title("Platform & Country Insights")

    st.caption(
        "Understand social media activity across platforms and countries"
    )

    st.divider()

    # Platform color legend
    st.markdown(
        f"""
        **Platform Colors**

        🟣 Instagram `#A7A0C8` &nbsp;&nbsp;
        🔵 Twitter `#91A8C7` &nbsp;&nbsp;
        🟣 Facebook `#B8B1D0`
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📱 Platform Distribution")

        platform_counts = (
            df["Platform"]
            .value_counts()
        )

        platform_colors = []

        for platform in platform_counts.index:

            if platform == "Instagram":
                platform_colors.append(INSTAGRAM)

            elif platform == "Twitter":
                platform_colors.append(TWITTER)

            elif platform == "Facebook":
                platform_colors.append(FACEBOOK)

            else:
                platform_colors.append(PRIMARY)

        fig, ax = plt.subplots()

        platform_counts.plot(
            kind="bar",
            ax=ax,
            color=platform_colors
        )

        ax.set_ylabel(
            "Number of Posts",
            color=TEXT
        )

        ax.set_title(
            "Platform Distribution",
            color=HEADING
        )

        ax.tick_params(
            colors=TEXT
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        st.subheader("🌍 Country Distribution")

        country_counts = (
            df["Country"]
            .value_counts()
            .head(10)
        )

        fig, ax = plt.subplots()

        country_counts.sort_values().plot(
            kind="barh",
            ax=ax,
            color=COUNTRY
        )

        ax.set_xlabel(
            "Number of Posts",
            color=TEXT
        )

        ax.set_title(
            "Top 10 Countries",
            color=HEADING
        )

        ax.tick_params(
            colors=TEXT
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    st.subheader("📈 Platform Engagement")

    platform_table = (
        df.groupby("Platform")
        .agg(
            Posts=("Platform", "count"),
            Average_Likes=("Likes", "mean"),
            Average_Retweets=("Retweets", "mean")
        )
        .reset_index()
    )

    platform_table["Average_Likes"] = (
        platform_table["Average_Likes"]
        .round(2)
    )

    platform_table["Average_Retweets"] = (
        platform_table["Average_Retweets"]
        .round(2)
    )

    fig, ax = plt.subplots()

    platform_table.plot(
        x="Platform",
        y=[
            "Average_Likes",
            "Average_Retweets"
        ],
        kind="bar",
        ax=ax,
        color=[
            INSTAGRAM,
            TWITTER
        ]
    )

    ax.set_ylabel(
        "Average Engagement",
        color=TEXT
    )

    ax.set_title(
        "Platform Engagement",
        color=HEADING
    )

    ax.tick_params(
        colors=TEXT
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Platform Statistics")

    st.dataframe(
        platform_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 4. SENTIMENT & EMOTION ANALYSIS
# ============================================================

elif page == "Sentiment & Emotion Analysis":

    st.title("Sentiment & Emotion Analysis")

    st.caption(
        "Analyze broad sentiment and detailed emotion categories"
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Positive", "349")

    with col2:
        st.metric("Neutral", "193")

    with col3:
        st.metric("Negative", "170")

    st.divider()

    st.subheader("Broad Sentiment")

    sentiment_data = pd.DataFrame(
        {
            "Sentiment": [
                "Positive",
                "Neutral",
                "Negative"
            ],
            "Posts": [
                349,
                193,
                170
            ]
        }
    )

    fig, ax = plt.subplots()

    ax.bar(
        sentiment_data["Sentiment"],
        sentiment_data["Posts"],
        color=[
            POSITIVE,
            NEUTRAL,
            NEGATIVE
        ]
    )

    ax.set_xlabel(
        "Sentiment",
        color=TEXT
    )

    ax.set_ylabel(
        "Number of Posts",
        color=TEXT
    )

    ax.set_title(
        "Positive / Neutral / Negative",
        color=HEADING
    )

    ax.tick_params(
        colors=TEXT
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Top Emotion Categories")

    emotion_counts = (
        df["Sentiment"]
        .value_counts()
        .head(15)
    )

    fig, ax = plt.subplots()

    emotion_counts.sort_values().plot(
        kind="barh",
        ax=ax,
        color=PRIMARY
    )

    ax.set_xlabel(
        "Number of Posts",
        color=TEXT
    )

    ax.set_ylabel(
        "Emotion",
        color=TEXT
    )

    ax.tick_params(
        colors=TEXT
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Detailed Emotion Distribution")

    show_chart(
        "sentiment_distribution.png"
    )


# ============================================================
# 5. TRENDING HASHTAGS
# ============================================================

elif page == "Trending Hashtags":

    st.title("Trending Hashtags")

    st.caption(
        "Identify popular hashtags and social media topics"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Top Hashtag",
            top_hashtag
        )

    with col2:

        st.metric(
            "Usage",
            top_hashtag_count
        )

    st.divider()

    st.subheader("🔥 Top Trending Hashtags")

    if len(hashtag_counts) > 0:

        top_hashtags = hashtag_counts.head(15)

        hashtag_colors = []

        for hashtag in top_hashtags.index:

            if hashtag == top_hashtag:
                hashtag_colors.append(HASHTAG_TOP)
            else:
                hashtag_colors.append(HASHTAG_MAIN)

        fig, ax = plt.subplots()

        top_hashtags.sort_values().plot(
            kind="barh",
            ax=ax,
            color=HASHTAG_MAIN
        )

        ax.set_xlabel(
            "Frequency",
            color=TEXT
        )

        ax.set_ylabel(
            "Hashtag",
            color=TEXT
        )

        ax.set_title(
            "Top 15 Hashtags",
            color=HEADING
        )

        ax.tick_params(
            colors=TEXT
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        hashtag_table = (
            top_hashtags
            .reset_index()
        )

        hashtag_table.columns = [
            "Hashtag",
            "Frequency"
        ]

        st.subheader("Hashtag Frequency")

        st.dataframe(
            hashtag_table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No hashtags found."
        )


# ============================================================
# 6. ACTIVITY & ENGAGEMENT
# ============================================================

elif page == "Activity & Engagement":

    st.title("Activity & Engagement Trends")

    st.caption(
        "Discover when users are most active and engaged"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("⏰ Hourly Activity")

        hourly_counts = (
            df["Hour"]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots()

        hourly_counts.plot(
            kind="bar",
            ax=ax,
            color=ACTIVITY
        )

        ax.set_xlabel(
            "Hour",
            color=TEXT
        )

        ax.set_ylabel(
            "Posts",
            color=TEXT
        )

        ax.set_title(
            "Hourly Activity",
            color=HEADING
        )

        ax.tick_params(
            colors=TEXT
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        st.subheader("📈 Engagement by Hour")

        engagement_hour = (
            df.groupby("Hour")
            .agg(
                Likes=("Likes", "mean"),
                Retweets=("Retweets", "mean")
            )
            .reset_index()
        )

        fig, ax = plt.subplots()

        engagement_hour.plot(
            x="Hour",
            y=[
                "Likes",
                "Retweets"
            ],
            kind="line",
            marker="o",
            ax=ax,
            color=[
                ENGAGEMENT,
                ACTIVITY
            ]
        )

        ax.set_xlabel(
            "Hour",
            color=TEXT
        )

        ax.set_ylabel(
            "Average Engagement",
            color=TEXT
        )

        ax.set_title(
            "Engagement by Hour",
            color=HEADING
        )

        ax.tick_params(
            colors=TEXT
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    st.subheader("📱 Platform Engagement")

    platform_engagement = (
        df.groupby("Platform")
        .agg(
            Likes=("Likes", "mean"),
            Retweets=("Retweets", "mean")
        )
        .reset_index()
    )

    fig, ax = plt.subplots()

    platform_engagement.plot(
        x="Platform",
        y=[
            "Likes",
            "Retweets"
        ],
        kind="bar",
        ax=ax,
        color=[
            ENGAGEMENT,
            ACTIVITY
        ]
    )

    ax.set_ylabel(
        "Average Engagement",
        color=TEXT
    )

    ax.set_title(
        "Platform Engagement",
        color=HEADING
    )

    ax.tick_params(
        colors=TEXT
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    df["Total Engagement"] = (
        df["Likes"].fillna(0)
        +
        df["Retweets"].fillna(0)
    )

    st.subheader("🏆 Top Engagement Posts")

    top_posts = (
        df[
            [
                "Text",
                "Platform",
                "Likes",
                "Retweets",
                "Total Engagement"
            ]
        ]
        .sort_values(
            "Total Engagement",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        top_posts,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 7. ML MODEL PERFORMANCE
# ============================================================

elif page == "ML Model Performance":

    st.title("ML Model Performance")

    st.caption(
        "Spark ML Logistic Regression sentiment classification"
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Algorithm",
            "Logistic Regression"
        )

    with col2:
        st.metric(
            "Training Rows",
            "598"
        )

    with col3:
        st.metric(
            "Testing Rows",
            "114"
        )

    with col4:
        st.metric(
            "Accuracy",
            "64.04%"
        )

    st.divider()

    st.subheader("🔄 Spark ML Pipeline")

    st.write(
        "1. **Text Cleaning** – "
        "Social media text was cleaned and prepared for analysis."
    )

    st.write(
        "2. **Tokenization** – "
        "Text was divided into individual words using Spark ML Tokenizer."
    )

    st.write(
        "3. **TF-IDF Feature Extraction** – "
        "HashingTF and IDF converted text into numerical features."
    )

    st.write(
        "4. **Logistic Regression** – "
        "Spark ML classified the text into three broad sentiment classes."
    )

    st.write(
        "5. **Model Evaluation** – "
        "The model achieved 64.04% test accuracy."
    )

    st.divider()

    st.subheader("Confusion Matrix")

    matrix = [
        [43, 12, 4],
        [7, 15, 7],
        [8, 3, 15]
    ]

    fig, ax = plt.subplots()

    ax.imshow(
        matrix,
        cmap="Purples"
    )

    ax.set_xlabel(
        "Predicted Class",
        color=TEXT
    )

    ax.set_ylabel(
        "Actual Class",
        color=TEXT
    )

    ax.set_title(
        "Spark ML Confusion Matrix",
        color=HEADING
    )

    ax.set_xticks(
        [0, 1, 2]
    )

    ax.set_yticks(
        [0, 1, 2]
    )

    ax.set_xticklabels(
        ["Class 0", "Class 1", "Class 2"],
        color=TEXT
    )

    ax.set_yticklabels(
        ["Class 0", "Class 1", "Class 2"],
        color=TEXT
    )

    for i in range(3):

        for j in range(3):

            ax.text(
                j,
                i,
                matrix[i][j],
                ha="center",
                va="center",
                color=HEADING,
                fontweight="bold"
            )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        "The model uses three broad sentiment classes: "
        "Positive, Neutral and Negative."
    )


# ============================================================
# 8. BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    st.title("Business Insights")

    st.caption(
        "Key findings generated from SparkPulse AI"
    )

    st.divider()

    st.info(
        f"📱 **Platform Insight:** "
        f"{most_active_platform} is the most active platform "
        f"with the highest number of posts."
    )

    st.info(
        f"⏰ **Time Insight:** "
        f"The highest posting activity occurs around "
        f"{most_active_hour:02d}:00."
    )

    st.info(
        f"🌍 **Country Insight:** "
        f"{most_active_country} has the highest number "
        f"of posts in the dataset."
    )

    st.info(
        f"🔥 **Hashtag Insight:** "
        f"{top_hashtag} is the most frequently used hashtag."
    )

    st.info(
        f"🤖 **Machine Learning Insight:** "
        f"The Spark ML Logistic Regression model achieved "
        f"{ml_accuracy} accuracy on the test dataset."
    )

    st.success(
        "💼 **Business Value:** "
        "The project can help businesses understand audience activity, "
        "monitor sentiment, identify trending topics and improve "
        "social media marketing strategies."
    )

    st.divider()

    st.subheader("🔗 Project Workflow")

    st.write(
        "Raw Data → Spark Preprocessing → Spark Analysis → "
        "Spark ML → Trend Detection → Visualization → "
        "Streamlit Dashboard"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SparkPulse AI • MSc Data Science Project • "
    "Apache Spark • Spark ML • Python • Streamlit"
)