import os
import pickle
import html
import gradio as gr
import numpy as np


# =========================================================
# 1. LOAD MOVIE DATA
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "movies.pkl"), "rb") as file:
    movies = pickle.load(file)

# Load the similarity matrix:
# - On Render, download it from the private Hugging Face dataset using HF_TOKEN.
# - Locally, use similarity.pkl from the project directory as a fallback.
HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    from huggingface_hub import hf_hub_download

    print("Downloading similarity matrix from Hugging Face...")

    similarity_path = hf_hub_download(
        repo_id="Bitsemmbler/movie-recommendation-assets",
        filename="similarity.pkl",
        repo_type="dataset",
        token=HF_TOKEN
    )
else:
    similarity_path = os.path.join(BASE_DIR, "similarity.pkl")

    if not os.path.isfile(similarity_path):
        raise RuntimeError(
            "Similarity matrix not found. Set the HF_TOKEN environment "
            "variable to download it from the private Hugging Face dataset, "
            "or place similarity.pkl in the project directory."
        )

with open(similarity_path, "rb") as file:
    similarity = pickle.load(file)

similarity = np.asarray(similarity)

if "title" not in movies.columns:
    raise ValueError("The movies DataFrame must contain a title column.")

if similarity.shape != (len(movies), len(movies)):
    raise ValueError(
        "Movie data and similarity matrix dimensions do not match."
    )

movie_choices = movies["title"].dropna().unique().tolist()

print("CineMatch data loaded successfully!")
print("Total movies:", len(movies))
print("Similarity matrix:", similarity.shape)


# =========================================================
# 2. MOVIE RECOMMENDATION FUNCTION
# =========================================================

def recommend(movie):

    if not movie:
        return "### Please select a movie to continue."

    print("\nSelected movie:", movie)

    # Find the selected movie's position
    matches = np.flatnonzero(
        movies["title"].eq(movie).to_numpy()
    )

    if len(matches) == 0:
        return "### Movie not found."

    movie_index = int(matches[0])

    # Get similarity scores
    distances = similarity[movie_index]

    # Sort by similarity score
    sorted_indices = np.argsort(distances)[::-1]

    recommendations = []

    for index in sorted_indices:

        if index == movie_index:
            continue

        title = movies.iloc[index]["title"]

        if not isinstance(title, str):
            continue

        recommendations.append(title)

        if len(recommendations) == 5:
            break

    if not recommendations:
        return "### No recommendations available."

    # Generate styled recommendation cards
    result = """
## Your movie picks

Here are five movies selected based on your choice.
"""

    for rank, title in enumerate(recommendations, start=1):

        safe_title = html.escape(title)

        result += f"""
<div class="movie-result">
    <div class="movie-number">{rank:02d}</div>
    <div class="movie-title">{safe_title}</div>
    <div class="movie-arrow">↗</div>
</div>
"""

    print("Recommendations:", recommendations)

    return result


# =========================================================
# 3. RESET FUNCTION
# =========================================================

def clear_selection():

    return (
        None,
        """
## Ready to explore?

Your recommendations will appear here.

Choose a movie and click **Discover Movies** to get started.
"""
    )


# =========================================================
# 4. CUSTOM CSS
# =========================================================

CUSTOM_CSS = """

/* MAIN APPLICATION */

.gradio-container {
    background: #090d18 !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

#cinematch-app {
    max-width: 1180px !important;
    margin: auto !important;
    padding: 28px 24px 45px !important;
}


/* HERO SECTION */

#hero-section {
    background:
        radial-gradient(
            ellipse at 85% 20%,
            rgba(124, 92, 255, 0.25),
            transparent 45%
        ),
        linear-gradient(
            120deg,
            #17152f 0%,
            #11172b 55%,
            #101b32 100%
        );

    border: 1px solid rgba(167, 139, 250, 0.23);
    border-radius: 24px;
    padding: 42px 44px;
    margin-bottom: 25px;
    overflow: hidden;
}

.brand {
    color: #a78bfa;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 18px;
}

.hero-title {
    font-size: clamp(32px, 5vw, 48px);
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #ffffff;
    margin: 0 0 15px 0;
}

.hero-gradient {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-description {
    color: #aebbd1;
    font-size: 16px;
    line-height: 1.8;
    max-width: 650px;
    margin-bottom: 24px;
}

.hero-tag {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 30px;
    background: rgba(167, 139, 250, 0.13);
    border: 1px solid rgba(167, 139, 250, 0.3);
    color: #c4b5fd;
    font-size: 12px;
    font-weight: 600;
}


/* STATISTICS CARDS */

.stat-card {
    background: linear-gradient(145deg, #151d30, #111827);
    border: 1px solid #27334a;
    border-radius: 17px;
    padding: 22px 24px;
    min-height: 105px;
    transition: 0.25s ease;
}

.stat-card:hover {
    border-color: #8170e8;
    transform: translateY(-3px);
}

.stat-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.stat-value {
    font-size: 29px;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 9px;
}


/* MAIN PANELS */

#selection-panel,
#results-panel {
    background: #121a2a !important;
    border: 1px solid #29354b !important;
    border-radius: 20px !important;
    padding: 26px !important;
    min-height: 330px;
}

.panel-heading {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 750;
    margin: 0 0 8px 0;
}

.panel-subheading {
    color: #94a3b8;
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 22px;
}


/* SEARCH LABEL - CENTERED */

#search-label {
    display: block !important;
    width: 100% !important;
    text-align: center !important;
    color: #cbd5e1 !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    margin-top: 8px;
    margin-bottom: 10px;
}


/* MOVIE DROPDOWN */

#movie-dropdown {
    width: 100% !important;
    margin-bottom: 0 !important;
}

#movie-dropdown input {
    text-align: center !important;
    color: #ffffff !important;
    background: #1d293d !important;
    border: 1px solid #35445f !important;
    border-radius: 11px !important;
    font-size: 15px !important;
}

#movie-dropdown [role="combobox"] {
    background: #1d293d !important;
    border: 1px solid #35445f !important;
    border-radius: 11px !important;
    color: #ffffff !important;
}

#movie-dropdown [role="option"] {
    text-align: center !important;
}


/* DISCOVER BUTTON */

#recommend-button {
    width: 100% !important;
    margin-top: 17px !important;
}

#recommend-button button {
    width: 100% !important;
    min-height: 52px !important;

    display: flex !important;
    justify-content: center !important;
    align-items: center !important;

    background: linear-gradient(100deg, #7558e8, #5c68eb) !important;
    border: none !important;
    border-radius: 11px !important;
    color: white !important;
    font-weight: 750 !important;
    transition: all 0.25s ease !important;
}

#recommend-button button:hover {
    background: linear-gradient(100deg, #886df7, #7180ff) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.25);
    transform: translateY(-2px);
}


/* RESET BUTTON */

#clear-button {
    width: 100% !important;
}

#clear-button button {
    width: 100% !important;
    min-height: 45px !important;

    display: flex !important;
    justify-content: center !important;
    align-items: center !important;

    background: #202b3e !important;
    border: 1px solid #35445f !important;
    border-radius: 11px !important;
    color: #cbd5e1 !important;
    font-weight: 650 !important;
}

#clear-button button:hover {
    background: #29364d !important;
}


/* RECOMMENDATION RESULTS */

#recommendation-output {
    background: transparent !important;
    border: none !important;
}

#recommendation-output h2 {
    color: #f8fafc !important;
    font-size: 21px !important;
    margin-top: 0 !important;
    margin-bottom: 7px !important;
}

#recommendation-output p {
    color: #94a3b8 !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
}

.movie-result {
    display: flex;
    align-items: center;
    gap: 17px;
    background: #1a2639;
    border: 1px solid #2c3b53;
    border-radius: 12px;
    padding: 16px 17px;
    margin: 12px 0;
    transition: all 0.2s ease;
}

.movie-result:hover {
    background: #222f47;
    border-color: #8b79ed;
    transform: translateX(4px);
}

.movie-number {
    width: 36px;
    height: 36px;
    min-width: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(139, 124, 246, 0.16);
    color: #b9aaff;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 800;
}

.movie-title {
    flex: 1;
    color: #f1f5f9;
    font-size: 15px;
    font-weight: 650;
    overflow-wrap: anywhere;
}

.movie-arrow {
    color: #8b79ed;
    font-size: 20px;
}


/* FOOTER */

#footer-section {
    text-align: center;
    padding: 22px 10px 5px;
    color: #64748b;
    font-size: 12px;
    border-top: 1px solid #263247;
    margin-top: 24px;
}


/* RESPONSIVE DESIGN */

@media (max-width: 700px) {

    #cinematch-app {
        padding: 15px 12px 30px !important;
    }

    #hero-section {
        padding: 28px 22px;
        border-radius: 17px;
    }

    .hero-title {
        font-size: 32px;
    }

    #selection-panel,
    #results-panel {
        padding: 19px !important;
        min-height: auto;
    }

    .stat-card {
        padding: 17px;
    }

    .stat-value {
        font-size: 24px;
    }
}

"""


# =========================================================
# 5. BUILD CINEMATCH INTERFACE
# =========================================================

with gr.Blocks(
    title="CineMatch | Discover Movies"
) as demo:

    with gr.Column(elem_id="cinematch-app"):

        # HERO SECTION

        gr.HTML("""
        <div id="hero-section">

            <div class="brand">🎬 CINEMATCH</div>

            <h1 class="hero-title">
                Discover your next
                <span class="hero-gradient">
                    favourite movie.
                </span>
            </h1>

            <div class="hero-description">
                Not sure what to watch next?
                Discover movies tailored to your taste
                using intelligent content-based recommendations.
            </div>

            <span class="hero-tag">
                ✦ AI-POWERED MOVIE DISCOVERY
            </span>

        </div>
        """)

        # STATISTICS

        with gr.Row():

            gr.HTML(f"""
            <div class="stat-card">
                <div class="stat-label">Movie Collection</div>
                <div class="stat-value">🎞️ {len(movies):,}</div>
            </div>
            """)

            gr.HTML("""
            <div class="stat-card">
                <div class="stat-label">Recommendations</div>
                <div class="stat-value">⭐ 05</div>
            </div>
            """)

            gr.HTML("""
            <div class="stat-card">
                <div class="stat-label">Recommendation Engine</div>
                <div class="stat-value">✦ Smart</div>
            </div>
            """)

        # MAIN CONTENT

        with gr.Row(equal_height=True):

            # MOVIE SELECTION PANEL

            with gr.Column(
                scale=1,
                elem_id="selection-panel"
            ):

                gr.HTML("""
                <div class="panel-heading">
                    🎥 Find a Movie
                </div>

                <div class="panel-subheading">
                    Select a movie you already love.
                    We'll find five similar movies for you.
                </div>
                """)

                # Custom centered label

                gr.HTML("""
                <div id="search-label">
                    Search Movie Title
                </div>
                """)

                # Dropdown without default label

                movie_dropdown = gr.Dropdown(
                    choices=movie_choices,
                    show_label=False,
                    interactive=True,
                    elem_id="movie-dropdown"
                )

                # Discover button

                recommend_button = gr.Button(
                    "✦  Discover Movies",
                    variant="primary",
                    size="lg",
                    elem_id="recommend-button"
                )

                # Reset button

                clear_button = gr.Button(
                    "↻  Reset Selection",
                    variant="secondary",
                    elem_id="clear-button"
                )

                gr.HTML("""
                <div style="
                    color:#64748b;
                    font-size:12px;
                    margin-top:15px;
                    line-height:1.7;
                    text-align:center;
                ">
                    <b style="color:#94a3b8;">TIP</b><br>
                    Search for your favourite movie
                    by typing its title in the dropdown.
                </div>
                """)

            # RECOMMENDATION PANEL

            with gr.Column(
                scale=1,
                elem_id="results-panel"
            ):

                gr.HTML("""
                <div class="panel-heading">
                    ✨ Your Recommendations
                </div>

                <div class="panel-subheading">
                    Five suggestions based on movie similarity.
                </div>
                """)

                recommendation_output = gr.Markdown(
                    value="""
## Ready to explore?

Your recommendations will appear here.

Choose a movie on the left and click
**Discover Movies** to get started.
""",
                    elem_id="recommendation-output"
                )

        # FOOTER

        gr.HTML("""
        <div id="footer-section">
            <div style="
                font-weight:700;
                color:#94a3b8;
                margin-bottom:6px;
            ">
                🎬 CineMatch
            </div>

            Movie Recommendation System
            <br>
            Powered by Content-Based Filtering
            & Cosine Similarity
        </div>
        """)

        # BUTTON EVENTS

        recommend_button.click(
            fn=recommend,
            inputs=movie_dropdown,
            outputs=recommendation_output
        )

        clear_button.click(
            fn=clear_selection,
            inputs=None,
            outputs=[
                movie_dropdown,
                recommendation_output
            ]
        )


# =========================================================
# 6. LAUNCH APPLICATION - GRADIO 6 COMPATIBLE
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 3000))

    print(f"\nStarting CineMatch on port {port}...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,

        # Gradio 6: theme and CSS belong in launch()
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate"
        ),

        css=CUSTOM_CSS
    )