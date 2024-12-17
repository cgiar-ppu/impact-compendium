import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
import plotly.express as px
import streamlit.components.v1 as components

# Load the model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Add a logo and title side by side
col_image, col_title = st.columns([1, 2])  # Adjust the ratio as needed

with col_image:
    st.image("static/CGIAR-logo.png", width=150)  # Adjust the width as needed

with col_title:
    # Use CSS to center the title vertically
    st.markdown(
        """
        <style>
        .centered-title {
            display: flex;
            align-items: center;
            height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="centered-title"><h1>IMPACT COMPENDIUM SEARCH</h1></div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_excel(
            "Evaluative studies (impact studies+evaluations) TAGGING LR.xlsx",
            sheet_name="Impact studies",
        )
    except Exception as e:
        st.write(f"Error reading Excel file: {e}")
        df = pd.DataFrame()  # Empty DataFrame
    return df

df = load_data()

# Initialize session state variables
if "selected_countries" not in st.session_state:
    st.session_state.selected_countries = ["All countries"]
if "selected_crops" not in st.session_state:
    st.session_state.selected_crops = ["All crop/stock types"]
if "search_performed" not in st.session_state:
    st.session_state.search_performed = False
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "cosine_scores" not in st.session_state:
    st.session_state.cosine_scores = None
if "titles" not in st.session_state:
    st.session_state.titles = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.35

def get_filtered_options(df, selected_countries, selected_crops):
    temp_df = df.copy()
    if selected_crops and "All crop/stock types" not in selected_crops:
        temp_df = temp_df[temp_df["Crop/stock type"].isin(selected_crops)]
    country_options = ["All countries"] + sorted(temp_df["Country"].dropna().unique())

    temp_df = df.copy()
    if selected_countries and "All countries" not in selected_countries:
        temp_df = temp_df[temp_df["Country"].isin(selected_countries)]
    crop_options = ["All crop/stock types"] + sorted(temp_df["Crop/stock type"].dropna().unique())

    return country_options, crop_options

country_options, crop_options = get_filtered_options(
    df, st.session_state.selected_countries, st.session_state.selected_crops
)

search_term = st.text_input("Search Term", placeholder="Used for semantic similarity search...")

col1, col2 = st.columns(2)
with col1:
    selected_countries = st.multiselect(
        "Countries",
        options=country_options,
        default=st.session_state.selected_countries,
        key="countries_filter",
    )
with col2:
    selected_crops = st.multiselect(
        "Crops",
        options=crop_options,
        default=st.session_state.selected_crops,
        key="crops_filter",
    )

st.session_state.selected_countries = selected_countries
st.session_state.selected_crops = selected_crops

filtered_df = df.copy()

if selected_countries and "All countries" not in selected_countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]

if selected_crops and "All crop/stock types" not in selected_crops:
    filtered_df = filtered_df[filtered_df["Crop/stock type"].isin(selected_crops)]

st.session_state.filtered_df = filtered_df

similarity_threshold = st.slider(
    "Search Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.get("similarity_threshold", 0.35),
    step=0.01
)

search_button_disabled = not search_term
col_search, col_summarize = st.columns(2)
with col_search:
    search_clicked = st.button("Search", disabled=search_button_disabled)

if search_clicked:
    st.session_state.similarity_threshold = similarity_threshold

    search_embedding = model.encode(search_term, convert_to_tensor=True)
    titles = st.session_state.filtered_df['Title'].fillna('').tolist()
    title_embeddings = model.encode(titles, convert_to_tensor=True)
    cosine_scores = util.cos_sim(search_embedding, title_embeddings)[0].cpu().numpy()

    st.session_state['cosine_scores'] = cosine_scores
    st.session_state['titles'] = titles
    st.session_state.search_performed = True
    st.session_state.summary = None

    cosine_scores = st.session_state['cosine_scores']
    titles = st.session_state['titles']
    filtered_df = st.session_state.filtered_df.copy()
    filtered_df['Similarity'] = cosine_scores
    result_df = filtered_df[filtered_df['Similarity'] >= st.session_state.similarity_threshold]
    result_df = result_df.sort_values(by='Similarity', ascending=False)

    columns = result_df.columns.tolist()
    if 'Title' in columns:
        columns.remove('Title')
        columns.insert(1, 'Title')
        result_df = result_df[columns]

    st.session_state.search_results = result_df

    if result_df.empty:
        st.write("No results found matching your query. Consider using a different search term or lowering the similarity threshold.")
    else:
        st.header("Search Results")
        st.write(result_df)

        countries = result_df['Country'].dropna().unique().tolist()
        if countries:
            map_df = pd.DataFrame({'country': countries, 'value': 1})
            fig = px.choropleth(
                map_df,
                locations='country',
                locationmode='country names',
                color='value',
                color_continuous_scale='Blues',
                title='Countries Present in Results'
            )
            st.plotly_chart(fig)

        # Build a carousel with HTML/CSS/JS
        # Extract values for the carousel
        items = []
        for i, row in result_df.iterrows():
            title_val = row.get("Title", "N/A")
            nrm_val = row.get("Natural resource management", "N/A")
            author_val = row.get("Author, date", "N/A")
            item_html = f"""
            <div class="carousel-item">
                <h3>{title_val}</h3>
                <p><strong>Natural resource management:</strong> {nrm_val}</p>
                <p><strong>Author, date:</strong> {author_val}</p>
            </div>
            """
            items.append(item_html)

        items_html = "\n".join(items)
        total_items = len(items)

        # IMPORTANT: Do not use f-string here, just a normal triple-quoted string
        # to avoid Python interpreting JavaScript template literals.
        carousel_html = """
        <style>
        .carousel-container {
            position: relative;
            width: 100%;
            max-width: 600px;
            margin: 0 auto;
            overflow: hidden;
            background: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
        }
        .carousel-wrapper {
            display: flex;
            transition: transform 0.5s ease;
        }
        .carousel-item {
            min-width: 100%;
            box-sizing: border-box;
            padding: 20px;
        }
        .carousel-controls {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
        }
        .carousel-controls button {
            background: #007bff;
            border: none;
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
        }
        .carousel-controls button:disabled {
            background: #cccccc;
            cursor: not-allowed;
        }
        .carousel-indicator {
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
        }
        </style>

        <div class="carousel-container">
            <div class="carousel-wrapper" id="carousel-wrapper">
                <!--ITEMS_PLACEHOLDER-->
            </div>
            <div class="carousel-controls">
                <button id="prev-btn">Previous</button>
                <button id="next-btn">Next</button>
            </div>
            <div class="carousel-indicator" id="carousel-indicator">1 of TOTAL_PLACEHOLDER</div>
        </div>

        <script>
        const wrapper = document.getElementById('carousel-wrapper');
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const indicator = document.getElementById('carousel-indicator');
        
        let currentIndex = 0;
        const total = TOTAL_PLACEHOLDER;

        function updateCarousel() {
            wrapper.style.transform = "translateX(-" + (currentIndex * 100) + "%)";
            indicator.textContent = (currentIndex + 1) + " of " + total;
            prevBtn.disabled = currentIndex === 0;
            nextBtn.disabled = currentIndex === total - 1;
        }

        prevBtn.addEventListener('click', () => {
            if (currentIndex > 0) {
                currentIndex--;
                updateCarousel();
            }
        });

        nextBtn.addEventListener('click', () => {
            if (currentIndex < total - 1) {
                currentIndex++;
                updateCarousel();
            }
        });

        updateCarousel();
        </script>
        """

        # Replace placeholders with actual HTML and counts
        carousel_html = carousel_html.replace("<!--ITEMS_PLACEHOLDER-->", items_html)
        carousel_html = carousel_html.replace("TOTAL_PLACEHOLDER", str(total_items))

        components.html(carousel_html, height=400, scrolling=False)

elif st.session_state.get('search_performed', False) and st.session_state.get('search_results') is not None:
    result_df = st.session_state.search_results
    if result_df.empty:
        st.write("No results found matching your query. Consider using a different search term or lowering the similarity threshold.")
    else:
        columns = result_df.columns.tolist()
        if 'Title' in columns:
            columns.remove('Title')
            columns.insert(1, 'Title')
            result_df = result_df[columns]
        st.header("Search Results")
        st.write(result_df)

        countries = result_df['Country'].dropna().unique().tolist()
        if countries:
            map_df = pd.DataFrame({'country': countries, 'value': 1})
            fig = px.choropleth(
                map_df,
                locations='country',
                locationmode='country names',
                color='value',
                color_continuous_scale='Blues',
                title='Countries Present in Results'
            )
            st.plotly_chart(fig)

        # Rebuild the carousel similarly
        items = []
        for i, row in result_df.iterrows():
            title_val = row.get("Title", "N/A")
            nrm_val = row.get("Natural resource management", "N/A")
            author_val = row.get("Author, date", "N/A")
            item_html = f"""
            <div class="carousel-item">
                <h3>{title_val}</h3>
                <p><strong>Natural resource management:</strong> {nrm_val}</p>
                <p><strong>Author, date:</strong> {author_val}</p>
            </div>
            """
            items.append(item_html)

        items_html = "\n".join(items)
        total_items = len(items)

        carousel_html = """
        <style>
        .carousel-container {
            position: relative;
            width: 100%;
            max-width: 600px;
            margin: 0 auto;
            overflow: hidden;
            background: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
        }
        .carousel-wrapper {
            display: flex;
            transition: transform 0.5s ease;
        }
        .carousel-item {
            min-width: 100%;
            box-sizing: border-box;
            padding: 20px;
        }
        .carousel-controls {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
        }
        .carousel-controls button {
            background: #007bff;
            border: none;
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
        }
        .carousel-controls button:disabled {
            background: #cccccc;
            cursor: not-allowed;
        }
        .carousel-indicator {
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
        }
        </style>

        <div class="carousel-container">
            <div class="carousel-wrapper" id="carousel-wrapper">
                <!--ITEMS_PLACEHOLDER-->
            </div>
            <div class="carousel-controls">
                <button id="prev-btn">Previous</button>
                <button id="next-btn">Next</button>
            </div>
            <div class="carousel-indicator" id="carousel-indicator">1 of TOTAL_PLACEHOLDER</div>
        </div>

        <script>
        const wrapper = document.getElementById('carousel-wrapper');
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const indicator = document.getElementById('carousel-indicator');
        
        let currentIndex = 0;
        const total = TOTAL_PLACEHOLDER;

        function updateCarousel() {
            wrapper.style.transform = "translateX(-" + (currentIndex * 100) + "%)";
            indicator.textContent = (currentIndex + 1) + " of " + total;
            prevBtn.disabled = currentIndex === 0;
            nextBtn.disabled = currentIndex === total - 1;
        }

        prevBtn.addEventListener('click', () => {
            if (currentIndex > 0) {
                currentIndex--;
                updateCarousel();
            }
        });

        nextBtn.addEventListener('click', () => {
            if (currentIndex < total - 1) {
                currentIndex++;
                updateCarousel();
            }
        });

        updateCarousel();
        </script>
        """

        carousel_html = carousel_html.replace("<!--ITEMS_PLACEHOLDER-->", items_html)
        carousel_html = carousel_html.replace("TOTAL_PLACEHOLDER", str(total_items))

        components.html(carousel_html, height=400, scrolling=False)

summarize_button_disabled = True
if st.session_state.get('search_performed', False) and st.session_state.get('search_results') is not None:
    if not st.session_state.search_results.empty:
        summarize_button_disabled = False

with col_summarize:
    summarize_clicked = st.button("Summarize", disabled=summarize_button_disabled)

if summarize_clicked:
    result_df = st.session_state.search_results
    titles = result_df['Title'].tolist()
    if len(titles) == 0:
        st.write("No titles available to summarize. Please perform a search that returns results.")
    else:
        system_prompt = """
You are an expert summarizer skilled in creating concise and relevant summaries of given titles. Your goal is to produce summaries that align with the specific objectives provided.

Guidelines:
1. **Clarity and Precision**: Ensure the summary is clear, precise, and easy to understand.
2. **Objective Alignment**: Tailor the summary to address any specific objectives provided.
3. **Coherence and Flow**: Maintain a logical flow and coherence in the summary.
4. **Conciseness**: Include only the most relevant information.
"""

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        else:
            try:
                llm = ChatOpenAI(api_key=openai_api_key, model_name='gpt-4o')
                all_titles = "\n".join(titles)
                user_prompt = f"Please provide a summary using a minimum of 1 paragraph and a maximum of 3 paragraphs, without mentioning the specific number of texts or details, since it needs to be a summary of all of the activities as a whole showcasing the impact generated in a narrative, worded in a way that describes the work performed and achievements of the text of the following titles:\n{all_titles}"
                system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                chain = LLMChain(llm=llm, prompt=chat_prompt)
                response = chain.run(user_prompt=user_prompt)
                summary = response.strip()
                st.header("Summarization Results")
                st.write(summary)
                st.session_state.summary = summary
            except Exception as e:
                st.write(f"An error occurred: {e}")
elif st.session_state.get('summary') is not None:
    st.header("Summarization Results")
    st.write(st.session_state.summary)
