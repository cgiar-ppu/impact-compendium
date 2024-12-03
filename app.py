import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import openai  # Import OpenAI module
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

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

# Read the Excel file
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
# Initialize variables to store embeddings and cosine scores
if "cosine_scores" not in st.session_state:
    st.session_state.cosine_scores = None
if "titles" not in st.session_state:
    st.session_state.titles = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.35  # Default value

# Function to get available options based on current selections
def get_filtered_options(df, selected_countries, selected_crops):
    temp_df = df.copy()

    # Apply crop filter to get available countries
    if selected_crops and "All crop/stock types" not in selected_crops:
        temp_df = temp_df[temp_df["Crop/stock type"].isin(selected_crops)]
    country_options = ["All countries"] + sorted(temp_df["Country"].dropna().unique())

    # Reset temp_df to original df
    temp_df = df.copy()

    # Apply country filter to get available crops
    if selected_countries and "All countries" not in selected_countries:
        temp_df = temp_df[temp_df["Country"].isin(selected_countries)]
    crop_options = ["All crop/stock types"] + sorted(temp_df["Crop/stock type"].dropna().unique())

    return country_options, crop_options

# Get available options based on current selections
country_options, crop_options = get_filtered_options(
    df, st.session_state.selected_countries, st.session_state.selected_crops
)

# Search Bar
search_term = st.text_input("Search Term", placeholder="Used for semantic similarity search...")

# Dropdown menus for filters
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

# Update session state
st.session_state.selected_countries = selected_countries
st.session_state.selected_crops = selected_crops

# Filter the DataFrame based on selections
filtered_df = df.copy()

if selected_countries and "All countries" not in selected_countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]

if selected_crops and "All crop/stock types" not in selected_crops:
    filtered_df = filtered_df[filtered_df["Crop/stock type"].isin(selected_crops)]

# Save filtered_df in session state
st.session_state.filtered_df = filtered_df

# Similarity Threshold Slider
similarity_threshold = st.slider(
    "Search Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.get("similarity_threshold", 0.35),
    step=0.01
)

# Define the disabled state for the Search button
search_button_disabled = not search_term

# Create two columns for the buttons
col_search, col_summarize = st.columns(2)

with col_search:
    # Search Button
    search_clicked = st.button("Search", disabled=search_button_disabled)

# Search Functionality
if search_clicked:
    # Store the similarity threshold at the time of search
    st.session_state.similarity_threshold = similarity_threshold

    # Perform semantic similarity search
    # Encode the search term and the titles
    search_embedding = model.encode(search_term, convert_to_tensor=True)
    titles = st.session_state.filtered_df['Title'].fillna('').tolist()
    title_embeddings = model.encode(titles, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(search_embedding, title_embeddings)[0].cpu().numpy()

    # Store embeddings and scores in session state
    st.session_state['cosine_scores'] = cosine_scores
    st.session_state['titles'] = titles

    # Update search_performed flag
    st.session_state.search_performed = True

    # Clear any existing summary
    st.session_state.summary = None

    # Filter and display results based on the stored similarity threshold
    cosine_scores = st.session_state['cosine_scores']
    titles = st.session_state['titles']
    filtered_df = st.session_state.filtered_df.copy()

    # Attach scores to the DataFrame
    filtered_df['Similarity'] = cosine_scores

    # Filter DataFrame by similarity threshold stored in session_state
    result_df = filtered_df[filtered_df['Similarity'] >= st.session_state.similarity_threshold]

    # Sort DataFrame by similarity scores
    result_df = result_df.sort_values(by='Similarity', ascending=False)

    # Rearrange columns so that 'Title' is in the 2nd position
    columns = result_df.columns.tolist()
    if 'Title' in columns:
        columns.remove('Title')
        columns.insert(1, 'Title')
        result_df = result_df[columns]

    # Store the result_df in session state
    st.session_state.search_results = result_df

    # Display the results
    if result_df.empty:
        st.write("No results found matching your query. Consider using a different search term or lowering the similarity threshold.")
    else:
        st.header("Search Results")
        st.write(result_df)

# Now, if search has been performed, display the stored results
elif st.session_state.get('search_performed', False) and st.session_state.get('search_results') is not None:
    result_df = st.session_state.search_results

    if result_df.empty:
        st.write("No results found matching your query. Consider using a different search term or lowering the similarity threshold.")
    else:
        # Rearrange columns so that 'Title' is in the 2nd position
        columns = result_df.columns.tolist()
        if 'Title' in columns:
            columns.remove('Title')
            columns.insert(1, 'Title')
            result_df = result_df[columns]
        st.header("Search Results")
        st.write(result_df)

# Now, calculate the disabled state for the Summarize button
summarize_button_disabled = True
if st.session_state.get('search_performed', False) and st.session_state.get('search_results') is not None:
    if not st.session_state.search_results.empty:
        summarize_button_disabled = False

with col_summarize:
    # Summarize Button
    summarize_clicked = st.button("Summarize", disabled=summarize_button_disabled)

# Summarization Functionality
if summarize_clicked:
    result_df = st.session_state.search_results
    titles = result_df['Title'].tolist()
    if len(titles) == 0:
        st.write("No titles available to summarize. Please perform a search that returns results.")
    else:
        # Define system prompt
        system_prompt = """
You are an expert summarizer skilled in creating concise and relevant summaries of given titles. Your goal is to produce summaries that align with the specific objectives provided.

Guidelines:
1. **Clarity and Precision**: Ensure the summary is clear, precise, and easy to understand.
2. **Objective Alignment**: Tailor the summary to address any specific objectives provided.
3. **Coherence and Flow**: Maintain a logical flow and coherence in the summary.
4. **Conciseness**: Include only the most relevant information.
"""

        # Initialize LLM
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        else:
            try:
                llm = ChatOpenAI(api_key=openai_api_key, model_name='gpt-4o')
                # Prepare prompts
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
                # Store the summary in session state
                st.session_state.summary = summary
            except Exception as e:
                st.write(f"An error occurred: {e}")
elif st.session_state.get('summary') is not None:
    # If the summary exists in session state, display it
    st.header("Summarization Results")
    st.write(st.session_state.summary)