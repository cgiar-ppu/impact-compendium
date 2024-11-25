import streamlit as st

# Title of the application
st.title("IMPACT COMPENDIUM")

# Search Bar
st.text_input("Search", placeholder="Type to search...")

# Dropdown menus for filters
col1, col2 = st.columns(2)
with col1:
    st.selectbox("Countries", options=["All countries", "Country 1", "Country 2"], key="countries_filter")
with col2:
    st.selectbox("Crops", options=["All crops", "Crop 1", "Crop 2"], key="crops_filter")

# Filter button
st.button("Filter on Impact Areas (summary)")

# Divider
st.write("---")

# Placeholder for search results title
st.subheader("SEARCH RESULTS TITLE")

# Placeholder for AI-made summary
st.text_area(
    "AI-made summary from all related resources",
    placeholder="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin non ligula quis magna fringilla pulvinar vitae eu magna...",
    height=200,
)

# Placeholder for content body
st.text_area(
    "Additional Information",
    placeholder=(
        "Duis a faucibus risus, porttitor consectetur dui. Aenean nisi ex, tempus in pharetra sed, lobortis a erat. "
        "Aenean sed ex at metus lobortis ullamcorper sed ut mauris..."
    ),
    height=400,
)

# Charts/graphs/numbers section
st.write("Charts / graphs / numbers")
st.line_chart([1, 2, 3, 4, 5])  # Placeholder for chart

# Footer or additional text
st.write("Footer information or additional text.")