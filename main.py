import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
from search_test import read_indicators_file, search_func, format_maturity_levels, create_spider_chart, fetch_indicators_from_web, fetch_indicator

# Streamlit UI

## Title
st.title("Smart City Indicators")

# Initialize session state for city inputs
if "city_inputs" not in st.session_state:
    st.session_state.city_inputs = [1]  # Start with one input field

if "add_city_clicks" not in st.session_state:
    st.session_state.add_city_clicks = 1

# Function to add a new city input field
def add_city_input():
    st.session_state.add_city_clicks += 1
    if len(st.session_state.city_inputs) < 5:
        st.session_state.city_inputs.append(len(st.session_state.city_inputs) + 1)

# Function to remove a city input field
def remove_city_input():
    st.session_state.add_city_clicks = st.session_state.city_inputs[-1] - 1
    if len(st.session_state.city_inputs) > 1:
        st.session_state.city_inputs.pop()

# Dynamic city input fields
st.subheader("Enter City Names")
for i in range(len(st.session_state.city_inputs)):
    st.text_input(f"City {i+1}:", key=f"city_{i+1}")


# Add and Remove buttons for dynamic fields
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Add City", on_click=add_city_input):
        if st.session_state.add_city_clicks > 5:
            st.warning("Only 5 cities allowed.")

with col2:
    if st.button("Remove City", on_click=remove_city_input):
        pass

# Get the list of indicators and their categories
combined_df = read_indicators_file()

# Option to fetch indicators dynamically
if st.checkbox("Fetch indicators dynamically from the web"):
    # Create the first dropdown for "Category"
    selected_category = st.text_input("Category:")
    indicator_list, maturity_levels_list = fetch_indicators_from_web(category=selected_category)
    # st.write(f"*Fetched indicators:* {'\n - '.join(indicator_list)}")
    filtered_df = pd.DataFrame({'Indicator': indicator_list, 
                                'Category': [selected_category] * len(indicator_list),
                                'Maturity Assessment (1-5)': maturity_levels_list})
    st.markdown("#### Indicators")
    st.markdown("\n\n".join([indicator for indicator in filtered_df["Indicator"].unique()]))
else:
    # Create the first dropdown for "Category" from the excel file
    selected_category = st.selectbox("Category", combined_df["Category"].unique(), index=None)
    # Filter the DataFrame based on the selected category
    filtered_df = combined_df[combined_df["Category"] == selected_category]
    st.markdown("#### Indicators:")
    st.markdown("\n\n".join([indicator for indicator in filtered_df["Indicator"].unique()]))



if st.button("Radar Graph"): 
    city_list = [st.session_state[f"city_{i+1}"] for i in range(len(st.session_state.city_inputs))]
    city_list = list(set(city for city in city_list if city))  # Remove empty fields
    # st.title(", ".join(city_list))
    if city_list and selected_category:
        with st.spinner(f"Generating radar graph for cities: {', '.join(city_list)}, please wait..."):
            combined_results = ""
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Parallelize over cities
                futures = [executor.submit(search_func, city=city, indicators=filtered_df["Indicator"].unique()) for city in city_list]
                results = [f.result() for f in futures]

            # Process results for each city
            radar_data = {}
            for i, city in enumerate(city_list):
                # st.title(f"{city}")
                perplexity_results, citations, indicator_values, maturity_scores = results[i]
                combined_results += f"\n\n---\n\n # {city}: \n\n"
                for j, indicator in enumerate(filtered_df["Indicator"].unique()):
                    combined_results += f"## {indicator}:  \n\n ### Indicator Value: {indicator_values[j]} \n\n ### Maturity Score: {maturity_scores[j]} \n\n ### Output Text: \n\n {perplexity_results[j]}\n\n\n\n"
                radar_data[city] = maturity_scores
            combined_results += "\n\n---\n\n"

            # Create a spider chart for the indicators
            create_spider_chart(
                indicators=filtered_df["Indicator"].unique(),
                values_dict=radar_data,
                title=f"Comparative Radar Chart for {selected_category} for cities: {', '.join(city_list)}",
            )

            # Render the Markdown content
            st.markdown(combined_results)
        st.success("Radar graph generated for both cities successfully!")
    else:
        st.warning("Please enter at least one city and select a category.")
