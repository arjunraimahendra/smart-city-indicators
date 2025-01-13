import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
from search import read_indicators_file, search_func, format_maturity_levels, create_spider_chart, fetch_indicators_from_web, fetch_indicator, check_for_data

# Streamlit UI

## Title
st.title("Smart City Indicators")

# Initialize session state for city inputs
if "city_inputs" not in st.session_state:
    st.session_state.city_inputs = [1]  # Start with one input field

if "add_city_clicks" not in st.session_state:
    st.session_state.add_city_clicks = 1

if "indicator_bool" not in st.session_state:
    st.session_state.indicator_bool = False

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


# Get the list of cities
city_list = [st.session_state[f"city_{i+1}"] for i in range(len(st.session_state.city_inputs))]
city_list = list(set(city for city in city_list if city))  # Remove empty fields
st.session_state.city_list = city_list

# Get the list of indicators and their categories
combined_df = read_indicators_file()

# Checkbox to dynamically fetch indicators from the web
use_web = st.checkbox("Fetch indicators dynamically from the web")

# Initialize session state
if "ascending" not in st.session_state:
    st.session_state["ascending"] = False  # Default to Bottom 5

col_indicators_1, col_indicators_2, col_indicators_3= st.columns([6, 4, 2])

with col_indicators_1:
    # Unified input for "Category" (either text input or dropdown)
    if use_web:
        selected_category = st.text_input("Category (type the indicator category):")
    else:
        selected_category = st.selectbox("Category (select the indicator category from dropdown):", combined_df["Category"].unique(), index=None)

with col_indicators_2:
    # Create a radio button toggle
    option = st.radio(
        "Choose to display Top 5 or Bottom 5 indicators:",
        ("Top 5 Indicators", "Bottom 5 Indicators")
    )
    # Update session state based on the selection
    st.session_state["ascending"] = option != "Top 5 Indicators"

with col_indicators_3:
    # Add space to align the button to the bottom
    st.markdown("<div style='height: 1.9em;'></div>", unsafe_allow_html=True) 
    indicator_button_clicked = st.button("Get Indicators", use_container_width=True)



# Fetch indicators based on the selected/typed category
if selected_category and indicator_button_clicked:
    st.session_state.indicator_bool = True
    with st.spinner("Generating the Indicator List"):
        indicator_list, maturity_levels_list = fetch_indicators_from_web(category=selected_category)
        filtered_df = pd.DataFrame({
            'Indicator': indicator_list,
            'Category': [selected_category] * len(indicator_list),
            'Maturity Assessment (1-5)': maturity_levels_list
        })
        top_indicators_df = check_for_data(filtered_df, st.session_state.city_list[0], ascending=st.session_state["ascending"])
        st.session_state.final_indicator_list = list(top_indicators_df["Indicator"])
        st.session_state.city0_outputs = list(top_indicators_df["Perplexity Output"])
        st.session_state.city0_maturity_scores = list(top_indicators_df["Maturity Score"])

# Show the list of final indicators
if st.session_state.indicator_bool:
    st.subheader("Indicators:")
    st.markdown("\n\n".join([indicator for indicator in st.session_state.final_indicator_list]))


# Generate the Radar Graph
st.subheader("Generate Radar Graph")
if st.button("Radar Graph"): 
    if st.session_state.city_list and selected_category:
        with st.spinner(f"Generating radar graph for cities: {', '.join(st.session_state.city_list)}, please wait..."):
            combined_results = ""
            radar_data = {}

            combined_results += f"\n\n---\n\n # {st.session_state.city_list[0]}: \n\n"
            for j, indicator in enumerate(st.session_state.final_indicator_list):
                combined_results += f"## {indicator}: \n\n ### Maturity Score: {st.session_state.city0_maturity_scores[j]} \n\n ### Output Text: \n\n {st.session_state.city0_outputs[j]}\n\n\n\n"
            radar_data[st.session_state.city_list[0]] = st.session_state.city0_maturity_scores

            if len(st.session_state.city_list) > 1:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Parallelize over cities
                    futures = [executor.submit(search_func, city=city, indicators=st.session_state.final_indicator_list) for city in st.session_state.city_list[1:]]
                    results = [f.result() for f in futures]

                # Process results for each city
                
                for i, city in enumerate(st.session_state.city_list[1:]):
                    perplexity_results, citations, indicator_values, maturity_scores = results[i]
                    combined_results += f"\n\n---\n\n # {city}: \n\n"
                    for j, indicator in enumerate(st.session_state.final_indicator_list):
                        combined_results += f"## {indicator}: \n\n ### Maturity Score: {maturity_scores[j]} \n\n ### Output Text: \n\n {perplexity_results[j]}\n\n\n\n"
                    radar_data[city] = maturity_scores

        
            combined_results += "\n\n---\n\n"

            # Create a spider chart for the indicators
            create_spider_chart(
                indicators=st.session_state.final_indicator_list,
                values_dict=radar_data,
                title=f"Comparative Radar Chart for {selected_category} for cities: {', '.join(st.session_state.city_list)}",
            )

            # Render the Markdown content
            st.markdown(combined_results)
        st.success("Successfully generated the Radar graph for the cities!")
    else:
        st.warning("Please enter at least one city and select a category.")
