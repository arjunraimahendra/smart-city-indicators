import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from search import read_indicators_file, search_func, format_maturity_levels, create_spider_chart, fetch_indicators_from_web

# Streamlit UI

## Title
st.title("Smart City Indicators")

# City inputs side by side
col1, col2 = st.columns(2)

with col1:
    city1 = st.text_input("City 1:")

with col2:
    city2 = st.text_input("City 2:")


# # Type the city
# city = st.text_input("Enter a city: ")

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
else:
    # Create the first dropdown for "Category" from the excel file
    selected_category = st.selectbox("Category", combined_df["Category"].unique())
    # Filter the DataFrame based on the selected category
    filtered_df = combined_df[combined_df["Category"] == selected_category]

# Create the second dropdown for "Indicator"
selected_indicator = st.selectbox("Indicator", filtered_df["Indicator"].unique())

# Add a submit button
if st.button("Submit"):
    if city1 and city2 and selected_category and selected_indicator:
        with st.spinner(f"\n\n ### Generating Information for Category: `{selected_category}` and Indicator: `{selected_indicator}` for `{city1}` and `{city2}`, please wait..."):
            combined_results = ""
            for city in [city1, city2]:
                maturity_scale_str = filtered_df[filtered_df["Indicator"] == selected_indicator]["Maturity Assessment (1-5)"].iloc[0]

                # Call the search function
                search_results = search_func(
                    city=city, 
                    indicator=selected_indicator, 
                    maturity_scale=maturity_scale_str
                )
                combined_results += f"\n\n---\n\n # {city}: \n\n ## {selected_indicator}: \n\n ### Indicator Value: {search_results[2]} \n\n ### Maturity Score: {search_results[3]} \n\n ### Output Text: \n\n {search_results[0]}"
            combined_results += "\n\n---\n\n"
            
            # Display the search results
            # st.text_area("Search Results", search_results[0], height=500)
            st.markdown(combined_results)
        st.success(f"Generated Information for Category: `{selected_category}` and Indicator: `{selected_indicator}` for both cities successfully!")
    else:
        st.warning("Please enter two city names and select both a category and an indicator.")




# Submit button for all indicators in the category
if st.button("Radar Graph"): 
    if city1 and city2 and selected_category:
        with st.spinner(f" \n\n ### Generating radar graph for {city1} and {city2}, please wait..."):
            combined_results = ""
            radar_data = {}

            for city in [city1, city2]:
                maturity_scores = []
                combined_results += f"\n\n---\n\n # {city}: \n\n"
                for indicator in filtered_df["Indicator"].unique():

                    maturity_scale_str = filtered_df[filtered_df["Indicator"] == indicator]["Maturity Assessment (1-5)"].iloc[0]

                    search_results = search_func(
                        city=city, 
                        indicator=indicator, 
                        maturity_scale=maturity_scale_str
                    )
                    
                    combined_results += f"## {indicator}:  \n\n ### Indicator Value: {search_results[2]} \n\n ### Maturity Score: {search_results[3]} \n\n ### Output Text: \n\n {search_results[0]}\n\n\n\n"
                    maturity_scores.append(search_results[3])
                radar_data[city] = maturity_scores
            combined_results += "\n\n---\n\n"
            
            
            # Create a spider chart for the indicators
            create_spider_chart(
                indicators=filtered_df["Indicator"].unique(),
                values_dict=radar_data,
                title=f"Comparative Radar Chart for {selected_category} for {city1} and {city2}",
            )

            # Render the Markdown content
            st.markdown(combined_results)
        st.success("Radar graph generated for both cities successfully!")
    else:
        st.warning("Please enter at least one city and select a category.")
