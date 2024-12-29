####################
##### Imports ######
####################

import pandas as pd
import time
import logging
import requests
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pydantic import BaseModel, Field

from dotenv import load_dotenv

from typing import Optional, Tuple, List, Union, Dict, Any, Annotated

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from requests.exceptions import ConnectionError, Timeout, RequestException
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

##################
##### Setup ######
##################
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

####################
##### Prompts ######
####################

indicator_prompt = """ 
**Objective:**  
Find the {indicator} for {city} from reputable and up-to-date online sources, then determine its maturity level based on the provided thresholds.

**Instructions:**  
1. **Data Extraction:**  
   - Perform a targeted web search using reliable and official sources (such as the city’s official data portal, government websites, recognized statistical agencies, trusted news outlets, or reputable research organizations).  
   - Identify the most recent and credible data available regarding the "{indicator}". For example, if the indicator is "Number of datasets available on city open data portal," find the current count of publicly available datasets from the official city open data website.

2. **Validation and Verification:**  
   - Confirm that the source is official or reputable. Cross-reference multiple sources if possible.  
   - Ensure the data pertains specifically to {city} and is not about another location with a similar name.  
   - Check the date of the source to ensure the data is as current as possible. If multiple data points are available, prefer the most recent credible figure.
   - If the current data is not available then find out the data that is most recent as far as 5 to 10 years old.

3. **Data Interpretation:**  
   - Extract the core number or value associated with the indicator.  
   - Example: If the indicator is "Number of employees trained in data analytics annually," find the exact or estimated number of employees trained in the last year.

4. Sources Section:
   - Use numbered sources in your answer (e.g., [1], [2]).
   - Provide a comprehensive list of all cited sources at the end of the answer.
   - Follow this format:
     ```markdown
     ### Sources
     [1] Full link - https://www.example-first.com/path  
     [2] Full link - https://www.example-second.com/path
     ```
   - Each cited source should be the full url mentioned in the context, with schema (example: https), domain (www.example-first.com), along with path, query and fragment provided in the context above.

5. **Maturity Assessment Mapping:**  
   - After obtaining the indicator value, compare it against the provided maturity scale.  
   - Use the thresholds given. The maturity assessment levels for the "{indicator}" are defined as:  
    {maturity_scale}
    
   Assign the correct maturity level based on where the identified value falls within these ranges.

6. **Formatting the Final Answer:**  
   - Present the final answer in a clear, concise manner.  
   - Provide:  
     - The indicator name  
     - The identified value (e.g., the current number or rate found)  
     - The assigned maturity level based on the thresholds
     - The set of urls used for gathering the information:
     ### Sources
     [1] https://www.example-first.com/path  
     [2] https://www.example-second.com/path

**Example of Final Output Format:**  
- Indicator: "[*Indicator Name*]"  
- Data Found: [*Exact numeric value or best estimate with source reference*]  
- Maturity Level: [*1-5, as determined by the thresholds*]

**Additional Guidance:**  
- If data cannot be found directly, look for related municipal reports, annual performance reviews, or recognized urban analytics services that may provide proxy indicators or related statistics.  
- If no credible data is found after a thorough check, note that no data is available and consider the default or lowest maturity level, if instructed.
"""


perplexity_system_prompt = """ 
You are an AI assistant tasked with gathering and searching for the most recent official statistics, reports, or reputable news sources that provide information on {indicator} for {city}.
"""


extraction_prompt = """ 
    You are an assistant with the ability to read structured answers and return them in a standardized format. You have been given a final answer that includes the following data:
    - An indicator name (e.g., “Number of datasets available on city open data portal”)
    - A data value for that indicator (a number)
    - A maturity level (an integer between 1 and 5) that corresponds to that data value based on predefined thresholds

    Your task:
    - Identify the numeric value of the indicator from the provided text. If the answer states something like "Data Found: 47 datasets," then indicator_value = 47.
    - Identify the integer maturity score assigned to that value. For example, if the answer states "Maturity Level: 4," then maturity_score = 4.
    - Validate that indicator_value is a float (it can be an integer type number as well, but represented as float is acceptable) and maturity_score is an integer between 1 and 5.

    Note:
    - If the given answer does not explicitly include the word "Data Found:" or "Maturity Level:" labels, infer the correct numeric values from the context.
    - If multiple values are presented, use the one that clearly aligns with the final indicator result.
    - If no data can be found, indicate that no valid extraction is possible and return 0 value for both: e.g., {"indicator_value": 0, "maturity_score": 0}
    """

find_indicators_prompt = """ 
You are a smart city assessment expert. Given a category, provide a comprehensive set of indicators and their corresponding maturity assessment levels for evaluating a city's performance in that category.

Requirements for the response:
1. Provide at least 5 distinct, measurable indicators for the given category
2. Each indicator should:
   - Be quantifiable and objective
   - Have clear units of measurement
   - Be relevant to the category
   - Be applicable across different city sizes and contexts
3. For each indicator, provide a 5-level maturity assessment scale where:
   - Level 1 represents basic/beginning implementation
   - Level 2 represents developing capability
   - Level 3 represents established capability
   - Level 4 represents advanced capability
   - Level 5 represents leading/exemplary performance
4. Format the response as follows:
   - List each indicator with its unit of measurement
   - Follow each indicator immediately with its 5-level maturity scale
   - Use numerical ranges or percentages where appropriate
   - Separate levels clearly using commas

Example format:
Category: [Category Name]
1. [Indicator 1] ([unit])
   Maturity levels: 1: [range], 2: [range], 3: [range], 4: [range], 5: [range]
2. [Indicator 2] ([unit])
   Maturity levels: 1: [range], 2: [range], 3: [range], 4: [range], 5: [range]
[...]

For reference, here is an example output:
Category: Connectivity
1. Fixed broadband subscriptions per 100 inhabitants
   Maturity levels: 1: <10, 2: 10-25, 3: 26-40, 4: 41-55, 5: >55
2. Percentage of population covered by at least a 4G mobile network
   Maturity levels: 1: <60%, 2: 60-75%, 3: 76-90%, 4: 91-98%, 5: >98%

Please provide indicators and maturity levels for the following category: {category}
"""

####################################
##### Perplexity Search Class ######
####################################

class PerplexitySearchError(Exception):
    """Custom exception for Perplexity search errors"""
    pass 

class PerplexityAPIError(Exception):
    """Custom exception for Perplexity API-specific errors"""    
    pass

class PerplexitySearchHandler:
    def __init__(
            self, 
            api_key: Optional[str] = None,
            model: Optional[str] = None,
            max_retries: int = 5,
            min_wait: float = 1,
            max_wait: float = 60,
            temperature: float = 0.2
    ):
        """
        Initialize PerplexitySearchHandler with configuration parameters.
        
        Args:
            api_key (str, optional): Perplexity API key, defaults to environment variable
            model (str, optional): Model to use for completions
            max_retries (int): Maximum number of retry attempts
            min_wait (float): Minimum wait time between retries in seconds
            max_wait (float): Maximum wait time between retries in seconds
            temperature (float): Temperature for response generation
        """
        self.api_key = api_key or os.getenv('PERPLEXITY_API')
        if not self.api_key:
            self.logger.error("__init__: Perplexity API key not found")
            raise PerplexityAPIError("Perplexity API key not found")
        
        self.model = model or os.getenv('MODEL')
        if not self.model:
            self.logger.error("__init__: Model not specified")
            raise PerplexityAPIError("Model not specified")
        
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait 
        self.temperature = temperature
        self.endpoint_url = "https://api.perplexity.ai/chat/completions"

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()


    def _setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - Line:%(lineno)d - %(message)s',
            # Optional: Add file handler to save logs
            handlers=[
                logging.FileHandler('perplexity_search.log'),
                logging.StreamHandler()  # Print to console as well
            ]
        )


    def _create_payload(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Create the API request payload"""
        print("------3-------")
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # "max_tokens": "Optional",
            "temperature": self.temperature,
            "top_p": 0.9,
            "return_citations": True,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            # "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
    

    def _handle_response(self, response: requests.Response) -> str:
        """Handle and validate API response"""
        try:
            if response.status_code != 200:
                self.logger.error(f"_handle_response: API returned status code {response.status_code}: {response.text}")
                raise PerplexityAPIError(f"_handle_response: API returned status code {response.status_code}: {response.text}")
            
            response_dict = response.json()
            response_choice = response_dict.get("choices", [])
            response_citations = response_dict.get("citations", [])
            print("------5-------")
            
            if not response_choice:
                self.logger.error("_handle_response: No choices found in response")
                raise PerplexityAPIError("_handle_response: No choices found in response")
            
            print("------6-------")
            response_message = response_choice[0].get("message", {})
            if not response_message:
                self.logger.error("_handle_response: No message found in response")
                raise PerplexityAPIError("_handle_response: No message found in response")
            
            print("------7-------")
            return response_message.get("content", ""), response_citations
            
        except json.JSONDecodeError as e:
            self.logger.error(f"_handle_response: Failed to parse API response: {str(e)}")
            raise PerplexityAPIError(f"_handle_response: Failed to parse API response: {str(e)}")
        
        except Exception as e: 
            self.logger.error(f"_handle_response: Error processing response: {str(e)}")
            raise PerplexityAPIError(f"_handle_response: Error processing response: {str(e)}")
        

    @classmethod
    def _get_retry_decorator(cls, logger):
        """Get a retry decorator with the specified logger"""
        return retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type((
                ConnectionError,
                Timeout,
                RequestException,
                PerplexityAPIError
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO),
            reraise=True
        )
    
    
    def _make_request_method(self, system_prompt: str, user_prompt: str) -> str:
        """Make request to Perplexity API with retry handling using decorator method"""
        retry_decorator = self._get_retry_decorator(self.logger)
        
        @retry_decorator
        def _make_request_inner():
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.endpoint_url,
                json=self._create_payload(system_prompt, user_prompt),
                headers=headers
            )
            print("------4-------")
            return self._handle_response(response)
            
        return _make_request_inner()
    

    def search(self, system_prompt: str, user_prompt: str) -> str:
        """
        Perform a search query with error handling and retries.
        
        Args:
            system_prompt (str): System prompt for the LLM
            user_prompt (str): User's search query
            
        Returns:
            str: Formatted search results
            
        Raises:
            PerplexitySearchError: If search operation fails
            ValueError: If inputs are invalid
        """
        # Input validation
        if not system_prompt or not isinstance(system_prompt, str):
            self.logger.error("search: System prompt must be a non-empty string")
            raise ValueError("search: System prompt must be a non-empty string")
        if not user_prompt or not isinstance(user_prompt, str):
            self.logger.error("search: User prompt must be a non-empty string")
            raise ValueError("search: User prompt must be a non-empty string")

        try:
            print("------2-------")
            self.logger.info(f"search: Executing search query: {user_prompt[-100:]}")
            return self._make_request_method(system_prompt, user_prompt)
            
        except Exception as e:
            self.logger.error(f"search: Search failed: {str(e)}")
            raise PerplexitySearchError(f"search: Search failed: {str(e)}")
        


# Example usage
def perplexity_search_func(system_prompt: str, user_prompt: str):
    try:
        # Initialize the search handler
        search_handler = PerplexitySearchHandler(
            max_retries=5,
            min_wait=1,
            max_wait=60,
            temperature=0.2
        )
        print("------1-------")
        # Perform search
        results, citations = search_handler.search(system_prompt, user_prompt)
        # print("Search Results:", results)
        return results, citations
        
    except ValueError as e:
        search_handler.logger.error(f"perplexity_search_func: Invalid input: {str(e)}")
        raise ValueError(f"perplexity_search_func: Invalid input: {str(e)}")
        
    except PerplexitySearchError as e:
        search_handler.logger.error(f"perplexity_search_func: Search failed: {str(e)}")
        raise PerplexitySearchError(f"perplexity_search_func: Search failed: {str(e)}")
        
    except Exception as e:
        search_handler.logger.error(f"perplexity_search_func: Unexpected error: {str(e)}")
        raise PerplexitySearchError(f"perplexity_search_func: Unexpected error: {str(e)}")


##########################################
##### Read the indicator Excel file ######
##########################################

def read_indicators_file():
    # Load the Excel file
    file_path = './Provisional indicator list.xlsx'

    # Define the sheet names and the desired columns
    sheets = [
        "Digital Transformation",
        "Policies and Regulations",
        "People and Digital Skills",
        "City Functions",
        "City",
        "Data"
    ]
    columns = ["Category", "Indicator", "City Level Source", "National Data Source", 
            "Maturity Assessment (1-5)", "Toolkit Source"]

    # Read each sheet into a DataFrame, selecting the desired columns, and concatenate them
    df_list = []
    for sheet in sheets:
        try:
            sheet_df = pd.read_excel(file_path, sheet_name=sheet, usecols=columns)
            # print(f"For sheet - {sheet}, the number of indicators are: {len(sheet_df)}")
            df_list.append(sheet_df)
        except Exception as e:
            print(f"Error reading sheet {sheet}: {e}")

    # Combine all the data into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # Return the combined DataFrame to the user
    return combined_df


#######################################
##### Format the Maturity Levels ######
#######################################

def format_maturity_levels(maturity_scale: str):

    # Prompt
    maturity_format_prompt = """ 
        Given an input of maturity scale for an indicator with thresholds. Format the maturity scale in Levels with thresholds mentioned in the example output shown below:

        Modify this maturity scale to below format:
        {maturity_scale}

        Example:
        Input: 1: <10, 2: 10-25, 3: 26-40, 4: 41-55, 5: >55
        
        Example of Final Output Format:
        Level 1: <10  
        Level 2: 10-25
        Level 3: 26-40 
        Level 4: 41-55 
        Level 5: >55
    """

    # System Prompt
    system_prompt = SystemMessage(content=maturity_format_prompt.format(maturity_scale=maturity_scale))

    # Invoke the LLM to generate query
    level_list = llm.invoke([system_prompt])

    return level_list.content

#######################################
##### MaturityScore Class ######
#######################################

class MaturityScore(BaseModel):
    indicator_value: float = Field(
        ...,
        description=(
            "The numeric value of the indicator obtained from the processed results. "
            "For example, if the indicator is 'Number of datasets available on the city open data portal', "
            "this field should capture the exact number of datasets found (e.g., 47.0). "
            "If a range or estimate is given, use the most precise numeric figure available. "
            "Ensure that this value corresponds directly to the indicator measured and is not a derived metric."
        )
    )
    maturity_score: int = Field(
        ...,
        description=(
            "The assigned maturity level of the indicator, expressed as an integer between 1 and 5, inclusive. "
            "This value is determined by comparing the extracted 'indicator_value' against predefined threshold ranges. "
            "For example, if the thresholds for this indicator are defined as: "
            "Level 1: <10"  
            "Level 2: 10-25"
            "Level 3: 26-40"
            "Level 4: 41-55" 
            "Level 5: >55"
            " and the indicator_value is 47, "
            "then the maturity_score would be 4. "
            "This field must be a whole number (no decimals) and must always fall within the range of 1 to 5."
        )
    )


def search_func(city: str, indicator: str, maturity_scale: str):
    perplexity_result, citations = perplexity_search_func(system_prompt=perplexity_system_prompt.format(indicator=indicator, city=city), user_prompt=indicator_prompt.format(indicator=indicator, city=city, maturity_scale=format_maturity_levels(maturity_scale)))

    # Structured LLM
    structured_llm = llm.with_structured_output(MaturityScore)

    # Invoke the LLM to get the list of economic levers
    maturity_value = structured_llm.invoke([SystemMessage(content=extraction_prompt)] + [HumanMessage(content=f"Extract the indicator value and maturity score from the output of perplexity search:\n {perplexity_result}")])

    return perplexity_result, citations, maturity_value.indicator_value, maturity_value.maturity_score



# Function to create a spider chart
def create_spider_chart_old(indicators, values, title, color):
    """
    Creates a professional spider (radar) chart.

    Parameters:
    - indicators: List of indicators.
    - values: List of values corresponding to each category.
    - title: Title of the chart.
    - color: Color for the area plot.
    """
    # Number of variables
    num_vars = len(indicators)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the circle
    values += values[:1]
    angles += angles[:1]
    
    # Create a radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw the outline of the chart
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.plot(angles, values, color=color, linewidth=2)
    
    # Add labels to the chart
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicators, fontsize=12)
    
    # Add title
    ax.set_title(title, fontsize=16, pad=20)
    
    # Add grid lines and make it visually appealing
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.spines['polar'].set_visible(False)
    
    st.pyplot(fig)

# # Example data format
# indicators = ['Communication', 'Teamwork', 'Problem-solving', 'Adaptability', 'Leadership']
# values = [2, 4, 5, 2, 1]

# # Generate the spider diagram
# create_spider_chart(indicators, values, 'Skill Assessment', 'red')

def create_spider_chart(indicators, values_dict, title):
    """
    Creates an overlapping radar (spider) chart for multiple cities.

    Parameters:
    - indicators: List of indicators.
    - values_dict: Dictionary with city names as keys and list of values as values.
    - title: Title of the chart.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Number of variables
    num_vars = len(indicators)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot data for each city
    for city, values in values_dict.items():
        # Complete the circle for each city
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=city)
        ax.fill(angles, values, alpha=0.25)

    # Add labels to the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicators, fontsize=12)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=10)
    
    # Add title
    ax.set_title(title, fontsize=16, pad=20)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0, 0), fontsize=10)
    
    # Add grid lines and make it visually appealing
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.spines['polar'].set_visible(False)

    # Display the radar chart in Streamlit
    st.pyplot(fig)


# indicators = ["Connectivity", "Sustainability", "Economy", "Healthcare", "Safety"]
# values_dict = {
#     "New York": [4, 3, 5, 4, 4],
#     "Los Angeles": [3, 4, 4, 5, 3]
# }

# create_spider_chart(indicators, values_dict, title="Comparative Radar Chart for Cities")

#######################################
##### Web Indicators Class ############
#######################################

class WebIndicators(BaseModel):
    indicator_list: List[str] = Field(
        ...,
        description=(
            "A comprehensive set of indicators identified for evaluating a city's performance in a given category."
            "For example, if the category is 'Sanitation' then the indicators are: "
            "['Access to Improved Sanitation Facilities (% of population)' Maturity levels', 'Wastewater Treatment Coverage (% of wastewater treated)', 'Frequency of Waste Collection (times per week)' , 'Public Toilet Availability (number of public toilets per 10,000 inhabitants)' , 'Recycling Rate (% of total waste recycled)' ]"
        )
    )
    maturity_levels_list: List[str] = Field(
        ...,
        description=(
            "The list of assigned maturity level for each of the indicators, expressed as an integer between 1 and 5, inclusive. "
            "This value is determined by comparing the extracted 'indicator_value' against predefined threshold ranges. "
            "For example, if the category is 'Sanitation' then the maturity levels for the identified indicators are: "
            "['1: <50%, 2: 50-70%, 3: 71-85%, 4: 86-95%, 5: >95%', '1: <30%, 2: 30-50%, 3: 51-70%, 4: 71-90%, 5: >90%', '1: <1, 2: 1-2, 3: 3-4, 4: 5-6, 5: Daily', '1: <1, 2: 1-3, 3: 4-6, 4: 7-9, 5: >9', '1: <10%, 2: 10-25%, 3: 26-40%, 4: 41-60%, 5: >60%']"
        )
    )


def fetch_indicators_from_web(category: str):
    user_prompt = find_indicators_prompt.format(category=category)

    # Invoke the LLM to generate query
    indicator_list = llm.invoke([user_prompt])

    # Structured LLM
    structured_llm = llm.with_structured_output(WebIndicators)

    # Invoke the LLM to get the list of economic levers
    web_indicators = structured_llm.invoke([HumanMessage(content=f"Extract the list of indicators and the list of their maturity scores from the output:\n {indicator_list.content}")])

    return web_indicators.indicator_list, web_indicators.maturity_levels_list
