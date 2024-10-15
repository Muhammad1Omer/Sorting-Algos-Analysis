import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sorting.views import get_timings, generate_dataset, MonogoToPostgres, GetData

def main():
    st.title("Sorting Times Visualization")

    # Custom array input functionality
    custom_array_input = st.text_input("Enter your custom array (comma-separated values):", "")
    if st.button("Create Custom Array"):
        try:
            # Split the input by commas and attempt to convert to integers
            custom_array = [int(x.strip()) for x in custom_array_input.split(',')]
            
            # Check for negative integers
            if any(x < 0 for x in custom_array):
                st.error("Array must contain only non-negative integers.")
            else:
                st.session_state['array'] = custom_array  # Save array in session state
                st.success("Custom array created successfully!")

        except ValueError:
            st.error("Jahil. You have entered letters or special characters.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    
    # Retrieve timing data
    timing_data = get_timings("00")  
    times = timing_data["times"]
    
    df = pd.DataFrame(times)
    df_melted = pd.melt(df, id_vars=["size"], 
                        value_vars=["insertion", "bubble", "selection", "quick", "merge", "heap", "bucket", "count", "radix"], 
                        var_name="algorithm", 
                        value_name="time")

    df_melted["log_time"] = df_melted["time"].apply(lambda x: np.log10(x + 1))  
    
    selected_algorithms = st.multiselect(
        "Select algorithms to display", 
        options=["insertion", "bubble", "selection", "quick", "merge", "heap", "bucket", "count", "radix"], 
        default=["insertion", "bubble", "selection", "quick", "merge", "heap", "bucket", "count", "radix"]
    )

    df_filtered = df_melted[df_melted["algorithm"].isin(selected_algorithms)]
    
    # If a custom array exists, add it to the dataset for plotting
    if 'array' in st.session_state:
        custom_array = st.session_state['array']
        custom_df = pd.DataFrame({
            'size': range(1, len(custom_array) + 1),  # Assuming the size corresponds to the index
            'algorithm': ['custom_array'] * len(custom_array),
            'time': custom_array
        })
        custom_df['log_time'] = custom_df['time'].apply(lambda x: np.log10(x + 1))
        
        # Combine the custom array data with the filtered data
        df_filtered = pd.concat([df_filtered, custom_df], ignore_index=True)

    # Plot the line chart
    line_chart = alt.Chart(df_filtered).mark_line().encode(
        x='size:Q',         # X-axis as size
        y='log_time:Q',     # Y-axis as log-transformed time
        color='algorithm:N'  # Different lines for each algorithm
    ).properties(
        title="Sorting Times for Selected Algorithms and Custom Array"
    )

    st.altair_chart(line_chart, use_container_width=True)
    
    # Random dataset generation and processing
    if st.button("Generate Random Dataset"):
        generate_dataset()
        MonogoToPostgres()
        st.success("Datasets and times calculated!")
    

if __name__ == "__main__":
    main()
