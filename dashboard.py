#Importing libraries

import streamlit as st
import pandas as pd
import io
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import ttest_ind





#setting the page title
st.title("LEARNERS DASHBOARD")


#Sidebar Navigation
st.sidebar.title("Navigation")
selected_section = st.sidebar.radio("Go to", ["Data Cleaning", "About the Dataset", "Data Insights"], index=2)


# Load the dataset
df = pd.read_excel("Coursera Dataset copy.xlsx")


def clean_data(df):
    df['Completed'].fillna('Not Started Yet', inplace=True)
    df_test = df
    df_test['Completion Time'] = pd.to_datetime(df_test['Completion Time'], errors='coerce', infer_datetime_format=True)
    df_test['Completion Time'] = pd.to_datetime(df_test['Completion Time'], errors='coerce')
    # Converting to the desired format
    df_test['Formatted Completion Time'] = df_test['Completion Time'].dt.strftime('%d-%m-%y %H:%M:%S')
    df['Completion Time'] = pd.to_datetime(df['Completion Time'], errors='coerce')
    df['Completion Time'] = df['Completion Time'].dt.strftime('%d-%m-%y %H:%M:%S')
    df['Enrollment Time'] = pd.to_datetime(df['Enrollment Time'], errors='coerce')
    df['Enrollment Time'] = df['Enrollment Time'].dt.strftime('%d-%m-%y %H:%M:%S')
    dummy_datetime = pd.to_datetime('1900-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')
    df_test['Enrollment Time'].fillna(dummy_datetime, inplace=True)
    dummy_datetime = pd.to_datetime('1900-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')
    df['Completion Time'].fillna(dummy_datetime, inplace=True)
    df_test['Learning Hours Spent'].fillna(0.0, inplace=True)
    df['Learning Hours Spent'].fillna(0.0, inplace=True)
    df['Course Grade'].fillna(0.0, inplace=True)
    df.drop(columns=['Formatted Completion Time'], inplace=True)
    df['Group'] = df['Group'].replace('#', 'Not Available')
    df['Department'] = df['Department'].replace('#', 'Not Available')
    df_test=df
    return df








# Capture the console output
buffer = io.StringIO()
df.info(buf=buffer)
# Set the buffer position to the beginning
buffer.seek(0)


#Section : Data Cleaning

if selected_section == "Data Cleaning":
    st.header("Data Cleaning")



    st.markdown("#### Fixing the 'Completed' Column")

    completed_column = df['Completed']
    st.write(completed_column)

    st.markdown("Missing values in the 'Completed' column indicate learners who have not yet started the course. "
                "To improve clarity, these missing values have been replaced with 'Not Started Yet'.")

    # Replace NaN values in 'Completed' column with 'Not Started Yet'
    df['Completed'].fillna('Not Started Yet', inplace=True)

    # Display the updated DataFrame
    st.markdown("#### Updated DataFrame with 'Completed' Column")
    st.write(df['Completed'])

    df_test = df
    df_test['Completion Time'] = pd.to_datetime(df_test['Completion Time'], errors='coerce', infer_datetime_format=True)
    st.markdown("### Fixing  All the 'Time' Column")

    st.write(df_test['Completion Time'])

    df_test['Completion Time'] = pd.to_datetime(df_test['Completion Time'], errors='coerce')

    # Converting to the desired format and creating a new column
    df_test['Formatted Completion Time'] = df_test['Completion Time'].dt.strftime('%d-%m-%y %H:%M:%S')

    st.markdown("#### Converting 'Completion Time' to Datetime")

    # Display the original 'Completion Time' and the new 'Formatted Completion Time'
    st.dataframe(df_test[['Completion Time', 'Formatted Completion Time']])


    # Provide explanation for the conversion
    st.markdown("The 'Completion Time' column has been converted to datetime format, and a new column "
                "'Formatted Completion Time' has been created to display the datetime values in the desired format.")

    st.markdown("#### Formatted Completion Time")
    st.write(df_test['Formatted Completion Time'])

    st.markdown("#### Completion Time")
    st.write(df_test['Completion Time'])

    # Converting 'Completion Time' to datetime
    df['Completion Time'] = pd.to_datetime(df['Completion Time'], errors='coerce')

    # Formatting 'Completion Time' to the desired format
    df['Completion Time'] = df['Completion Time'].dt.strftime('%d-%m-%y %H:%M:%S')
    st.markdown("#### Converting and Formatting 'Completion Time'")
    st.dataframe(df[['Completion Time']])
    st.markdown("The 'Completion Time' column has been converted to datetime format, and the datetime values have "
                "been formatted to the desired format ('%d-%m-%y %H:%M:%S').")
    st.markdown("#### Fixing the Enrollment Time")
    st.dataframe(df_test['Enrollment Time'])
    df['Enrollment Time'] = pd.to_datetime(df['Enrollment Time'], errors='coerce')
    df['Enrollment Time'] = df['Enrollment Time'].dt.strftime('%d-%m-%y %H:%M:%S')
    dummy_datetime = pd.to_datetime('1900-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')
    df_test['Enrollment Time'].fillna(dummy_datetime, inplace=True)
    st.markdown(" Filling all the NaN values with the dummy values of Enrollment time by filling them with the time 1 January 1900 and time will be midnight timming.")
    st.dataframe(df_test['Enrollment Time'])
    dummy_datetime_2 = pd.to_datetime('1900-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S')
    df['Completion Time'].fillna(dummy_datetime_2, inplace=True)
    st.markdown("Filling all the NaN values with the dummy values of Completion time by filling them with the time 1 January 1900 and time will be midnight timming.")
    # Display missing values count
    st.subheader("Missing Values Count : ")
    missing_values = df.isnull().sum()
    st.bar_chart(missing_values)
    st.markdown("#### Fixed the completion time column")
    st.dataframe(df_test['Completion Time'])

    st.markdown("#### Fixing the 'Learning Hours Spent' column")
    st.dataframe(df['Learning Hours Spent'])
    df_test['Learning Hours Spent'].fillna(0.0, inplace=True)
    df['Learning Hours Spent'].fillna(0.0, inplace=True)
    st.markdown("Filled all the not availible values with 0.00 in the learning hours column")
    st.markdown("#### Fixed the Learning Hours Column")

    #st.dataframe(df['Course Grade'])

    st.markdown("#### Fixing the 'Course Grade' column")
    st.dataframe(df['Course Grade'])
    df['Course Grade'].fillna(0.0, inplace=True)
    st.markdown("Filled all the not availible values with 0.00 in the Course Grade column")
    st.markdown("#### Fixed the 'Course Grade' Column")

    st.dataframe(df['Course Grade'])
    # Display missing values count
    st.subheader("Missing Values Count : ")
    missing_values = df.isnull().sum()
    st.bar_chart(missing_values)
    st.markdown("Dropping the unnecessay column named 'Formatted Completion Time' is same as 'Completion Time'")
    # Display missing values count
    df.drop(columns=['Formatted Completion Time'], inplace=True)
    st.subheader("Missing Values Count : ")
    missing_values = df.isnull().sum()
    st.bar_chart(missing_values)
    st.markdown("#### The Formatted dataset ")
    st.dataframe(df)


    st.markdown("Now moving on to fix the '#' values in the dataset")
    st.markdown("#### Hash Value Count")
    hash_counts_per_column = df.apply(lambda x: x.eq('#').sum())
    st.bar_chart(hash_counts_per_column)
    st.markdown("Replacing the '#' values everywhere with the 'Not Availible' value")
    df['Group'] = df['Group'].replace('#', 'Not Available')
    df['Department'] = df['Department'].replace('#', 'Not Available')
    hash_counts_per_column_2 = df.apply(lambda x: x.eq('#').sum())
    st.markdown("The Modified Group Column")
    st.dataframe(df['Group'])
    st.bar_chart(hash_counts_per_column_2)
    df_test = df

    #Final Formatted DataFrame

    st.markdown("### Final Formatted Dataset")
    st.dataframe(df)














#Section : About the Dataset
elif selected_section == "About the Dataset":
    st.header("About the Dataset")

    st.subheader("Dataset Overview")

    st.dataframe(df)
    # Display information about columns and rows
    st.markdown(f"- **Columns:** {len(df.columns)}")
    st.markdown(f"- **Rows:** {len(df)}")
    # Display information about columns
    st.write("### Columns:")
    st.markdown("Here are the columns in the dataset along with their data types:")

    # Create a table for columns and data types
    columns_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes
    })
    st.table(columns_info)

    # Describe the dataset
    st.write("### Descriptive Statistics:")
    st.markdown("Here are the descriptive statistics for the numerical columns:")
    st.table(df.describe())


    # Display information about the dataset
    st.write("### Detailed Dataset Information:")
    st.text(buffer.read())

    # Display missing values count
    st.subheader("Missing Values Count : ")
    missing_values = df.isnull().sum()
    st.bar_chart(missing_values)

#Section : Data Insights
elif selected_section == "Data Insights":

    df=clean_data(df)
    st.header("Data Insights")
    st.subheader("Data Demographics")
    # Create demographics DataFrame
    # Create demographics dictionary
    demographics = {
        'Persons': 1409,
        'Divisions': 22,
        'Group': 63,
        'Department': 289,
        'Programs': 134,
        'Courses': 291,
        'Skills': 357,
        'Total Learning Hours': 5160
    }
    # Display metrics in two rows with four columns each
    col1, col2, col3, col4  = st.columns(4)

    # Display metrics in the first row
    with col1:
        st.metric(label="Persons", value=demographics['Persons'], delta=None)
        st.metric(label="Divisions", value=demographics['Divisions'], delta=None)

    # Display metrics in the second row
    with col2:
        st.metric(label="Programs", value=demographics['Programs'], delta=None)
        st.metric(label="Courses", value=demographics['Courses'], delta=None)

    with col3:
        st.metric(label="Group", value=demographics['Group'], delta=None)
        st.metric(label="Department", value=demographics['Department'], delta=None)
    with col4:
        st.metric(label="Skills", value=demographics['Skills'], delta=None)
        st.metric(label="Total Learning Hours", value=demographics['Total Learning Hours'], delta=None)
    #No of Unique Divisions in the dataset
    st.markdown("#### Unique Divisions DataFrame")
    st.markdown("The following table contains the list of unique divisions:")
    unique_divisions = df['Division'].unique()
    df_unique_divisions = pd.DataFrame({'Unique Divisions': unique_divisions})
    st.dataframe(df_unique_divisions)

    #No of Unique Learners in the Dataset
    st.markdown("#### Unique Groups DataFrame")
    st.markdown("The following table contains the list of unique groups:")
    unique_groups = df['Group'].unique()
    df_unique_groups = pd.DataFrame({'Unique Groups': unique_groups})
    st.dataframe(df_unique_groups)

    #No of Unique Departments in the Dataset
    st.markdown("#### Unique Departments DataFrame")
    st.markdown("The following table contains the list of unique Deartments:")
    unique_Departments = df['Department'].unique()
    df_unique_Departments = pd.DataFrame({'Unique Departments': unique_Departments})
    st.dataframe(df_unique_Departments)



    #No of Unique Programs in the Dataset
    st.markdown("#### Unique Programs DataFrame")
    st.markdown("The following table contains the list of unique Programs:")
    unique_Programs = df['Program Name'].unique()
    df_unique_Programs = pd.DataFrame({'Unique Programs': unique_Programs})
    st.dataframe(df_unique_Programs)

    # No of Unique Courses in the Dataset
    st.markdown("#### Unique Courses DataFrame")
    st.markdown("The following table contains the list of unique Courses:")
    unique_Courses= df['Course Name'].unique()
    df_unique_Courses = pd.DataFrame({'Unique Courses': unique_Courses})
    st.dataframe(df_unique_Courses)

    # No of Unique Skills in the Dataset
    st.markdown("#### Unique Skills DataFrame")
    st.markdown("The following table contains the list of unique Skills:")
    skills_counts = df['Skills Learned'].str.split('; ').explode().value_counts()
    df_skills_counts = pd.DataFrame({'Skill': skills_counts.index, 'Count': skills_counts.values})
    st.dataframe(df_skills_counts)


    st.markdown("#### Learning Hours Dataframe")

    df['Learning Hours Spent'].fillna(0.0, inplace=True)
    st.dataframe(df['Learning Hours Spent'])
    #df['Learning Hours Spent'].fillna(0.0, inplace=True)
    total_learning_hours = df['Learning Hours Spent'].sum()
    print("Total Learning Hours Spent:", total_learning_hours)

    #st.dataframe(df)

    st.markdown("## Learners Count Insights")

    col5, col6 = st.columns(2)

    with col5:
        st.metric(label="Maximum Learners Enrolled", value="329", delta=None)
        st.markdown("One IT Division & Automation Dept.")
        st.markdown("")
        st.markdown("")
        st.metric(label="Maximum Learners Enrolled", value="2080", delta=None)
        st.markdown("Records of Learners Not Tagged To A Specific Group.")
        st.markdown("")
        st.markdown("")
        st.metric(label="Minimum Learners Enrolled", value="1", delta=None)
        st.markdown("Minimum Number Of Learners Enrolled in Any Possible Group, Department Or Division")

    with col6:
        st.metric(label="Enrolled Learners Percentage", value="61%", delta=None)
        st.markdown("One IT Division")
        st.markdown("")
        st.markdown("")
        st.metric(label="Enrolled Learners Percentage", value="71%", delta=None)
        st.markdown("TSM Division")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.metric(label="Maximum Learners Percentage Across Any Department", value="23%", delta=None)
        st.markdown("Automation")

    distribution_counts_division_group_department = df.groupby(['Division', 'Group', 'Department']).size().reset_index(
        name='Learner Count')
    # Sort the distribution counts by 'Learner Count' in descending order
    distribution_counts_sorted = distribution_counts_division_group_department.sort_values(by='Learner Count',
                                                                                           ascending=False)
    # Display distribution counts
    # Reset the index before using st.bar_chart
    distribution_counts_sorted_top10 = distribution_counts_sorted.head(10).reset_index(drop=True)

    # Select the top 10 records
    distribution_counts_sorted_top10 = distribution_counts_sorted.head(10)
    st.markdown("#### Top 10 Learner Count Distribution between Divsion And Department")
    # Display the top 10 records as a bar chart using Altair
    st.altair_chart(alt.Chart(distribution_counts_sorted_top10).mark_bar().encode(
        x=alt.X('Learner Count', title='Number of Learners'),
        y=alt.Y('Department', title='Department'),
        color='Division:N',
        tooltip=['Division', 'Group', 'Department', 'Learner Count']
    ).properties(
        width=800,  # Set the width of the chart
        height=400  # Set the height of the chart
    ).interactive())

    st.markdown("#### Division Group Department DataFrame")
    st.dataframe(distribution_counts_sorted)

    distribution_counts_division_group = df.groupby(['Division', 'Group']).size().reset_index(name='Learner Count')
    distribution_counts_division_group_sorted = distribution_counts_division_group.sort_values(by='Learner Count',
                                                                                               ascending=False)
    chart = alt.Chart(distribution_counts_division_group_sorted.head(10)).mark_bar().encode(
        x=alt.X('Learner Count:Q', title='Number of Learners'),
        y=alt.Y('Group:N', title='Group'),
        color=alt.Color('Division:N', scale=alt.Scale(scheme='category20b')),
        tooltip=['Division', 'Group', 'Learner Count']
    ).properties(
        width=800,
        height=400
    ).interactive()
    # Display the chart using Streamlit
    st.markdown("#### Top 10 Learner Count Distribution between Divsion And Group")
    st.altair_chart(chart)
    st.markdown("#### Division Group DataFrame")
    st.dataframe(distribution_counts_division_group_sorted)

    distribution_counts_division_Department = df.groupby(['Division', 'Department']).size().reset_index(
        name='Learner Count')
    distribution_counts_division_Department_sorted = distribution_counts_division_Department.sort_values(
        by='Learner Count', ascending=False)
    # Create a bar chart for distribution_counts_division_Department_sorted using Altair
    chart_department = alt.Chart(distribution_counts_division_Department_sorted.head(10)).mark_bar().encode(
        x=alt.X('Learner Count:Q', title='Number of Learners'),
        y=alt.Y('Department:N', title='Department'),
        color=alt.Color('Division:N', scale=alt.Scale(scheme='set1')),
        tooltip=['Division', 'Department', 'Learner Count']
    ).properties(
        width=800,
        height=400
    ).interactive()
    # Display the chart using Streamlit
    st.markdown("#### Top 10 Learner Count Distribution between Divsion And Department")
    st.altair_chart(chart_department)

    st.markdown("#### Division Department Dataframe")
    st.dataframe(distribution_counts_division_Department_sorted)

    distribution_counts_group_Department = df.groupby(['Group', 'Department']).size().reset_index(name='Learner Count')
    distribution_counts_group_Department_sorted = distribution_counts_group_Department.sort_values(by='Learner Count',

                                                                                                   ascending=False)

    # Create a bar chart for distribution_counts_group_Department_sorted using Altair

    chart_group_department = alt.Chart(distribution_counts_group_Department_sorted.head(10)).mark_bar().encode(
        x=alt.X('Learner Count:Q', title='Number of Learners'),
        y=alt.Y('Department:N', title='Department'),
        color=alt.Color('Group:N', scale=alt.Scale(scheme='category10')),
        tooltip=['Group', 'Department', 'Learner Count']
    ).properties(
        width=800,
        height=400
    ).interactive()
    st.markdown("#### Top 10 Learner Count Distribution between Groups And Departments")
    # Display the chart using Streamlit
    st.altair_chart(chart_group_department)
    st.markdown("#### Group Department Dataframe")
    st.dataframe(distribution_counts_group_Department_sorted)

    # Group by 'Group' and get the count of enrolled learners
    enrolled_learners = df
    enrolled_by_group = enrolled_learners.groupby('Group').size().reset_index(name='Enrolled Learner Count')

    # Create a vertical bar chart using
    st.markdown("### GroupWise Learner Records Distribution")
    chart_grp = alt.Chart(enrolled_by_group).mark_bar().encode(
        x='Group:N',
        y='Enrolled Learner Count:Q',
        color=alt.value('#3182bd')  # You can customize the color as needed
    ).properties(
        width=alt.Step(80)  # You can adjust the width as needed
    )

    # Display the chart using Streamlit
    st.altair_chart(chart_grp, use_container_width=True)  # Adjust 'use_container_width' as needed

    enrolled_by_department = enrolled_learners.groupby('Department').size().reset_index(name='Enrolled Learner Count')
    # Sort the DataFrame in decreasing order
    enrolled_by_department = enrolled_by_department.sort_values(by='Enrolled Learner Count', ascending=False)
    # Create a vertical bar chart using Altair with the Streamlit theme color
    st.markdown("### Department Wise Learner Records Distribution")
    chart_dept = alt.Chart(enrolled_by_department).mark_bar().encode(
        x='Department:N',
        y='Enrolled Learner Count:Q',
        color=alt.value('#3182bd')  # Use the Streamlit theme color (light blue)
    ).properties(
        width=alt.Step(80)  # You can adjust the width as needed
    )

    # Display the chart using Streamlit
    st.altair_chart(chart_dept, use_container_width=True)  # Adjust 'use_container_width' as needed


    st.markdown("## Program Insights")

    col_pro_1, col_pro_2 = st.columns(2)

    with col_pro_1:
        st.metric(label="Maximum Learners Enrolled in a Program", value="2026", delta=None)
        st.markdown("School Of Analytic Basic")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.metric(label="Second Most Maximum Learners Enrolled in a Program", value="588", delta="-71%")
        st.markdown("School Of Blockchain")
        st.markdown("71% less than most popular one")
        st.markdown("")
        st.markdown("")
        st.markdown("")
    with col_pro_2:
        st.metric(label="School Of Analytic Basic", value="30%", delta=None)
        st.markdown("Most Popular among Learners having 30% percent of the records ")
        st.markdown("")
        st.markdown("")
        st.metric(label="5 Most Programs with 62% of Records", value="62%", delta=None)
        st.markdown("School of Analytics-Basic , School Of Blockchain, 	Project Mgmt Learning Program-Generic, Business Communication Learning Program, Six Sigma Black Belt Learning Program")
        st.markdown("")
        st.markdown("")

    program_counts = df.groupby('Program Name').size().reset_index(name='Learner Count')

    # Sort the DataFrame by 'Learner Count' in descending order
    most_popular_programs = program_counts.sort_values(by='Learner Count', ascending=False)

    most_popular_programs_sorted = most_popular_programs.sort_values(by='Learner Count', ascending=False)

    # Create an Altair chart
    st.markdown("### Program Wise Learner Records Distribution")
    chart_program = alt.Chart(most_popular_programs_sorted).mark_bar().encode(
        x=alt.X('Program Name:N', title='Program Name'),
        y=alt.Y('Learner Count:Q', title='Learner Count'),
        color=alt.value('#3182bd')  # Use the Streamlit theme color (light blue)
    ).properties(
        width=alt.Step(80)  # You can adjust the width as needed
    )

    # Display the chart using Streamlit
    st.altair_chart(chart_program, use_container_width=True)

    # Display the sorted DataFrame

    st.markdown("### Most Popular Program Wise Dataframe")
    st.dataframe(most_popular_programs)

    st.markdown("## Course Insights")
    st.markdown("")

    col_course1, col_course2 = st.columns(2)

    with col_course1:
        st.metric(label="Maximum Completion Percentage in a Course", value="100%", delta=None)
        st.markdown("20 Course Records have full course completion percentage ")
        st.markdown("")
        st.metric(label="High Completion Rate Courses", value="18", delta=None)
        st.markdown("Courses Have Completion Rate >= 85%")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.metric(label="Low Completion Rate Courses", value="185", delta=None)
        st.markdown("Courses Have Completion Rate <= 0.4%")
        st.markdown("")
        st.markdown("")

    with col_course2:
        st.metric(label="Course Completion Rate", value="73.6%", delta=None)
        st.markdown("Course Records have NO enrolled learners who have completed the course")
        st.markdown("")
        st.metric(label="Number of Courses with 0% Completion Rate", value="206", delta="930%")
        st.markdown("930 percent more than as compared to courses with 100% completion rate")
        st.markdown("")
        st.markdown("")
        st.markdown("")

    course_completion_rates = df.groupby('Course Name')['Completed'].apply(
        lambda x: x.eq('Yes').sum() / x.notna().sum()).reset_index(name='Completion Rate')

    st.markdown("### Course Wise Dataframe With Completion Rates")
    st.dataframe(course_completion_rates)

    enrolled_learners_course = df[df['Enrollment Time'] != 'Not Started Yet']

    # Grouping by the 'Program Name' and 'Course Name'
    program_completion_summary = enrolled_learners_course.groupby(['Program Name', 'Course Name'])['Completed'].agg(
        TotalLearners='count',
        CompletedLearners=lambda x: x.eq('Yes').sum(),
        CompletionRate=lambda x: x.eq('Yes').sum() / x.notna().sum(),
        CompletionRatePercentage=lambda x: (x.eq('Yes').sum() / x.notna().sum()) * 100
    ).reset_index()
    program_completion_summary_sorted = program_completion_summary.sort_values(by='CompletionRatePercentage', ascending=False)
    st.markdown("### Program and Couse Completion Summary Dataframe")
    st.dataframe(program_completion_summary_sorted)
    completion_rate_counts = program_completion_summary_sorted['CompletionRatePercentage'].round(
        1).value_counts().reset_index()
    completion_rate_counts.columns = ['CompletionRatePercentage', 'Frequency']

    # Convert to dictionary
    completion_rate_dict = dict(
        zip(completion_rate_counts['CompletionRatePercentage'], completion_rate_counts['Frequency']))

    rounded_completion_rate_counts = completion_rate_counts.copy()
    rounded_completion_rate_counts['CompletionRatePercentage'] = rounded_completion_rate_counts[
        'CompletionRatePercentage'].round().astype(int)
    rounded_completion_rate_dict = dict(
        zip(rounded_completion_rate_counts['CompletionRatePercentage'], rounded_completion_rate_counts['Frequency']))

    rounded_completion_rate_df = pd.DataFrame(list(rounded_completion_rate_dict.items()),
                                              columns=['CompletionRatePercentage', 'Frequency'])

    ranges = [
        {'label': '0%', 'start': -0.1, 'end': 0},
        {'label': '1-30%', 'start': 0.1, 'end': 30},
        {'label': '31-65%', 'start': 30.1, 'end': 65},
        {'label': '66-99%', 'start': 65.1, 'end': 99.9},
        {'label': '100%', 'start': 100, 'end': 100}
    ]

    # Categorize completion rates
    rounded_completion_rate_df['Category'] = pd.cut(
        rounded_completion_rate_df['CompletionRatePercentage'],
        bins=[range['start'] for range in ranges] + [float('inf')],
        labels=[range['label'] for range in ranges],
        right=False
    )

    # Group by category and sum the frequencies
    category_counts = rounded_completion_rate_df.groupby('Category')['Frequency'].sum().reset_index()
    formal_palette = ["#4C72B0", "#7F8C8D", "#95A5A6", "#BDC3C7", "#ECF0F1"]
    # Plotly Express pie chart
    fig = px.pie(category_counts, names='Category', values='Frequency',
                 title="Distribution of Completion Rates Showing Percentage Of Learners",
                 color_discrete_sequence=["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],  # Vibrant colors
                 width=800, height=600)  # Larger size

    # Display the chart using Streamlit
    st.plotly_chart(fig)

    enrolled_learners = df[df['Enrollment Time'] != 'Not Started Yet']

    # Group by 'Course Name' and calculate completion rates
    course_completion_rates = enrolled_learners.groupby('Course Name')['Completed'].agg(
        TotalLearners='count',
        CompletedLearners=lambda x: x.eq('Yes').sum(),
        CompletionRate=lambda x: x.eq('Yes').sum() / x.notna().sum()
    ).reset_index()

    # Set your threshold for significantly higher or lower completion rates (adjust as needed)
    threshold_higher = 0.8
    threshold_lower = 0.04

    # Identify courses with significantly higher completion rates
    higher_completion_courses = course_completion_rates[course_completion_rates['CompletionRate'] > threshold_higher]

    # Identify courses with significantly lower completion rates
    lower_completion_courses = course_completion_rates[course_completion_rates['CompletionRate'] < threshold_lower]

    st.markdown("### Higher Completion Rate Courses DataFrame")
    st.markdown("The Courses having completion rate greater than 80% are in the following list")
    st.dataframe(higher_completion_courses)
    st.markdown("### Lower Completion Rate Courses DataFrame")
    st.markdown("The Courses having completion rate lower than 0.4% are in the following list")
    st.dataframe(lower_completion_courses)

    st.markdown("## Learning Hours Analysis")
    st.markdown("")
    completed_learners = df[(df['Completed'] == 'Yes') & df['Learning Hours Spent'].notna()]
    average_learning_hours = completed_learners['Learning Hours Spent'].mean()
    rounded_average_learning_hours = round(average_learning_hours, 2)
    completed_learners_all = df[df['Learning Hours Spent'].notna()]
    average_learning_hours_all = completed_learners_all['Learning Hours Spent'].mean()
    rounded_average_learning_hours_all = round(average_learning_hours_all,2)

    col_correlation_1, col_correlation_2 = st.columns(2)

    with col_correlation_1:
        st.metric(label="Average Learning Hours", value=rounded_average_learning_hours, delta=None)
        st.markdown("For Learners Who Have Completed the Course")
        st.markdown("")
        st.markdown("")
        st.metric(label="Correlation Between Learning Hours And Course Grade", value=0.577, delta=None)
        st.markdown("Positive Correlation ,  More The Learning Hours Higher Chnace Of Getting Good Grade")
        st.markdown("")
        st.markdown("")

    with col_correlation_2:
        st.metric(label="Average Learning Hours", value=rounded_average_learning_hours_all, delta=None)
        st.markdown("For Learners Who Have Started the Course")
        st.markdown("")
        st.markdown("")
        st.metric(label="Percentage Of Learners With Negative Correlation", value="2.8%", delta=None)
        st.markdown("For Learners Who Have Started the Course")
        st.markdown("")
        st.markdown("")


    st.markdown("#### Avg Learning Hours Dataframe For Completed Courses")
    average_learning_hours_per_course = completed_learners.groupby('Course Name')[
        'Learning Hours Spent'].mean().reset_index(name='Average Learning Hours')
    st.dataframe(average_learning_hours_per_course)
    # Sort the DataFrame by 'Average Learning Hours' in descending order
    # Sort the DataFrame by 'Average Learning Hours' in descending order
    top_10_courses = average_learning_hours_per_course.sort_values(by='Average Learning Hours', ascending=False).head(
        10)

    # Create an Altair chart
    chart_courses = alt.Chart(top_10_courses).mark_bar().encode(
        x=alt.X('Average Learning Hours:Q', title='Average Learning Hours'),
        y=alt.Y('Course Name:N', title='Course Name'),
        color=alt.value('#3182bd')  # Use the Streamlit theme color (light blue)
    ).properties(
        width=800,
        height=400
    ).interactive()

    # Display the chart using Streamlit
    st.markdown("#### Top 10 Courses with the Most Average Learning Hours")
    st.altair_chart(chart_courses)
    # Group by 'Course Name' and calculate the mean of 'Learning Hours Spent'
    # Group by 'Course Name' and calculate the mean of 'Learning Hours Spent'
    average_learning_hours_per_course_all = completed_learners_all.groupby('Course Name')[
        'Learning Hours Spent'].mean().reset_index(name='Average Learning Hours')

    # Sort the DataFrame by 'Average Learning Hours' in descending order
    top_courses_all = average_learning_hours_per_course_all.sort_values(by='Average Learning Hours',
                                                                        ascending=False).head(10)

    # Define a custom color palette
    color_palette = alt.Scale(
        range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
               '#17becf'])

    # Create an Altair chart with the custom color palette
    chart_courses_all = alt.Chart(top_courses_all).mark_bar().encode(
        x=alt.X('Average Learning Hours:Q', title='Average Learning Hours'),
        y=alt.Y('Course Name:N', title='Course Name', sort='-x'),
        color=alt.Color('Course Name:N', scale=color_palette),
        tooltip=['Course Name', 'Average Learning Hours']
    ).properties(
        width=800,
        height=400
    ).interactive()

    # Display the chart using Streamlit
    st.markdown("#### Top 10 Courses with the Most Average Learning Hours (All Learners)")
    st.altair_chart(chart_courses_all)
    st.markdown("")
    st.markdown("### Avg Learning Hours Dataframe For All Started Courses")
    st.dataframe(average_learning_hours_per_course_all)

    # Convert 'Completed' column to numeric
    learners_atleast_started_the_course = df[(df['Completed'] != 'Not Started Yet')]
    # Convert 'Completed' column to numeric
    learners_atleast_started_the_course['Completed'] = learners_atleast_started_the_course['Completed'].map(
        {'Yes': 1, 'No': 0})

    # Add a constant to the independent variable


    st.markdown("")
    st.markdown("")
   # st.pyplot()
    local_image_path = "img.png"
    st.image(local_image_path, caption='Your Image Caption', use_column_width=True)
    st.markdown("")
    st.markdown("#### Correlation Value increases from Not Completed To Completed Courses Steepingly")
    st.markdown("Not Completed Courses Have been generally aborted after leaning few hours")
    st.markdown("Completed Courses tend to Have More Leaning Hours Than Not Completed Ones")
    st.markdown("")

    course_wise_correlation = learners_atleast_started_the_course.groupby('Course Name').apply(
        lambda group: group['Learning Hours Spent'].corr(group['Course Grade'])).reset_index(name='Correlation')
    st.markdown("#### Course Wise Correlation DataFrame")
    st.dataframe(course_wise_correlation)


    # Define the bins for categorization
    bins = [-np.inf, -1, 0, 1, np.inf]

    # Categorize the correlation values into bins and convert to strings
    correlation_bins = pd.cut(course_wise_correlation['Correlation'], bins=bins).astype(str)
    correlation_counts = correlation_bins.value_counts(dropna=False)

    # Set a custom color palette
    custom_palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261"]

    # Create a Pie chart using Plotly Express
    fig_2 = px.pie(names=correlation_counts.index, values=correlation_counts,
                 title="Distribution of Correlation Values",
                 color=correlation_counts.index,
                 color_discrete_sequence=custom_palette)

    # Create a legend
    fig_2.update_layout(
        legend=dict(title='Correlation Range', orientation='h', yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Display the chart using Streamlit
    st.plotly_chart(fig_2)
    st.markdown("")
    st.markdown("## Completion Status Analysis")
    #st.dataframe(df['Completed'])
    df_pro = pd.DataFrame({'Completed': ['Not Started Yet', 'No', 'Yes'],
                       'proportion': [73.805576, 16.021100, 10.173323]})

    # Altair Bar Chart
    chart_pro = alt.Chart(df_pro).mark_bar().encode(
        x='Completed:N',
        y='proportion:Q',
        color=alt.Color('Completed:N', scale=alt.Scale(scheme='category10')),  # You can use a different color scheme
        tooltip=['Completed:N', 'proportion:Q']
    ).properties(
        width=alt.Step(80),
        title='Percentage of Learners by Completion Status'
    )

    # Display the chart using Streamlit
    st.markdown("")
    st.markdown("")
    st.altair_chart(chart_pro, use_container_width=True)
    st.markdown("")
    st.markdown("")


    st.markdown("## Skills Insights")
    skills_counts = df['Skills Learned'].str.lower().str.split(';').explode().str.strip().value_counts()
    df_skills_counts = pd.DataFrame({'Skill': skills_counts.index, 'Count': skills_counts.values})
    st.markdown("")
    st.markdown("#### Skills DataFrame")
    st.dataframe(df_skills_counts)
    df_cleaned = df.dropna(subset=['Skills Learned'])

    # Creating a dictionary to store the count of programs for each skill present in the dataset
    skill_popularity = {}

    # Iterate over rows and populate the dictionary
    for index, row in df_cleaned.iterrows():
        skills = set(row['Skills Learned'].split('; '))

        for skill in skills:
            if skill not in skill_popularity:
                skill_popularity[skill] = 1
            else:
                skill_popularity[skill] += 1

    # Converting the dictionary to a DataFrame
    popularity_df = pd.DataFrame(list(skill_popularity.items()), columns=['Skill', 'Popularity'])

    # Sort the DataFrame by popularity in descending order
    popularity_df = popularity_df.sort_values(by='Popularity', ascending=False)

    # Display the top skills
    top_skills = popularity_df.head(11)
    custom_color_scheme = alt.Scale(
        range=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
               "#17becf"])

    # Create an Altair bar chart
    chart_skills = alt.Chart(top_skills).mark_bar().encode(
        x=alt.X('Skill:N', title='Skill'),
        y=alt.Y('Popularity:Q', title='Popularity'),
        color=alt.Color('Skill:N', scale=custom_color_scheme),
        tooltip=['Skill:N', 'Popularity:Q']
    ).properties(
        width=alt.Step(80),
        height=600,
        # You can adjust the width as needed
        title='Top 10 Most Popular Skills Across Programs'
    )
    st.markdown("")
    # Display the chart using Streamlit
    st.altair_chart(chart_skills, use_container_width=True)

    st.markdown("")
    st.markdown("")
    st.markdown("### Enrollment Time & Completion Time Distribution Graph")# Convert the datetime columns to year-month format
    df['Enrollment Time'] = pd.to_datetime(df['Enrollment Time'], errors='coerce')
    df['Completion Time'] = pd.to_datetime(df['Completion Time'], errors='coerce')

    # Extract year and month from 'Enrollment Time' and 'Completion Time' columns
    df['Enrollment YearMonth'] = df['Enrollment Time'].dt.to_period('M')
    df['Completion YearMonth'] = df['Completion Time'].dt.to_period('M')

    # Filter out records with '1900-01-01' in either 'Enrollment Time' or 'Completion Time'
    df_filtered = df[(df['Enrollment YearMonth'].dt.to_timestamp() != '1900-01-01') &
                     (df['Completion YearMonth'].dt.to_timestamp() != '1900-01-01')]
    df['Enrollment Time'] = pd.to_datetime(df['Enrollment Time'], errors='coerce')
    df['Completion Time'] = pd.to_datetime(df['Completion Time'], errors='coerce')
    df_filtered = df[
        ~((df['Enrollment Time'] == '1900-01-01 00:00:00') & (df['Completion Time'] == '1900-01-01 00:00:00'))]

    # Group by enrollment and completion time and count the number of records
    enrollment_counts = df_filtered.groupby('Enrollment YearMonth').size()
    completion_counts = df_filtered.groupby('Completion YearMonth').size()

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({'Enrollment': enrollment_counts, 'Completion': completion_counts})

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(plot_data.index.astype(str), plot_data['Enrollment'], marker='o', label='Enrollment')
    plt.plot(plot_data.index.astype(str), plot_data['Completion'], marker='o', label='Completion')

    # Annotate each marker with the number of learners
    for x, y_enroll, y_complete in zip(plot_data.index.astype(str), plot_data['Enrollment'], plot_data['Completion']):
        plt.annotate(f'{y_enroll}', (x, y_enroll), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.annotate(f'{y_complete}', (x, y_complete), textcoords="offset points", xytext=(0, -15), ha='center')

    plt.xlabel('Year-Month')
    plt.ylabel('Number of Learners')
    plt.title('Enrollment and Completion Trends Over Time (Excluding 1900-01)')
    plt.legend()
    plt.xticks(rotation=45, ha='right')

    # Change the label on the x-axis tick corresponding to '1900-01'
    labels = [item.get_text() if item.get_text() != '1900-01' else 'Not Completed Yet' for item in
              plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(labels)

    plt.tight_layout()
    st.markdown("")
    st.markdown("")
    # Display the Matplotlib plot in Streamlit
    st.pyplot()
    st.markdown("")
    st.markdown("")
    st.markdown("Above Plot shows overall trend in the enrollment and the completeion time by the learners who are enrolled in the courses ")
    st.markdown("")
    st.markdown("")
    st.markdown("## Correlation Analysis")
    st.markdown("")
    st.markdown("")

    df_not_completed_status = df[(df['Completed'] == 'No') & (df['Course Grade'] != 0)].copy()
    correlation_not_completed = df_not_completed_status['Learning Hours Spent'].corr(
        df_not_completed_status['Course Grade'])
    df_completed_status = df[(df['Completed'] == 'Yes') & (df['Course Grade'] != 0)].copy()
    correlation_completed = df_completed_status['Learning Hours Spent'].corr(df_completed_status['Course Grade'])
    df_all_courses = df[(df['Completed'] != 'Not Started Yet') & (df['Course Grade'] != 0)].copy()
    correlation_all_courses = df_all_courses['Learning Hours Spent'].corr(df_all_courses['Course Grade'])

    correlation_not_completed_round=round(correlation_not_completed,2)
    correlation_completed_round = round(correlation_completed, 2)
    correlation_all_courses_round = round(correlation_all_courses, 2)
    st.metric(label="Correlation Between Course Grade & Duration For Not Completed Courses", value=correlation_not_completed_round, delta=None)
    st.markdown("")
    st.markdown("")
    st.metric(label="Correlation Between Course Grade & Duration For Completed Courses",
                  value=correlation_completed_round, delta=None)
    st.markdown("")
    st.markdown("")
    st.metric(label="Correlation Between Course Grade & Duration For All Courses",
                  value=correlation_all_courses_round, delta=None)
    st.markdown("")
    st.markdown("")
    st.markdown("#### Correlation for Not Completed Courses: 0.38")
    st.markdown("##### Interpretation")
    st.markdown("There is a moderate positive correlation between learning hours and course grades for learners who have not completed their courses. As learning hours increase, there is a tendency for grades to improve.")
    st.markdown("")
    st.markdown("")
    st.markdown("#### Correlation for Completed Courses: 0.01")
    st.markdown("##### Interpretation")
    st.markdown("There is a very weak positive correlation between learning hours and course grades for learners who have completed their courses. The relationship between learning hours and grades for completed courses is nearly negligible.")
    st.markdown("")
    st.markdown("")
    st.markdown("#### Correlation for All Courses: 0.27")
    st.markdown("##### Interpretation")
    st.markdown("When considering all courses, including both completed and not completed, there is a moderate positive correlation between learning hours and course grades. The overall relationship suggests that, on average, more learning hours are associated with higher grades.")
    st.markdown("")
    st.markdown("#### Comparison")
    st.markdown("The highest correlation is observed for not completed courses, indicating a relatively stronger relationship between learning hours and grades for learners still in progress.")
    st.markdown("The correlation for completed courses is very low, suggesting that the time spent learning may not be a significant predictor of grades for those who have already completed the course.")
    st.markdown("")
    # Create a scatter plot using Altair
    scatter_plot = alt.Chart(df).mark_circle().encode(
        x='Learning Hours Spent:Q',
        y='Course Grade:Q',
        color='Completed:N'
    ).properties(
        title='Learning Hours vs Course Grade',
        width=800,
        height=600
    )

    # Display the chart using Streamlit
    st.altair_chart(scatter_plot)
    st.markdown("")
    st.markdown("")
    st.markdown("## Top Performing Learners Analysis")
    st.markdown("")
    st.markdown("A Learner is said to be a top performer iff his course grade>=85 and no of courses completed>=3")
    # The criteria for being a top performer in the dataset
    st.markdown("")
    col_hi_1, col_hi_2 = st.columns(2)
    with col_hi_1:
        st.metric(label="Average Learning Hours", value=5.31, delta=-0.17)
        st.markdown("High Performers Take 0.17 hrs less on an avg per course")
        st.markdown("")
        st.markdown("")
        st.metric(label="High Perfromers", value=67, delta=None)
        st.markdown("Unique High Performing Learners among total of 1409")
        st.markdown("")
        st.markdown("")
        st.metric(label="TQM & E&P", value="25.4%", delta=None)
        st.markdown("Division With Highest percentage of top perfromers")
        st.markdown("")
        st.markdown("")
        st.metric(label="Automation", value="12%", delta=None)
        st.markdown("Department With Highest percentage of top perfromers")
        st.markdown("")
        st.markdown("")



    with col_hi_2:
        st.metric(label="Most Popular Skills", value="33%", delta=None)
        st.markdown("High performers go for courses with skills Analytics,Leadership & Management")
        st.markdown("")
        st.markdown("")
        st.metric(label="Most Popular Program", value="62%", delta=None)
        st.markdown("Most Common Program Among High Perfromers is School Of Anlytic-Basic")
        st.markdown("")
        st.markdown("")
        st.metric(label="Most Popular Course", value="14.4%", delta=None)
        st.markdown("Most Common Course Among High Perfromers is Data-What It Is And What We Can Do With It")
        st.markdown("")
        st.markdown("")
        st.metric(label="Percenatge Of High Perfromers", value="26%", delta=None)
        st.markdown("Of Top Performers Not Tagged To Any Group")
        st.markdown("")
        st.markdown("")





    score_threshold = 85
    courses_completed_threshold = 3

    # Filtering the DataFrame on the basis of course grade here
    high_performers_df = df[(df['Course Grade'] > score_threshold) & (df['Completed'] == 'Yes')]

    # Filtering the DataFrame on the basis of no of skills in that course here
    high_performers_df = high_performers_df.groupby('Name').filter(lambda x: len(x) >= courses_completed_threshold)
    st.markdown("#### Top Performers Dataframe")
    st.dataframe(high_performers_df)
    top_programs = high_performers_df['Program Name'].value_counts().nlargest(10).reset_index()

    # Rename the columns
    top_programs.columns = ['Program Name', 'Number of Learners']

    # Create a bar chart using Altair
    chart_hi = alt.Chart(top_programs).mark_bar().encode(
        x='Number of Learners:Q',
        y=alt.Y('Program Name:N',
                sort=alt.EncodingSortField(field='Number of Learners', op='values', order='descending')),
        color=alt.Color('Number of Learners:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Program Name', 'Number of Learners']
    ).properties(
        width=800,
        height=500,
        title='Most Popular Programs Among High-Performing Learners'
    )

    # Display the chart using Streamlit
    st.markdown("")
    st.markdown("")
    st.altair_chart(chart_hi)

    # Convert to a DataFrame for Altair
    plt.figure(figsize=(10, 8))
    top_skills = high_performers_df['Skills Learned'].str.split('; ').explode().value_counts().nlargest(10)
    plt.pie(top_skills, labels=top_skills.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Blues"))
    plt.title('Most Popular Skills Among High-Performing Learners')
    #plt.show()

    # Display the chart using Streamlit
    st.markdown("")
    st.markdown("### Top 10 Skills Among High Performers")
    st.markdown("")
    st.pyplot()
    st.markdown("")
    st.markdown("")

    data_hi = high_performers_df['Course Name'].value_counts().nlargest(10).reset_index()
    data_hi.columns = ['Course Name', 'Number of Learners']

    # Streamlit App
    st.markdown('### Most Popular Courses Among High-Performing Learners')

    # Altair Chart
    chart_pop_hi = alt.Chart(data_hi).mark_bar().encode(
        x='Number of Learners:Q',
        y=alt.Y('Course Name:N', sort='-x'),
        color=alt.Color('Number of Learners:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Course Name', 'Number of Learners']
    ).properties(
        width=600,
        height=400
    )

    # Display the chart using Streamlit
    st.altair_chart(chart_pop_hi, use_container_width=True)
    st.markdown("")
    program_skills_dict = {}

    for index, row in df.iterrows():
        program_name = row['Program Name']
        skills = row['Skills Learned'].split('; ')

        if program_name not in program_skills_dict:
            program_skills_dict[program_name] = {'total_courses': 0, 'skills': {}}

        program_skills_dict[program_name]['total_courses'] += 1

        for skill in skills:
            if skill in program_skills_dict[program_name]['skills']:
                program_skills_dict[program_name]['skills'][skill] += 1
            else:
                program_skills_dict[program_name]['skills'][skill] = 1

    # Creating a list of dictionaries to store program name, the most popular skill, and the percentage
    data = []

    for program, program_data in program_skills_dict.items():
        most_popular_skill = max(program_data['skills'], key=program_data['skills'].get)
        percentage_courses_with_skill = (program_data['skills'][most_popular_skill] / program_data[
            'total_courses']) * 100
        data.append({
            'Program Name': program,
            'Most Popular Skill': most_popular_skill,
            'Popularity Percentage': percentage_courses_with_skill
        })

    # Creating a dataframe from the list of dictionaries
    program_most_popular_skill = pd.DataFrame(data)

    chart = alt.Chart(program_most_popular_skill.head(10)).mark_bar().encode(
        x='Popularity Percentage:Q',
        y=alt.Y('Program Name:N', sort='-x'),
        color=alt.Color('Most Popular Skill:N', scale=alt.Scale(scheme='viridis')),
        tooltip=['Program Name', 'Most Popular Skill', 'Popularity Percentage']
    ).properties(
        width=800,
        height=400
    )
    st.markdown("#### Most Popular Skills Across Programs Preferred By Top Performers")
    # Display the chart using Streamlit
    st.altair_chart(chart, use_container_width=True)
    st.markdown("")
    df['Completion Status'] = df['Completed'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Group by person and aggregate information
    person_data = df.groupby('Name').agg({
        'Course Name': 'count',  # Number of courses enrolled
        'Completion Status': 'sum'  # Number of courses completed
    }).reset_index()

    # Calculate completion percentage
    person_data['Completion Percentage'] = (person_data['Completion Status'] / person_data['Course Name']) * 100

    # Rename columns for clarity
    person_data.columns = ['Person', 'Enrolled Courses', 'Completed Courses', 'Completion Percentage']

    #person_data
    st.markdown("#### List Of Learners Who Have Completed Atleast One Course")
    completed_persons_new_dataframe = person_data[person_data['Completed Courses'] > 0]
    st.dataframe(completed_persons_new_dataframe)
    unique_persons_count = high_performers_df['Name'].nunique()
   # print("Number of unique persons:", unique_persons_count)
    person_courses_dict = {}

    # Populate the dictionary
    for _, row in high_performers_df.iterrows():
        person = row['Name']
        course = row['Course Name']

        if person in person_courses_dict:
            person_courses_dict[person].append(course)
        else:
            person_courses_dict[person] = [course]
    #person_courses_dict
    division_unique_persons = high_performers_df.groupby('Division')['Name'].nunique().reset_index()

    # Rename the column for clarity
    division_unique_persons.columns = ['Division', 'Number of Unique Persons']

    # Calculate the percentage of high performers in each division
    division_unique_persons['Percentage of High Performers'] = (division_unique_persons[
                                                                    'Number of Unique Persons'] / 67) * 100
    #division_unique_persons
    chart_div_u = alt.Chart(division_unique_persons).mark_bar().encode(
        x='Percentage of High Performers:Q',
        y=alt.Y('Division:N', sort='-x'),
        color=alt.Color('Percentage of High Performers:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Division', 'Percentage of High Performers']
    ).properties(
        width=800,
        height=500
    )
    st.markdown("")
    st.markdown("")
    st.markdown("### High Performers across Divisions")
    # Display the chart using Streamlit
    st.altair_chart(chart_div_u, use_container_width=True)
    st.markdown("")
    st.markdown("")
    st.markdown("### High Performers across Groups")
    st.markdown("")
    st.markdown("")
    group_unique_persons = high_performers_df.groupby('Group')['Name'].nunique().reset_index()
    group_unique_persons.columns = ['Group', 'Number of Unique Persons']
    group_unique_persons['Percentage of High Performers'] = (group_unique_persons[
                                                                 'Number of Unique Persons'] / 67) * 100
    #group_unique_persons
    chart = alt.Chart(group_unique_persons).mark_bar().encode(
        x=alt.X('Percentage of High Performers:Q', title='Percentage'),  # Explicitly specifying data type
        y=alt.Y('Group:N', sort='-x', title='Group'),
        color=alt.Color('Percentage of High Performers:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Group', 'Percentage of High Performers']
    ).properties(
        width=800,
        height=400
    )

    # Display the chart using Streamlit
    st.altair_chart(chart, use_container_width=True)
    st.markdown("")
    st.markdown("")
    st.markdown("### High Performers across Departments")
    st.markdown("")
    st.markdown("")
    length = 67  # Total number of unique high performers
    department_unique_persons = high_performers_df.groupby('Department')['Name'].nunique().reset_index()
    department_unique_persons.columns = ['Department', 'Number of Unique Persons']
    department_unique_persons['Percentage of High Performers'] = (department_unique_persons[
                                                                      'Number of Unique Persons'] / length) * 100
    #department_unique_persons

    chart_d = alt.Chart(department_unique_persons).mark_bar().encode(
        x=alt.X('Percentage of High Performers:Q', title='Percentage'),  # Explicitly specifying data type
        y=alt.Y('Department:N', sort='-x', title='Department'),
        color=alt.Color('Percentage of High Performers:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Department', 'Percentage of High Performers']
    ).properties(
        width=800,
        height=500
    )

    # Display the chart using Streamlit
    st.altair_chart(chart_d, use_container_width=True)
    st.markdown("")
    st.markdown("")
    st.markdown("### High Performers across Programs")
    st.markdown("")
    st.markdown("")
    total_high_performers = 67
    program_unique_persons = high_performers_df.groupby('Program Name')['Name'].nunique().reset_index()
    program_unique_persons.columns = ['Program Name', 'Number of Unique Persons']
    program_unique_persons['Percentage of High Performers'] = (program_unique_persons[
                                                                   'Number of Unique Persons'] / total_high_performers) * 100
    #program_unique_persons
    chart_p = alt.Chart(program_unique_persons).mark_bar().encode(
        x=alt.X('Percentage of High Performers:Q', title='Percentage'),  # Explicitly specifying data type
        y=alt.Y('Program Name:N', sort='-x', title='Program Name'),
        color=alt.Color('Percentage of High Performers:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Program Name', 'Percentage of High Performers']
    ).properties(
        width=800,
        height=500
    )

    # Display the chart using Streamlit
    st.altair_chart(chart_p, use_container_width=True)
    st.markdown("")
    st.markdown("")
    st.markdown("#### Percentage Of High Performers Across Courses")
    course_stats = high_performers_df.groupby('Course Name')['Name'].agg(
        ['nunique', lambda x: len(x) / len(high_performers_df) * 100]).reset_index()
    course_stats.columns = ['Course Name', 'Number of Unique Persons', 'Percentage of High Performers']
    st.dataframe(course_stats)
    st.markdown("")
    st.markdown("")
