import streamlit as st
import pandas as pd
from algorithm import genetic_algorithm
from algorithm import plot_flight_paths

# Set page configuration to "wide" layout
st.set_page_config(layout="wide")

# Initialize session state for tasks and UAVs
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'UAVs' not in st.session_state:
    st.session_state.UAVs = []
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False  # For file upload status

# File upload in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload Excel File for Tasks (Optional)", type=["xlsx"])

# Process uploaded file and validate columns
if uploaded_file:
    df_tasks = pd.read_excel(uploaded_file)
    expected_columns = ['location x', 'location y', 'type', 'tasktime(min)', 'deadline']
    if all(col in df_tasks.columns for col in expected_columns):
        st.session_state.tasks = df_tasks.to_dict('records')
        st.session_state.file_uploaded = True
        st.session_state.num_tasks = len(st.session_state.tasks)
    else:
        st.error(f"The uploaded file must contain the following columns: {expected_columns}")
        st.session_state.file_uploaded = False
else:
    if 'num_tasks' not in st.session_state:
        st.session_state.num_tasks = 5

# Sidebar configuration for tasks and UAVs
st.sidebar.title("Settings")
st.sidebar.header("Task and UAV Configuration")

num_tasks = st.sidebar.number_input('Number of Tasks', min_value=1, max_value=100, key="num_tasks")
num_UAVs = st.sidebar.number_input('Number of UAVs', min_value=1, max_value=20, value=st.session_state.get('num_UAVs', 3), key="num_UAVs")

# Run algorithm button
if st.sidebar.button("Run Algorithm"):
    tasks = st.session_state.tasks
    UAVs = st.session_state.UAVs

    # Run the algorithm
    best_solution, fitness_history = genetic_algorithm(tasks, UAVs)

    # Group tasks by UAV based on best_solution
    uav_task_allocation = {uav["NO"]: [] for uav in UAVs}
    for task_idx, uav_idx in enumerate(best_solution):
        uav_task_allocation[UAVs[uav_idx - 1]["NO"]].append(task_idx + 1)  # Task index +1 for 1-based indexing

    # Display the task allocation table for each UAV
    uav_task_df = pd.DataFrame([
        {"UAV NO": uav_no, "Assigned Tasks": ", ".join(map(str, tasks))}
        for uav_no, tasks in uav_task_allocation.items()
    ])

    st.title('UAV Task Assignment')
    # Use a container to ensure full width for both the table and chart
    with st.container():
        # Create two equal-width columns for the table and the chart
        col1, col2 = st.columns(2)


        with col1:
            st.subheader("Best Solution Task Allocation")

            # 使用HTML和Markdown格式化表格的输出
            st.markdown("""
            <style>
            table {
                width: 80%;
                border-collapse: collapse;
            }
            th, td {
                padding: 10px;
                border: 1px solid #ddd;
                text-align: center;
                background-color: white;
                color: black;
            }
            </style>
            """, unsafe_allow_html=True)

            # 构建表格HTML内容
            table_html = "<table><tr><th>UAV NO</th><th>Assigned Tasks</th></tr>"
            for _, row in uav_task_df.iterrows():
                # 将每个任务单独放入一个框中
                tasks_html = " ".join([
                                          f"<span style='display:inline-block; border:1px solid #ddd; padding:5px; margin:2px;'>{task}</span>"
                                          for task in row['Assigned Tasks'].split(", ")])
                table_html += f"<tr><td>{row['UAV NO']}</td><td>{tasks_html}</td></tr>"
            table_html += "</table>"

            # 输出表格
            st.markdown(table_html, unsafe_allow_html=True)

        with col2:
            st.subheader("Fitness History")
            st.line_chart(fitness_history)  # Display the fitness history chart

    plot_flight_paths(best_solution, UAVs, tasks)

    for uav_no, task_indices in uav_task_allocation.items():
        if task_indices:
            last_task_index = task_indices[-1] - 1
            st.session_state.UAVs[uav_no - 1]['location x'] = st.session_state.tasks[last_task_index]["location x"]
            st.session_state.UAVs[uav_no - 1]['location y'] = st.session_state.tasks[last_task_index]["location y"]

# Title and data input handling
st.title('UAV Task Assignment')

# Task and UAV data initialization
if not st.session_state.file_uploaded:
    while len(st.session_state.tasks) < num_tasks:
        st.session_state.tasks.append({
            "NO": len(st.session_state.tasks) + 1,
            "location x": 0.0,
            "location y": 0.0,
            "type": 'O',
            "tasktime(min)": 60,
            "deadline": 120
        })
    while len(st.session_state.tasks) > num_tasks:
        st.session_state.tasks.pop()

while len(st.session_state.UAVs) < num_UAVs:
    st.session_state.UAVs.append({
        "NO": len(st.session_state.UAVs) + 1,
        "speed": 1.0,
        "location x": 0.0,
        "location y": 0.0,
        "type": 'X'
    })
while len(st.session_state.UAVs) > num_UAVs:
    st.session_state.UAVs.pop()

# Display Task Data Table and Task Inputs in two columns with 1:2 width ratio
st.subheader('Task Data and Inputs')
task_col1, task_col2 = st.columns([1, 2])

with task_col1:
    st.write(pd.DataFrame(st.session_state.tasks))

with task_col2:
    with st.expander("Task Inputs", expanded=True):
        for i, task in enumerate(st.session_state.tasks):
            st.subheader(f'Task {i + 1}')
            col1, col2, col3, col4, col5 = st.columns(5)
            task['location x'] = col1.number_input('Location X', value=float(task['location x']), step=0.1, key=f'x_{i}')
            task['location y'] = col2.number_input('Location Y', value=float(task['location y']), step=0.1, key=f'y_{i}')
            task['type'] = col3.selectbox('Type', ['O', 'C', 'G', 'F'], key=f'type_{i}', index=['O', 'C', 'G', 'F'].index(task['type']))
            task['tasktime(min)'] = col4.number_input('Duration (min)', value=int(task['tasktime(min)']), key=f'time_{i}')
            task['deadline'] = col5.number_input('Deadline', value=int(task['deadline']), key=f'deadline_{i}')

# Display UAV Data Table and UAV Inputs in two columns below with 1:2 width ratio
st.subheader('UAV Data and Inputs')
uav_col1, uav_col2 = st.columns([1, 2])

with uav_col1:
    st.write(pd.DataFrame(st.session_state.UAVs))

with uav_col2:
    with st.expander("UAV Inputs", expanded=True):
        for i, uav in enumerate(st.session_state.UAVs):
            st.subheader(f'UAV {i + 1}')
            col1, col2, col3, col4 = st.columns(4)
            uav['speed'] = col1.number_input('Speed', value=float(uav['speed']), step=1.0, key=f'speed_{i}')
            uav['location x'] = col2.number_input('Location X', value=float(uav['location x']), step=0.1, key=f'UAVx_{i}')
            uav['location y'] = col3.number_input('Location Y', value=float(uav['location y']), step=0.1, key=f'UAVy_{i}')
            uav['type'] = col4.selectbox('Type', ['X', 'Y', 'Z'], key=f'type_UAV_{i}', index=['X', 'Y', 'Z'].index(uav['type']))
