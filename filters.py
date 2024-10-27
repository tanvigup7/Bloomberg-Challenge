# filters.py
def apply_filters(data):
    filtered_data = data.copy()
    for column in filtered_data.columns:
        filtered_data = filtered_data[filtered_data[column] != ""].dropna()
    return filtered_data
