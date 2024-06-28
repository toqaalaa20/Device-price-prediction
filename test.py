import requests

# Define the API endpoint URL
api_url = 'http://127.0.0.1:5000/predict'  # Replace with your API endpoint URL

# Define the device specifications for prediction (similar to your test data)
device_data = {
    'battery_power': 842,
    'blue': 0,
    'clock_speed': 2.2,
    'dual_sim': 0,
    'fc': 1,
    'four_g': 0,
    'int_memory': 7,
    'm_dep': 0.6,
    'mobile_wt': 188,
    'n_cores': 2,
    'pc': 2,
    'px_height': 20,
    'px_width': 756,
    'ram': 2549,
    'sc_h': 9,
    'sc_w': 7,
    'talk_time': 19,
    'three_g': 0,
    'touch_screen': 0,
    'wifi': 1
}

# Send POST request to the API endpoint with device data
response = requests.post(api_url, json=device_data)

# Print the API response
print(response.json())
