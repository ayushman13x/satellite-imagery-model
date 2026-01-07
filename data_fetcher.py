import pandas as pd
import os
import requests

def download_all_images(csv_path, output_folder, api_token, zoom=16, size="400x400"):
    """
    Downloads satellite images based on lat/long from an Excel/CSV file.
    """
    df = pd.read_excel(csv_path) if csv_path.endswith('.xlsx') else pd.read_csv(csv_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Starting download for {len(df)} rows...")

    for index, row in df.iterrows():
        house_id = row['id']
        lat = row['lat']
        lon = row['long']
        
        filepath = os.path.join(output_folder, f"{house_id}.jpg")
        
        if os.path.exists(filepath):
            continue
            
        url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom},0/{size}?access_token={api_token}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed ID {house_id}: {response.status_code}")
        
        except Exception as e:
            print(f"Error at ID {house_id}: {e}")

        if (index + 1) % 50 == 0:
            print(f"Progress: {index + 1}/{len(df)} images processed.")

    print("Download process finished!")