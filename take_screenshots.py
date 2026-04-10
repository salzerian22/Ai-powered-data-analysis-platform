import time
from playwright.sync_api import sync_playwright
import os

pages = [
    ("", "home"),
    ("Data_Cleaning", "data_cleaning"),
    ("Outlier_Detection", "outlier_detection"),
    ("Data_Quality", "data_quality"),
    ("Visualization", "visualization"),
    ("AI_Insights", "ai_insights"),
    ("Predictions", "predictions"),
    ("Export_Report", "export_report"),
]

output_dir = "screenshots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        for path, name in pages:
            url = f"http://localhost:8501/{path}"
            print(f"Navigating to {url}...")
            
            # Wait with a timeout
            try:
                page.goto(url, timeout=30000, wait_until="networkidle")
            except Exception as e:
                print(f"Goto error for {name}: {e}")
                
            # Streamlit often has a small delay for its UI elements to render completely.
            # wait for specific streamlit elements to avoid capturing loading screens
            try:
                page.wait_for_selector('.stApp', state='visible', timeout=15000)
                # Additional wait for dynamic content
                page.wait_for_timeout(3000)
            except Exception as e:
                print(f"Wait error for {name}: {e}")

            screenshot_path = os.path.join(output_dir, f"{name}.png")
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"Saved: {screenshot_path}")

        browser.close()
        print("Done!")

if __name__ == "__main__":
    # Wait for Streamlit server to be ready
    time.sleep(10)
    run()
