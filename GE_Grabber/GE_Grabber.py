import json
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time

EXECUTABLE_PATH = "drivers/chromedriver.exe"
CHROME_OPTIONS = webdriver.ChromeOptions()
CHROME_OPTIONS.add_experimental_option("excludeSwitches", ['enable-automation'])

class GE_Grabber():
    def __init__(self, locations_json):
        with open(locations_json) as f:
            self.locations = json.load(f)
        self.driver = None
        self.image_count = 0
        # self.refresh_driver()

    def refresh_driver(self):
        if self.driver is not None:
            self.driver.quit()
        self.driver = webdriver.Chrome(executable_path=EXECUTABLE_PATH, options=CHROME_OPTIONS)

    def expand_shadow_element(self, element):
        shadow_root = self.driver.execute_script('return arguments[0].shadowRoot', element)
        return shadow_root

    def get_canvas_element(self):
        shadow_root1 = self.expand_shadow_element(
            self.driver.find_element(By.TAG_NAME, "earth-app")
        )
        shadow_root2 = self.expand_shadow_element(
            shadow_root1.find_element(By.TAG_NAME, "earth-view")
        )
        canvas = shadow_root2.find_element(By.TAG_NAME, "canvas")
        return canvas

    def click_clean_map_style(self):
        self.get_canvas_element().send_keys("m")

        shadow_root1 = self.expand_shadow_element(
            self.driver.find_element(By.TAG_NAME, "earth-app")
        )
        shadow_root2 = self.expand_shadow_element(
            shadow_root1.find_element(By.ID, "drawer-container")
        )
        shadow_root3 = self.expand_shadow_element(
            shadow_root2.find_element(By.ID, "mapstyle")
        )
        shadow_root3.find_element(By.TAG_NAME, "earth-radio-card").click()
        
        self.get_canvas_element().send_keys("m")

    def get_percentage_text(self):
        shadow_root1 = self.expand_shadow_element(
            self.driver.find_element(By.TAG_NAME, "earth-app")
        )
        shadow_root2 = self.expand_shadow_element(
            shadow_root1.find_element(By.TAG_NAME, "earth-view-status")
        )
        percentage_text_element = shadow_root2.find_element(By.ID, "percentage-text")
        
        return percentage_text_element.text

    def set_visibility_hidden(self, shadow_root, locator):
        self.driver.execute_script("arguments[0].style.visibility='hidden'", shadow_root.find_element(*locator))

    def hide_elements(self):
        shadow_root1 = self.expand_shadow_element(
            self.driver.find_element(By.TAG_NAME, "earth-app")
        )
        self.set_visibility_hidden(shadow_root1, (By.TAG_NAME, "earth-toolbar"))
        self.set_visibility_hidden(shadow_root1, (By.TAG_NAME, "earth-view-status"))
        self.set_visibility_hidden(shadow_root1, (By.ID, "earth-relative-elements"))

    def count(self):
        if self.image_count % 100 == 0:
            self.refresh_driver()
        self.image_count = self.image_count + 1            

    def visit_url(self, url):
        self.count()
        self.driver.get(url)
        self.driver.fullscreen_window()
        try:
            WebDriverWait(self.driver, 20).until(
                # might be good to wait on text element to equal "100%" instead
                # EC.invisibility_of_element((By.ID, "splash-screen-frame"))
                lambda x: self.get_percentage_text() == "100%"
            )
        finally:
            self.click_clean_map_style()
            self.hide_elements()

    # https://earth.google.com/web/@37.81991479,-122.4786796,-0.51074999a,2000d,35y,90.00158624h,45.00204381t,-0
    def generate_and_visit_urls(self, position, image_dir):
        lat = position["lat"]
        lon = position["lon"]
        distance = position["distance"]
        alt_range = position["alt"]
        tilt_range = position["tilt"]
        heading_range = position["heading"]
        fov = 35
        rotation = 0

        for altitude in range (alt_range["start"], alt_range["stop"], alt_range["step"]):
            for tilt in range(tilt_range["start"], tilt_range["stop"], tilt_range["step"]):
                for heading in range(heading_range["start"], heading_range["stop"], heading_range["step"]):
                    url = f"https://earth.google.com/web/@{lat},{lon},{altitude}a,{distance}d,{fov}y,{heading}h,{tilt}t,{rotation}r/"
                    file_name = f"{altitude}_{distance}_{fov}_{heading}_{tilt}_{rotation}.png"
                    file_path = os.path.join(image_dir, file_name)

                    if not os.path.exists(file_path):
                        self.visit_url(url)
                        time.sleep(0.5)
                        self.driver.save_screenshot(file_path)

    def run(self):
        for location, position in self.locations.items():
            print(f'Writing images to {location}')

            path = os.path.join("images", location)
            if not os.path.exists(path):
                os.mkdir(path)

            self.image_count = 0
            start = time.time()
            self.generate_and_visit_urls(position, path)
            end = time.time()
            total_time = end - start

            print(f'Wrote {self.image_count} images in {total_time} seconds')
