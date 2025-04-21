# Data extraction: scraping housing price and information, join with census demographic data
import json
import traceback
import time
import urllib.parse
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from shapely.geometry import Polygon

# Extract all property links
def scroll(element):
    viewport_height = browser.execute_script("return window.innerHeight;")
    element_position = element.location_once_scrolled_into_view["y"]
    current_scroll_position = browser.execute_script("return window.scrollY;")
    target_scroll_position = max(0, min(element_position - (viewport_height / 2), current_scroll_position))
    step = 20  # Adjust scroll step here
    while current_scroll_position != target_scroll_position:
        if current_scroll_position < target_scroll_position:
            current_scroll_position = min(current_scroll_position + step, target_scroll_position)
        else:
            current_scroll_position = max(current_scroll_position - step, target_scroll_position)
        browser.execute_script("window.scrollTo(0, arguments[0]);", current_scroll_position)
        time.sleep(0.00000000000001)  # Adjust scroll speed here

# Loop through each page
def get_hrefs(href_list):
    try:
        while True:
            time.sleep(3)
            next_button = browser.find_element(By.CSS_SELECTOR, "a[aria-label='Next page']")
            scroll(next_button)
            time.sleep(0.5)
            browser.execute_script("arguments[0].scrollIntoView({ behavior: 'auto', block: 'center', inline: 'center' });",
                                  next_button)

            elem = browser.find_elements(By.CLASS_NAME, "sc-1r1odlb-27.fZlqIb")
            hrefs = [div.get_attribute("href") for div in elem if div.get_attribute("href")]
            href_list.extend(hrefs)
            # print(all_hrefs)
            print(f"len:{len(all_hrefs)}")

            if next_button.get_attribute("aria-disabled") == "true":
                return href_list  # Break the loop if the 'Next' button is disabled (last page)
            else:
                next_button.click()
    except Exception as e:
        print(f"Errored: {e}")
        return href_list

url = "https://www.midland.com.hk/zh-hk/list/buy"
browser = webdriver.Chrome()
browser.get(url)

all_hrefs = []
all_href_ls = [[i] for i in get_hrefs(all_hrefs)]
print(all_href_ls)
with open('all_hrefs_midland.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(all_href_ls)


# Extract variables from each link
count = 0
with open('data_all.csv', 'w', newline='', encoding="utf-8") as data:
    writer = csv.writer(data)
    writer.writerow(['price', 'district', 'age', 'developer', 'area', 'efficiency', 'floor', 'rooms', 'bathrooms',
                     'storerooms', 'direction', 'duplex', 'sea', 'balcony', 'garden', 'clubhouse', 'pool', 'mtr', 'mall',
                     'park', 'url'])

    with open('all_hrefs_midland.csv', mode='r') as f:
        csvFile = csv.reader(f)
        try:
            for href in csvFile:
                count += 1
                print(f"----- Count: {count} -----")
                print(href[0])
                browser = webdriver.Chrome()
                browser.get(href[0])
                browser.implicitly_wait(2)
                html = browser.page_source
                soup = BeautifulSoup(html, 'html.parser')

                # --- TARGET: PRICE ---
                prices = soup.find_all(class_="sc-kdvk35-8 fweMHf")
                if not prices:
                    continue
                prices = prices[0]
                try:
                    price = int(prices.find_all(class_="sc-kdvk35-3 jjGIIk")[0].text[1:].replace(",", ""))
                except ValueError:
                    price = -999

                # --- FEATURES (building, property, facilities) ---
                district = href[0].split("property/")[1].split("-")[1]
                district = urllib.parse.unquote(district, encoding='utf-8')

                developer = soup.find(class_="sc-bku5wy-2 jUZLpR")
                if developer:
                    developer = developer.text
                else:
                    developer = "NA"

                area = prices.find(class_="sc-kdvk35-1 bxNwSa")
                if area:
                    try:
                        area = int(area.text.replace(",", ""))
                    except ValueError:
                        area = -999
                else:
                    area = -999

                age = -999
                efficiency = -999
                rooms = 0
                brooms = 0
                srooms = 0
                direction = "NA"

                row = soup.find_all(class_="sc-4q1ceo-5 fFFfkH")
                for i in row:
                    if i.text == "樓齡":
                        age = i.find_previous_sibling()
                        age = int(age.text[:-1])
                    elif i.text == "實用率":
                        efficiency = i.find_previous_sibling()
                        efficiency = int(efficiency.text[:-1])
                    elif i.text == "間隔":
                        rooms = i.find_previous_siblings()
                        rooms.reverse()
                        rooms = int(rooms[0].text[0])
                    elif i.text == "浴室/洗手間":
                        brooms = i.find_previous_sibling()
                        brooms = int(brooms.text[0])
                    elif i.text == "工人房/儲物室/":
                        srooms = i.find_previous_sibling()
                        srooms = int(srooms.text[0])
                    elif i.text == "座向":
                        direction = i.find_previous_sibling()
                        direction = direction.text

                floor = soup.find(class_="sc-3l1lqj-11 enpoHH")
                if floor:
                    floor = floor.contents[0].text
                else:
                    floor = "NA"

                duplex = 0
                sea = 0
                balcony = 0
                garden = 0

                row2 = soup.find_all(class_="sc-16dnbe-1 hJhJTv")
                if row2:
                    for i in row2:
                        if i.text == "複式":
                            duplex = 1
                        elif i.text == "海景":
                            sea = 1
                        elif i.text == "連露台":
                            balcony = 1
                        elif i.text == "連花園":
                            garden = 1

                clubhouse = 0
                pool = 0

                row3 = soup.find_all(class_="sc-txcvyl-0 czvEFU sc-85wejs-11 fYOzei")
                if row3:
                    for i in row3:
                        if i.text == "會所":
                            clubhouse = 1
                        elif i.text == "泳池":
                            pool = 1

                if soup.find(class_="sc-xs4x3m-0 sc-y966pp-1 kEwXKw"):
                    mtr = 1
                else:
                    mtr = 0
                if soup.find(class_="sc-xs4x3m-0 sc-y966pp-1 iezLtf"):
                    mall = 1
                else:
                    mall = 0
                if soup.find(class_="sc-xs4x3m-0 sc-y966pp-1 iwYagD"):
                    park = 1
                else:
                    park = 0

                print(f"Price: {price}")

                print(f"District: {district}")
                print(f"Age: {age}")
                print(f"Developer: {developer}")

                print(f"Area: {area}")
                print(f"Efficiency: {efficiency}")
                print(f"Floor: {floor}")
                print(f"Rooms: {rooms}")
                print(f"Bathrooms: {brooms}")
                print(f"Storerooms: {srooms}")
                print(f"Direction: {direction}")
                print(f"Duplex: {duplex}")
                print(f"Sea: {sea}")
                print(f"Balcony: {balcony}")
                print(f"Garden: {garden}")
                print(f"Clubhouse: {clubhouse}")
                print(f"Pool: {pool}")

                print(f"MTR: {mtr}")
                print(f"Mall: {mall}")
                print(f"Park: {park}")

                writer.writerow(
                    [price, district, age, developer, area, efficiency, floor, rooms, brooms, srooms, direction,
                     duplex, sea, balcony, garden, clubhouse, pool, mtr, mall, park, href[0]])

                browser.quit()

        except Exception:
            print(traceback.format_exc())


# Census demographic data extraction
# Extract property district boundaries, aggregated with census data in QGIS
with open('district_translate.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    d_lookup = list(reader)[1:]
d_chi = [d[0] for d in d_lookup]
d_eng = [d[1] for d in d_lookup]

url = "https://www.midland.com.hk/en/district"
browser = webdriver.Chrome()
browser.get(url)
html = browser.page_source
soup = BeautifulSoup(html, 'html.parser')
script = soup.find(id="__NEXT_DATA__").string
script_json = json.loads(script)
features = script_json["props"]["pageProps"]["mapData"]["features"]
features_list = [[i["properties"]["name"], i["geometry"]["coordinates"][0][0]] for i in features]
# print(features_list)
with open('district_boundaries.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    for i in features_list:
        try:
            d_index = d_chi.index(i[0].split("/")[0].strip())
            name = d_eng[d_index]
            boundary = [(c[0], c[1]) for c in i[1]]  # get list of (lon, lat) for all vertices in each boundary
            boundary = Polygon(boundary).wkt
            writer.writerow([name, boundary])
        except ValueError:
            continue
