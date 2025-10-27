from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from time import sleep

driver = webdriver.Chrome()
for year in range(21,25):
    for period in range(1,5):
        driver.get('https://www.nba.com/stats/teams/boxscores-traditional?Season=20{}-{}&Period={}'.format(year, year + 1, period))
        sleep(1)
        driver.execute_script("window.scrollBy(0, 300);")
        sleep(5)

        for i in range (50):
            right_click_button = driver.find_element(By.XPATH, "//button[@title='Next Page Button']")
            html = driver.page_source
            with open('html/20{}-Q{}-{}.html'.format(year, period, i), 'w+') as f:
                f.write(html)
            right_click_button.click()
driver.quit()