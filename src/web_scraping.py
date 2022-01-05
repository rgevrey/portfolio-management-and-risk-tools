from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Last updated 5/01/2022 19:35


def url_scrap(url, path):
    """
    Saves the full content of a web page on the drive, bypassing disclaimer pages.
    Works only with Chrome and the following websites re disclaimers:
        - lgim.com
    Args:
        url : url to be scraped
        path : where to save the file on the drive
    """
    
    # Bypassing disclaimers on the page to get to the real content
    if url.find("lgim.com") > 0:
        driver = webdriver.Chrome() # TO DO - ADD A SELECTOR TO RECOGNISE THE BROWSER IN USE
        driver.get(url)
    
        # ADD ASSERTIONS FOR WHEN MORE FLEXIBLE
        
        # Cookies banner at the top of LGIM
        cookie_accept = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler")))
        cookie_accept.click()
        
        # Terms and conditions and confirmation we are professional investors
        terms_tick = driver.find_element(By.ID, "popup-checkbox-$tools.math.add($velocityCount,-1)")
        terms_tick.click()
        terms_accept = driver.find_element(By.CLASS_NAME, "btn.btn-secondary.btn-accept") 
        terms_accept.click()
    
    # Getting the source code
    WebDriverWait(driver,10).until(EC.staleness_of(terms_accept))
    print("wait complete") # DEBUG
    html = driver.page_source 
    driver.close() 
    
    # Saving 
    page_save = open(path, "w")
    page_save.write(html)
    page_save.close()
    
    print("File saved at " + path)