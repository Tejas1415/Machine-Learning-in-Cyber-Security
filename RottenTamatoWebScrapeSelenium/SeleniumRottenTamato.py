# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 00:27:16 2022

@author: Tejas

Using selenium to scrape rotten tamatoes 
"""

import selenium
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time


url = 'https://www.rottentomatoes.com/m/exodus_gods_and_kings/reviews'

driver = webdriver.Chrome('C:/Users/Tejas/Downloads/Github/Manasa Assignments/chromedriver.exe')

driver.get('https://www.rottentomatoes.com/m/exodus_gods_and_kings/reviews')


def scrape(driver, url):
    movie_reviews = driver.find_elements(by=By.CSS_SELECTOR,value='div[class="row review_table_row"]')
    
    
    name1 = []
    date1 = []
    text1 = []
    polarity1 = []
    
    for review in movie_reviews:
        ## name
        review_name = review.find_element(by=By.CSS_SELECTOR, value='div[class="col-sm-17 col-xs-32 critic_name"]').text
        review_name = review_name.split('\n')[0]
        name1.append(review_name)
        
        ##date and text
        rev = review.find_element(by=By.CSS_SELECTOR,value='div[class="review_area"]')
        review_text = rev.text
        review_date = review_text.split('\n')[0]
        review_msg = review_text.split('\n')[1]
        date1.append(review_date)
        text1.append(review_msg)
    
        ##polarity
        polarity = review.find_element(by=By.CSS_SELECTOR, value="div[class^='review_icon icon']").get_attribute("class")
    
        if 'fresh' in polarity:
            pol = 'fresh'
        elif 'rotten' in polarity:
            pol = 'rotten'
            
        polarity1.append(pol)
        ### save the text into csv
        
    
    df = pd.DataFrame()
    df['reviewer_name'] = name1
    df['review_date'] = date1
    df['review_text'] = text1
    df['review_polarity'] = polarity1
    
    return df

#df1 = scrape(driver, url)


dfs = []
for i in range(100000):
    
    df1 = scrape(driver, url)
    dfs.append(df1)
    
    button = driver.find_element(by=By.CSS_SELECTOR, value="button[class^='js-prev-next-paging-next btn prev-next-paging__button prev-next-paging__button']").get_attribute("class")
    
    if 'hide' not in button:
        driver.find_element(by=By.CSS_SELECTOR, value="button[class='js-prev-next-paging-next btn prev-next-paging__button prev-next-paging__button-right']").click()
        time.sleep(5)
    else:
        break
    
    

result_df = pd.concat(dfs, axis = 0)   
    



















