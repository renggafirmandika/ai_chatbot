import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import SitemapLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.utilities import ApifyWrapper
from langchain_core.documents import Document
from langchain_community.document_loaders import ApifyDatasetLoader
from streamlit import _bottom
from time import sleep
import requests
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from seleniumrequests import Chrome
from selenium.webdriver.support.ui import WebDriverWait

apify = ApifyWrapper()
# url = "https://babel.beta.bps.go.id/api/pressrelease?lang=id&page=1&sort=latest"

load_dotenv()

tahun = [2020, 2021, 2022, 2023]
bulan = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
url = []

driver = webdriver.Chrome()
driver.get("https://babel.beta.bps.go.id/id/pressrelease")

# for x in tahun:
#     for y in bulan:
#         url.append(f"https://babel.beta.bps.go.id/id/pressrelease?month={y}&year={x}&sort=latest")

# url.append("https://babel.beta.bps.go.id/id/pressrelease?month=01&year=2024&sort=latest")
# url.append("https://babel.beta.bps.go.id/id/pressrelease?month=02&year=2024&sort=latest")
# url.append("https://babel.beta.bps.go.id/id/pressrelease?month=03&year=2024&sort=latest")
# url.append("https://babel.beta.bps.go.id/id/pressrelease?month=04&year=2024&sort=latest")
# url.append("https://babel.beta.bps.go.id/id/pressrelease?month=05&year=2024&sort=latest")
# url.append("https://babel.beta.bps.go.id/id/pressrelease?month=06&year=2024&sort=latest")

# url_json = json.dumps([{'url':url2} for url2 in url])
# json_array = json.loads(url_json)
# print(json_array)


# loader = apify.call_actor(
# actor_id="apify/website-content-crawler",
# run_input={
#     "aggressivePrune": False,
#     "clientSideMinChangePercentage": 15,
#     "crawlerType": "playwright:adaptive",
#     "debugLog": False,
#     "debugMode": False,
#     "dynamicContentWaitSecs": 60000,
#     "includeUrlGlobs": [
#         {
#             "glob": "https://babel.beta.bps.go.id/id/pressrelease/**"
#         }
#     ],
#     "htmlTransformer": "none",
#     "ignoreCanonicalUrl": True,
#     "dynamicContentWaitSecs": 60,
#     "maxCrawlDepth": 1,
#     "maxCrawlPages": 1000,
#     "maxRequestRetries": 3,
#     "requestTimeoutSecs": 30,
#     "proxyConfiguration": {
#         "useApifyProxy": True,
#         "apifyProxyGroups": [
#             "RESIDENTIAL"
#         ],
#         "apifyProxyCountry": "ID"
#     },
#     "readableTextCharThreshold": 100,
#     "removeCookieWarnings": True,
#     "removeElementsCssSelector": "nav, footer, script, style, noscript, svg,\n.rc-pagination,\n.py-6.mt-4.flex,\n.p-5,\n.top-0,\n.overflow-text-ellipsis,\n.text-white,\n.bottom-0,\n.h-fit,\n.download-product,\n.download-infographic,\n.download-slide,\n.caption,\n[role=\"alert\"],\n[role=\"banner\"],\n[role=\"dialog\"],\n[role=\"alertdialog\"],\n[role=\"region\"][aria-label*=\"skip\" i],\n[aria-modal=\"true\"]",
#     "renderingTypeDetectionPercentage": 10,
#     "saveFiles": False,
#     "saveHtml": False,
#     "saveMarkdown": True,
#     "saveScreenshots": False,
#     "startUrls": [{"url": response}],
# },
# dataset_mapping_function=lambda item: Document(
#     page_content=item["text"] or "", metadata={"source": item["url"]}))

# elems = driver.find_elements(By.CSS_SELECTOR, "a.p-5")
# for elem in elems:
#     url.append(str(elem.get_attribute("href")))

# url_json = json.dumps([{'url':url2} for url2 in url])
# json_array = json.loads(url_json)
# print(json_array)

page = 1

while True:
    sleep(5)
    btn_next = driver.find_element(By.CLASS_NAME, 'rc-pagination-next')
    elems = driver.find_elements(By.CSS_SELECTOR, "a.p-5")
    for elem in elems:
        url.append(str(elem.get_attribute("href")))

    if page==20:
        break
    
    page += 1
    btn_next.send_keys(Keys.RETURN)

url_json = json.dumps([{'url':url2} for url2 in url])
json_array = json.loads(url_json)

loader = apify.call_actor(
actor_id="apify/website-content-crawler",
run_input={
    "aggressivePrune": False,
    "clientSideMinChangePercentage": 15,
    "crawlerType": "playwright:adaptive",
    "debugLog": False,
    "debugMode": False,
    "dynamicContentWaitSecs": 60000,
    "htmlTransformer": "none",
    "ignoreCanonicalUrl": True,
    "dynamicContentWaitSecs": 60,
    "maxCrawlDepth": 0,
    "maxCrawlPages": 1000,
    "maxRequestRetries": 3,
    "requestTimeoutSecs": 30,
    "proxyConfiguration": {
        "useApifyProxy": True,
        "apifyProxyGroups": [
            "RESIDENTIAL"
        ],
        "apifyProxyCountry": "ID"
    },
    "readableTextCharThreshold": 100,
    "removeCookieWarnings": True,
    "removeElementsCssSelector": "nav, footer, script, style, noscript, svg,\n.rc-pagination,\n.py-6.mt-4.flex,\n.p-5,\n.top-0,\n.overflow-text-ellipsis,\n.text-white,\n.bottom-0,\n.h-fit,\n.download-product,\n.download-infographic,\n.download-slide,\n.caption,\n[role=\"alert\"],\n[role=\"banner\"],\n[role=\"dialog\"],\n[role=\"alertdialog\"],\n[role=\"region\"][aria-label*=\"skip\" i],\n[aria-modal=\"true\"]",
    "renderingTypeDetectionPercentage": 10,
    "saveFiles": False,
    "saveHtml": False,
    "saveMarkdown": True,
    "saveScreenshots": False,
    "startUrls": json_array,
},
dataset_mapping_function=lambda item: Document(
    page_content=item["text"] or "", metadata={"source": item["url"]}))
