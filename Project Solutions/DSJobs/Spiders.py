# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 21:10:35 2017

@author: juan9
"""
import scrapy
import urllib
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess



class IndeedSpider(scrapy.Spider):
    name = "Indeed"
    
    def Get_URLS(self):
        base_url = "https://www.indeed.com"
        original_url = "https://www.indeed.com/q-Data-Science-jobs.html"
        urls= []
        urls.append(original_url)
        page = urllib.request.urlopen(original_url).read()
        fpage_parser = BeautifulSoup(page,"lxml")
        pages = fpage_parser.find("div", attrs = {"class":"pagination"})
        page_As = pages.find_all("a",attrs = {"href":True})
        page_urls = [base_url + a['href'] for a  in page_As]
        urls.extend(page_urls)
        return urls[:-1]

    
    def start_requests(self):
        urls = self.Get_URLS()
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self, response):
        
        for posting in response.css('div.row.result'):
            yield {
                'JobTitle': posting.css('h2.jobtitle a.turnstileLink::attr(title)').extract_first(),
                'Company': posting.css('span.company span a::text').extract_first(),
                'Text': posting.css('span.summary::text').extract(),
            }
         

class KaggleSpider(scrapy.Spider):
    name = "Kaggle"
    DOWNLOAD_DELAY = 5
    
    def Get_URLS(self):
        base_url = "https://www.kaggle.com"
        original_url = "https://www.kaggle.com/jobs"
        urls= []
        page = urllib.request.urlopen(original_url).read()
        fpage_parser = BeautifulSoup(page,"lxml")
        pages = fpage_parser.find_all("a", attrs = {"class":"job-post-row"})
        page_urls = [base_url + a['href'] for a  in pages]
        urls.extend(page_urls)
        return urls

    
    def start_requests(self):
        urls = self.Get_URLS()
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self, response):
        
        for posting in response.css('div.jobs-board-post'):
            yield {
                'JobTitle': posting.css('div.jobs-board-post-header div.title h1::text').extract_first(),
                'Company': posting.css('div.jobs-board-post-header div.title h2::text').extract_first(),
                'Text': posting.css('div.jobs-board-post-content p::text').extract()[0:5],
            }
         

process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
    'FEED_FORMAT': 'jsonlines',
    'FEED_URI': 'results.jl'
})
            

process.crawl(IndeedSpider)
process.crawl(KaggleSpider)
process.start()
