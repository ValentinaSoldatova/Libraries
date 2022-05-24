
import re

import scrapy
from itemloaders.processors import TakeFirst, MapCompose, Join


def clear_string(s):
    return s.strip()


def get_price(price):
    return int(re.sub(r'(\D)', '', price))


class ChitaiGorodItem(scrapy.Item):
    _id = scrapy.Field()
    title = scrapy.Field(input_processor=MapCompose(clear_string),
                         output_processor=Join(separator=" "))
    url = scrapy.Field(output_processor=TakeFirst())
    price = scrapy.Field(input_processor=TakeFirst(),
                         output_processor=MapCompose(get_price))
    img_urls = scrapy.Field()
    img_info = scrapy.Field()
    features_books = scrapy.Field(output_processor=TakeFirst()from scrapy import signals


class ChitaiGorodSpiderMiddleware:


    @classmethod
    def from_crawler(cls, crawler):
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        return None

    def process_spider_output(self, response, result, spider):
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
       pass

    def process_start_requests(self, start_requests, spider):
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


class ChitaiGorodDownloaderMiddleware:


    @classmethod
    def from_crawler(cls, crawler):
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):

        return None

    def process_response(self, request, response, spider):

        return response

    def process_exception(self, request, exception, spider):


    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


import os
from urllib.parse import urlparse

import pymongo
from slugify import slugify
from itemadapter import ItemAdapter
from scrapy.http import Request
from scrapy.pipelines.images import ImagesPipeline


class ChitaiGorodPipeline:
    collection_name = 'scrapy_books'

    def __init__(self, mongo_uri, mongo_db):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mongo_uri=crawler.settings.get('MONGO_URI'),
            mongo_db=crawler.settings.get('MONGO_DATABASE', 'items')
        )

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.mongo_db]

    def close_spider(self, spider):
        self.client.close()

    def process_item(self, item, spider):
        self.db[self.collection_name].update_one(
            {'url': item['url']},
            {'$set': ItemAdapter(item).asdict()},
            upsert=True,
        )
        return item


class ChitaiGorodImagesPipeline(ImagesPipeline):
    def get_media_requests(self, item, info):
        img_urls = []
        img_urls.extend(item["img_urls"])
        img_urls = set(img_urls)

        if img_urls:
            for img_url in img_urls:
                try:
                    yield Request(img_url)
                except Exception as e:
                    print(e)

    def file_path(self, request, response=None, info=None, *, item=None):
       slug = slugify(item['title'])
        return f'full/{slug}/' + os.path.basename(urlparse(request.url).path)

    def item_completed(self, results, item, info):
        if results:
            item["img_info"] = [r[1]['path'] for r in results if r[0]]
            del item["img_urls"]
        return item
© 2022 GitHub, Inc.
Terms
Priva

BOT_NAME = 'chitai_gorod'

SPIDER_MODULES = ['chitai_gorod.spiders']
NEWSPIDER_MODULE = 'chitai_gorod.spiders'



USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.72 Safari/537.36'


ROBOTSTXT_OBEY = False


CONCURRENT_REQUESTS = 16


DOWNLOAD_DELAY = 5


COOKIES_ENABLED = True


IMAGES_STORE = "images"

ITEM_PIPELINES = {
   'chitai_gorod.pipelines.ChitaiGorodImagesPipeline': 149,
   'chitai_gorod.pipelines.ChitaiGorodPipeline': 150,
}


MONGO_URI: str = 'mongodb://localhost:27017'
MONGO_DATABASE: str = 'mydb'

from shutil import which

SELENIUM_DRIVER_NAME = 'chrome'
SELENIUM_DRIVER_EXECUTABLE_PATH = which('chromedriver')
SELENIUM_DRIVER_ARGUMENTS = ['--headless', '--start-maximized']

DOWNLOADER_MIDDLEWARES = {
   'scrapy_selenium.SeleniumMiddleware': 600
}

LOG_ENABLED = True
LOG_FILE = 'log.txt'
lOG_LEVEL = 'DEBUG'
© 2022 GitHub, Inc.