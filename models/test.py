from bs4 import BeautifulSoup
import requests
import time



# 构造函数

url_saves = 'http://www.tripadvisor.cn/Saves#207971'
url = 'http://www.tripadvisor.cn/Attractions-g60763-Activities-New_York_City_New_York.html'
# 连续爬取30页的网页信息，发现每一页的区别在于 oa30 这个参数上，每一页的站点在这个数值的基础上增加30
# 所以可以用以下的格式进行爬取
urls = ['http://www.tripadvisor.cn/Attractions-g60763-Activities-oa{}-New_York_City_New_York.html#ATTRACTION_LIST'.format(str(i)) for i in range(30,930,30)]
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.109 Safari/537.36',
    'Cookie':'__utma=55896302.1244431692.1455603774.1455603774.1455603774.1; __utmc=55896302; =',
}

def get_attractions(url, data=None):
    wb_data   = requests.get(url)
    time.sleep(4)
    soup      = BeautifulSoup(wb_data.text, 'lxml')
    titles    = soup.select('div.property_title > a[target="_blank"]')
    imgs      = soup.select('img[width="160"]')
    cates     = soup.select('div.p13n_reasoning_v2')
    for title, img, cate in zip(titles, imgs, cates):
        data = {
            'title':title.get_text(),
            'img': img.get('src'),
            'cate':list(cate.stripped_strings),
        }
        print data,'\n','*'*50,'\n'


def get_favs(url, data=None):
    wb_data = requests.get(url, headers=headers)
    soup    = BeautifulSoup(wb_data.text, 'lxml')
    titles  = soup.select('a.location-name')
    imgs    = soup.select('img.photo_image')
    metas   = soup.select('span.format_address')

    if data == None:
        for title, img, meta in zip(titles, imgs, metas):
            data = {
                'title':title.get_text(),
                'img':img.get('src'),
                'meta':list(meta.stripped_strings),
            }
            print data,'\n','*'*50,'\n'

for single_url in urls:
    get_attractions(single_url)
