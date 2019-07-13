import requests

from bs4 import BeautifulSoup

def get_html(url):
    headers={'User-Agent':'Mozzila/5.0(Windows NT 6.1;Win64;x64) AppleWebkit/537.36(KHTML,like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    resp=requests.get(url,headers=headers).text
    return resp

def html_parser():
    for url in all_page():
        soup=BeautifulSoup(get_html(url),'html.parser')
        alldiv=soup.find_all('div',class_='p12')
        names=[a.find('a')['title'] for a in alldiv]
        allp=soup.find_all('p',class_='p1')
        authors=[s.get_txt() for s in allp]
        starspan=soup.find_all('span',class_='inq')
        scores=[s.get_text() for s in starspan]
        sumspan=soup.find_all('span',class_='rating numbers')
        sums=[i.get_text() for i in sumspan]
        for name,author,score,sum in zip(names,authors,scores,sums):
            name = '书名：' + str(name) + '\n'
            author = '作者：' + str(author) + '\n'
            score = '评分：' + str(score) + '\n'
            sum = '简介：' + str(sum) + '\n'
            data = name + author + score + sum
            f.writelines(data+'============================'+'\n')

def all_page():
    base_url='https://book.douban.com/top250?start='
    urllist=[]
    for page in range(0,250,25):
        allurl=base_url+str(page)
        urllist.append(allurl)
    return urllist

filename='doubantop250books.txt'
f=open(filename,'w',encoding='utf-8')
html_parser()
f.close()
print('saved successfully')