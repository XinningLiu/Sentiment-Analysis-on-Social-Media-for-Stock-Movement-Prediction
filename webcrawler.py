#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Xinning Liu'

from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import NoSuchElementException
import time

s_sticker="T"

driver = webdriver.Chrome("/Users/liuxinning/Documents/python/chromedriver")

s_path=r"https://finance.yahoo.com/quote/"+s_sticker+"/community?p="+s_sticker
s_time=time.strftime("%Y%m%d%H%M")
s_file=r"/Users/liuxinning/Documents/python/data/"+s_sticker+"_"+s_time+".txt"


driver.get(s_path)

time.sleep(0.1)
driver.find_element_by_xpath("""//button[@class='sort-filter-button O(n) Fz(14px) Fw(600) M(0) P(0) Trsdu(.5s) C($c-fuji-grey-l)']""").click()
time.sleep(0.2)
driver.find_element_by_xpath("""//button[@class='sort-by-createdAt Px(16px) Py(12px) O(n) Cur(p) C($c-fuji-grey-g) Fz(14px) Fw(500) C($c-fuji-blue-1-a):h selected_C($c-fuji-blue-1-a) selected_Fw(b)']""").click()
time.sleep(0.2)


start_time = time.time()
#find ul
ul_element=driver.find_element_by_css_selector("ul[class='comments-list List(n) Ovs(touch) Pos(r) Mstart(-12px) Pt(5px)']")

f=open(s_file,'w+',encoding='utf8')

lines=ul_element.find_elements_by_css_selector("li[class='comment Pend(2px) Mt(5px) P(12px) ']")

length=len(lines)

for line in lines:
    timestamp=line.find_element_by_css_selector("span[class='C($c-fuji-grey-g) Fz(12px)']").text
    wow=line.find_element_by_css_selector("div[class='Wow(bw)']")
    try:
        comment=wow.find_element_by_css_selector("[class='C($c-fuji-grey-l) Mb(2px) Fz(14px) Lh(20px)']").text
    except NoSuchElementException:
        comment="    "
    s=timestamp+"!! "+comment+".!\n"
    t=f.write(s)    

b_endoflist=False
for i in range(1000):
    if b_endoflist:
        break
    
    b_badClick=True
    while b_badClick:
        try:
            ShowMore_buttom=driver.find_element_by_xpath("""//button[@class='Ff(ss) Fz(14px) Fw(b) Bdw(2px) Ta(c) Cur(p) Va(m) Bdrs(4px) O(n)! Lh(n) Bgc(#fff) C($c-fuji-blue-1-a) Bdc($c-fuji-blue-1-a) Bd C(#fff):h Bgc($c-fuji-blue-1-a):h My(20px) Px(30px) Py(10px) showMore D(b) Mx(a) Pos(r) Tt(c)']""")
        except NoSuchElementException:
            b_endoflist=True
            break
        try:
            ShowMore_buttom=driver.find_element_by_xpath("""//button[@class='Ff(ss) Fz(14px) Fw(b) Bdw(2px) Ta(c) Cur(p) Va(m) Bdrs(4px) O(n)! Lh(n) Bgc(#fff) C($c-fuji-blue-1-a) Bdc($c-fuji-blue-1-a) Bd C(#fff):h Bgc($c-fuji-blue-1-a):h My(20px) Px(30px) Py(10px) showMore D(b) Mx(a) Pos(r) Tt(c)']""")
        except NoSuchElementException:
            b_endoflist=True
            break
        
        ShowMore_buttom.click()
        time.sleep(0.01)

        b_stillLoading=True
        while b_stillLoading:
            try:
                if ShowMore_buttom.find_element_by_tag_name("span").text!="Loading":
                    b_stillLoading=False
            except StaleElementReferenceException:
                b_endoflist=True
                b_stillLoading=False
            except NoSuchElementException:
                b_endoflist=True
                b_stillLoading=False
                
        lines=ul_element.find_elements_by_css_selector("li[class='comment Pend(2px) Mt(5px) P(12px) ']")[length:]

        if len(lines)>0:
            b_badClick=False
        else:
            time.sleep(0.1)

    for line in lines:
        timestamp=line.find_element_by_css_selector("span[class='C($c-fuji-grey-g) Fz(12px)']").text
        wow=line.find_element_by_css_selector("div[class='Wow(bw)']")
        try:
            comment=wow.find_element_by_css_selector("[class='C($c-fuji-grey-l) Mb(2px) Fz(14px) Lh(20px)']").text
        except NoSuchElementException:
            comment="    "
        s=timestamp+"!! "+comment+".!\n"
        t=f.write(s)
    length+=len(lines)

f.close()
driver.close()
driver.quit()
print("time=",(time.time()-start_time)/60,"minutes")
