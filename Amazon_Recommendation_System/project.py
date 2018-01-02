import csv
import gzip
import json
from collections import Counter
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#import requests
#from io import BytesIO
#from PIL import Image  
#from UICreator import *
import sys
from UICreator import UIQCreator
#from PyQt4 import QtCore
from PyQt4 import QtGui


class Transaction():
    def __init__(self, asin, title, imUrl, categories, price, brand, also_bought = [], also_viewed = [], bought_together = [], rating = 0):
        self.asin = asin # product ID
        self.title = title
        self.imUrl = imUrl # URL of the product image
        self.categories = categories # list of categories
        self.price = price
        self.brand = brand
        self.also_bought = also_bought # list of products bought by the one who bought this product, default emply list as no other product may have been bought earlier
        self.also_viewed = also_viewed # list of products viewed while buying this product, default emply list as no other product may be viewed
        self.bought_together = bought_together # list of products bought with this product, default emply list as no other product may be bought
        self.rating = rating # average customer rating for this product, default as zero if the product has not been reviewed yet
        

# read amazon compressed metadata file and convert to strict json format- output is "output strict" file
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

#The below is run only once to get strict json file    
'''f = open("output.strict", 'w')
for l in parse("metadata.json.gz"):
  f.write(l + '\n')'''
    
# read ratings from csv file
def read_ratings():
    ratings = {} # {productID: rating}
    with open("ratings.csv") as datafile:
        rows = csv.reader(datafile)
        
        for row in rows:
            try:
                if row[1] in ratings:
                    ratings[row[1]].append(float(row[2]))
                else:
                    ratings[row[1]] = [float(row[2])]
            except IndexError:
                continue
    
    for key in ratings:
        ratings[key] = round(sum(ratings[key]) / len(ratings[key]), 2)      
    
    return ratings
        
# returns a list of objects of class Transaction
def get_data():  
    ratings = read_ratings()
    
    fields = ['asin', 'title', 'imUrl', 'categories', 'price', 'brand'] # required fields
    other_fields = ['also_bought', 'also_viewed', 'bought_together'] # optional fields
    
    data = []
    for line in open("output.strict", 'r'):
        row_map = json.loads(line)
        row = {'also_bought': [], 'also_viewed': [], 'bought_together': [], 'rating': 0}
            
        # check if metadata rows has the fields we want to generate class and those fields have a value   
        if not(set(fields).issubset(row_map.keys())) or any(map(lambda x: row_map[x] == '', fields)):
            continue
        
        # processes the below if all required fields are in row_map
        try:
            for i in fields:
                if i == 'categories':
                    del row_map[i][0][0] # 1st item in categories is the main category like Electronics for this dataset, removed that so that category similarity is checked better- for complete data may need to keep it
                    row[i] = row_map[i][0]
                else:
                    row[i] = row_map[i]
            
          
            for i in other_fields:
                if i in row_map['related']:
                    row[i] = row_map['related'][i]
            
            if row['asin'] in ratings.keys():                      
                row['rating'] = ratings[row['asin']]
                
            obj = Transaction(row['asin'], row['title'], row['imUrl'], row['categories'], 
                          row['price'], row['brand'], row['also_bought'], 
                            row['also_viewed'],row['bought_together'], row['rating'])
            
        except Exception:
            pass
        
        
        data.append(obj)
        
    return data # data has 45195 objects
    


# returns list of items with categories similar to the categories of item being viewed

def sim_categories(item, data):
    similar_objs = [] 
    a = item.categories
    
    a_vals = Counter(a)
    for i in data:
        # filter out the items already bought or the item being viewed so that they are not recommended
        if i == item or i.asin in (item.also_bought + item.bought_together):
            continue
        b = i.categories
        b_vals = Counter(b)
        words  = list(a_vals.keys() | b_vals.keys())
        
        # Convert a and b into vectors as cosine similarity finds cosine of the two vectors
        a_vect = [a_vals.get(word, 0) for word in words]        
        b_vect = [b_vals.get(word, 0) for word in words] 
        len_a  = sum(av*av for av in a_vect) ** 0.5             
        len_b  = sum(bv*bv for bv in b_vect) ** 0.5             
        dot    = sum(av*bv for av,bv in zip(a_vect, b_vect))    
        cosine = dot / (len_a * len_b) # higher the cosine value more similar the categories are
        if cosine > 0.85: # threshold chosen after increasing it incrementally from 0.5 to 0.95 (cosine similarity range is 0-1)
            similar_objs.append(i)

        
    #print(len(similar_objs))
    #print(a)
    return similar_objs
    
# from the list of 10 or less items with similar categories, returns list of items with title similar to the title of item being viewed
    
def sim_title(item, data):
    """similar_title = [] 
   
    for i in data:
        documents = [item.title,i.title]
        tfidf_matrix=TfidfVectorizer().fit_transform(documents)
        cs = cosine_similarity(tfidf_matrix)
        if cs.item(1) > 0.1: # threshold chosen after increasing it incrementally from 0 to 0.95 (cosine similarity range is 0-1)
            similar_title.append(i)"""
            
    simi_title = {}
    
   
    for i in data:
        documents = [item.title,i.title]
        tfidf_matrix=TfidfVectorizer().fit_transform(documents)
        cs = cosine_similarity(tfidf_matrix)
        if cs.item(1) > 0.1: # threshold chosen after increasing it incrementally from 0 to 0.95 (cosine similarity range is 0-1)
            simi_title[i] = cs.item(1)
    
    # sorts the dictionary simi_title by value (cosine similarity value) and returns a list of items only        
    similar_title = sorted(simi_title, key=simi_title.get, reverse = True)
     

    if len(similar_title) < 2:
        print("No items to recommend")
     
    return similar_title
    
    # code to return items in order of ratings- can be used if we add sorting feature in UI   
    """if len(similar_title) < 2:
        print("No items to recommend")
        return
    else:
        return similar_title.sort(key = lambda x: x.rating, reverse = True)"""
    


app = QtGui.QApplication([])
#QtCore.QCoreApplication.addLibraryPath("/Users/monicadabas/anaconda3/pkgs/pyqt-4.11.4-py35_3/")

#print(QImageReader.supportedImageFormats())

data = get_data()
item_being_viewed = random.choice(data)

items_by_categories = sim_categories(item_being_viewed, data)
items_by_title= sim_title(item_being_viewed, items_by_categories) 
  

if len(items_by_title) > 1:
    window = UIQCreator(item_being_viewed, items_by_title)
    window.show()
    sys.exit(app.exec_())             