
from pymongo import MongoClient


client = MongoClient('mongodb://root:root@localhost:27017/?authSource=admin')
db = client['AlgoAss1']
# collection = db['RandomData']

