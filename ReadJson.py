import json

try:
    with open("test.json", "r", encoding = "utf-8") as json_file:
        data = json.load(json_file)
        print("Data: %s" % data)
except IOError:
    print("Could not read file")

reviews = []
for review in data:
    reviews.append(review['stars'])
print(reviews)

reviewsAgg = []
for i in {1.0, 2.0, 3.0, 4.0, 5.0}:
    reviewsAgg.append(reviews.count(i))

print(reviewsAgg)