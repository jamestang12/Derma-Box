from time import sleep
from picamera import PiCamera
from google.cloud import storage
from tempfile import NamedTemporaryFile
from twilio.rest import Client

import RPi.GPIO as GPIO
import LCD
import os
import mysql.connector
import random
import cv2
import numpy as np

from AcneVulgaris.AcneVulgarisModel import AcneVulgarisModel
from AtopicEczema.AtopicEczemaModel import AtopicEczemaModel
import AcneVulgaris.SingleFileFilter1
import AtopicEczema.SingleFileFilter2

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


lower_threshold_colour = np.array([190, 100, 100])
upper_threshold_colour = np.array([255, 165, 165])

def analyze_name1(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name

def read_single_file1(path, filtered_path):
    print("Starting to filter path")
    raw = cv2.imread(path)
    mask = cv2.inRange(raw, lower_threshold_colour, upper_threshold_colour)
    # result = cv2.bitwise_and(raw, raw, mask=mask)
    cv2.imwrite(filtered_path, mask)
    count = cv2.countNonZero(mask)
    return float(count) / (raw.shape[0] * raw.shape[1])


def getPercentRed1(path):
    data = [[], [], []]

    img_path = ''
    filtered_path = ''

    name = analyze_name1(path)
    raw = cv2.imread(img_path + name + '.jpg')
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    # raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    # preparing the mask to overlay
    mask = cv2.inRange(rgb, lower_threshold_colour, upper_threshold_colour)
    count = cv2.countNonZero(mask)
    return float(count) / (raw.shape[0] * raw.shape[1])

lower_threshold_colour2 = np.array([190, 100, 100])
upper_threshold_colour2 = np.array([255, 165, 165])

def analyze_name2(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name

def read_single_file2(path, filtered_path):
    print("Starting to filter path")
    raw = cv2.imread(path)
    mask = cv2.inRange(raw, lower_threshold_colour2, upper_threshold_colour2)
    # result = cv2.bitwise_and(raw, raw, mask=mask)
    cv2.imwrite(filtered_path, mask)
    count = cv2.countNonZero(mask)
    return float(count) / (raw.shape[0] * raw.shape[1])


def getPercentRed2(path):
    data = [[], [], []]

    img_path = ''
    filtered_path = ''

    name = analyze_name2(path)
    raw = cv2.imread(img_path + name + '.jpg')
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    # raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    # preparing the mask to overlay
    mask = cv2.inRange(rgb, lower_threshold_colour2, upper_threshold_colour2)
    count = cv2.countNonZero(mask)
    return float(count) / (raw.shape[0] * raw.shape[1])

# Total images in dataset
"""total_images = 75
file = "filtered_image_data.csv"
model = AcneVulgarisModel(file, total_images)
model.setup()
# model.load_from_disk()
model.train_new_model()
# model.runTests(file)
print("aaa")
model.export_current_model()
print("aaa")
print(model.runTest(0.2))
print("aaa")
print("ESTIMATE")
print(model.runTest(0.8))"""

total_images = 75
file = "filtered_image_data.csv"

model1 = AcneVulgarisModel(file, total_images)
model1.setup()
model1.train_new_model()
print(model1.runTest(0.2))

"""total_images = 101
file = "filtered_image_data.csv"
model = AtopicEczemaModel(file, total_images)
model.setup()
# model.load_from_disk()
model.train_new_model()
# model.runTests(file)
print("aaa")
model.export_current_model()
print("aaa")
print(model.runTest(0.2))
print("aaa")
print("ESTIMATE")
print(model.runTest(0.8))"""

total_images2 = 101
file2 = "filtered_image_data.csv"

model2 = AtopicEczemaModel(file2, total_images2)
model2.setup()
model2.train_new_model()
print(model2.runTest(0.2))

camera = PiCamera ()
camera.resolution = (1024, 768)

GPIO.setmode (GPIO.BCM)
GPIO.setup (4, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

lcd = LCD.LCD ()

counter = 0

lcd.clearScreen ()
lcd.printOut ("Hello")

os.environ ["GOOGLE_APPLICATION_CREDENTIALS"] = "../token.json"

mydb = mysql.connector.connect(
    host="34.70.40.173",
    user="root",
    passwd="password123",
    database="test",

)

def getLink(person, count, disease):
    letter_Set = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    new_Name = ""
    for i in range(10):
        new_Name += letter_Set[random.randint(0, 10)]
    new_Name += ".png"

    client = storage.Client()
    bucket = client.get_bucket('jamestang')
    with NamedTemporaryFile() as temp:
        iName = "".join([str(person), ".jpg"])
        count = count + 1
        name = person + ".png"
        blob = bucket.blob(new_Name)  # here
        blob.upload_from_filename(iName, content_type='image/jpeg')
        url = blob.public_url
        sendMeassage(disease, url)
        return url

def sendMeassage(disease, url):
    client = Client("ACd6c7bfff08bea174c9cbe8c5a97d7838", "d60176b90e6a7853d540b73c97fae232")
    to = 'whatsapp:+85263031883'
    from1 = 'whatsapp:+14155238886'
    if disease == "acne vulgaris":
        client.messages.create(body="Unfortunately you might got Acne Vulgaris", from_=from1, to=to, media_url=url)
        client.messages.create(
            body="Acne, also known as acne vulgaris, is a long-term skin disease that occurs when dead skin cells and oil from the skin clog hair follicles",
            from_=from1, to=to)
        
    if disease == "atopic eczema":
        client.messages.create(body="Unfortunately you might got Atopic Eczema", from_=from1, to=to, media_url=url)
        client.messages.create(
            body="Atopic dermatitis, also known as atopic eczema, is a type of inflammation of the skin. It results in itchy, red, swollen, and cracked skin.",
            from_=from1, to=to)
        
    else:
        client.messages.create(body="Your clear!!", from_=from1, to=to, media_url=url)
        
        

my_cursor = mydb.cursor()

""" sqlStuff = "INSERT INTO newhack(username, password, id, image, data) VALUES (%s,%s,%s,%s,%s)"
record1 = ("Eric3", "Eric123", 40, getLink ("../eye", 1, "corneal ulcer"),"data")
my_cursor.execute(sqlStuff, record1)
mydb.commit()"""

lcd.clearScreen ()
lcd.printOut ("Press to\nPhotograph")

while True:
        while True:
            if (GPIO.input (4) == GPIO.HIGH):
                counter += 1
                break

        if (counter == 1):
                camera.capture ("eye.jpg")
                lcd.clearScreen ()
                lcd.printOut ("Picture taken")
                sleep (1)
                red1 = getPercentRed1("../eye.jpg")
                acne_result = model1.runTest(red1)
                red2 = getPercentRed2("../eye.jpg")
                atopic_result = model2.runTest(red2)
                data = ""
                sqlStuff = "INSERT INTO newhack(username, password, id, image, data) VALUES (%s,%s,%s,%s,%s)"
                if (acne_result < 0.5):
                    data = "Acne Vulgaris: negative"
                    record1 = ("Eric3", "Eric123", 40, getLink ("eye", 1, "2"),data)
                    my_cursor.execute(sqlStuff, record1)
                    mydb.commit()
                    lcd.clearScreen ()
                    lcd.printOut ("Acne Vulgaris: -")  

                else:
                    data = "Acne Vulgaris: positive"
                    record1 = ("Eric3", "Eric123", 40, getLink ("eye", 1, "acne vulgaris"),data)
                    my_cursor.execute(sqlStuff, record1)
                    mydb.commit()
                    lcd.clearScreen ()
                    lcd.printOut ("Acne Vulgaris: +")                    

                if (atopic_result < 0.5):
                    data = "Atopic Eczema: negative"
                    record1 = ("Eric3", "Eric123", 40, getLink ("eye", 1, "2"),data)
                    my_cursor.execute(sqlStuff, record1)
                    mydb.commit() 
                    lcd.clearScreen ()
                    lcd.printOut ("Atopic Eczema: -")                   

                else:
                    data = "Atopic Eczema: positive"
                    record1 = ("Eric3", "Eric123", 40, getLink ("eye", 1, "atopic eczema"),data)
                    my_cursor.execute(sqlStuff, record1)
                    mydb.commit()
                    lcd.clearScreen ()
                    lcd.printOut ("Atopic Eczema: +")  
                

                
                

        elif (counter == 2):
                lcd.clearScreen ()
                lcd.printOut ("Finished")
                counter = 0
                sleep (1)

