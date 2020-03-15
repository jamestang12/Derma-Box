import mysql.connector
from google.cloud import storage
import os
import random

from twilio.rest import Client
from tempfile import NamedTemporaryFile

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Owner/Downloads/MyProject.json"

mydb = mysql.connector.connect(
    host="34.70.40.173",
    user="root",
    passwd="password123",
    database="test",

)

my_cursor = mydb.cursor()


# my_cursor.execute("CREATE TABLE newhackdb (username VARCHAR(255), password VARCHAR (255), id INTEGER (255), image VARCHAR(255))")  #Create new table


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
    client = Client("ACd6c7bfff08bea174c9cbe8c5a97d7838", "548b3cb3e50cdda49c3d242f0c04ba4f")
    to = 'whatsapp:+85263031883'
    from1 = 'whatsapp:+14155238886'
    if disease == "corneal ulcer":
        client.messages.create(body="Unfortunately you might got corneal ulcer", from_=from1, to=to, media_url=url)
        client.messages.create(
            body="A corneal ulcer is an open sore of the cornea. There are a wide variety of causes of corneal ulcers, including infection, physical and chemical trauma, corneal drying and exposure, and contact lens overwear and misuse. Corneal ulcers are a serious problem and may result in loss of vision or blindness",
            from_=from1, to=to)
        client.messages.create(
            body="Here is so tips for you: Apply cool compresses to the affected eye. Do not touch or rub your eye with your fingers. Limit spread of infection by washing your hands often and drying them with a clean towel. Take over-the-counter pain medications, such as acetaminophen (Tylenol) or ibuprofen (Motrin)",
            from_=from1, to=to)
    elif disease == "high blood pressure":
        client.messages.create(body="Unfortunately you might got high blood pressure", from_=from1, to=to,
                               media_url=url)
        client.messages.create(
            body="High blood pressure is the number one risk factor for stroke and a major risk factor for heart disease. High blood pressure is when the blood pressure in your arteries is elevated and your heart has to work harder than normal to pump blood through the blood vessels.",
            from_=from1, to=to)
        client.messages.create(
            body="Here is some tips for you: Lose extra pounds and watch your waistline Blood pressure often increases as weight increases. Being overweight also can cause disrupted breathing while you sleep (sleep apnea), which further raises your blood pressure. ",
            from_=from1, to=to)
    elif disease == "sickle cell":
        client.messages.create(body="Unfortunately you might got high sickle cell", from_=from1, to=to, media_url=url)
        client.messages.create(
            body="Sickle cell disease is a group of disorders that affects hemoglobin, the molecule in red blood cells that delivers oxygen to cells throughout the body",
            from_=from1, to=to)
        client.messages.create(
            body="Here is some tips for you: Drink water or other fluids when your symptoms start. Staying hydrated can help you head off the worst of an attack.",
            from_=from1, to=to)


sqlStuff = "INSERT INTO newhack(username, password, id, image, data) VALUES (%s,%s,%s,%s,%s)"
record1 = ("Eric2", "Eric123", 40, getLink("eye", 1, "corneal ulcer"),"data")
my_cursor.execute(sqlStuff, record1)
mydb.commit()
