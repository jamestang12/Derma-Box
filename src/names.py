import random

# Sets do not provide O(1) for randomly accessing elements while lists do
random_names = []

def load_array ():
    with open ("alpha-names.txt") as f:

        for line in f:
            random_names.append (line.rstrip ("\r\n"))
    
    print ("Successfully loaded the names.")

# Method to generate a random name for the database
def random_name ():
    global random_names
    return random_names [int (random.random () * len (random_names))] #randrange is too slow