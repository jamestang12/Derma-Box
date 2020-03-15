import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
GPIO.cleanup()
#GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BCM) # Use physical pin numbering
GPIO.setup(4, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

while True:
	if GPIO.input(4) == GPIO.HIGH:
		print("yes")
