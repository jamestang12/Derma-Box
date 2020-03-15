from RPLCD.gpio import CharLCD
import RPi.GPIO as GPIO
import time

class LCD:

	def __init__ (self):
		self.lcd = CharLCD (pin_rs=16, pin_rw=None, pin_e=20, pins_data=[26, 19, 13, 6], numbering_mode=GPIO.BCM, rows=2, cols=16)
		#self.lcd.write_string(u"Hello")
		#self.lcd = CharLCD (pin_rs=36, pin_rw=None, pin_e=38, pins_data=[31, 33, 35, 37], numbering_mode=GPIO.BOARD, rows=2, cols=16)
		self.lcd.cursor_mode = "hide"

	def clearScreen (self):
		self.lcd.clear ()

	def printOut (self, string):
		self.lcd.write_string (u"" + string)

	def println (self, string):
		self.lcd.write_string (u"\r\n" + string)
