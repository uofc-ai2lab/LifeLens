import Jetson.GPIO as GPIO
import time

BUTTON_PIN = 15
LED_PIN = 7

GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.HIGH)

try:
    print("Press the button to toggle the LED.")
    while True:
        button_state = GPIO.input(BUTTON_PIN)
        if button_state == GPIO.LOW:
            GPIO.output(LED_PIN, GPIO.HIGH)
            print("LED ON")
            print("Button Pressed!")
        else:
            GPIO.output(LED_PIN, GPIO.LOW)
            print("LED OFF")
            print("Button Released!")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    GPIO.cleanup()


