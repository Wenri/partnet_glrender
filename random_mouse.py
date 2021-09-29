import time

from pynput.mouse import Controller

mouse = Controller()

while True:
    time.sleep(10)
    x, y = mouse.position
    mouse.position = (x + 1, y + 1)
    time.sleep(1)
    mouse.position = (x, y)
