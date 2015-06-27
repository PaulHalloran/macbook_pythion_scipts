import time

counter = 0

while True:
	if counter < 5:
		print counter * 5
		time.sleep(1)		
		counter = counter + 1

