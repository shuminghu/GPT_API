import time 

def retry(func):
    while True:
        try:
            func()
            return
        except Exception as e:
            print(e)
            time.sleep(5)
            continue