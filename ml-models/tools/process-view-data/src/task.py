import json
import params
from threading import Thread
import requests
def task_proccessing_percent(percentage):
    print("Percentage:",percentage)
    if params.TASK_ID == -1 or not params.BROADCAST:
        return
    broadcast = json.loads(params.BROADCAST)
    sever = broadcast['server']
    app_id = broadcast['app_id']
    key = broadcast['key']
    URL = "{}/apps/{}/events?auth_key={}".format(sever, app_id, key)
    try:
        payload = {
            "channel": "private-task.{}".format(params.TASK_ID),
            "name": "App\\Events\\TaskProccessingPercent",
            "data": {
                'percentage': percentage
            }
        }
        def request(url, payload):
            requests.post(url=url, json=payload)
        thread = Thread(target=request, args=(URL, payload))
        thread.start()
    except Exception as e:
        print(e)
    return