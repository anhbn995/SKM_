from threading import Thread
import requests
import params
import json


def on_training_proccessing(percentage, task_id):
    data = {
        'percentage': percentage
    }
    task_proccessing_percent(task_id, data)


def request(url, payload):
    requests.post(url=url, json=payload)


def task_proccessing_percent(task_id, data):
    if task_id == -1 or not params.BROADCAST:
        return
    broadcast = json.loads(params.BROADCAST)
    sever = broadcast['server']
    app_id = broadcast['app_id']
    key = broadcast['key']
    URL = "{}/apps/{}/events?auth_key={}".format(sever, app_id, key)
    try:
        payload = {
            "channel": "private-task.{}".format(task_id),
            "name": "App\\Events\\TaskProccessingPercent",
            "data": data
        }
        thread = Thread(target=request, args=(URL, payload))
        thread.start()
    except Exception as e:
        print(e)
    return
