# 1. Посмотреть документацию к API GitHub, разобраться как вывести список репозиториев для конкретного пользователя, сохранить JSON-вывод в файле *.json.

import requests
import json

params = {'q': 'name', 'id': '44767046'}

url = 'https://api.github.com'

user = 'ValentinaSoldatova'

responce = requests.get(f'{url}/users/{user}/repos')

with open('data.json', 'w') as f:
    json.dump(responce.json(), f)

for i in responce.json():
    print(i['full_name'])


# 2. Зарегистрироваться на https://openweathermap.org/api и написать функцию, которая получает погоду в данный момент для города, название которого получается через input.


import requests


s_city = "Moscow(RU)"
city_id = 524901
appid = "325179fc0c2e6efed3b039f417e4033b"

try:
    res = requests.get("http://api.openweathermap.org/data/2.5/weather",
                 params={'id': city_id, 'units': 'metric', 'lang': 'ru', 'APPID': appid})
    data = res.json()
    print("conditions:", data['weather'][0]['description'])
    print("temp:", data['main']['temp'])
    print("temp_min:", data['main']['temp_min'])
    print("temp_max:", data['main']['temp_max'])
except Exception as e:
    print("Exception (weather):", e)
    pass

