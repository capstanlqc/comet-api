import json

import requests

model = "Unbabel/wmt22-cometkiwi-da"
mode = "mock"  # mock or prod
parameters = f"?model={model}&mode={mode}"

# url = "http://127.0.0.1:8000/api/scores"
url = "http://192.168.67.37:8000/api/scores"
# url = "https://mtpe-mock.up.railway.app/api/scores"

# data example:

data = [
    {
        "src": "How to Demonstrate Your Strategic Thinking Skills",
        "mt": "Cómo demostrar su capacidad de pensamiento estratégico",
    },
    {
        "src": "Why is Accuracy important in the workplace?",
        "mt": "¿Por qué es importante la precisión en el trabajo",
    },
    {
        "src": "When faced with a large amount of analysis ask for support setting up a team to approach the issue in different ways.",
        "mt": "Cuando se enfrente a una gran cantidad de análisis, pida ayuda para crear un equipo que aborde la cuestión de diferentes maneras.",
    },
]


def prepare_data(bitexts):
    return [{"src": k, "mt": v} for k, v in bitexts.items()]


def add_scores(bitexts):
    data = prepare_data(bitexts)
    # print(f"{data=}")

    payload = json.dumps(data)
    headers = {"Content-Type": "application/json"}
    response = requests.get(f"{url}{parameters}", headers=headers, data=payload)

    if response.status_code == 200:
        return json.loads(response.text)


bitexts = dict(zip([x["src"] for x in data], [x["mt"] for x in data]))
x = add_scores(bitexts)

print(x)
