

from fastapi import FastAPI, WebSocket
from random import choice, randint
from scipy.stats import spearmanr
import asyncio

app = FastAPI()


# put list of symbols in end point call
@app.websocket("/sample/{symbols}")
async def websocket_endpoint(websocket: WebSocket, symbols: str):
    list_symbols = symbols.split('-')
    await websocket.accept()
    while True:
        await websocket.send_json({
            # security 1
            "channel_0": list_symbols[0],
            "data_0": randint(1, 10),
            # security 2
            "channel_1": list_symbols[1],
            "data_1": randint(1, 10),
            # corr(1,2)
            'corr_channel': 'corr_1',
            'corr_1_data': return_corr([randint(1,10) for _ in range(31)], 30)
            }
        )
        await asyncio.sleep(0.5)


def return_corr(data, window):
    if len(data)>window:
        return spearmanr(data[0:window], data[1:window+1])[0]
    else:
        return 0



