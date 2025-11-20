from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    item_id_berekend = item_id * 2
    return {"item_id_origineel": item_id, "item_id_berekend": item_id_berekend, "q": q}

@app.get("/users/{username}")
def read_user(username: str):
    return {"username": username, "message": f"Welkom, {username}!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)