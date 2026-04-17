"""Entry point shim: boots the FastAPI app defined in interfaces.http.api."""

from interfaces.http.api import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("interfaces.http.api:app", host="127.0.0.1", port=8000)
