from fastapi import FastAPI, Response, status
import uvicorn
from recommendation_system import get_recommendations

app = FastAPI()

@app.get("/recommendation/{movie_title}")
async def main(movie_title: str, response: Response):
    try:
        recommendations = get_recommendations(movie_title)
        if len(recommendations) > 0:
            response.status_code = status.HTTP_200_OK
            return {"total": len(recommendations), "items": recommendations}
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"total" : 0, "items": []}
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"message": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)