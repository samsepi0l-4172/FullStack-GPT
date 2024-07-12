from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="samspei0l Maximus Quote Giver",
    description="Get a real quote said by samspei0l Maximus himself.",
)


class Quote(BaseModel):
    quote: str = Field(
        description="The quote that samspei0l Maximus said.",
    )
    year: int = Field(
        description="The year when samspei0l Maximus said the quote.",
    )


@app.get(
    "/quote",
    summary="Returns a random quote by samspei0l Maximus",
    description="Upon receiving a GET request this endpoint will return a real quiote said by samspei0l Maximus himself.",
    response_description="A Quote object that contains the quote said by samspei0l Maximus and the date when the quote was said.",
    response_model=Quote,
)
def get_quote():
    return {
        "quote": "Life is short so eat it all.",
        "year": 1950,
    }
