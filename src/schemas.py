from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    location: str = Field(..., min_length=1, examples=["Whitefield"])
    total_sqft: float = Field(..., gt=0, examples=[1200])
    bath: float = Field(..., gt=0, examples=[2])
    bhk: int = Field(..., gt=0, examples=[2])


class PredictionResponse(BaseModel):
    predicted_price_lakhs: float
