from typing import Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class PaperAccount(BaseModel):
    id: int
    balance: float
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class PaperPosition(BaseModel):
    id: Optional[int] = None
    ticker: str
    shares: float
    avg_cost: float
    current_value: Optional[float] = None
    status: str = "OPEN"
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, extra="ignore")

class PaperTrade(BaseModel):
    id: Optional[int] = None
    ticker: str
    action: str
    shares: float
    price: float
    total_value: float
    reasoning: Optional[str] = None
    macro_risk_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    pnl_percent: Optional[float] = None
    realized_pnl_usd: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    invalidation_condition: Optional[str] = None
    chain_of_thought: Optional[str] = None
    cycle_event: Optional[str] = None
    emergency: Optional[bool] = None
    was_punished: bool = False
    lesson_written: bool = False
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, extra="ignore")
