from typing import Dict, Tuple

AssetsAvgPrice = Dict[str, float]
AssetsMarketPrice = Dict[str, float]
AssetsPnL = Dict[str, float]
ActionType = Dict[str, str]
AvaiableAction = Tuple[bool, Dict[str, float]]
PortfolioTradeReturn = Dict[str, float]
SizeOfAssets = Dict[str, float]
ExecuteResult = Tuple[PortfolioTradeReturn, ActionType]