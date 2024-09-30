from pydantic import BaseModel, Field, ConfigDict
import yfinance as yf
import datetime
import pandas as pd
from langchain_core.tools import tool

DICT_TICKER_NAMES ={
    "Microsoft" : "MSFT",
    "Apple" : "AAPL",
    "Google" : "GOOG"
}

class get_ticker_data_input(BaseModel):
    ticker : str = Field(..., description="Acronym of the ticker")

class get_ticker_data_response(BaseModel):
    answer : str = Field(..., description="The final answer to respond to the user.")
    data : pd.DataFrame = Field(...,description="Dataframe with the data of the ticker")

    model_config = ConfigDict(arbitrary_types_allowed=True)

@tool(args_schema=get_ticker_data_input)
def get_ticker_data(ticker:str) -> get_ticker_data_response:
    """
    Get the data of a ticker given it's acronym.

    Args : 
        -   ticker : Acronym of the ticker
    """
    print("\nExecuting get_ticker_data tool.\n")
    stock = yf.Ticker(ticker=ticker)
    start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    interval = "1h"
    data = stock.history(start=start_date,
                         end=end_date,
                         interval=interval)
    
    if data.empty:
        print("Not data found.")
        return get_ticker_data_response(
            answer=f"No data found for ticker {ticker}",
            sources=pd.DataFrame()
        )
    
    answer = f"Historical data for ticker {ticker} from {start_date} to {end_date}."
    return get_ticker_data_response(
        answer=answer,
        sources=data  # Aquí se retorna el DataFrame con los datos
    )

@tool()
def get_current_date() -> str:
    """
    Get the current date.
    Returns: 
        - answer : Text response with the current date.
    """
    print("\nExecuting get_current_date tool.\n")
    answer = f"The current date is: {datetime.datetime.now().strftime('%Y-%m-%d')}"
    return answer
