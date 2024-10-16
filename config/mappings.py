_STOCK_MAPPING = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
}


def stock_mapping(ticker):
    if ticker in _STOCK_MAPPING:
        return _STOCK_MAPPING[ticker]
    return ticker
