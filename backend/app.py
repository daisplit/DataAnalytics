import pandas as pd
import numpy as np
import datetime
from pyramid.arima import auto_arima

class Modelling:
    def timeseries_forecast(self,data):
        print(data['expenses'])
        model = auto_arima(data['expenses'], start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
        model.fit(data['expenses'])
        return model

    def forecast(self, data):
        model = self.timeseries_forecast(data)
        prediction = model.predict(n_periods=3)
        forecast = pd.DataFrame(prediction).reset_index()
        forecast.columns = ['index', 'expenses']
        result = pd.concat([data,forecast])
        return result['expenses']
