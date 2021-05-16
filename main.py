import pandas as pd
import numpy as np
from math import floor
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.models import Sequential

# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    import pandas as pd

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return


if __name__ == "__main__":
    args = config()

    try:
        # read model
        model = load_model('./night.h5')

        consumption = pd.read_csv(args.consumption)
        generation = pd.read_csv(args.generation)

        if generation.shape[0] < 168:
            data = []
            output(args.output, data)
            quit()
            # data not enough

        from datetime import datetime, timedelta
        yesterday = datetime.strptime(generation.iloc[-1]['time'], '%Y-%m-%d %H:%M:%S')
        today = yesterday + timedelta(days=1)

        # generation.iloc[-1]['time']
        # a + timedelta(days=1)

        generation['buy'] = pd.DataFrame({'buy':(generation['generation'] - consumption['consumption'])})
        generation['time'] = generation['time'].map(lambda x: int(x[-8:-6]))
        
        hour0 = generation[generation['time']==0]['buy'].to_numpy().reshape(-1, 7, 1)
        ans = model.predict(hour0)

        data = []
        hours = [0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21, 22, 23]
        for i in hours:
            _data = generation[generation['time']==i]['buy'].to_numpy().reshape(-1, 7, 1)
            ans = model.predict(_data)
            print(f'time = {i} and ans = {ans}')
            if ans[0][0] < 0:
                cur_time = today.replace(hour=i)
                data.append([cur_time.strftime('%Y-%m-%d %H:%M:%S'), "buy", 2.2, floor(-ans[0][0])])

    except:
        print("CATCH SOMETHING")
        data = []



    

    # data = [["2018-01-01 00:00:00", "buy", 2.5, 3],
    #         ["2018-01-01 01:00:00", "sell", 3, 5]]
    
    output(args.output, data)
