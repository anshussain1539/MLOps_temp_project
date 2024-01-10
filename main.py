import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import joblib


Category_Names = pd.read_csv('./datafiles/item_categories.csv')
Shop_Names = pd.read_csv('./datafiles/shops.csv')
Item_Names = pd.read_csv('./datafiles/items.csv')
Training_Data = pd.read_csv('./datafiles/sales_train.csv')
Sample_submission = pd.read_csv('./datafiles/sample_submission.csv')

rows = Training_Data.shape[0]

def cityextractor():
    
    lists = []
    count = 0
    for i in Shop_Names['shop_name']:
        temp = i
        start_pt = temp.find("\"")
        end_pt = temp.find("\"", start_pt + 1) 
        quote = temp[start_pt + 1: end_pt]

        if(len(i)-1 != len(quote)):
            lists.append(quote)
        else:
            lists.append('N/A')
            
    # labeling city names
    cities = []
    indexes = []

    for i, value in enumerate(lists):

        if value not in cities:
            cities.append(value)
            indexes.append(len(cities)-1)
        else:
            for x, v in enumerate(cities):
                if v == value:
                    indexes.append(x)
    
    return indexes     

Shop_Names['city_labels'] = cityextractor()

def categoryextractor():
    
    maincategory, subcategory = [], []

    for i in Category_Names.values[:,0]:
        splited = i.split(' - ')
        if len(splited) == 2:
            for j , v in enumerate(splited):
                if v == 'Аксессуары':
                    splited[j] = 'Accessories'
                elif v == 'Игровые консоли':
                    splited[j] = 'Game consoles'
                elif v == 'Игры':
                    splited[j] = 'Games'
                elif v == 'Карты оплаты':
                    splited[j] = 'Payment cards'
                elif v == 'Кино':
                    splited[j] = 'Cinema'
                elif v == 'Книги':
                    splited[j] = 'Books'
                elif v == 'Музыка':
                    splited[j] = 'Music'
                elif v == 'Подарки':
                    splited[j] = 'Present'
                elif v == 'Программы':
                    splited[j] = 'Programs'
                elif v == 'Служебные':
                    splited[j] = 'Service'
                elif v == 'Игры PC':
                    splited[j] = 'PC Games' 
                elif v == 'Игры Android':
                    splited[j] = 'Android Games' 
                elif v == 'Игры MAC':
                    splited[j] = 'MAC Games'

            maincategory.append(splited[0])
            subcategory.append(splited[1])

        else:
            if 'Карты оплаты (Кино, Музыка, Игры)' in splited:
                maincategory.append('Payment cards')
                subcategory.append('Movie, Music, Games')
            elif 'Билеты (Цифра)' in splited:
                maincategory.append('Tickets')
                subcategory.append('Digital')
            elif 'Чистые носители (шпиль)' in splited:
                maincategory.append('Pure media')
                subcategory.append('spire')    
            elif 'Чистые носители (штучные)' in splited:
                maincategory.append('Pure media')
                subcategory.append('piece')   
            elif 'Доставка товара' in splited:
                maincategory.append('Delivery')
                subcategory.append('Goods') 
            elif 'Служебные' in splited:
                maincategory.append('Service')
                subcategory.append('N/A') 
            elif 'Элементы питания' in splited:
                maincategory.append('Batteries')
                subcategory.append('N/A')
                
        return maincategory, subcategory


def dateseparator():

    day, month, year = [], [], []

    for i, value in enumerate(Training_Data['date']):
        date = value.split('.')
        day.append(int(date[0])); month.append(int(date[1])); year.append(int(date[2]));
    
    return day, month, year

Training_Data['Day'], Training_Data['Month'], Training_Data['Year'] = dateseparator()


Training_Data = pd.merge(Training_Data, Item_Names, how = "inner", on = "item_id")
Training_Data = pd.merge(Training_Data, Shop_Names, how = "inner", on = "shop_id")
Training_Data = pd.merge(Training_Data, Category_Names, how = "inner", on = "item_category_id")


Training_Data = Training_Data[Training_Data['item_cnt_day'] >= 0]
Training_Data = Training_Data[Training_Data['item_price'] >= 0]
Training_Data = Training_Data[Training_Data['item_price'] < 100000]

print("Rows removed : ", rows - Training_Data.shape[0])
rows = Training_Data.shape[0]


Training_Data.drop(['date','item_name','shop_name','item_category_name','Day','item_price'], inplace=True, axis=1)



print("Number Of Shops : " , len(Training_Data['shop_id'].unique()))
print("Number Of Cities : " , len(Training_Data['city_labels'].unique()))
print("Number Of Items : " , len(Training_Data['item_id'].unique()))
print("Number Of Items' Categories : " , len(Training_Data['item_category_id'].unique()))


Training_Data = Training_Data.groupby(['shop_id','item_id', 'date_block_num', 'Month', 'Year', 'item_category_id','city_labels']).sum().reset_index()
rows = Training_Data.shape[0]

Training_Data = Training_Data.sort_values(by=['date_block_num','shop_id','item_id']).reset_index()
Training_Data.drop(['index'], inplace=True, axis=1)


def outliers(feature):
    
    sum_value = 0
    for i in feature:
        sum_value += i
    
    mean = sum_value / len(feature)
    
    std = 0
    value = 0
    for i in feature:
        value += (i - mean) ** 2
        
    std = (value / len(feature)) ** 0.5
    
    upperlimit = mean + 3* std
    lowerlimit = mean - 3* std
    
    outlier, index = [], []
    for i, value in enumerate(feature):
        if value >= upperlimit or value <= lowerlimit:
            index.append(i)
            outlier.append(value)
            
    return index, outlier


index, outlier = (outliers(Training_Data['item_cnt_day']))
Training_Data.drop(index, axis=0, inplace=True)
print("Rows removed : ", rows - Training_Data.shape[0])
rows = Training_Data.shape[0]

q75, q25 = np.percentile(Training_Data, [75, 25], axis = 0)
inter_quartile_range = q75 - q25

lower = q25 - (1.5 * inter_quartile_range)
upper = q75 + (1.5 * inter_quartile_range)

Training_Data[Training_Data < lower] = np.nan
Training_Data[Training_Data > upper] = np.nan
Training_Data = Training_Data.dropna()


Training_Data = Training_Data.reset_index()
Training_Data.drop('index', inplace=True, axis=1)
print("Rows removed : ", rows - Training_Data.shape[0])
rows = Training_Data.shape[0]


from itertools import product
def gridder(sale_train):
    grid = []

    for block_num in sale_train['date_block_num'].unique():

        cur_shops = sale_train[sale_train['date_block_num']==block_num]['shop_id'].unique()

        cur_items = sale_train[sale_train['date_block_num']==block_num]['item_id'].unique()

        grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))



    #turn the grid into pandas dataframe

    index_cols = ['shop_id', 'item_id', 'date_block_num']

    grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
    return grid



expanded_df = pd.DataFrame()
expanded_df['shop_id'] = Training_Data['shop_id'] 
expanded_df['item_id'] = Training_Data['item_id'] 
expanded_df['date_block_num'] = Training_Data['date_block_num'] 


grid = gridder(expanded_df)
grid.shape



grid = pd.merge(grid, Item_Names, how = "inner", on = "item_id")
grid.drop(['item_name'], inplace=True, axis=1)

grid = pd.merge(grid, Shop_Names, how = "inner", on = "shop_id")
grid.drop(['shop_name'], inplace=True, axis=1)


grid["Month"] = grid["date_block_num"].apply(lambda x: (x+1) % 12)
grid["Year"] = grid["date_block_num"].apply(lambda x: ((x+1) // 12) + 2013)


index_cols = ['date_block_num', 'shop_id', 'item_id','item_category_id','city_labels','Month','Year']
train = pd.merge(grid, Training_Data, how = 'left', on = index_cols)


train['item_cnt_month'] = train['item_cnt_day'].fillna(0).clip(0,20).astype(int)


train.drop(['item_cnt_day','city_labels'], inplace=True, axis=1)


def scale_features(X):
        
        df = pd.DataFrame(X)
        
        mini , maxi = [] , []
        for i in df.columns:

            mini.append(min(df[i]))
            maxi.append(max(df[i]))
            
        xmin , xmax = np.array(mini) , np.array(maxi)
        X = (X - xmin) / (xmax - xmin)

        return X


train = train.sort_values(by=['date_block_num','shop_id','item_id']).reset_index()
train.drop(['index'], inplace=True, axis=1)



print('Final Dataset Before Splitting ', train.shape)
train



X = train.drop('item_cnt_month', axis=1)
Y = train['item_cnt_month']

split = int(len(X) * 0.8) #  * 0.8)


X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]


model = DecisionTreeRegressor(random_state = 0) 
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

print('MSE : ' , mean_squared_error(Y_test , predictions))

# Save the trained model
joblib.dump(model, 'decision_tree_model.joblib')

