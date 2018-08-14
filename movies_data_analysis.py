import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# % matplotlib inline
import seaborn as sns
from subprocess import call

df = pd.read_csv('tmdb-movies.csv')
df.head()

df.info()
df.dtypes
df.isnull().sum()

describe_numeric = df.describe()
describe_numeric

df.columns

describe_object = df.describe(include=[np.object])
describe_object

df.isnull().describe()

#抽取原始数据的几个样本来观察
df.sample(10)

df.columns

df = df.fillna('missing')

describe_null_no = df.isnull().describe()
describe_null_no

df.info()

df.sample(10)

#检查重复值
df.duplicated().sum()

#删除含有重复值的项
df = df.drop_duplicates()

#再次检查重复值的数量，应该为零
df.duplicated().sum()

#对电影类型进行整理
genres = df['genres']

df.columns

genres = df.genres
genres.head()

genres.sample(10)

#电影类型的数据的清理
genres = df['genres'].str.split('|',expand = True).stack().reset_index(level = 1,drop = True).rename('genres2')

genres.head(5)

#检查genres的唯一值
genres.unique()

#检查genres的描述性统计
genres.describe()

#合并这一新的电影分组到数据集中
new_df = df.join(genres)

#检查确认连接情况
new_df[['genres','genres2','id']].head(5)

#检查数据集new_df的列情况
new_df.columns

#根据分组genres2，统计出popularity的每个电影类型的平均
new_df_genres2 = new_df.groupby('genres2')['popularity'].mean().sort_values(ascending=False)

#可视化genres2的平均popularity
new_df_genres2.plot(kind='bar',figsize=(15,15))

#选择人气popularity当做评估指标，对每年最受欢迎电影进行评估
popularity = new_df.groupby(['genres2'],as_index=False)['popularity'].mean()

popularity.head()

x = popularity.index
y = popularity
plt.ylabel("value of popularity")
plt.title("popularity per kind of movie type")
popularity.hist()
popularity.plot()

df.groupby('release_year')['popularity'].mean().plot(kind = 'line')
plt.ylabel('Average Revenue')
plt.title('Average popularity VS release year')

#选择vote_count作为评估每年最受欢迎的电影类型的指标
# vote_count_plt = new_df.groupby(['release_year','genres2'],as_index=False)['vote_count'].mean().plot()
new_df.groupby(['genres2'],as_index=False)['vote_count'].mean().plot(kind = 'line')
plt.ylabel('average og vote_count')
plt.title('average vote count VS genres2')

#根据分组genres2，统计出vote_count的每个电影类型的平均
new_df_genres2_vote_count = new_df.groupby('genres2')['vote_count'].mean().sort_values(ascending=False)

#按照genres2分组，计算revenue的均值,这可以分析票房高的电影的特点
new_df_genres2 = new_df.groupby('genres2')['revenue'].mean().sort_values(ascending=False)

#画出每个电影类型的收入平均值的条形图
new_df_genres2.plot(kind='bar',figsize=(10,10))

#可视化genres2的平均popularity
new_df_genres2_vote_count.plot(kind='bar',figsize=(10,10))

#选择评分vote_average作为每年最受欢迎的电影的评估指标
vote_average = new_df.groupby(['genres2'],as_index=False)['vote_average'].mean()

#画出分布直方图
vote_average.plot(kind='bar')
plt.xlabel('index of genres')
plt.ylabel('vote average')
plt.title(' genres2 vs vote average')


vote_average.plot(kind='line')
plt.xlabel('index of genres')
plt.ylabel('vote average')
plt.title(' genres2 vs vote average')

#根据分组genres2，统计出vote_average的每个电影类型的平均
new_df_genres2_vote_average = new_df.groupby('genres2')['vote_average'].mean().sort_values(ascending=False)

#可视化genres2的平均popularity
new_df_genres2_vote_average.plot(kind='bar',figsize=(10,10))
plt.xlabel('index of genres')
plt.ylabel('vote average')
plt.title(' genres2 vs vote average')

#选择popularity，vote_count，vote_average，三个特征作为每年最受欢迎的电影类型平哥指标
above_favorite = new_df.groupby(['genres2'],as_index=False)['popularity','vote_count','vote_average'].mean()

#打印每年最受欢迎的电影类型前几项
above_favorite.head()

above_favorite.hist()

above_favorite.plot(kind='line')
#最好对数据标准化后画图
plt.xlabel('index of genres')
plt.ylabel('popular level')
plt.title(' genres2 vs vote average')

#新合并后的数据集的列表名称打印
new_df.columns

df.columns

#通过popularity，vote_count，vote_average每年最受欢迎的电影的列举
genres_agg_pca = new_df.groupby('genres').agg({'popularity':'max','vote_count':['max'],'vote_average':'max'})

#打印前几项
genres_agg_pca.head()

#最受欢迎的电影类别，先分电影类别，在列举年份
genres_release_year = new_df.groupby(['genres','release_year']).agg({'popularity':'max','vote_count':['max'],'vote_average':'max'})

genres_release_year.head()

genres_release_year.hist()

#每年最受欢迎的电影类型列表
release_year_genres_pca = new_df.groupby(['release_year','genres2']).agg({'popularity':'max','vote_count':'max','vote_average':'max'})

#显示列表的前面部分
release_year_genres_pca.head()

#列表样本数量统计
release_year_genres_pca.count()

#画出列表张三种指标的直方图
release_year_genres_pca.hist()

#查看数据的10个样本
df.sample(5)

#查看数据集的描述性特征
df.describe()

df['revenue'].describe()

#绘制散点图
x = df.index
y = df['revenue']

# 计算颜色值
color = np.arctan2(y, x)
# 绘制散点图
plt.scatter(x, y, s = 75, c = color, alpha = 0.5)
# 设置坐标轴范围
plt.xlim((0, 10865))
plt.ylim((0.000000e+00, 2.781506e+09))

# 不显示坐标轴的值
# plt.xticks(())
# plt.yticks(())
plt.xlabel('index')
plt.ylabel('revenue')
plt.title(' scatter of revenue')
plt.show()

#对人气popularity进行分等级
#使用的函数qcut可以参考链接http://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.qcut.html#pandas.qcut
df['popularity_level'] = pd.qcut(df['popularity'],4,labels=['low','medium','high','very high'])

#对人气popularity进行分等级
#使用的函数qcut可以参考链接http://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.qcut.html#pandas.qcut
df['vote_average_level'] = pd.qcut(df['vote_average'],4,labels=['low','medium','high','very high'])

#使用cut对票房等级进行分组，分为四组
#cut的用法请看链接http://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.cut.html
df['revenue_adj_levels'] = pd.cut(df['revenue_adj'],4,labels=['low','medium','high','evry high'])

df.info()

#过滤调试
#过滤筛选票房等级高的电影
#query 的用法参见http://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.query.html#pandas.DataFrame.query
# revenue_adj_levels_high = df.query('revenue_adj_levels ==">high" ')
# revenue_adj_levels_high
# revenue_adj_levels_high.head()
#实际票房收入revenue_adj的折线图
# revenue_adj_levels_mean.plot()
# plt.xlabel('index')
# plt.ylabel('revenue_adj_levels_mean')
# plt.title(' line of revenue_adj')

#使用query进行过滤，把票房分文两部分，一部分是高的，一部分是低等级的
median = df['revenue_adj'].median()
low = df.query('revenue_adj < {}'.format(median))
high = df.query('revenue_adj >= {}'.format(median))

mean_quality_low = low['revenue_adj'].mean()
mean_quality_high = high['revenue_adj'].mean()

#我们来看看各个等级的描述性统计
revenue_adj_levels_describe = df.groupby('revenue_adj_levels').revenue_adj.describe()

df.groupby('revenue_adj_levels').revenue_adj.describe().plot(kind='line')
plt.xlabel('revenue_adj_levels')
plt.ylabel('revenue_adj')
plt.title(' revenue_adj vs revenue_adj_levels')

df.groupby('revenue_adj_levels').revenue_adj.describe().plot(kind='bar',figsize=(15,15))
plt.xlabel('revenue_adj_levels')
plt.ylabel('revenue_adj')
plt.title(' revenue_adj vs revenue_adj_levels')

df.groupby('revenue_adj_levels').popularity.describe().plot(kind='bar',figsize=(15,15))
plt.xlabel('revenue_adj_levels')
plt.ylabel('popularity')
plt.title(' popularity vs revenue_adj_levels')

#通过可视化每个票房等级的票数来分析
df.groupby('revenue_adj_levels').vote_count.describe().plot(kind='bar',figsize=(15,15))
plt.xlabel('revenue_adj_levels')
plt.ylabel('vote_count')
plt.title(' vote_count vs revenue_adj_levels')

#通过可视化每个票房等级的票数来分析
df.groupby('revenue_adj_levels').vote_average.plot(kind='bar',figsize=(15,15))
plt.xlabel('revenue_adj_levels')
plt.ylabel('vote_average')
plt.title(' vote_average vs revenue_adj_levels')

#通过可视化每个票房等级的来分析每个等级的人气特点
df.groupby('revenue_adj_levels').popularity.plot(kind='bar',figsize=(15,15))

#按照收入进行分组聚合
two_revenue = df.groupby(['revenue','revenue_adj'],as_index=False).max()

#画出直方图
df.groupby(['revenue','revenue_adj'],as_index=False).max().hist()

#根据年份分组，聚合列出，popularity，vote_count，vote_average的最大值
agg_p_c_a = new_df.groupby('revenue_adj').agg({'popularity':'max','vote_count':['max'],'vote_average':'max'})

df.groupby(['revenue','revenue_adj'],as_index=False).max().plot()
plt.xlabel('index')
plt.ylabel('values of renenue')
plt.title('compare with two renenue ')

x = df.index
y = df

#使用query进行过滤，把票房分文两部分，一部分是高的，一部分是低等级的
median = df['budget'].median()
low = df.query('budget < {}'.format(median))
high = df.query('budget >= {}'.format(median))

mean_quality_low = low['budget'].mean()
mean_quality_high = high['budget'].mean()

#对票房进行等级划分
locations = [1, 2]
heights = [mean_quality_low, mean_quality_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average Quality Ratings by Alcohol Content')
plt.xlabel('Alcohol Content')
plt.ylabel('Average Quality Rating');

#研究问题 3（这些年电影票房的变化趋势是什么样的？）

df.columns

df.groupby('release_year',as_index=False).budget_adj.plot(figsize=(20,20))

#每一年的电影数量的变化趋势
df.groupby('release_year').original_title.count().plot()

#研究问题 4（电影时长有什么特点）
#查看数据集的列，方便输入
df.columns

#将数据集的索引设置为下标
x = df.index
#将时长设置为y值
y = df.runtime

plt.plot(x,y)

#电影时长分布散点图
plt.scatter(x,y)
df['runtime_level'] = pd.qcut(df['runtime'],4,labels=['low','medium','high','very high'])

#画出箱线图
df.boxplot(by = 'runtime_level',column = 'runtime')

df.runtime.describe()

df.runtime_level.unique()

df.runtime_level.describe()

df.runtime.plot(kind='bar',figsize=(15,15))


call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])



