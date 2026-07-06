import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#code pour charger le dataset Titanic (Inclus dans seaborn)
df = sns.load_dataset('titanic')
df.to_csv('titanic.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

df.groupby('sex')['survived'].mean().plot(kind='bar', ax=axes[0], color=['salmon', 'skyblue'])
axes[0].set_title('Taux de survie par sexe')
axes[0].tick_params(rotation=0)

df.groupby('pclass')['survived'].mean().plot(kind='bar', ax=axes[1], color='seagreen')
axes[1].set_title('Taux de survie par classe')
axes[1].tick_params(rotation=0)

plt.tight_layout()
plt.show()


'''premier apercu
print(df.shape)
print(df.head())
print(df.dtypes)


print(
    (df.isnull().sum() / len(df) * 100).round(2).sort_values(ascending=False)
)

print(df['survived'].value_counts(normalize=True).round(3))
'''

