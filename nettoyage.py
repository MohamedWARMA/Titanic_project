import pandas as pd

df = pd.read_csv('titanic.csv')


#supprimer les colonnes inutilisables
df = df.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'])

# Imputer l'age manquant
df['age'] = df['age'].fillna(df['age'].median())

#Imputer les 2 valeurs manquantes de 'embarked
df['embarked'] = df['embarked'].fillna(df['embarked'].mode([0]))

#Encoder les variables texte en nombres
df['sex'] = df['sex'].map({ 'male' : 0 , 'female': 1})
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

#Creation d'une feature "Taille de la famille"
df['family_size'] = df['sibsp'] + df['parch'] + 1 # +1 pour la personne elle meme

#Sauvegarder le dataset nettoye pour la suite
df.to_csv('titanic_clean.csv', index=False)

print("Sauvegarder : " , df.shape)


print(df.isnull().sum())
print(df.head())