
import joblib
import re
import string

def clean_text(text):
    text=text.lower()
    text=re.sub(r"\W", " ", text)#Remplace tout caractère non-alphanumérique par un espace
    text=re.sub(r"https?://\S+", "", text)#Supprime les liens web.
    text=re.sub("<.*?>","",text)#Supprime les balises HTM
    text=re.sub("[%s]" % re.escape(string.punctuation),"",text)#Supprime toute la ponctuation (.,!?; etc.)
    text=re.sub(r"\n", "", text)#pour supprimer les sauts de ligne.
    text=re.sub("/w*/d/w*","",text)#Supprime les mots contenant des chiffres.
    return text

def manual_test_article(article_text):
    text = clean_text(article_text)
    Xv = vectoriser.transform([text])
    pred = model.predict(Xv)[0]
    prob = model.predict_proba(Xv)[0][pred]
    label = "FAKE" if pred == 0 else "REAL"
    return {"label": label, "prob": prob, "cleaned": text}


import joblib

vectoriser = joblib.load("C:\\Downloads\\fake-news\\models\\Vec.jb")
model = joblib.load("C:\\Downloads\\fake-news\\models\\Model.jb")


while True:
    article = input("Collez un article à tester (ou tapez 'quit' pour arrêter) :\n")
    if article.lower() == "quit":
        print("Fin du programme.")
        break
    res = manual_test_article(article)
    print("Label :", res['label'])
    print(f"Confiance : {res['prob']*100:.2f}%")

