from transformers import pipeline

import os

import spacy
import re

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity as cosine

import fasttext

def obtenir_frequencia(cami_diccionari_frequencies):
    frequencies = {}
    with open(cami_diccionari_frequencies, "r", encoding='latin-1') as reader:
        while True:
            linia = reader.readline()
            if not linia:
                break
            paraula,frequencia = linia.strip().split('\t',1)
            frequencies[paraula] = frequencia
    return frequencies

def obtenir_prevalenca_lexica(cami_prevalenca_lexica):
  percentatges = {}
  with open(cami_prevalenca_lexica, "r", encoding='latin-1') as reader:
    while True:
      linia = reader.readline()
      if not linia:
        break
      paraula, percentatge = linia.strip().split('\t',1)
      percentatges[paraula] = percentatge
  return percentatges

def comptar_sillabes(paraula):
    paraula = paraula.lower()
    vocals = "aeiouàèéíòóúü"
    diftongs = ["ai", "ei", "ii", "oi", "ui", "au", "eu", "iu", "ou", "ua", "ue", "uo", "uu"]
    hiats = ["aí", "aï", "eí", "eï", "oí", "oï", "uí", "uï", "aú", "eú", "oú", "aà", "eè", "oò", "ií", "uú"]

    vocals_trobades = re.findall(f'[{vocals}]', paraula)
    num_vocals = len(vocals_trobades)

    for diftong in diftongs:
        if diftong in paraula:
            num_vocals -= paraula.count(diftong)

    for hiat in hiats:
        if hiat in paraula:
            num_vocals += paraula.count(hiat)

    return max(1, num_vocals)

def mateix_genere_i_nombre(tokenitzador, paraula1, paraula2):

    doc = tokenitzador(f"{paraula1} {paraula2}")

    if len(doc) < 2:
        return False

    morph1 = doc[0].morph
    morph2 = doc[1].morph

    gender1 = morph1.get("Gender")
    gender2 = morph2.get("Gender")

    number1 = morph1.get("Number")
    number2 = morph2.get("Number")

    gender_ok = (not gender1) or (not gender2) or (gender1 == gender2)
    number_ok = (not number1) or (not number2) or (number1 == number2)

    return gender_ok and number_ok

def filtrar_candidats(paraula_original, candidats, tokenitzador, frequencies, percentatges, diccionari_vectors, morfologia):
    candidats_valids = []
    frequencies_candidats = []
    percentatges_coneixement_candidats = []
    puntuacions_vectors_candidats = []

    isFast = True

    if paraula_original not in diccionari_vectors:
        isFast = False
    else:
        vector_paraula_original = diccionari_vectors[paraula_original]

    for candidat in candidats:

        if morfologia:
            if not mateix_genere_i_nombre(tokenitzador, paraula_original, candidat):
                print("Hem eliminat " + candidat + " perquè no té el mateix gènere i nombre que " + paraula_original)
                continue

        # if candidat not in frequencies:
        #     print("Hem eliminat " + candidat + " perquè no apareix al diccionari de freqüències | " + paraula_original)
        #     continue
        # else:
        #     frequencia_candidat = float(frequencies[candidat])
            
        # if (frequencia_candidat < 3.608):
        #     print("Hem eliminat " + candidat + " perquè la seva freqüència és molt baixa | " + paraula_original)
        #     continue
        
        # if candidat not in percentatges:
        #     tokenitzat = tokenitzador(candidat)
        #     lema_candidat = tokenitzat[0].lemma_
        #     if lema_candidat not in percentatges:
        #         print("Hem eliminat " + candidat + " perquè no apareix al diccionari de word prevalence | " + paraula_original)
        #         continue
        #     else:
        #         percentatge_coneixement_candidat = float(percentatges[lema_candidat])
        # else:
        #     percentatge_coneixement_candidat = float(percentatges[candidat])

        # if (percentatge_coneixement_candidat < 0.985):
        #     print("Hem eliminat " + candidat + " perquè el percentatge de coneixement és molt baix | " + paraula_original)
        #     continue

        if isFast:
            if candidat not in diccionari_vectors:
                print("Hem eliminat " + candidat + " perquè no apareix al diccionari de vectors | " + paraula_original)
                continue

            vector_candidat = diccionari_vectors[candidat]
            puntuacio_vector_candidat = cosine(vector_paraula_original.reshape(1,-1), vector_candidat.reshape(1,-1)).item()
            if puntuacio_vector_candidat < 0.422:
                print("Hem eliminat " + candidat + " perquè la semblança és molt baixa | " + paraula_original)
                continue
            puntuacions_vectors_candidats.append(puntuacio_vector_candidat)
        
        candidats_valids.append(candidat)
                
    return candidats_valids, frequencies_candidats, percentatges_coneixement_candidats, puntuacions_vectors_candidats

def obtenir_millors_candidats(frase, paraula_original, tokenitzador, frequencies, percentatges, unmasker, diccionari_vectors, morfologia):
    masked = frase.replace(paraula_original, "<mask>")
    contextualized_masked = f"{frase} </s> {masked}"
    resultats = unmasker(contextualized_masked, top_k=10)

    candidats = []
    score_candidats_model = []
    num_sillabes_candidats = []

    if resultats:
        for resultat in resultats:
            if isinstance(resultat, dict) and 'score' in resultat:
                candidat = resultat['token_str'].strip().lower()
                candidats.append(candidat)
                num_sillabes = comptar_sillabes(candidat)
                num_sillabes_candidats.append(num_sillabes)
                score_candidats_model.append(resultat['score'])

    candidats_filtrats, frequencies_candidats, percentatges_coneixement_candidats, puntuacions_vectors_candidats = filtrar_candidats(paraula_original, candidats, tokenitzador, frequencies, percentatges, diccionari_vectors, morfologia)
    
    # Si no hi ha cap candidat, retornem la paraula complexa
    if len(candidats_filtrats) == 0:
        return [paraula_original]
    
    if len(puntuacions_vectors_candidats) > 0:
        seq = sorted(puntuacions_vectors_candidats, reverse = True )
        sis_rank = [seq.index(v)+1 for v in puntuacions_vectors_candidats]
    else:
       print("No hi ha vectors")

    rank_sillabes = sorted(num_sillabes_candidats, reverse = False)
    sillabes_rank = [rank_sillabes.index(v)+1 for v in num_sillabes_candidats]

    rank_score = sorted(score_candidats_model, reverse = True)
    score_rank = [rank_score.index(v)+1 for v in score_candidats_model]

    rank_count = sorted(frequencies_candidats, reverse = True )
    count_rank = [rank_count.index(v)+1 for v in frequencies_candidats]

    rank_perc = sorted(percentatges_coneixement_candidats, reverse = True)
    perc_rank = [rank_perc.index(v)+1 for v in percentatges_coneixement_candidats]

    if len(puntuacions_vectors_candidats) > 0:
        all_ranks = [0.5*score+0.5*sis  for sis,score in zip(sis_rank,score_rank)] 
    else:
        all_ranks = [score  for score in zip(score_rank)] 

    candidats_ordenats = [x for _, x in sorted(zip(all_ranks, candidats_filtrats))]

    return [candidat for candidat in candidats_ordenats]

def simplificar_text(text, tokenitzador, frequencies, percentatges, unmasker, diccionari_vectors):
    text_simplificat = text
    text_tokenitzat = tokenitzador(text)

    # Les úniques categories gramaticals que podem substituir són noms comuns, verbs no auxiliars, adjectius i adverbis
    categories = ['VERB', 'NOUN', 'ADJ', 'ADV']

    for token in text_tokenitzat:
        # Comprovem que la categoria gramatical del token és alguna de les que hem definit
        if (token.pos_ in categories):
            millors_candidats = obtenir_millors_candidats(text, token.text, tokenitzador, frequencies, percentatges, unmasker, diccionari_vectors, True)
            text_simplificat = text_simplificat.replace(token.text, millors_candidats[0])
    return text_simplificat

def avaluacio(tokenitzador, frequencies, percentatges, unmasker, diccionari_vectors):
    nom_avaluacio = input("Anomena l'avaluació: ")
    f = open("candidates/" + nom_avaluacio + ".tsv", "w", encoding='utf-8')
    # Llegim el conjunt de dades
    with open("evaluation/multilex_test_ca_ls_unlabelled.tsv", "r", encoding='utf-8') as reader:
        print("Reading file\n")
        while True:
            linia = reader.readline()
            if not linia:
                break
            frase, paraula_objectiu = linia.strip().split('\t',1)
            f.write(frase + "\t" + paraula_objectiu + " ")
            millors_candidats = obtenir_millors_candidats(frase, paraula_objectiu, tokenitzador, frequencies, percentatges, unmasker, diccionari_vectors, False)
            for candidat in millors_candidats:
                f.write("\t" + candidat)
            f.write("\n")
        f.close()
    print("Cridar shell script")
    # Executem el codi de l'avaluació
    os.system(f"run_tsar_eval.sh {nom_avaluacio}")

def main():
    print("Executant el codi\n")

    # Carreguem els recursos
    nlp = spacy.load("ca_core_news_sm")
    frequencies = obtenir_frequencia("resources/cat/diccionarifreq/SUBTLEX-CAT_form_zipf.txt")
    percentatges = obtenir_prevalenca_lexica("resources/cat/wordprevalence/word-prevalence.txt")
    unmasker = pipeline('fill-mask', model='projecte-aina/roberta-base-ca-v2')
    print("Llegint els vectors\n")
    diccionari_vectors = fasttext.load_model("resources/cat/fasttext/cc.ca.300.bin")

    while True:
        print("\n--------------------\n0) Avaluar\n1) Simplificar text\n2) Sortir\n--------------------")
        try:
            accio = int(input("Tria una opció: "))
            if accio == 0:
                avaluacio(nlp, frequencies, percentatges, unmasker, diccionari_vectors)
            elif accio == 1:
                text = input("Introdueix el text a simplificar: ")
                text_simplificat = simplificar_text(text, nlp, frequencies, percentatges, unmasker, diccionari_vectors)
                print("El text simplificat és:", text_simplificat)
            elif accio == 2:
                exit()
            else:
                print("Opció no vàlida. Torna-ho a intentar.")
        except ValueError:
            print("Entrada errònia. Si us plau, introdueix una de les opcions.")
          

if __name__ == "__main__":
    main()