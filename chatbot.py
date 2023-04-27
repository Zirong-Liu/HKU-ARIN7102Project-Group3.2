import numpy as np
import json
import os
import pickle
import nltk
from rank_bm25 import BM25Okapi
from transformers import pipeline


def start_chat(query=''):
    # Check if the 'punkt' resource is present in the NLTK data directory
    if not os.path.exists(os.path.join(nltk.data.path[0], 'tokenizers/punkt')):
        nltk.download('punkt')

    def create_temp_json_file(context, question):
        # Create the temp directory if it doesn't exist
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        data = {
            "version": "temp",
            "data": [
                {
                    "title": f"Temp_Context",
                    "paragraphs": [
                        {
                            "context": context,
                            "qas": [
                                {
                                    "question": question,
                                    "id": f"0",
                                    "answers": [
                                        {
                                            "text": "",
                                            "answer_start": 0
                                        }
                                    ],
                                    "is_impossible": False,
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        # Save the JSON file to the temp directory
        file_name = os.path.join(temp_dir, f"context.json")
        with open(file_name, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    '''
    # corpus format example
    corpus = [
        "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.",
        "The Norman dynasty had a major political, cultural and military impact on medieval Europe and even the Near East. The Normans were famed for their martial spirit and eventually for their Christian piety, becoming exponents of the Catholic orthodoxy into which they assimilated. They adopted the Gallo-Romance language of the Frankish land they settled, their dialect becoming known as Norman, Normaund or Norman French, an important literary language. The Duchy of Normandy, which they formed by treaty with the French crown, was a great fief of medieval France, and under Richard I of Normandy was forged into a cohesive and formidable principality in feudal tenure. The Normans are noted both for their culture, such as their unique Romanesque architecture and musical traditions, and for their significant military accomplishments and innovations. Norman adventurers founded the Kingdom of Sicily under Roger II after conquering southern Italy on the Saracens and Byzantines, and an expedition on behalf of their duke, William the Conqueror, led to the Norman conquest of England at the Battle of Hastings in 1066. Norman cultural and military influence spread from these new European centres to the Crusader states of the Near East, where their prince Bohemond I founded the Principality of Antioch in the Levant, to Scotland and Wales in Great Britain, to Ireland, and to the coasts of north Africa and the Canary Islands.",
        "In the course of the 10th century, the initially destructive incursions of Norse war bands into the rivers of France evolved into more permanent encampments that included local women and personal property. The Duchy of Normandy, which began in 911 as a fiefdom, was established by the treaty of Saint-Clair-sur-Epte between King Charles III of West Francia and the famed Viking ruler Rollo, and was situated in the former Frankish kingdom of Neustria. The treaty offered Rollo and his men the French lands between the river Epte and the Atlantic coast in exchange for their protection against further Viking incursions. The area corresponded to the northern part of present-day Upper Normandy down to the river Seine, but the Duchy would eventually extend west beyond the Seine. The territory was roughly equivalent to the old province of Rouen, and reproduced the Roman administrative structure of Gallia Lugdunensis II (part of the former Gallia Lugdunensis).",
        "Before Rollo's arrival, its populations did not differ from Picardy or the \u00cele-de-France, which were considered \"Frankish\". Earlier Viking settlers had begun arriving in the 880s, but were divided between colonies in the east (Roumois and Pays de Caux) around the low Seine valley and in the west in the Cotentin Peninsula, and were separated by traditional pagii, where the population remained about the same with almost no foreign settlers. Rollo's contingents who raided and ultimately settled Normandy and parts of the Atlantic coast included Danes, Norwegians, Norse\u2013Gaels, Orkney Vikings, possibly Swedes, and Anglo-Danes from the English Danelaw under Norse control.",
        "The descendants of Rollo's Vikings and their Frankish wives would replace the Norse religion and Old Norse language with Catholicism (Christianity) and the Gallo-Romance language of the local people, blending their maternal Frankish heritage with Old Norse traditions and customs to synthesize a unique \"Norman\" culture in the north of France. The Norman language was forged by the adoption of the indigenous langue d'o\u00efl branch of Romance by a Norse-speaking ruling class, and it developed into the regional language that survives today.",
        "The Normans thereafter adopted the growing feudal doctrines of the rest of France, and worked them into a functional hierarchical system in both Normandy and in England. The new Norman rulers were culturally and ethnically distinct from the old French aristocracy, most of whom traced their lineage to Franks of the Carolingian dynasty. Most Norman knights remained poor and land-hungry, and by 1066 Normandy had been exporting fighting horsemen for more than a generation. Many Normans of Italy, France and England eventually served as avid Crusaders under the Italo-Norman prince Bohemund I and the Anglo-Norman king Richard the Lion-Heart.",
        "Soon after the Normans began to enter Italy, they entered the Byzantine Empire and then Armenia, fighting against the Pechenegs, the Bulgars, and especially the Seljuk Turks. Norman mercenaries were first encouraged to come to the south by the Lombards to act against the Byzantines, but they soon fought in Byzantine service in Sicily. They were prominent alongside Varangian and Lombard contingents in the Sicilian campaign of George Maniaces in 1038\u201340. There is debate whether the Normans in Greek service actually were from Norman Italy, and it now seems likely only a few came from there. It is also unknown how many of the \"Franks\", as the Byzantines called them, were Normans and not other Frenchmen.",
        "Several families of Byzantine Greece were of Norman mercenary origin during the period of the Comnenian Restoration, when Byzantine emperors were seeking out western European warriors. The Raoulii were descended from an Italo-Norman named Raoul, the Petraliphae were descended from a Pierre d'Aulps, and that group of Albanian clans known as the Maniakates were descended from Normans who served under George Maniaces in the Sicilian expedition of 1038.",
        "A few years after the First Crusade, in 1107, the Normans under the command of Bohemond, Robert's son, landed in Valona and besieged Dyrrachium using the most sophisticated military equipment of the time, but to no avail. Meanwhile, they occupied Petrela, the citadel of Mili at the banks of the river Deabolis, Gllavenica (Ballsh), Kanina and Jericho. This time, the Albanians sided with the Normans, dissatisfied by the heavy taxes the Byzantines had imposed upon them. With their help, the Normans secured the Arbanon passes and opened their way to Dibra. The lack of supplies, disease and Byzantine resistance forced Bohemond to retreat from his campaign and sign a peace treaty with the Byzantines in the city of Deabolis.",
        "The Normans were in contact with England from an early date. Not only were their original Viking brethren still ravaging the English coasts, they occupied most of the important ports opposite England across the English Channel. This relationship eventually produced closer ties of blood through the marriage of Emma, sister of Duke Richard II of Normandy, and King Ethelred II of England. Because of this, Ethelred fled to Normandy in 1013, when he was forced from his kingdom by Sweyn Forkbeard. His stay in Normandy (until 1016) influenced him and his sons by Emma, who stayed in Normandy after Cnut the Great's conquest of the isle.",
        "One of the claimants of the English throne opposing William the Conqueror, Edgar Atheling, eventually fled to Scotland. King Malcolm III of Scotland married Edgar's sister Margaret, and came into opposition to William who had already disputed Scotland's southern borders. William invaded Scotland in 1072, riding as far as Abernethy where he met up with his fleet of ships. Malcolm submitted, paid homage to William and surrendered his son Duncan as a hostage, beginning a series of arguments as to whether the Scottish Crown owed allegiance to the King of England.",
        "Even before the Norman Conquest of England, the Normans had come into contact with Wales. Edward the Confessor had set up the aforementioned Ralph as earl of Hereford and charged him with defending the Marches and warring with the Welsh. In these original ventures, the Normans failed to make any headway into Wales."
    ]
    '''
    if not os.path.exists('corpus.pkl'):
        with open('dataset/medquad.json') as f:
            data = json.load(f)
        data = data['data']
        corpus = []
        for i in range(len(data)):
            para = data[i]['paragraphs']
            for j in range(len(para)):
                corpus.append(para[j]['context'])
        # print(len(corpus))

        tokenized_corpus = [list(nltk.word_tokenize(doc)) for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)

        with open('dataset/corpus.pkl', 'wb') as f_corpus:
            pickle.dump(corpus, f_corpus)
        with open('dataset/bm25.pkl', 'wb') as f_bm25:
            pickle.dump(bm25, f_bm25)

    else:
        with open('corpus.pkl', 'rb') as f_corpus:
            corpus = pickle.load(f_corpus)
        with open('bm25.pkl', 'rb') as f_bm25:
            bm25 = pickle.load(f_bm25)

    tokenized_query = list(nltk.word_tokenize(query))

    doc_scores = bm25.get_scores(tokenized_query)

    # print ('scores of corpus:', doc_scores)

    # retrieve number of corpus k
    k = 1
    best_docs = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]

    if doc_scores[best_docs[0]] < 17:
        print(
            "Answer: I'm sorry, but I'm having trouble understanding your question. Could you please rephrase it or "
            "provide more context? I'll do my best to help you.")
        return "I'm sorry, but I'm having trouble understanding your question. Could you please rephrase it or " \
               "provide more context? I'll do my best to help you. "
    create_temp_json_file(corpus[best_docs[0]], query)

    command = "python main.py --train_file dataset/train.json --predict_file temp/context.json --model_type bert " \
              "--model_name_or_path output/  --output_dir output/eval/ --max_seq_length 512 --max_query_length 60 " \
              "--max_answer_length 450 --do_eval --do_lower_case --overwrite_output --save_steps 0"
    os.system(command)
    with open('output/eval/predictions_.json') as f:
        data = json.load(f)
    answer = data["0"]
    print("Answer:", answer)
    return answer
